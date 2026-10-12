"""Bit-reproducibility of perception training across processes.

Regression test for the 2026-10-10 nondeterminism (LAB_NOTEBOOK.md): exp05
gave first-epoch train loss 2.0872 vs 2.4561 for the same seed and config.
Cause: `stack_scenarios` built its output dict by iterating a `set` of
node-type strings, so key order depended on PYTHONHASHSEED (randomized per
process); HGTConv(in_channels=-1) lazily initializes per-type weights in
x_dict order, so the initial weights differed between processes.

An in-process repeat cannot catch this (both runs share one hash seed), so
each training run here is a fresh subprocess with a different PYTHONHASHSEED.
Training goes through exp05's own `train_model` on small synthetic inputs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_TRAIN_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/experiments")
import numpy as np, torch
import exp05_perception as E
from src.perception.encoder import TARGETS, EncoderConfig, PerceptionEncoder
from tests.test_perception_encoder import COUNTS, F_DIMS, NODE_TYPES, _edge_index_dict

S, N_SCEN = 12, 4
E.set_all_seeds(0)
gen = torch.Generator().manual_seed(0)
scenarios = []
for i in range(N_SCEN):
    x = {t: torch.randn(S, COUNTS[t], F_DIMS[t], generator=gen) for t in NODE_TYPES}
    labels = {t: (torch.rand(S, generator=gen) < 0.5).float() for t in TARGETS}
    mask = {t: torch.ones(S, dtype=torch.bool) for t in TARGETS}
    scenarios.append(E.ScenarioData(
        run_id=i, discrete=None, x_dict=x, globals_=torch.randn(S, 6, generator=gen),
        labels=labels, mask=mask, violations=[],
    ))
ei = _edge_index_dict(np.random.default_rng(0))
torch.manual_seed(123)
model = PerceptionEncoder(EncoderConfig(
    node_types=tuple(NODE_TYPES), edge_types=tuple(ei), hidden=16, heads=2,
    n_gnn_layers=2, tcn_dilations=(1, 2),
))
cfg = {"learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 2, "n_epochs": 2,
       "early_stopping_patience_epochs": 10, "grad_clip_norm": 1.0}
_, rows = E.train_model(model, scenarios[:3], scenarios[3:], ei, cfg, seed=123)
print("ROWS=" + json.dumps([[r["train_loss"], r["val_loss"]] for r in rows]))
"""


def _train_in_subprocess(hash_seed: int) -> list[list[float]]:
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed)}
    proc = subprocess.run(
        [sys.executable, "-c", _TRAIN_SCRIPT, str(REPO_ROOT)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(l for l in proc.stdout.splitlines() if l.startswith("ROWS="))
    return json.loads(line[len("ROWS="):])


def test_two_epoch_losses_identical_across_processes_and_hash_seeds():
    # Hash seeds 0 and 1 give different set orders for these five node types
    # (verified: the pre-fix code produced different losses for this pair).
    a = _train_in_subprocess(0)
    b = _train_in_subprocess(1)
    assert len(a) == len(b) == 2
    assert a == b, f"losses differ across processes: {a} vs {b}"
