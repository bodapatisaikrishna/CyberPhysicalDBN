"""Scenario 1 KL(EX||FF) under one-shot latched reactions.

Tests the hypothesis recorded in LAB_NOTEBOOK.md (2026-07-31): Cerotti et al.'s
own figures are mutually inconsistent under any memoryless reaction. Fig. 5a's
flat-0.7 CorrReact plateau requires no memory, while Fig. 6c states the FF
divergence "is not able to converge to the exact solution (it stabilizes just
below 0.12)", which requires memory. A one-shot latched reaction satisfies both.

exp01 already showed the memoryless model's CorrReact KL DECAYS to 0,
contradicting Fig. 6c, and that FF-latched CorrReact locks permanently at
0.5079. This script measures the full KL(EX||FF) trajectories so the three
Scenario 1 targets can be checked directly.

Scope: Scenario 1 only, and only to slice MAX_SLICES rather than the paper's
full 722. EX-latched costs ~7.5 s/slice at 2^15 = 32768 joint states, so the
full horizon would be ~90 min for EX alone.

The horizon was initially 250 slices (t~69) on the reasoning that the paper's
Scenario 1 KL peaks all occur early. That was too aggressive: the latched
UnstablePS KL turns out to be TWO-humped -- a large early peak (~0.29 at t~8,
driven by CorrReact) and a second rise beginning around t~40 that was still
climbing at t=69 when the run stopped. The second hump is the ModCtrlLogic ->
WrongLogicExec branch entering, which is exactly where Fig. 6a puts the
paper's peak, and it is invisible to the memoryless model (KL for that branch
is identically 0 there, since WrongLogicExec is computed intra-slice from
ModCtrlLogic and that sub-process is structurally disjoint from the rest).
Extended to t~120 so the second hump's peak is actually captured. Behaviour
past t~120 is still NOT measured.

Run: .venv/bin/python experiments/exp02_latched_kl.py
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from src.attack_graph.graph import build_attack_graph
from src.dbn.inference import (
    DBNInference,
    InferenceConfig,
    exact_clustering,
    fully_factorized_clustering,
)
from src.eval.provenance import git_sha
from src.eval.metrics import binary_kl, m_kl

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
RESULTS_DIR = REPO_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

TIME_UNIT_SECONDS = 600
DELTA_T_OVERRIDE = 166.13 / TIME_UNIT_SECONDS
SLICES_PER_TIME_UNIT = 1.0 / DELTA_T_OVERRIDE
MAX_SLICES = 433  # t ~= 120 time units

M = 1.0
P_POS = 1e-4
P_NEG = 1e-4
QUERY_NODES = ["UnstablePS", "CorrReact", "MITM"]

# Cerotti et al. Figs. 6a/6c/6e, for reference in the printed comparison only.
# Not thresholds -- nothing here adjusts to fit them.
PAPER_TARGETS = {
    "UnstablePS": "peak ~2e-2 around t~40-50",
    "CorrReact": "stabilizes just below 0.12, does NOT converge to 0",
    "MITM": "peak ~1.8e-2 early (t~10), decays to ~0",
}


def run(clustering_name: str):
    ag = build_attack_graph(reaction_mode="latched")
    interface = sorted(n for n, d in ag.nodes(data=True) if d["self_loop"])
    clustering = (
        fully_factorized_clustering(interface)
        if clustering_name == "FF"
        else exact_clustering(interface)
    )
    engine = DBNInference(
        ag,
        InferenceConfig(
            clustering=clustering,
            m=M,
            p_pos=P_POS,
            p_neg=P_NEG,
            delta_t_override=DELTA_T_OVERRIDE,
        ),
    )
    print(f"  running {clustering_name} ({len(interface)} interface nodes) ...", flush=True)
    return engine.run({}, MAX_SLICES)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    SUMMARIES_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)

    ff = run("FF")
    ex = run("EX")

    rows = []
    kl_traj: dict[str, dict[int, float]] = {n: {} for n in QUERY_NODES}
    for i, (ff_m, ex_m) in enumerate(zip(ff.marginals, ex.marginals), start=1):
        t_units = i * DELTA_T_OVERRIDE
        for node in QUERY_NODES:
            kl = binary_kl(ex_m[node], ff_m[node])
            kl_traj[node][i] = kl
            rows.append(
                {
                    "slice": i,
                    "t_units": t_units,
                    "node": node,
                    "p_ex": ex_m[node],
                    "p_ff": ff_m[node],
                    "kl_ex_ff": kl,
                    "reaction_mode": "latched",
                    "git_sha": sha,
                    "seed": seed,
                    "m": M,
                    "delta_t": DELTA_T_OVERRIDE,
                    "p_pos": P_POS,
                    "p_neg": P_NEG,
                    "max_slices": MAX_SLICES,
                }
            )

    frame = pd.DataFrame(rows)
    path = RESULTS_DIR / f"exp02_latched_kl_scenario1_{timestamp}.csv"
    frame.to_csv(path, index=False)
    print(f"\nwrote {path}")

    summary = []
    print("\n=== SCENARIO 1 KL(EX||FF), LATCHED REACTIONS ===")
    print(f"(horizon truncated at slice {MAX_SLICES} = t~{MAX_SLICES*DELTA_T_OVERRIDE:.0f})")
    print(f"{'node':<14}{'M_KL':>12}{'at t':>9}{'final KL':>12}   paper (Fig. 6)")
    for node in QUERY_NODES:
        value, argmax_slice = m_kl(kl_traj[node])
        final = kl_traj[node][MAX_SLICES]
        print(
            f"{node:<14}{value:>12.5g}{argmax_slice*DELTA_T_OVERRIDE:>9.1f}"
            f"{final:>12.5g}   {PAPER_TARGETS[node]}"
        )
        summary.append(
            {
                "scenario": "scenario1",
                "reaction_mode": "latched",
                "node": node,
                "m_kl": value,
                "argmax_t_units": argmax_slice * DELTA_T_OVERRIDE,
                "final_kl": final,
                "paper_reference": PAPER_TARGETS[node],
                "git_sha": sha,
                "seed": seed,
                "m": M,
                "delta_t": DELTA_T_OVERRIDE,
                "max_slices": MAX_SLICES,
            }
        )

    summary_path = SUMMARIES_DIR / f"exp02_latched_kl_summary_{timestamp}.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
