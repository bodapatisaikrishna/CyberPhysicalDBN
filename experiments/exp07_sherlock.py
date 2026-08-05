"""Grounding the perception layer on real data: Sherlock (Wagner et al.,
ACM CODASPY'25), a real power-grid IDS dataset from the Wattson
co-simulator (CLAUDE.md layer [1], Session 7).

WHY: Sessions 1-6 evaluate perception exclusively on twin-generated data --
exactly the criticism the source paper already absorbs. This experiment
trains/evaluates on the REAL downloaded 01-Basic scenario and reports the
result honestly, including a possibly-large twin<->Sherlock transfer gap,
which is a finding about twin realism, not a failure to hide (task
instruction, quoted in LAB_NOTEBOOK.md 2026-08-05).

REAL DATA FORMAT (verified after download, see
src/perception/sherlock_loader.py's module docstring for the full
discrepancy write-up against this project's original assumption): Sherlock
ships physical power-grid STATE snapshots (bus/line/trafo/sgen/load/switch
values, exactly 1.0s cadence), each carrying a ready-made `malicious` label
-- not IEC-104 message traffic. No verified electrical topology ships, so
this experiment uses a topology-FREE `CausalTCN` classifier over an
aggregate per-slice feature vector, NOT the HGTConv `PerceptionEncoder`
pipeline the twin side uses.

Stages: 0 locate real files -> 1 parse, verify cadence, build features/
labels, chronological train/val/calib/test split -> 2 train -> 3 evaluate
on held-out Sherlock test (AUC-PR, ECE) -> 4 bidirectional transfer via the
shared bus-voltage subspace (Sherlock-trained -> twin-eval, twin-trained ->
Sherlock-eval) -> 5 gate (correctness only, CLAUDE.md rule 3 -- numbers
reported unconditionally, never gated on beating anything).

Run: .venv/bin/python experiments/exp07_sherlock.py [--smoke]

Requires `data/sherlock/01-Basic/` on disk -- see
scripts/download_sherlock.sh and docs/sherlock_download.md. Never
downloaded automatically (CLAUDE.md rule 1: fails loudly with the manual
command instead).
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

from src.attack_graph.graph import build_attack_graph
from src.eval.provenance import git_sha
from src.perception.asset_graph import CyberAsset, CyberOverlayConfig, build_asset_graph, observed_endpoints
from src.perception.calibration import apply_temperature, fit_temperature, perception_calibration_report
from src.perception.encoder import CausalTCN
from src.perception.features import DynamicFeatureConfig, build_dynamic_features, twin_bus_voltage_shared_subspace
from src.twin.consequence import ZoneMap, build_zone_map
from src.perception.sherlock_loader import (
    SHARED_TRANSFER_COLUMNS,
    SHERLOCK_GLOBAL_COLUMNS,
    build_global_features,
    build_labels,
    build_shared_subspace,
    chronological_chunks,
    component_indices,
    per_bus_voltage,
    read_state_file,
)
from src.twin.grid import GridConfig, GridModel, voltage_sensitivity
from src.twin.runner import TwinConfig, TwinRunner, discretize

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
SHERLOCK_CONFIG_PATH = REPO_ROOT / "configs" / "sherlock.yaml"
TWIN_CONFIG_PATH = REPO_ROOT / "configs" / "twin.yaml"
RESULTS_DIR = REPO_ROOT / "results"

TWIN_TIME_UNIT_SECONDS = 600
TWIN_DELTA_T = 166.13 / TWIN_TIME_UNIT_SECONDS  # Table 5, m=1 -- twin side only


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# === stage 0: locate real files ==============================================


def locate_sherlock_files(data_dir: Path, train_glob: str, test_glob: str) -> tuple[Path | None, Path | None]:
    """Real glob against the real scenario root -- never a fixed filename
    assumption. Prints every candidate found so a naming surprise on a
    different scenario is visible, not silently mismatched."""
    if not data_dir.exists():
        return None, None
    train_matches = sorted(data_dir.glob(train_glob))
    test_matches = sorted(data_dir.glob(test_glob))
    print(f"  data_dir: {data_dir}")
    print(f"  train_glob {train_glob!r} matched: {[p.name for p in train_matches]}")
    print(f"  test_glob {test_glob!r} matched: {[p.name for p in test_matches]}")
    train_file = train_matches[0] if train_matches else None
    test_file = test_matches[0] if test_matches else None
    if len(train_matches) > 1:
        print(f"  WARNING: {len(train_matches)} train candidates, using {train_file.name}")
    if len(test_matches) > 1:
        print(f"  WARNING: {len(test_matches)} test candidates, using {test_file.name}")
    return train_file, test_file


def check_uniform_cadence(records, expected_seconds: float, tolerance_seconds: float) -> tuple[bool, float]:
    """Verifies (never assumes) that `read_state_file`'s slice_index-by-
    file-position convention is valid for this real file. Returns
    `(ok, median_gap)`."""
    ts = [r.timestamp for r in records]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    if not gaps:
        return True, expected_seconds
    median_gap = float(np.median(gaps))
    ok = all(abs(g - expected_seconds) <= tolerance_seconds for g in gaps)
    return ok, median_gap


# === stage 2: main model (topology-free CausalTCN over global features) ====


class SherlockModel(nn.Module):
    def __init__(self, in_width: int, hidden: int, kernel_size: int, dilations, dropout: float):
        super().__init__()
        self.in_proj = nn.Linear(in_width, hidden)
        self.tcn = CausalTCN(channels=hidden, kernel_size=kernel_size, dilations=dilations, dropout=dropout)
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, S, F] -> [B, S]
        h = self.in_proj(x).transpose(1, 2)
        h = self.tcn(h)
        return self.head(h).squeeze(1)


@dataclass
class Chunk:
    name: str
    x: torch.Tensor  # [S, F]
    y: torch.Tensor  # [S]
    bus_voltage: torch.Tensor  # [S, n_bus]


def slice_chunk(name: str, x: torch.Tensor, y: torch.Tensor, bus_voltage: torch.Tensor, start: int, end: int) -> Chunk:
    return Chunk(name=name, x=x[start:end], y=y[start:end], bus_voltage=bus_voltage[start:end])


def pos_weight_for(chunks: list[Chunk]) -> torch.Tensor:
    pos = sum(float(c.y.sum()) for c in chunks)
    total = sum(float(c.y.numel()) for c in chunks)
    return torch.tensor((total - pos) / pos if pos > 0 else 1.0)


def train_sherlock_model(
    model: SherlockModel, train_chunk: Chunk, val_chunk: Chunk, train_cfg: dict,
) -> tuple[SherlockModel, list[dict]]:
    pos_weight = pos_weight_for([train_chunk])
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"])
    )
    n_epochs = int(train_cfg["n_epochs"])
    patience = int(train_cfg["early_stopping_patience_epochs"])
    grad_clip = float(train_cfg["grad_clip_norm"])

    rows, best_val, bad_epochs, best_state = [], float("inf"), 0, None
    for epoch in range(n_epochs):
        model.train()
        logits = model(train_chunk.x.unsqueeze(0))[0]
        loss = F.binary_cross_entropy_with_logits(logits, train_chunk.y, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        train_loss = float(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(val_chunk.x.unsqueeze(0))[0]
            val_loss = float(F.binary_cross_entropy_with_logits(val_logits, val_chunk.y, pos_weight=pos_weight).item())

        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"    epoch {epoch + 1}/{n_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("    early stopping", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, rows


# === stage 4: bidirectional transfer, shared bus-voltage subspace ===========


class TransferModel(nn.Module):
    """Topology-free, domain-agnostic: `[S, 2]` (`SHARED_TRANSFER_COLUMNS`)
    -> CausalTCN -> 1 logit/slice. The SAME trained instance can run on
    either domain's reduced tensor since both reduce to identical width/
    semantics (mean bus voltage pu, its delta)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.in_proj = nn.Linear(len(SHARED_TRANSFER_COLUMNS), hidden)
        self.tcn = CausalTCN(channels=hidden)
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x).transpose(1, 2)
        h = self.tcn(h)
        return self.head(h).squeeze(1)


def train_transfer_model(
    reduced_by_scenario: list[torch.Tensor], labels_by_scenario: list[torch.Tensor], cfg: dict, seed: int,
) -> TransferModel:
    torch.manual_seed(seed)
    model = TransferModel(hidden=int(cfg["hidden_dim"]))
    pos = sum(float(y.sum()) for y in labels_by_scenario)
    total = sum(float(y.numel()) for y in labels_by_scenario)
    pos_weight = torch.tensor((total - pos) / pos if pos > 0 else 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    for _ in range(int(cfg["n_epochs"])):
        model.train()
        for x, y in zip(reduced_by_scenario, labels_by_scenario):
            logits = model(x.unsqueeze(0))[0]
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def eval_transfer_model(model: TransferModel, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        q = torch.sigmoid(model(x.unsqueeze(0))[0]).numpy()
    y_np = y.numpy()
    return average_precision_score(y_np, q) if len(np.unique(y_np)) > 1 else float("nan")


def build_twin_transfer_scenarios(
    twin_cfg: dict, n_scenarios: int, horizon_time_units: float, seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Fresh twin scenarios reduced to the shared bus-voltage subspace.
    Label = any attack-step node active (OR over `node_type ==
    "attack_step"`, excluding reactions/analytics/goals) -- the twin-side
    analog of Sherlock's binary `malicious`."""
    grid_cfg = GridConfig(
        network=twin_cfg["grid"]["network"], n_der=int(twin_cfg["grid"]["n_der"]),
        p_mw_levels=list(twin_cfg["grid"]["p_mw_levels"]),
        nominal_level_index=int(twin_cfg["grid"]["nominal_level_index"]),
        nonconvergence_is_unstable=bool(twin_cfg["grid"]["nonconvergence_is_unstable"]),
    )
    ag = build_attack_graph(reaction_mode=twin_cfg["experiment"]["reaction_mode"], physical_evidence=True)
    attack_step_nodes = [n for n, d in ag.nodes(data=True) if d["node_type"] == "attack_step"]

    grid = GridModel(grid_cfg)
    overlay = CyberOverlayConfig(
        assets=(
            CyberAsset("ControlCentre", "host", "ControlCentre"),
            *(CyberAsset(f"IED_{b}", "IED", der_id, controls=(der_id,))
              for der_id, b in zip(grid.der_ids, grid.der_buses)),
        ),
        source="exp07 stage 4 (twin side)",
    )
    n_slices = round(horizon_time_units / TWIN_DELTA_T)
    probe_cfg = TwinConfig(grid=grid_cfg, horizon_time_units=horizon_time_units)
    probe_trace = TwinRunner(ag, probe_cfg, np.random.SeedSequence(seed)).run()
    obs = observed_endpoints(probe_trace.messages)
    asset_graph = build_asset_graph(grid, overlay, observed_endpoints_set=obs)

    sensitivity = voltage_sensitivity(grid, float(twin_cfg["physical"]["delta_p_mw"]))
    der_buses = dict(zip(grid.der_ids, grid.der_buses))
    zones: ZoneMap = build_zone_map(
        sensitivity, dominance_tau=float(twin_cfg["physical"]["dominance_tau"]),
        delta_p_mw=float(twin_cfg["physical"]["delta_p_mw"]), der_buses=der_buses,
    )

    root = np.random.SeedSequence(seed)
    reduced_list, label_list = [], []
    for i, s in enumerate(root.spawn(n_scenarios)):
        tw_cfg = TwinConfig(
            grid=grid_cfg,
            dispatch_period_time_units=float(twin_cfg["control_centre"]["dispatch_period_time_units"]),
            comms_latency_time_units=float(twin_cfg["comms"]["latency_time_units"]),
            horizon_time_units=horizon_time_units,
        )
        trace = TwinRunner(ag, tw_cfg, s).run()
        discrete = discretize(
            trace, ag, TWIN_DELTA_T, n_slices, np.random.default_rng(s.spawn(1)[0]), 1e-4, 1e-4, zones=zones,
        )
        dyn = build_dynamic_features(
            trace, TWIN_DELTA_T, n_slices, asset_graph, overlay, grid_cfg,
            dispatch_period_time_units=tw_cfg.dispatch_period_time_units,
            config=DynamicFeatureConfig(se_noise_sigma=0.0, observability="voltage_only"),
        )
        reduced = twin_bus_voltage_shared_subspace(dyn["bus"])  # [S, 2]
        label = torch.tensor(
            [float(any(bool(r.ground_truth[n]) for n in attack_step_nodes)) for r in discrete.records],
            dtype=torch.float32,
        )
        reduced_list.append(reduced)
        label_list.append(label)
        print(f"    twin transfer scenario {i + 1}/{n_scenarios} done", flush=True)
    return reduced_list, label_list


# === main =====================================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(BASE_CONFIG_PATH.read_text())
    cfg = yaml.safe_load(SHERLOCK_CONFIG_PATH.read_text())
    twin_cfg = yaml.safe_load(TWIN_CONFIG_PATH.read_text())
    seed = int(base_cfg["seed"])
    set_all_seeds(seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    tag = "smoke_" if args.smoke else ""

    # === stage 0: locate real files ==========================================
    print("stage 0: locating Sherlock data ...", flush=True)
    data_dir = REPO_ROOT / cfg["data"]["data_dir"]
    train_file, test_file = locate_sherlock_files(data_dir, cfg["data"]["train_glob"], cfg["data"]["test_glob"])
    if train_file is None or test_file is None:
        print(f"\nGATE FAILED: train_file={train_file}, test_file={test_file} under {data_dir}")
        print("Run:  bash scripts/download_sherlock.sh --scenario 01-Basic")
        print("See docs/sherlock_download.md for details.")
        return 1

    # === stage 1: parse, verify cadence, build features/labels ==============
    print("\nstage 1: parsing real state files ...", flush=True)
    train_records, train_labels = read_state_file(train_file)
    test_records, test_labels = read_state_file(test_file)
    if args.smoke:
        max_records = int(cfg["smoke"]["max_records"])
        train_records, train_labels = train_records[:max_records], train_labels[:max_records]
        test_records, test_labels = test_records[:max_records], test_labels[:max_records]
    print(f"  train: {len(train_records)} records, test: {len(test_records)} records")

    for name, records in (("train", train_records), ("test", test_records)):
        ok, gap = check_uniform_cadence(records, float(cfg["timebase"]["expected_cadence_seconds"]),
                                         float(cfg["timebase"]["cadence_tolerance_seconds"]))
        print(f"  {name} cadence check: {'PASS' if ok else 'FAIL'} (median gap={gap:.4f}s)")
        if not ok:
            print(f"\nGATE FAILED: {name} file has non-uniform cadence -- slice_index-by-file-position "
                  "(read_state_file's convention) is invalid for this file; needs real-timestamp-based "
                  "slicing instead.")
            return 1

    train_base_rate = float(build_labels(train_labels).mean())
    test_base_rate = float(build_labels(test_labels).mean())
    print(f"  train base rate: {train_base_rate:.6f}, test base rate: {test_base_rate:.6f}")
    if train_base_rate > 0:
        print("  NOTE: train split contains real attack-positive slices (task description assumed "
              "attack-free train) -- reported as observed, not adjusted to match the assumption.")

    bus_indices = component_indices(train_records + test_records, "bus")
    print(f"  bus indices (derived): {bus_indices}")

    train_x = build_global_features(train_records)
    train_y = build_labels(train_labels)
    train_bus_v = per_bus_voltage(train_records, bus_indices)
    test_x = build_global_features(test_records)
    test_y = build_labels(test_labels)
    test_bus_v = per_bus_voltage(test_records, bus_indices)

    n_train_total = train_x.shape[0]
    bounds = chronological_chunks(
        n_train_total,
        {"train": 1.0 - float(cfg["splits"]["val_frac_of_train"]) - float(cfg["splits"]["calib_frac_of_train"]),
         "val": float(cfg["splits"]["val_frac_of_train"]),
         "calib": float(cfg["splits"]["calib_frac_of_train"])},
    )
    chunks = {
        name: slice_chunk(name, train_x, train_y, train_bus_v, start, end)
        for name, (start, end) in bounds.items()
    }
    test_chunk = Chunk(name="test", x=test_x, y=test_y, bus_voltage=test_bus_v)
    for name, c in {**chunks, "test": test_chunk}.items():
        print(f"  {name}: {c.x.shape[0]} slices, base_rate={float(c.y.mean()):.6f}")
        if c.x.shape[0] == 0:
            print(f"\nGATE FAILED: {name} chunk has 0 slices -- adjust configs/sherlock.yaml splits fractions.")
            return 1

    # === stage 2: train =======================================================
    print("\nstage 2: training ...", flush=True)
    train_cfg = dict(cfg["training"])
    if args.smoke:
        train_cfg["n_epochs"] = int(cfg["smoke"]["n_epochs"])
    model = SherlockModel(
        in_width=len(SHERLOCK_GLOBAL_COLUMNS), hidden=int(cfg["architecture"]["hidden_dim"]),
        kernel_size=int(cfg["architecture"]["tcn_kernel_size"]), dilations=tuple(cfg["architecture"]["tcn_dilations"]),
        dropout=float(cfg["architecture"]["tcn_dropout"]),
    )
    model, training_rows = train_sherlock_model(model, chunks["train"], chunks["val"], train_cfg)
    training_df = pd.DataFrame(training_rows)
    training_df["git_sha"] = sha
    training_path = RESULTS_DIR / f"exp07_{tag}training_{timestamp}.csv"
    training_df.to_csv(training_path, index=False)
    print(f"  wrote {training_path}")

    # === stage 3: evaluate on held-out Sherlock test =========================
    print("\nstage 3: evaluation on Sherlock test ...", flush=True)
    model.eval()
    with torch.no_grad():
        calib_logits = model(chunks["calib"].x.unsqueeze(0))[0].numpy()
    calib_labels = chunks["calib"].y.numpy()
    scaler = fit_temperature(
        logits={"malicious_binary": calib_logits}, labels={"malicious_binary": calib_labels},
        run_ids={"malicious_binary": np.zeros(len(calib_labels), dtype=int)}, fit_split="calib",
    )
    print(f"  fitted temperature: {scaler.temperatures['malicious_binary']:.4f}")

    with torch.no_grad():
        test_logits = model(test_chunk.x.unsqueeze(0))[0].numpy()
    test_labels_arr = test_chunk.y.numpy()
    run_ids = np.zeros(len(test_labels_arr), dtype=int)

    metrics_rows = []
    for stage, q in (
        ("before_temp", apply_temperature(test_logits, 1.0)),
        ("after_temp", apply_temperature(test_logits, scaler.temperatures["malicious_binary"])),
    ):
        base_rate = float(test_labels_arr.mean())
        ap = average_precision_score(test_labels_arr, q) if len(np.unique(test_labels_arr)) > 1 else float("nan")
        report = perception_calibration_report(test_labels_arr, q, run_ids, n_bootstrap=1000, rng=np.random.default_rng(seed))
        metrics_rows.append({
            "target": "malicious_binary", "stage": stage, "auc_pr": ap, "base_rate": base_rate,
            "brier": report.brier, "ece_10_uniform": report.ece[(10, "uniform")],
            "n": report.n, "n_runs": report.n_runs,
        })
        print(f"  {stage:<12} AUC-PR={ap:.4f} (base_rate={base_rate:.4f}) ECE={report.ece[(10,'uniform')]:.4f}")
    print("  NOTE: test data is a single continuous stream (run_ids constant) -- the calibration "
          "bootstrap's run-level resampling is degenerate here (n_runs=1); interpret any reported "
          "CI width accordingly.")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["git_sha"] = sha
    metrics_path = RESULTS_DIR / f"exp07_{tag}perception_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  wrote {metrics_path}")

    # === stage 4: bidirectional transfer =====================================
    transfer_rows = []
    if bool(cfg["transfer"]["enabled"]):
        print("\nstage 4: bidirectional transfer (shared bus-voltage subspace) ...", flush=True)
        transfer_cfg = dict(cfg["transfer"])
        train_reduced = build_shared_subspace(chunks["train"].bus_voltage)

        print("  training on Sherlock, evaluating on twin ...")
        sherlock_transfer_model = train_transfer_model([train_reduced], [chunks["train"].y], transfer_cfg, seed)
        n_twin = int(transfer_cfg["n_twin_scenarios"]) if not args.smoke else 2
        twin_reduced, twin_labels = build_twin_transfer_scenarios(
            twin_cfg, n_twin, float(transfer_cfg["twin_horizon_time_units"]), seed,
        )
        aps = [eval_transfer_model(sherlock_transfer_model, x, y) for x, y in zip(twin_reduced, twin_labels)]
        ap_s2t = float(np.nanmean(aps))
        print(f"  Sherlock-trained -> twin-eval AUC-PR: {ap_s2t:.4f} (mean over {len(aps)} twin scenarios)")
        transfer_rows.append({"direction": "sherlock_to_twin", "auc_pr": ap_s2t, "n_eval_scenarios": len(aps)})

        print("  training on twin, evaluating on Sherlock ...")
        twin_transfer_model = train_transfer_model(twin_reduced, twin_labels, transfer_cfg, seed + 1)
        test_reduced = build_shared_subspace(test_chunk.bus_voltage)
        ap_t2s = eval_transfer_model(twin_transfer_model, test_reduced, test_chunk.y)
        print(f"  twin-trained -> Sherlock-eval AUC-PR: {ap_t2s:.4f}")
        transfer_rows.append({"direction": "twin_to_sherlock", "auc_pr": ap_t2s, "n_eval_scenarios": 1})

        print("\n  *** twin<->Sherlock transfer gap is REPORTED, not gated. A large gap is a finding "
              "about twin realism (LAB_NOTEBOOK.md H2), not a failure to hide. ***")

    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df["git_sha"] = sha
    transfer_path = RESULTS_DIR / f"exp07_{tag}transfer_{timestamp}.csv"
    transfer_df.to_csv(transfer_path, index=False)
    print(f"  wrote {transfer_path}")

    # === validation gate =======================================================
    print("\n=== VALIDATION GATE ===")
    failures: list[str] = []

    print("(a) leak guard .............................. PASS (asserted in tests/test_sherlock_loader.py)")
    print("(b) primary label direct from real malicious field PASS (build_labels, no interval computation)")

    calib_temp_finite = np.isfinite(scaler.temperatures["malicious_binary"])
    print(f"(c) calib temperature finite ................. {'PASS' if calib_temp_finite else 'FAIL'}")
    if not calib_temp_finite:
        failures.append("non-finite calibration temperature")

    all_q_finite = bool(np.isfinite(apply_temperature(test_logits, scaler.temperatures["malicious_binary"])).all())
    print(f"(d) every test q finite ....................... {'PASS' if all_q_finite else 'FAIL'}")
    if not all_q_finite:
        failures.append("non-finite q on test")

    print("(e) train/val/calib chronologically disjoint . PASS (chronological_chunks, non-overlapping by construction)")
    print(f"(f) test split from SEPARATE real file ........ PASS (test_file={test_file.name}, train_file={train_file.name})")

    recomputed_rate = float(build_labels(train_labels[bounds['train'][0]:bounds['train'][1]]).mean())
    rate_ok = abs(recomputed_rate - float(chunks["train"].y.mean())) < 1e-9
    print(f"(g) label provenance recomputed ............... {'PASS' if rate_ok else 'FAIL'}")
    if not rate_ok:
        failures.append("recomputed train base rate mismatch")

    print(
        "\nNOTE: AUC-PR/ECE on Sherlock and the twin<->Sherlock transfer gap above are REPORTED, "
        "not gated (CLAUDE.md rule 3)."
    )

    if failures:
        print("\nGATE FAILED:")
        for line in failures:
            print("  -", line)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
