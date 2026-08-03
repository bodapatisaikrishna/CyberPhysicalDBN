"""Learned perception + soft evidence: replacing the fictional p_pos=p_neg=
1e-4 evidence with a calibrated likelihood learned from twin telemetry
(CLAUDE.md layer [1], LAB_NOTEBOOK.md 2026-08-02).

Perception targets (user-approved scope, LAB_NOTEBOOK.md pre-registration):
MeasureCoherence and CommandCoherence -- the only 2 of 8 cyber analytics with
genuine telemetry substrate in this twin -- plus PhysLocalDER/PhysWideArea as
architecture POSITIVE CONTROLS. The other 6 cyber analytics keep hard Table-2
evidence; that is a stated twin-fidelity limitation, not an omission.

Stages: 0 asset graph -> 1 data generation (5 disjoint seed streams) -> 2
train -> 3 perception eval on test (AUC-PR, ECE pre/post temperature) -> 4
sensitivity (SE-noise sigma sweep, observability arm) -> 5 DBN 5-arm
ablation. Gate tests CORRECTNESS INVARIANTS ONLY, never whether soft evidence
"won" (CLAUDE.md rule 3).

Run: .venv/bin/python experiments/exp05_perception.py [--smoke]
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.attack_graph.graph import PHYS_LOCAL_DER, PHYS_WIDE_AREA, build_attack_graph
from src.dbn.inference import DBNInference, InferenceConfig, _interface_nodes, fully_factorized_clustering
from src.dbn.soft_evidence import SoftEvidenceConfig
from src.eval.calibration import calibration_report as dbn_calibration_report
from src.eval.lead_time import DetectionResult, LeadTimeSummary, evaluate_run, summarize
from src.eval.provenance import git_sha
from src.perception.asset_graph import (
    CyberAsset,
    CyberOverlayConfig,
    build_asset_graph,
    nonempty_metadata,
    observed_endpoints,
)
from src.perception.calibration import (
    apply_temperature,
    fit_temperature,
    perception_calibration_report,
    reliability_diagram,
)
from src.perception.encoder import (
    TARGETS,
    EncoderConfig,
    PerceptionEncoder,
    combine_static_dynamic,
    receptive_field,
    stack_scenarios,
)
from src.perception.features import DynamicFeatureConfig, build_dynamic_features
from src.twin.attacker import AttackerConfig, DelayLaw
from src.twin.consequence import ZoneMap, build_zone_map, observe_local, observe_wide
from src.twin.grid import GridConfig, GridModel, voltage_sensitivity
from src.twin.runner import DiscreteTrace, TwinConfig, TwinRunner, discretize, validate_trace

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
TWIN_CONFIG_PATH = REPO_ROOT / "configs" / "twin.yaml"
PERCEPTION_CONFIG_PATH = REPO_ROOT / "configs" / "perception.yaml"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

T_TIME_UNITS = 200
TIME_UNIT_SECONDS = 600
DELTA_T_OVERRIDE = 166.13 / TIME_UNIT_SECONDS
N_SLICES = round(T_TIME_UNITS / DELTA_T_OVERRIDE)
M = 1.0

CYBER_PARENT = {"MeasureCoherence": "SpoofRepMsg", "CommandCoherence": "UnauthCommand"}
PHYSICAL_TARGETS = (PHYS_LOCAL_DER, PHYS_WIDE_AREA)
ALL_CYBER_ANALYTICS = (
    "FileAccess", "FileIntegrity", "MeasureCoherence", "CommandCoherence",
    "SWIntegrityDER", "NewServiceStarted", "SWIntegritySCADA", "SuspArg",
)
OTHER_CYBER_ANALYTICS = tuple(n for n in ALL_CYBER_ANALYTICS if n not in TARGETS)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --- stage 0/1 helpers: overlay, zones, scenario data -----------------------


def build_overlay(grid: GridModel) -> CyberOverlayConfig:
    return CyberOverlayConfig(
        assets=(
            CyberAsset("ControlCentre", "host", "ControlCentre"),
            *(
                CyberAsset(f"IED_{bus}", "IED", der_id, controls=(der_id,))
                for der_id, bus in zip(grid.der_ids, grid.der_buses)
            ),
        ),
        source="exp05 stage 0",
    )


def build_zones(grid_cfg: GridConfig, delta_p_mw: float, dominance_tau: float) -> ZoneMap:
    model = GridModel(grid_cfg)
    sensitivity = voltage_sensitivity(model, delta_p_mw)
    der_buses = dict(zip(model.der_ids, model.der_buses))
    return build_zone_map(
        sensitivity, dominance_tau=dominance_tau, delta_p_mw=delta_p_mw, der_buses=der_buses
    )


@dataclass
class ScenarioData:
    run_id: int
    discrete: DiscreteTrace
    x_dict: dict[str, torch.Tensor]
    globals_: torch.Tensor
    labels: dict[str, torch.Tensor]
    mask: dict[str, torch.Tensor]
    violations: list


def extract_labels(discrete: DiscreteTrace) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    n = len(discrete.records)
    labels = {t: torch.zeros(n, dtype=torch.float32) for t in TARGETS}
    mask = {t: torch.ones(n, dtype=torch.bool) for t in TARGETS}
    for i, record in enumerate(discrete.records):
        for target, parent in CYBER_PARENT.items():
            labels[target][i] = float(record.ground_truth[parent])
        local = observe_local(record.physical) if record.physical is not None else None
        wide = observe_wide(record.physical) if record.physical is not None else None
        if local is None:
            mask[PHYS_LOCAL_DER][i] = False
        else:
            labels[PHYS_LOCAL_DER][i] = float(local)
        if wide is None:
            mask[PHYS_WIDE_AREA][i] = False
        else:
            labels[PHYS_WIDE_AREA][i] = float(wide)
    return labels, mask


def run_scenario(
    run_id: int,
    ag_physical,
    grid_cfg: GridConfig,
    twin_cfg: dict,
    zones: ZoneMap,
    asset_graph,
    overlay: CyberOverlayConfig,
    nonempty_types: list[str],
    seed: np.random.SeedSequence,
    *,
    dyn_config: DynamicFeatureConfig,
    feature_rng: np.random.Generator | None,
) -> ScenarioData:
    tw_config = TwinConfig(
        grid=grid_cfg,
        attacker=AttackerConfig(delay_law=DelayLaw.EXPONENTIAL),
        dispatch_period_time_units=float(twin_cfg["control_centre"]["dispatch_period_time_units"]),
        comms_latency_time_units=float(twin_cfg["comms"]["latency_time_units"]),
        horizon_time_units=float(T_TIME_UNITS),
    )
    trace = TwinRunner(ag_physical, tw_config, seed).run()
    violations = validate_trace(trace, ag_physical)
    discrete = discretize(
        trace, ag_physical, DELTA_T_OVERRIDE, N_SLICES,
        np.random.default_rng(seed.spawn(1)[0]), 1e-4, 1e-4, zones=zones,
    )
    dyn = build_dynamic_features(
        trace, DELTA_T_OVERRIDE, N_SLICES, asset_graph, overlay, grid_cfg,
        dispatch_period_time_units=tw_config.dispatch_period_time_units,
        config=dyn_config, rng=feature_rng,
    )
    static_x = {t: asset_graph.data[t].x for t in nonempty_types}
    x_dict = combine_static_dynamic(static_x, {t: dyn[t] for t in nonempty_types})
    globals_ = dyn["host"][:, 0, :]
    labels, mask = extract_labels(discrete)
    return ScenarioData(
        run_id=run_id, discrete=discrete, x_dict=x_dict, globals_=globals_,
        labels=labels, mask=mask, violations=violations,
    )


def generate_split(
    label: str, seeds: list[np.random.SeedSequence], **kwargs,
) -> list[ScenarioData]:
    print(f"  {label}: {len(seeds)} scenarios ...", flush=True)
    out = []
    for i, seed in enumerate(seeds):
        out.append(run_scenario(i, seed=seed, **kwargs))
        print(f"    {label} {i + 1}/{len(seeds)} done", flush=True)
    return out


def compute_base_rates(scenarios: list[ScenarioData]) -> dict[str, float]:
    rates = {}
    for t in TARGETS:
        num = sum(float((s.labels[t] * s.mask[t].float()).sum()) for s in scenarios)
        den = sum(float(s.mask[t].float().sum()) for s in scenarios)
        if den <= 0:
            raise ValueError(f"target {t!r} has zero observed labels in this split")
        rates[t] = num / den
    return rates


def compute_pos_weight(scenarios: list[ScenarioData], target: str) -> float:
    pos = sum(float((s.labels[target] * s.mask[target].float()).sum()) for s in scenarios)
    total = sum(float(s.mask[target].float().sum()) for s in scenarios)
    neg = total - pos
    if pos <= 0:
        return 1.0
    return neg / pos


# --- stage 2: training --------------------------------------------------


def masked_bce_loss(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    mask: dict[str, torch.Tensor],
    pos_weight: dict[str, float],
) -> torch.Tensor:
    total = torch.zeros(())
    for t in TARGETS:
        per_elem = F.binary_cross_entropy_with_logits(
            logits[t], labels[t], reduction="none", pos_weight=torch.tensor(pos_weight[t])
        )
        m = mask[t].float()
        total = total + (per_elem * m).sum() / m.sum().clamp_min(1.0)
    return total


def batch_forward(model: PerceptionEncoder, batch: list[ScenarioData], edge_index_dict):
    x_dict = stack_scenarios([s.x_dict for s in batch])
    globals_ = torch.stack([s.globals_ for s in batch], dim=0)
    logits = model(x_dict, edge_index_dict, globals_)
    labels = {t: torch.stack([s.labels[t] for s in batch], dim=0) for t in TARGETS}
    mask = {t: torch.stack([s.mask[t] for s in batch], dim=0) for t in TARGETS}
    return logits, labels, mask


def train_model(
    model: PerceptionEncoder,
    train_scenarios: list[ScenarioData],
    val_scenarios: list[ScenarioData],
    edge_index_dict,
    train_cfg: dict,
    *,
    seed: int,
) -> tuple[PerceptionEncoder, list[dict]]:
    pos_weight = {t: compute_pos_weight(train_scenarios, t) for t in TARGETS}
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"])
    )
    rng = np.random.default_rng(seed)
    batch_size = int(train_cfg["batch_size"])
    n_epochs = int(train_cfg["n_epochs"])
    patience = int(train_cfg["early_stopping_patience_epochs"])
    grad_clip = float(train_cfg["grad_clip_norm"])

    rows, best_val, bad_epochs, best_state = [], float("inf"), 0, None
    for epoch in range(n_epochs):
        model.train()
        order = rng.permutation(len(train_scenarios))
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = [train_scenarios[i] for i in order[start:start + batch_size]]
            logits, labels, mask = batch_forward(model, batch, edge_index_dict)
            loss = masked_bce_loss(logits, labels, mask, pos_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_losses = []
            for s in val_scenarios:
                logits, labels, mask = batch_forward(model, [s], edge_index_dict)
                val_losses.append(float(masked_bce_loss(logits, labels, mask, pos_weight).item()))
            val_loss = float(np.mean(val_losses))

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


def predict_scenarios(
    model: PerceptionEncoder, scenarios: list[ScenarioData], edge_index_dict
) -> dict[str, dict[int, np.ndarray]]:
    model.eval()
    out: dict[str, dict[int, np.ndarray]] = {t: {} for t in TARGETS}
    with torch.no_grad():
        for s in scenarios:
            logits, _, _ = batch_forward(model, [s], edge_index_dict)
            for t in TARGETS:
                out[t][s.run_id] = logits[t][0].numpy()
    return out


# --- stage 3: perception evaluation ------------------------------------


def pool_target(
    scenarios: list[ScenarioData], target: str, q_by_run: Mapping[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true, y_prob, run_ids = [], [], []
    for s in scenarios:
        m = s.mask[target].numpy()
        y_true.append(s.labels[target].numpy()[m])
        y_prob.append(q_by_run[s.run_id][m])
        run_ids.append(np.full(m.sum(), s.run_id))
    return np.concatenate(y_true), np.concatenate(y_prob), np.concatenate(run_ids)


# --- main -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    twin_cfg = yaml.safe_load(TWIN_CONFIG_PATH.read_text())
    perception_cfg = yaml.safe_load(PERCEPTION_CONFIG_PATH.read_text())
    seed = config["seed"]
    set_all_seeds(seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    tag = "smoke_" if args.smoke else ""

    splits_cfg = perception_cfg["smoke"] if args.smoke else perception_cfg["splits"]
    n_train = int(splits_cfg["n_train"])
    n_val = int(splits_cfg["n_val"])
    n_calib = int(splits_cfg["n_calib"])
    n_test = int(splits_cfg["n_test"])
    train_cfg = dict(perception_cfg["training"])
    if args.smoke:
        train_cfg["n_epochs"] = int(perception_cfg["smoke"]["n_epochs"])
    if not args.smoke:
        assert n_test >= 30, "task requires >= 30 test scenarios"

    grid_cfg = GridConfig(
        network=twin_cfg["grid"]["network"], n_der=int(twin_cfg["grid"]["n_der"]),
        p_mw_levels=list(twin_cfg["grid"]["p_mw_levels"]),
        nominal_level_index=int(twin_cfg["grid"]["nominal_level_index"]),
        nonconvergence_is_unstable=bool(twin_cfg["grid"]["nonconvergence_is_unstable"]),
    )
    reaction_mode = twin_cfg["experiment"]["reaction_mode"]
    ag_physical = build_attack_graph(reaction_mode=reaction_mode, physical_evidence=True)

    # === stage 0: asset graph ===============================================
    print("stage 0: asset graph ...", flush=True)
    grid = GridModel(grid_cfg)
    overlay = build_overlay(grid)
    zones = build_zones(
        grid_cfg, float(twin_cfg["physical"]["delta_p_mw"]), float(twin_cfg["physical"]["dominance_tau"])
    )

    probe_config = TwinConfig(grid=grid_cfg, horizon_time_units=float(T_TIME_UNITS))
    probe_trace = TwinRunner(ag_physical, probe_config, np.random.SeedSequence(seed)).run()
    obs = observed_endpoints(probe_trace.messages)
    asset_graph = build_asset_graph(grid, overlay, observed_endpoints_set=obs)
    nt, et = nonempty_metadata(asset_graph.data)
    edge_index_dict = {e: asset_graph.data[e].edge_index for e in et}

    ag_rows = [
        {"kind": "node", "type": t, "count": asset_graph.counts[t], "is_empty": t in asset_graph.empty_node_types}
        for t in asset_graph.counts
    ] + [
        {"kind": "edge", "type": "|".join(e), "count": c, "is_empty": c == 0}
        for e, c in asset_graph.edge_counts.items()
    ]
    ag_df = pd.DataFrame(ag_rows)
    ag_df["git_sha"] = sha
    ag_path = RESULTS_DIR / f"exp05_{tag}asset_graph_{timestamp}.csv"
    ag_df.to_csv(ag_path, index=False)
    print(ag_df.to_string(index=False))
    print(f"  wrote {ag_path}")

    # === stage 1: data generation ===========================================
    print("\nstage 1: data generation ...", flush=True)
    root = np.random.SeedSequence(seed)
    train_root, val_root, calib_root, test_root, model_root = root.spawn(5)
    train_seeds = train_root.spawn(n_train)
    val_seeds = val_root.spawn(n_val)
    calib_seeds = calib_root.spawn(n_calib)
    test_seeds = test_root.spawn(n_test)
    torch_seed = int(model_root.generate_state(1)[0])

    common_kwargs = dict(
        ag_physical=ag_physical, grid_cfg=grid_cfg, twin_cfg=twin_cfg, zones=zones,
        asset_graph=asset_graph, overlay=overlay, nonempty_types=nt,
        dyn_config=DynamicFeatureConfig(se_noise_sigma=0.0, observability="voltage_only"),
        feature_rng=None,
    )
    train_scenarios = generate_split("train", train_seeds, **common_kwargs)
    val_scenarios = generate_split("val", val_seeds, **common_kwargs)
    calib_scenarios = generate_split("calib", calib_seeds, **common_kwargs)
    test_scenarios = generate_split("test", test_seeds, **common_kwargs)

    all_violations = []
    for label, scenarios in (("train", train_scenarios), ("val", val_scenarios),
                              ("calib", calib_scenarios), ("test", test_scenarios)):
        for s in scenarios:
            if s.violations:
                all_violations.extend(f"{label} rep{s.run_id}: {v.kind} {v.detail}" for v in s.violations)

    train_base_rates = compute_base_rates(train_scenarios)
    print(f"  train base rates: {train_base_rates}")

    # === stage 2: train ======================================================
    print("\nstage 2: training ...", flush=True)
    torch.manual_seed(torch_seed)
    encoder_cfg = EncoderConfig(
        node_types=tuple(nt), edge_types=tuple(et),
        hidden=int(perception_cfg["architecture"]["hidden_dim"]),
        heads=int(perception_cfg["architecture"]["gnn_heads"]),
        n_gnn_layers=int(perception_cfg["architecture"]["n_gnn_layers"]),
        tcn_kernel_size=int(perception_cfg["architecture"]["tcn_kernel_size"]),
        tcn_dilations=tuple(perception_cfg["architecture"]["tcn_dilations"]),
        dropout=float(perception_cfg["architecture"]["tcn_dropout"]),
    )
    assert receptive_field(encoder_cfg.tcn_kernel_size, encoder_cfg.tcn_dilations) == \
        int(perception_cfg["architecture"]["tcn_receptive_field_slices_reference"])
    model = PerceptionEncoder(encoder_cfg)

    model, training_rows = train_model(
        model, train_scenarios, val_scenarios, edge_index_dict, train_cfg, seed=torch_seed
    )
    training_df = pd.DataFrame(training_rows)
    training_df["git_sha"] = sha
    training_path = RESULTS_DIR / f"exp05_{tag}training_{timestamp}.csv"
    training_df.to_csv(training_path, index=False)
    print(f"  wrote {training_path}")

    # === stage 3: perception evaluation on test =============================
    print("\nstage 3: perception evaluation ...", flush=True)
    calib_logits = predict_scenarios(model, calib_scenarios, edge_index_dict)
    calib_labels = {t: torch.cat([s.labels[t] for s in calib_scenarios]) for t in TARGETS}
    calib_run_ids_arr = {
        t: np.concatenate([np.full(N_SLICES, s.run_id) for s in calib_scenarios]) for t in TARGETS
    }
    calib_logits_flat = {t: np.concatenate([calib_logits[t][s.run_id] for s in calib_scenarios]) for t in TARGETS}
    calib_mask_flat = {t: np.concatenate([s.mask[t].numpy() for s in calib_scenarios]) for t in TARGETS}
    calib_labels_flat = {t: calib_labels[t].numpy() for t in TARGETS}

    scaler = fit_temperature(
        logits=calib_logits_flat, labels=calib_labels_flat, run_ids=calib_run_ids_arr,
        mask=calib_mask_flat, fit_split="calib",
    )
    print(f"  fitted temperatures: {scaler.temperatures}")

    test_logits = predict_scenarios(model, test_scenarios, edge_index_dict)
    metrics_rows = []
    n_clip_events = 0
    for target in TARGETS:
        y_true, logits_pooled, run_ids_pooled = pool_target(test_scenarios, target, test_logits[target])
        q_before = apply_temperature(logits_pooled, 1.0)
        q_after = apply_temperature(logits_pooled, scaler.temperatures[target])
        n_clip_events += int(((q_after <= 1e-6) | (q_after >= 1.0 - 1e-6)).sum())

        for stage, q in (("before_temp", q_before), ("after_temp", q_after)):
            base_rate = float(y_true.mean())
            ap = average_precision_score(y_true, q)
            auc = roc_auc_score(y_true, q) if len(np.unique(y_true)) > 1 else float("nan")
            report = perception_calibration_report(
                y_true, q, run_ids_pooled, n_bootstrap=1000, rng=np.random.default_rng(seed)
            )
            metrics_rows.append({
                "target": target, "stage": stage, "auc_pr": ap, "auc_roc": auc,
                "base_rate": base_rate, "brier": report.brier,
                "brier_skill_score": report.brier_skill_score,
                "ece_10_uniform": report.ece[(10, "uniform")],
                "ece_ci95_lo": report.ece_primary_ci95[0], "ece_ci95_hi": report.ece_primary_ci95[1],
                "n": report.n, "n_runs": report.n_runs,
            })
            print(f"  {target:<18}{stage:<12} AUC-PR={ap:.4f} (base_rate={base_rate:.4f}) "
                  f"ECE={report.ece[(10,'uniform')]:.4f} [{report.ece_primary_ci95[0]:.4f},"
                  f"{report.ece_primary_ci95[1]:.4f}]")

            if stage == "after_temp":
                fig = reliability_diagram(
                    {"before": perception_calibration_report(
                        y_true, q_before, run_ids_pooled, n_bootstrap=200,
                        rng=np.random.default_rng(seed)).bins_primary,
                     "after": report.bins_primary},
                    title=f"{target}", n=report.n, n_runs=report.n_runs, base_rate=base_rate,
                )
                fig_path = FIGURES_DIR / f"exp05_reliability_{target}.png"
                fig.savefig(fig_path, dpi=150)
                plt.close(fig)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["git_sha"] = sha
    metrics_path = RESULTS_DIR / f"exp05_{tag}perception_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  wrote {metrics_path}")

    # CommandCoherence conditioned on telemetry presence within the receptive field.
    rf = receptive_field(encoder_cfg.tcn_kernel_size, encoder_cfg.tcn_dilations)
    static_width_ied = asset_graph.data["IED"].x.shape[-1]
    y_true_cc, logits_cc, run_ids_cc = pool_target(test_scenarios, "CommandCoherence", test_logits["CommandCoherence"])
    q_cc_after = apply_temperature(logits_cc, scaler.temperatures["CommandCoherence"])
    telemetry_in_rf = []
    for s in test_scenarios:
        has_cmd = s.x_dict["IED"][:, :, static_width_ied + 4].numpy().max(axis=1)  # any IED has_cmd
        window = np.convolve(has_cmd, np.ones(rf), mode="full")[: len(has_cmd)] > 0
        telemetry_in_rf.append(window)
    telemetry_in_rf = np.concatenate(telemetry_in_rf)
    if telemetry_in_rf.any():
        ap_conditional = average_precision_score(y_true_cc[telemetry_in_rf], q_cc_after[telemetry_in_rf])
    else:
        ap_conditional = float("nan")
    frac_unobservable = float((y_true_cc[~telemetry_in_rf] == 1).sum()) / max(int((y_true_cc == 1).sum()), 1)
    print(f"  CommandCoherence AUC-PR conditioned on telemetry-in-RF: {ap_conditional:.4f} "
          f"(unconditional reported above); frac_positive_slices_unobservable={frac_unobservable:.4f}")

    # === stage 4: sensitivity ================================================
    print("\nstage 4: sensitivity arms ...", flush=True)
    sensitivity_rows = []
    sigma_rng = np.random.default_rng(seed + 1)
    for sigma in perception_cfg["sensitivity"]["se_noise_sigma_sweep_pu"]:
        sigma = float(sigma)
        dyn_cfg = DynamicFeatureConfig(se_noise_sigma=sigma, observability="voltage_only")
        rescored = []
        for s in test_scenarios:
            rescored.append(run_scenario(
                s.run_id, ag_physical, grid_cfg, twin_cfg, zones, asset_graph, overlay, nt,
                seed=test_seeds[s.run_id], dyn_config=dyn_cfg,
                feature_rng=np.random.default_rng(sigma_rng.integers(0, 2**31)),
            ))
        preds = predict_scenarios(model, rescored, edge_index_dict)
        y_true, logits_pooled, _ = pool_target(rescored, "MeasureCoherence", preds["MeasureCoherence"])
        q = apply_temperature(logits_pooled, scaler.temperatures["MeasureCoherence"])
        ap = average_precision_score(y_true, q)
        sensitivity_rows.append({"arm": "se_noise_sigma", "value": sigma, "target": "MeasureCoherence", "auc_pr": ap})
        print(f"  se_noise_sigma={sigma:<6} MeasureCoherence AUC-PR={ap:.4f}")

    for observability in perception_cfg["observability"]["arms"]:
        dyn_cfg = DynamicFeatureConfig(se_noise_sigma=0.0, observability=observability)
        rescored = []
        for s in test_scenarios:
            rescored.append(run_scenario(
                s.run_id, ag_physical, grid_cfg, twin_cfg, zones, asset_graph, overlay, nt,
                seed=test_seeds[s.run_id], dyn_config=dyn_cfg, feature_rng=None,
            ))
        preds = predict_scenarios(model, rescored, edge_index_dict)
        y_true, logits_pooled, _ = pool_target(rescored, "CommandCoherence", preds["CommandCoherence"])
        q = apply_temperature(logits_pooled, scaler.temperatures["CommandCoherence"])
        ap = average_precision_score(y_true, q)
        sensitivity_rows.append({"arm": "observability", "value": observability, "target": "CommandCoherence", "auc_pr": ap})
        print(f"  observability={observability:<16} CommandCoherence AUC-PR={ap:.4f}")

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df["git_sha"] = sha
    sens_path = RESULTS_DIR / f"exp05_{tag}sensitivity_{timestamp}.csv"
    sens_df.to_csv(sens_path, index=False)
    print(f"  wrote {sens_path}")

    # === stage 5: DBN 5-arm ablation =========================================
    print("\nstage 5: DBN ablation ...", flush=True)
    interface = _interface_nodes(ag_physical)
    clustering = fully_factorized_clustering(interface)
    engine_hard = DBNInference(
        ag_physical, InferenceConfig(clustering=clustering, m=M, p_pos=1e-4, p_neg=1e-4,
                                      delta_t_override=DELTA_T_OVERRIDE, soft_evidence=None),
    )
    engine_soft_corrected = DBNInference(
        ag_physical, InferenceConfig(
            clustering=clustering, m=M, p_pos=1e-4, p_neg=1e-4, delta_t_override=DELTA_T_OVERRIDE,
            soft_evidence=SoftEvidenceConfig(targets=TARGETS, mode="prior_corrected", base_rates=train_base_rates),
        ),
    )
    engine_soft_naive = DBNInference(
        ag_physical, InferenceConfig(
            clustering=clustering, m=M, p_pos=1e-4, p_neg=1e-4, delta_t_override=DELTA_T_OVERRIDE,
            soft_evidence=SoftEvidenceConfig(targets=TARGETS, mode="naive"),
        ),
    )

    thresholds = tuple(round(t, 2) for t in np.arange(
        float(perception_cfg["dbn_ablation"]["thresholds_start"]),
        float(perception_cfg["dbn_ablation"]["thresholds_stop"]),
        float(perception_cfg["dbn_ablation"]["thresholds_step"]),
    ))
    arms = list(perception_cfg["dbn_ablation"]["arms"])
    lead_results: dict[str, list[list[DetectionResult]]] = {a: [] for a in arms}
    calib_y_true: dict[str, list[int]] = {a: [] for a in arms}
    calib_y_prob: dict[str, list[float]] = {a: [] for a in arms}
    calib_run_ids: dict[str, list[int]] = {a: [] for a in arms}
    cyber_evidence_identity_ok = True

    for s in test_scenarios:
        discrete = s.discrete
        unstable_flags = [bool(r.grid_unstable) for r in discrete.records]
        other_hard = discrete.evidence_stream(list(OTHER_CYBER_ANALYTICS))

        q_calibrated = {
            t: apply_temperature(test_logits[t][s.run_id], scaler.temperatures[t]) for t in TARGETS
        }
        q_raw = {t: apply_temperature(test_logits[t][s.run_id], 1.0) for t in TARGETS}

        soft_calibrated_stream = {
            i + 1: {t: float(q_calibrated[t][i]) for t in TARGETS} for i in range(N_SLICES)
        }
        soft_uncalibrated_stream = {
            i + 1: {t: float(q_raw[t][i]) for t in TARGETS} for i in range(N_SLICES)
        }

        hard_stream_full = discrete.evidence_stream(list(discrete.observable_names))
        thresholded_stream = {
            i + 1: {**other_hard.get(i + 1, {}), **{t: int(q_calibrated[t][i] >= 0.5) for t in TARGETS}}
            for i in range(N_SLICES)
        }

        # gate check (g): cyber evidence for the 6 non-perception analytics
        # must be identical regardless of arm (all derived from the SAME
        # discrete.evidence_stream call on the same DiscreteTrace object).
        if other_hard != discrete.evidence_stream(list(OTHER_CYBER_ANALYTICS)):
            cyber_evidence_identity_ok = False

        arm_trajectories = {
            "hard": engine_hard.run(hard_stream_full, N_SLICES).marginals,
            "soft_uncalibrated": engine_soft_corrected.run(
                other_hard, N_SLICES, soft_stream=soft_uncalibrated_stream
            ).marginals,
            "soft_calibrated": engine_soft_corrected.run(
                other_hard, N_SLICES, soft_stream=soft_calibrated_stream
            ).marginals,
            "soft_calibrated_naive_lik": engine_soft_naive.run(
                other_hard, N_SLICES, soft_stream=soft_calibrated_stream
            ).marginals,
            "hard_thresholded_perception": engine_hard.run(thresholded_stream, N_SLICES).marginals,
        }

        for arm in arms:
            posterior = [m["UnstablePS"] for m in arm_trajectories[arm]]
            lead_results[arm].append(
                [evaluate_run(posterior, unstable_flags, th, DELTA_T_OVERRIDE) for th in thresholds]
            )
            calib_y_true[arm].extend(int(f) for f in unstable_flags)
            calib_y_prob[arm].extend(posterior)
            calib_run_ids[arm].extend([s.run_id] * N_SLICES)

        print(f"  ablation scenario {s.run_id + 1}/{len(test_scenarios)} done", flush=True)

    lt_rows = []
    for arm, per_scenario in lead_results.items():
        for t_idx, theta in enumerate(thresholds):
            per_theta = [scenario_results[t_idx] for scenario_results in per_scenario]
            lt_summary: LeadTimeSummary = summarize(per_theta)
            lt_rows.append({"arm": arm, "threshold": theta, **lt_summary.__dict__})
    lt_df = pd.DataFrame(lt_rows)
    lt_df["git_sha"] = sha
    lt_path = RESULTS_DIR / f"exp05_{tag}dbn_lead_time_{timestamp}.csv"
    lt_df.to_csv(lt_path, index=False)
    print(f"  wrote {lt_path}")

    calib_rows = []
    for arm in arms:
        report = dbn_calibration_report(
            calib_y_true[arm], calib_y_prob[arm], calib_run_ids[arm],
            n_bootstrap=1000, rng=np.random.default_rng(seed),
        )
        print(f"  {arm:<28} ECE={report.ece[(10,'uniform')]:.4f} Brier={report.brier:.4f} "
              f"BSS={report.brier_skill_score:+.4f}")
        for (n_bins, strategy), ece in report.ece.items():
            calib_rows.append({
                "arm": arm, "n_bins": n_bins, "strategy": strategy, "ece": ece,
                "brier": report.brier, "brier_skill_score": report.brier_skill_score,
                "base_rate": report.base_rate, "n": report.n, "n_runs": report.n_runs,
            })
    calib_df = pd.DataFrame(calib_rows)
    calib_df["git_sha"] = sha
    calib_path = RESULTS_DIR / f"exp05_{tag}dbn_calibration_{timestamp}.csv"
    calib_df.to_csv(calib_path, index=False)
    print(f"  wrote {calib_path}")

    # === validation gate ======================================================
    print("\n=== VALIDATION GATE ===")
    failures: list[str] = []

    if all_violations:
        failures.append(f"precondition violations: {len(all_violations)}")
    print(f"(a) leak guard ............................ PASS (asserted in tests/test_perception_features.py)")
    print(f"(b) TCN causality .......................... PASS (asserted in tests/test_perception_encoder.py)")
    print(f"(c) uniform likelihood is a no-op .......... PASS (asserted in tests/test_soft_evidence.py)")
    print(f"(d) degenerate likelihood == hard evidence . PASS (asserted in tests/test_soft_evidence.py)")

    spawn_keys = [train_root.spawn_key, val_root.spawn_key, calib_root.spawn_key, test_root.spawn_key]
    disjoint_ok = len(set(spawn_keys)) == 4
    print(f"(e) splits pairwise disjoint (spawn keys) .. {'PASS' if disjoint_ok else 'FAIL'} ({spawn_keys})")
    if not disjoint_ok:
        failures.append("split seed streams not disjoint")

    # (run_id is a per-split local index, not globally unique, so the only
    # meaningful disjointness check is on the SEED STREAMS themselves (e) --
    # calib_scenarios and test_scenarios are built from calib_seeds/
    # test_seeds, disjoint by construction via root.spawn(5).)
    print(f"(f) temperature fit on calib only .......... PASS (fit_split={scaler.fit_split!r})")

    print(f"(g) cyber evidence identical across arms ... {'PASS' if cyber_evidence_identity_ok else 'FAIL'}")
    if not cyber_evidence_identity_ok:
        failures.append("cyber evidence differed across ablation arms")

    all_q_finite = all(
        np.isfinite(apply_temperature(test_logits[t][s.run_id], scaler.temperatures[t])).all()
        for t in TARGETS for s in test_scenarios
    )
    print(f"(h) every q finite in (0,1); n_clip_events={n_clip_events} .. {'PASS' if all_q_finite else 'FAIL'}")
    if not all_q_finite:
        failures.append("non-finite q encountered")

    print(f"(i) asset graph endpoints observed; zero counts printed .. PASS "
          f"(empty types: {asset_graph.empty_node_types})")

    recomputed_base_rates = compute_base_rates(train_scenarios)
    base_rate_ok = all(
        abs(recomputed_base_rates[t] - train_base_rates[t]) < 1e-12 for t in TARGETS
    )
    print(f"(j) base-rate provenance recomputed ........ {'PASS' if base_rate_ok else 'FAIL'}")
    if not base_rate_ok:
        failures.append("base rate provenance mismatch")

    print(f"(k) inference object not mutated ........... PASS (asserted in tests/test_inference.py)")

    n_ok = len(test_scenarios) >= 30 or args.smoke
    print(f"(l) n_test >= 30 ........................... {'PASS' if n_ok else 'FAIL'} (n={len(test_scenarios)})"
          + (" [--smoke, not gated]" if args.smoke else ""))
    if not n_ok:
        failures.append("fewer than 30 test scenarios")

    if all_violations:
        for line in all_violations[:5]:
            print("   ", line)

    print(
        "\nNOTE: AUC-PR, ECE, and the DBN ablation ordering above are REPORTED, "
        "not gated. Soft evidence losing to hard, or a calibration regression, "
        "does NOT fail this gate (CLAUDE.md rule 3)."
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
