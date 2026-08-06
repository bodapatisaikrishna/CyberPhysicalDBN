"""GNN vs. per-asset MLP perception-encoder ablation (Session 10, ablation
axis 4): isolates whether the GNN's graph structure is what actually drives
detection, or whether an encoder with ZERO message-passing over the same
per-asset features does just as well.

Reuses `experiments/exp05_perception.py`'s stage 0/1 (asset graph, scenario
generation) and its training/eval harness UNCHANGED, imported by path (the
same pattern `experiments/exp06_baselines.py` already uses for exp05) --
both encoders train on the BYTE-IDENTICAL train/val/calib/test split and
training budget, so the only thing that differs between arms is
`EncoderConfig.encoder_type` ("gnn" vs. "mlp", `src/perception/encoder.py`).
Reuses `configs/perception.yaml` directly (splits/training/architecture) --
no new config file is needed, since fairness requires identical
splits/training budgets for both arms anyway (an unequal budget would
confound the comparison).

Stages: 0 asset graph -> 1 data generation (SHARED by both arms, one twin
run per scenario) -> 2 train GNN + train MLP (same split, same
hyperparameters) -> 3 perception-level AUC-PR comparison per target
(`MeasureCoherence`, `CommandCoherence`, `PhysLocalDER`, `PhysWideArea`) on
the held-out test split -> 4 DBN-level lead-time/ECE comparison
(`hard` / `soft_calibrated_gnn` / `soft_calibrated_mlp`, reusing exp05
stage 5's exact ablation pattern with only the encoder swapped). Gate tests
structural correctness only -- same split, same budget, valid ranges --
never "does the GNN win" (CLAUDE.md rule 3); the GNN-vs-MLP comparison
itself is reported unconditionally.

Run: .venv/bin/python experiments/exp11_perception_ablation.py [--smoke]
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score

from src.dbn.inference import DBNInference, InferenceConfig, _interface_nodes, fully_factorized_clustering
from src.dbn.soft_evidence import SoftEvidenceConfig
from src.eval.calibration import calibration_report as dbn_calibration_report
from src.eval.lead_time import evaluate_run, summarize
from src.eval.provenance import git_sha
from src.perception.asset_graph import build_asset_graph, nonempty_metadata, observed_endpoints
from src.perception.calibration import apply_temperature, fit_temperature
from src.perception.encoder import EncoderConfig, PerceptionEncoder, receptive_field
from src.twin.grid import GridConfig, GridModel
from src.twin.runner import TwinConfig, TwinRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
TWIN_CONFIG_PATH = REPO_ROOT / "configs" / "twin.yaml"
PERCEPTION_CONFIG_PATH = REPO_ROOT / "configs" / "perception.yaml"
RESULTS_DIR = REPO_ROOT / "results"


def _load_exp05():
    """Import experiments/exp05_perception.py by path -- the same pattern
    experiments/exp06_baselines.py already uses, so this script shares
    exp05's exact scenario-generation/training/eval helpers rather than
    re-deriving them and risking drift."""
    path = REPO_ROOT / "experiments" / "exp05_perception.py"
    spec = importlib.util.spec_from_file_location("exp05_perception", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp05_perception"] = module
    spec.loader.exec_module(module)
    return module


exp05 = _load_exp05()
T_TIME_UNITS = exp05.T_TIME_UNITS
DELTA_T_OVERRIDE = exp05.DELTA_T_OVERRIDE
N_SLICES = exp05.N_SLICES
TARGETS = exp05.TARGETS
OTHER_CYBER_ANALYTICS = exp05.OTHER_CYBER_ANALYTICS

ENCODER_ARMS = ("gnn", "mlp")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    ag_physical = exp05.build_attack_graph(reaction_mode=reaction_mode, physical_evidence=True)

    # === stage 0: asset graph (shared by both arms) =========================
    print("stage 0: asset graph ...", flush=True)
    grid = GridModel(grid_cfg)
    overlay = exp05.build_overlay(grid)
    zones = exp05.build_zones(
        grid_cfg, float(twin_cfg["physical"]["delta_p_mw"]), float(twin_cfg["physical"]["dominance_tau"])
    )

    probe_config = TwinConfig(grid=grid_cfg, horizon_time_units=float(T_TIME_UNITS))
    probe_trace = TwinRunner(ag_physical, probe_config, np.random.SeedSequence(seed)).run()
    obs = observed_endpoints(probe_trace.messages)
    asset_graph = build_asset_graph(grid, overlay, observed_endpoints_set=obs)
    nt, et = nonempty_metadata(asset_graph.data)
    edge_index_dict = {e: asset_graph.data[e].edge_index for e in et}

    # === stage 1: data generation (shared by both arms) =====================
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
        dyn_config=exp05.DynamicFeatureConfig(se_noise_sigma=0.0, observability="voltage_only"),
        feature_rng=None,
    )
    train_scenarios = exp05.generate_split("train", train_seeds, **common_kwargs)
    val_scenarios = exp05.generate_split("val", val_seeds, **common_kwargs)
    calib_scenarios = exp05.generate_split("calib", calib_seeds, **common_kwargs)
    test_scenarios = exp05.generate_split("test", test_seeds, **common_kwargs)

    all_violations = []
    for label, scenarios in (("train", train_scenarios), ("val", val_scenarios),
                              ("calib", calib_scenarios), ("test", test_scenarios)):
        for s in scenarios:
            if s.violations:
                all_violations.extend(f"{label} rep{s.run_id}: {v.kind} {v.detail}" for v in s.violations)

    train_base_rates = exp05.compute_base_rates(train_scenarios)
    print(f"  train base rates: {train_base_rates}")

    # === stage 2: train both encoders on the SAME split/budget ==============
    print("\nstage 2: training (gnn, then mlp) ...", flush=True)
    models: dict[str, PerceptionEncoder] = {}
    training_dfs = []
    for arm in ENCODER_ARMS:
        print(f"  training {arm} ...", flush=True)
        torch.manual_seed(torch_seed)
        encoder_cfg = EncoderConfig(
            node_types=tuple(nt), edge_types=tuple(et), encoder_type=arm,
            hidden=int(perception_cfg["architecture"]["hidden_dim"]),
            heads=int(perception_cfg["architecture"]["gnn_heads"]),
            n_gnn_layers=int(perception_cfg["architecture"]["n_gnn_layers"]),
            tcn_kernel_size=int(perception_cfg["architecture"]["tcn_kernel_size"]),
            tcn_dilations=tuple(perception_cfg["architecture"]["tcn_dilations"]),
            dropout=float(perception_cfg["architecture"]["tcn_dropout"]),
        )
        model = PerceptionEncoder(encoder_cfg)
        model, training_rows = exp05.train_model(
            model, train_scenarios, val_scenarios, edge_index_dict, train_cfg, seed=torch_seed
        )
        models[arm] = model
        df = pd.DataFrame(training_rows)
        df["arm"] = arm
        training_dfs.append(df)

    training_df = pd.concat(training_dfs, ignore_index=True)
    training_df["git_sha"] = sha
    training_path = RESULTS_DIR / f"exp11_{tag}training_{timestamp}.csv"
    training_df.to_csv(training_path, index=False)
    print(f"  wrote {training_path}")

    # === stage 3: perception-level AUC-PR comparison =========================
    print("\nstage 3: perception evaluation (gnn vs. mlp) ...", flush=True)
    scalers = {}
    test_logits = {}
    metrics_rows = []
    for arm in ENCODER_ARMS:
        calib_logits = exp05.predict_scenarios(models[arm], calib_scenarios, edge_index_dict)
        calib_labels_flat = {t: np.concatenate([s.labels[t].numpy() for s in calib_scenarios]) for t in TARGETS}
        calib_run_ids_arr = {
            t: np.concatenate([np.full(N_SLICES, s.run_id) for s in calib_scenarios]) for t in TARGETS
        }
        calib_logits_flat = {t: np.concatenate([calib_logits[t][s.run_id] for s in calib_scenarios]) for t in TARGETS}
        calib_mask_flat = {t: np.concatenate([s.mask[t].numpy() for s in calib_scenarios]) for t in TARGETS}
        scaler = fit_temperature(
            logits=calib_logits_flat, labels=calib_labels_flat, run_ids=calib_run_ids_arr,
            mask=calib_mask_flat, fit_split="calib",
        )
        scalers[arm] = scaler
        test_logits[arm] = exp05.predict_scenarios(models[arm], test_scenarios, edge_index_dict)

        for target in TARGETS:
            y_true, logits_pooled, run_ids_pooled = exp05.pool_target(test_scenarios, target, test_logits[arm][target])
            q = apply_temperature(logits_pooled, scaler.temperatures[target])
            ap = average_precision_score(y_true, q)
            metrics_rows.append({
                "arm": arm, "target": target, "auc_pr": ap, "base_rate": float(y_true.mean()),
            })
            print(f"  {arm:<4} {target:<18} AUC-PR={ap:.4f} (base_rate={float(y_true.mean()):.4f})")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["git_sha"] = sha
    metrics_path = RESULTS_DIR / f"exp11_{tag}perception_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  wrote {metrics_path}")

    # === stage 4: DBN-level lead-time/ECE comparison =========================
    # Same 3-way structure as exp05 stage 5 (hard / soft_calibrated_<arm>),
    # condensed to the arms this ablation needs.
    print("\nstage 4: DBN ablation (hard vs. soft_calibrated_gnn vs. soft_calibrated_mlp) ...", flush=True)
    interface = _interface_nodes(ag_physical)
    clustering = fully_factorized_clustering(interface)
    engine_hard = DBNInference(
        ag_physical, InferenceConfig(clustering=clustering, m=1.0, p_pos=1e-4, p_neg=1e-4,
                                      delta_t_override=DELTA_T_OVERRIDE, soft_evidence=None),
    )
    engines_soft = {
        arm: DBNInference(
            ag_physical, InferenceConfig(
                clustering=clustering, m=1.0, p_pos=1e-4, p_neg=1e-4, delta_t_override=DELTA_T_OVERRIDE,
                soft_evidence=SoftEvidenceConfig(targets=TARGETS, mode="prior_corrected", base_rates=train_base_rates),
            ),
        )
        for arm in ENCODER_ARMS
    }

    thresholds = tuple(round(t, 2) for t in np.arange(
        float(perception_cfg["dbn_ablation"]["thresholds_start"]),
        float(perception_cfg["dbn_ablation"]["thresholds_stop"]),
        float(perception_cfg["dbn_ablation"]["thresholds_step"]),
    ))
    dbn_arms = ["hard"] + [f"soft_calibrated_{arm}" for arm in ENCODER_ARMS]
    lead_results: dict[str, list] = {a: [] for a in dbn_arms}
    calib_y_true: dict[str, list] = {a: [] for a in dbn_arms}
    calib_y_prob: dict[str, list] = {a: [] for a in dbn_arms}
    calib_run_ids: dict[str, list] = {a: [] for a in dbn_arms}

    for s in test_scenarios:
        discrete = s.discrete
        unstable_flags = [bool(r.grid_unstable) for r in discrete.records]
        other_hard = discrete.evidence_stream(list(OTHER_CYBER_ANALYTICS))
        hard_stream_full = discrete.evidence_stream(list(discrete.observable_names))

        arm_trajectories = {"hard": engine_hard.run(hard_stream_full, N_SLICES).marginals}
        for arm in ENCODER_ARMS:
            q_calibrated = {
                t: apply_temperature(test_logits[arm][t][s.run_id], scalers[arm].temperatures[t]) for t in TARGETS
            }
            soft_stream = {i + 1: {t: float(q_calibrated[t][i]) for t in TARGETS} for i in range(N_SLICES)}
            arm_trajectories[f"soft_calibrated_{arm}"] = engines_soft[arm].run(
                other_hard, N_SLICES, soft_stream=soft_stream
            ).marginals

        for arm in dbn_arms:
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
            lt_summary = summarize(per_theta)
            lt_rows.append({"arm": arm, "threshold": theta, **lt_summary.__dict__})
    lt_df = pd.DataFrame(lt_rows)
    lt_df["git_sha"] = sha
    lt_path = RESULTS_DIR / f"exp11_{tag}dbn_lead_time_{timestamp}.csv"
    lt_df.to_csv(lt_path, index=False)
    print(f"  wrote {lt_path}")

    calib_rows = []
    for arm in dbn_arms:
        report = dbn_calibration_report(
            calib_y_true[arm], calib_y_prob[arm], calib_run_ids[arm],
            n_bootstrap=1000, rng=np.random.default_rng(seed),
        )
        print(f"  {arm:<24} ECE={report.ece[(10,'uniform')]:.4f} Brier={report.brier:.4f} "
              f"BSS={report.brier_skill_score:+.4f}")
        for (n_bins, strategy), ece in report.ece.items():
            calib_rows.append({
                "arm": arm, "n_bins": n_bins, "strategy": strategy, "ece": ece,
                "brier": report.brier, "brier_skill_score": report.brier_skill_score,
                "base_rate": report.base_rate, "n": report.n, "n_runs": report.n_runs,
            })
    calib_df = pd.DataFrame(calib_rows)
    calib_df["git_sha"] = sha
    calib_path = RESULTS_DIR / f"exp11_{tag}dbn_calibration_{timestamp}.csv"
    calib_df.to_csv(calib_path, index=False)
    print(f"  wrote {calib_path}")

    # === validation gate ======================================================
    print("\n=== VALIDATION GATE ===")
    failures: list[str] = []

    if all_violations:
        failures.append(f"precondition violations: {len(all_violations)}")
    print(f"(a) leak guard .............................. PASS (asserted in tests/test_perception_features.py)")
    print(f"(b) TCN causality ............................ PASS (asserted in tests/test_perception_encoder.py)")
    print(f"(c) both arms trained on byte-identical split . PASS (single train/val/calib/test generation, shared)")

    spawn_keys = [train_root.spawn_key, val_root.spawn_key, calib_root.spawn_key, test_root.spawn_key]
    disjoint_ok = len(set(spawn_keys)) == 4
    print(f"(d) splits pairwise disjoint (spawn keys) .... {'PASS' if disjoint_ok else 'FAIL'} ({spawn_keys})")
    if not disjoint_ok:
        failures.append("split seed streams not disjoint")

    all_scores_valid = all(
        0.0 <= v <= 1.0 for arm in dbn_arms for v in calib_y_prob[arm]
    ) and all(len(calib_y_prob[arm]) == len(test_scenarios) * N_SLICES for arm in dbn_arms)
    print(f"(e) every DBN score in [0,1], correct length .. {'PASS' if all_scores_valid else 'FAIL'}")
    if not all_scores_valid:
        failures.append("invalid DBN posterior scores")

    n_test_ok = args.smoke or n_test >= 30
    print(f"(f) n_test >= 30 ({n_test}) {'[--smoke, not gated]' if args.smoke else ''} " f"{'PASS' if n_test_ok else 'FAIL'}")
    if not n_test_ok:
        failures.append(f"n_test={n_test} < 30")

    print(
        "\nNOTE: the GNN-vs-MLP AUC-PR/lead-time/ECE comparison above is "
        "REPORTED, not gated (CLAUDE.md rule 3). If the MLP matches or "
        "beats the GNN -- including on PhysWideArea, the target this "
        "session's pre-registered H3 specifically expects the GNN to win "
        "on -- that is the valid, reportable result. See LAB_NOTEBOOK.md."
    )

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
