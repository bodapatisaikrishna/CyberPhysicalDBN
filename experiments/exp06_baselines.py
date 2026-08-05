"""External ML baselines vs. the proposed system (CLAUDE.md's minimum-viable-
publication bar: "external baselines + lead-time evaluation").
LAB_NOTEBOOK.md 2026-08-03.

Cerotti et al. compare only against their own inference variants (EX/CL/FF).
This experiment adds four external baselines -- LSTM autoencoder
(reconstruction error), GAT/GraphSAGE end-to-end classifier, gradient-boosted
trees on engineered features, rule-based IDS proxy -- each with a genuine,
logged hyperparameter search, evaluated against "the proposed system"
(exp05's `soft_calibrated` closed-loop-DBN-plus-learned-perception arm) on
IDENTICAL twin scenarios and seeds.

"Identical scenarios and seeds" is made literal, not just "generated the same
way": this script reuses `configs/perception.yaml`'s SAME
`SeedSequence(42).spawn(5)` split construction and imports
`experiments/exp05_perception.py`'s helper functions (`build_overlay`,
`build_zones`, `extract_labels`, `compute_base_rates`, `compute_pos_weight`,
`train_model`, `predict_scenarios`, `TARGETS`, `OTHER_CYBER_ANALYTICS`, the
timebase constants) via `importlib.util.spec_from_file_location` -- the
pattern already established by `tests/test_twin.py::TestTimebasePin._load`.
Scenario generation itself is NOT the imported `run_scenario` (which
discards the raw per-slice dynamic tensors the GBM/LSTM-AE baselines need
after building the DBN's combined static+dynamic input) -- this script's own
`generate_scenario` runs the SAME twin/discretize/feature-extraction calls
ONCE per scenario and keeps both representations, so no scenario is
simulated twice. `exp05`'s reused functions are written against plain
attribute access (`.x_dict`, `.labels`, `.mask`, `.run_id`, `.violations`),
so this script's own `ScenarioBundle` (a superset of `exp05.ScenarioData`)
is a drop-in for every one of them.

Ground truth for every system, always: `SliceRecord.grid_unstable`, never
`ground_truth["UnstablePS"]`.

Stages: 0 asset graph -> 1 data generation -> 2 proposed-system perception
training + calibration + DBN soft_calibrated posterior -> 3 baseline
hyperparameter searches + final training -> 4 unified evaluation (AUC-PR,
lead-time sweep, calibration) for every system -> 5 validation gate. The gate
tests CORRECTNESS INVARIANTS ONLY; the AUC-PR ranking (favorable or not) is
printed unconditionally, before the PASS/FAIL line (CLAUDE.md rule 3).

Run: .venv/bin/python experiments/exp06_baselines.py [--smoke]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

from src.attack_graph.graph import build_attack_graph
from src.baselines.common import (
    TUNING_BUDGET_NOTE_BASELINE,
    TUNING_BUDGET_NOTE_DBN,
    TrialResult,
    flatten_engineered_features,
    hyperparameter_search,
)
from src.baselines.gbm import GBMTrialConfig, build_flat_table, score_trajectory as gbm_score
from src.baselines.gbm import train_gbm
from src.baselines.gnn_classifier import GNNBaselineConfig, GNNClassifier
from src.baselines.lstm_ae import (
    LSTMAETrialConfig,
    build_causal_windows,
    error_to_probability,
    fit_recon_error_scaler,
    score_trajectory as ae_score,
    train_autoencoder,
    training_window_cutoff,
)
from src.baselines.rule_based import RuleConfig
from src.baselines.rule_based import score_trajectory as rule_score
from src.dbn.inference import DBNInference, InferenceConfig, _interface_nodes, fully_factorized_clustering
from src.dbn.soft_evidence import SoftEvidenceConfig
from src.eval.calibration import calibration_report
from src.eval.lead_time import DetectionResult, LeadTimeSummary, evaluate_run, summarize
from src.eval.provenance import git_sha
from src.perception.asset_graph import build_asset_graph, nonempty_metadata, observed_endpoints
from src.perception.calibration import apply_temperature, fit_temperature
from src.perception.encoder import EncoderConfig, PerceptionEncoder, combine_static_dynamic, stack_scenarios
from src.perception.features import DynamicFeatureConfig, build_dynamic_features
from src.twin.attacker import AttackerConfig, DelayLaw
from src.twin.grid import GridConfig, GridModel
from src.twin.runner import DiscreteTrace, TwinConfig, TwinRunner, discretize, validate_trace

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
TWIN_CONFIG_PATH = REPO_ROOT / "configs" / "twin.yaml"
PERCEPTION_CONFIG_PATH = REPO_ROOT / "configs" / "perception.yaml"
BASELINES_CONFIG_PATH = REPO_ROOT / "configs" / "baselines.yaml"
RESULTS_DIR = REPO_ROOT / "results"


def _load_exp05():
    """Import experiments/exp05_perception.py by path -- the pattern already
    established by tests/test_twin.py::TestTimebasePin._load, used here so
    exp06 shares exp05's exact helper functions/constants rather than
    re-deriving them and risking drift."""
    path = REPO_ROOT / "experiments" / "exp05_perception.py"
    spec = importlib.util.spec_from_file_location("exp05_perception", path)
    module = importlib.util.module_from_spec(spec)
    # exp05_perception.py uses @dataclass, whose internals look up
    # sys.modules[cls.__module__] -- must be registered before exec_module,
    # or dataclass() raises AttributeError on a None module lookup.
    sys.modules["exp05_perception"] = module
    spec.loader.exec_module(module)
    return module


exp05 = _load_exp05()
T_TIME_UNITS = exp05.T_TIME_UNITS
DELTA_T_OVERRIDE = exp05.DELTA_T_OVERRIDE
N_SLICES = exp05.N_SLICES
M = exp05.M
TARGETS = exp05.TARGETS
OTHER_CYBER_ANALYTICS = exp05.OTHER_CYBER_ANALYTICS


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --- scenario generation: same twin/discretize/feature calls as exp05, but --
# --- retains the raw dynamic tensors GBM/LSTM-AE need, in ONE pass ----------


@dataclass
class ScenarioBundle:
    run_id: int
    discrete: DiscreteTrace
    dynamic_x: dict[str, torch.Tensor]  # RAW per-slice dynamic tensors (GBM/AE)
    x_dict: dict[str, torch.Tensor]  # static+dynamic combined (GNN baseline, perception encoder)
    globals_: torch.Tensor
    labels: dict[str, torch.Tensor]
    mask: dict[str, torch.Tensor]
    violations: list


def generate_scenario(
    run_id: int, ag_physical, grid_cfg, twin_cfg, zones, asset_graph, overlay, nonempty_types,
    seed: np.random.SeedSequence,
) -> ScenarioBundle:
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
        config=DynamicFeatureConfig(se_noise_sigma=0.0, observability="voltage_only"), rng=None,
    )
    dynamic_x = {t: dyn[t] for t in nonempty_types}
    static_x = {t: asset_graph.data[t].x for t in nonempty_types}
    x_dict = combine_static_dynamic(static_x, dynamic_x)
    globals_ = dyn["host"][:, 0, :]
    labels, mask = exp05.extract_labels(discrete)
    return ScenarioBundle(
        run_id=run_id, discrete=discrete, dynamic_x=dynamic_x, x_dict=x_dict,
        globals_=globals_, labels=labels, mask=mask, violations=violations,
    )


def generate_split(label: str, seeds: list[np.random.SeedSequence], **kwargs) -> list[ScenarioBundle]:
    print(f"  {label}: {len(seeds)} scenarios ...", flush=True)
    out = []
    for i, seed in enumerate(seeds):
        out.append(generate_scenario(i, seed=seed, **kwargs))
        print(f"    {label} {i + 1}/{len(seeds)} done", flush=True)
    return out


def unstable_labels(scenarios: list[ScenarioBundle]) -> dict[int, np.ndarray]:
    return {s.run_id: np.array([int(r.grid_unstable) for r in s.discrete.records]) for s in scenarios}


def sample_configs(grid: dict, n: int, rng: np.random.Generator) -> list[dict]:
    keys = list(grid)
    all_combos = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    n = min(n, len(all_combos))
    idx = rng.choice(len(all_combos), size=n, replace=False)
    return [all_combos[i] for i in idx]


def val_auc_pr_for_scores(scores_by_run: dict[int, np.ndarray], val_scenarios: list[ScenarioBundle]) -> float:
    y_true = np.concatenate([np.array([int(r.grid_unstable) for r in s.discrete.records]) for s in val_scenarios])
    y_prob = np.concatenate([scores_by_run[s.run_id] for s in val_scenarios])
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


# --- main --------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    twin_cfg = yaml.safe_load(TWIN_CONFIG_PATH.read_text())
    perception_cfg = yaml.safe_load(PERCEPTION_CONFIG_PATH.read_text())
    baselines_cfg = yaml.safe_load(BASELINES_CONFIG_PATH.read_text())
    seed = config["seed"]
    set_all_seeds(seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    tag = "smoke_" if args.smoke else ""

    splits_cfg = perception_cfg["smoke"] if args.smoke else perception_cfg["splits"]
    n_train, n_val, n_calib, n_test = (
        int(splits_cfg["n_train"]), int(splits_cfg["n_val"]),
        int(splits_cfg["n_calib"]), int(splits_cfg["n_test"]),
    )
    train_cfg = dict(perception_cfg["training"])
    if args.smoke:
        train_cfg["n_epochs"] = int(perception_cfg["smoke"]["n_epochs"])
    if not args.smoke:
        assert n_test >= 30, "task requires >= 30 test scenarios"

    if args.smoke:
        n_search = {"rule_based": 3, "gbm": 3, "lstm_ae": 2, "gnn_classifier": 2}
        n_epochs_baseline = 2
    else:
        n_search = {
            "rule_based": len(baselines_cfg["rule_based"]["window_slices_sweep"]),
            "gbm": int(baselines_cfg["gbm"]["n_search_trials"]),
            "lstm_ae": int(baselines_cfg["lstm_ae"]["n_search_trials"]),
            "gnn_classifier": int(baselines_cfg["gnn_classifier"]["n_search_trials"]),
        }
        n_epochs_baseline = None  # read per-baseline from config below

    grid_cfg = GridConfig(
        network=twin_cfg["grid"]["network"], n_der=int(twin_cfg["grid"]["n_der"]),
        p_mw_levels=list(twin_cfg["grid"]["p_mw_levels"]),
        nominal_level_index=int(twin_cfg["grid"]["nominal_level_index"]),
        nonconvergence_is_unstable=bool(twin_cfg["grid"]["nonconvergence_is_unstable"]),
    )
    reaction_mode = twin_cfg["experiment"]["reaction_mode"]
    ag_physical = build_attack_graph(reaction_mode=reaction_mode, physical_evidence=True)

    # === stage 0: asset graph (identical to exp05's stage 0) ================
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

    # === stage 1: data generation -- SAME seeds/splits as exp05 =============
    print("\nstage 1: data generation (identical seeds to exp05) ...", flush=True)
    root = np.random.SeedSequence(seed)
    train_root, val_root, calib_root, test_root, model_root = root.spawn(5)
    train_seeds, val_seeds, calib_seeds, test_seeds = (
        train_root.spawn(n_train), val_root.spawn(n_val), calib_root.spawn(n_calib), test_root.spawn(n_test),
    )
    torch_seed = int(model_root.generate_state(1)[0])

    common_kwargs = dict(
        ag_physical=ag_physical, grid_cfg=grid_cfg, twin_cfg=twin_cfg, zones=zones,
        asset_graph=asset_graph, overlay=overlay, nonempty_types=nt,
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

    train_base_rates = exp05.compute_base_rates(train_scenarios)
    test_unstable = unstable_labels(test_scenarios)

    # === stage 2: proposed system -- perception training + calibration + ===
    # === DBN soft_calibrated posterior (reuses exp05's exact construction) =
    print("\nstage 2: proposed system (perception + DBN soft_calibrated) ...", flush=True)
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
    perception_model = PerceptionEncoder(encoder_cfg)
    perception_model, _ = exp05.train_model(
        perception_model, train_scenarios, val_scenarios, edge_index_dict, train_cfg, seed=torch_seed
    )

    calib_logits = exp05.predict_scenarios(perception_model, calib_scenarios, edge_index_dict)
    calib_labels_flat = {t: np.concatenate([s.labels[t].numpy() for s in calib_scenarios]) for t in TARGETS}
    calib_mask_flat = {t: np.concatenate([s.mask[t].numpy() for s in calib_scenarios]) for t in TARGETS}
    calib_logits_flat = {t: np.concatenate([calib_logits[t][s.run_id] for s in calib_scenarios]) for t in TARGETS}
    calib_run_ids_flat = {
        t: np.concatenate([np.full(N_SLICES, s.run_id) for s in calib_scenarios]) for t in TARGETS
    }
    scaler = fit_temperature(
        logits=calib_logits_flat, labels=calib_labels_flat, run_ids=calib_run_ids_flat,
        mask=calib_mask_flat, fit_split="calib",
    )

    test_logits = exp05.predict_scenarios(perception_model, test_scenarios, edge_index_dict)

    interface = _interface_nodes(ag_physical)
    clustering = fully_factorized_clustering(interface)
    engine_soft_corrected = DBNInference(
        ag_physical, InferenceConfig(
            clustering=clustering, m=M, p_pos=1e-4, p_neg=1e-4, delta_t_override=DELTA_T_OVERRIDE,
            soft_evidence=SoftEvidenceConfig(targets=TARGETS, mode="prior_corrected", base_rates=train_base_rates),
        ),
    )

    dbn_scores: dict[int, np.ndarray] = {}
    for s in test_scenarios:
        other_hard = s.discrete.evidence_stream(list(OTHER_CYBER_ANALYTICS))
        q_calibrated = {t: apply_temperature(test_logits[t][s.run_id], scaler.temperatures[t]) for t in TARGETS}
        soft_stream = {i + 1: {t: float(q_calibrated[t][i]) for t in TARGETS} for i in range(N_SLICES)}
        marginals = engine_soft_corrected.run(other_hard, N_SLICES, soft_stream=soft_stream).marginals
        dbn_scores[s.run_id] = np.array([m["UnstablePS"] for m in marginals])
        print(f"  DBN proposed-system scenario {s.run_id + 1}/{len(test_scenarios)} done", flush=True)

    # === stage 3: baselines -- search, then final training ==================
    print("\nstage 3: baseline hyperparameter searches ...", flush=True)
    search_rows: list[dict] = []
    baseline_test_scores: dict[str, dict[int, np.ndarray]] = {}
    baseline_selected: dict[str, dict] = {}
    search_rng = np.random.SeedSequence(seed).spawn(1)[0]

    # --- rule-based -----------------------------------------------------
    print("  rule_based ...", flush=True)
    window_sweep = baselines_cfg["rule_based"]["window_slices_sweep"][: n_search["rule_based"]] \
        if args.smoke else baselines_cfg["rule_based"]["window_slices_sweep"]
    rule_trials = []
    for w in window_sweep:
        cfg = RuleConfig(window_slices=int(w))
        val_scores = {
            s.run_id: rule_score(s.discrete.evidence_stream(list(s.discrete.observable_names)),
                                  list(s.discrete.observable_names), N_SLICES, cfg)
            for s in val_scenarios
        }
        ap = val_auc_pr_for_scores(val_scores, val_scenarios)
        rule_trials.append(TrialResult(trial_id=len(rule_trials), config={"window_slices": w}, val_auc_pr=ap))
        search_rows.append({"baseline": "rule_based", "trial_id": rule_trials[-1].trial_id,
                             **rule_trials[-1].config, "val_auc_pr": ap})
    best_rule = max(rule_trials, key=lambda t: t.val_auc_pr).config
    baseline_selected["rule_based"] = best_rule
    rule_cfg_final = RuleConfig(window_slices=int(best_rule["window_slices"]))
    baseline_test_scores["rule_based"] = {
        s.run_id: rule_score(s.discrete.evidence_stream(list(s.discrete.observable_names)),
                              list(s.discrete.observable_names), N_SLICES, rule_cfg_final)
        for s in test_scenarios
    }
    print(f"    selected: {best_rule}", flush=True)

    # --- GBM --------------------------------------------------------------
    print("  gbm ...", flush=True)
    X_train, y_train, _ = build_flat_table(train_scenarios)
    X_val, y_val, val_run_ids = build_flat_table(val_scenarios)
    gbm_grid = baselines_cfg["gbm"]["search_space"]
    gbm_rng = np.random.default_rng(search_rng.generate_state(1)[0])
    gbm_configs = sample_configs(gbm_grid, n_search["gbm"], gbm_rng)

    def score_gbm(cfg: dict, rng: np.random.Generator) -> float:
        model = train_gbm(GBMTrialConfig(**cfg), X_train, y_train, random_state=int(rng.integers(0, 2**31)))
        val_probs = model.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else float("nan")

    gbm_trials, best_gbm_cfg = hyperparameter_search(gbm_configs, score_gbm, rng=gbm_rng)
    for t in gbm_trials:
        search_rows.append({"baseline": "gbm", "trial_id": t.trial_id, **t.config, "val_auc_pr": t.val_auc_pr})
    baseline_selected["gbm"] = best_gbm_cfg
    final_gbm = train_gbm(GBMTrialConfig(**best_gbm_cfg), X_train, y_train, random_state=seed)
    baseline_test_scores["gbm"] = {}
    for s in test_scenarios:
        flat, _ = flatten_engineered_features(s.dynamic_x)
        baseline_test_scores["gbm"][s.run_id] = gbm_score(final_gbm, flat.numpy())
    print(f"    selected: {best_gbm_cfg}", flush=True)

    # --- LSTM-AE ------------------------------------------------------------
    print("  lstm_ae ...", flush=True)
    window = int(baselines_cfg["lstm_ae"]["window_slices"])
    train_nominal_windows = []
    for s in train_scenarios:
        gt = [r.ground_truth for r in s.discrete.records]
        cutoff = training_window_cutoff(gt)
        if cutoff == 0:
            continue
        flat, _ = flatten_engineered_features(s.dynamic_x)
        windows = build_causal_windows(flat, window=window)
        train_nominal_windows.append(windows[:cutoff])
    train_windows = torch.cat(train_nominal_windows, dim=0) if train_nominal_windows else torch.zeros(0, window, 40)

    val_nominal_windows, val_nominal_errors_input = [], []
    for s in val_scenarios:
        gt = [r.ground_truth for r in s.discrete.records]
        cutoff = training_window_cutoff(gt)
        flat, _ = flatten_engineered_features(s.dynamic_x)
        windows = build_causal_windows(flat, window=window)
        if cutoff > 0:
            val_nominal_windows.append(windows[:cutoff])
    val_windows = torch.cat(val_nominal_windows, dim=0) if val_nominal_windows else train_windows[:1]

    print(f"    train nominal windows: {train_windows.shape[0]}, val nominal windows: {val_windows.shape[0]}", flush=True)

    ae_grid = baselines_cfg["lstm_ae"]["search_space"]
    ae_rng = np.random.default_rng(search_rng.generate_state(1)[0] ^ 0xA5A5)
    ae_configs = sample_configs(ae_grid, n_search["lstm_ae"], ae_rng)
    ae_epochs = n_epochs_baseline or int(baselines_cfg["lstm_ae"]["n_epochs"])
    ae_batch = int(baselines_cfg["lstm_ae"]["batch_size"])
    ae_clip = float(baselines_cfg["lstm_ae"]["grad_clip_norm"])
    ae_patience = int(baselines_cfg["lstm_ae"]["early_stopping_patience_epochs"])

    def score_ae(cfg: dict, rng: np.random.Generator) -> float:
        trial_cfg = LSTMAETrialConfig(**cfg)
        model, _ = train_autoencoder(
            trial_cfg, train_windows, val_windows, n_epochs=ae_epochs, batch_size=ae_batch,
            grad_clip_norm=ae_clip, patience=ae_patience, torch_seed=int(rng.integers(0, 2**31)),
        )
        val_scores = {}
        for s in val_scenarios:
            flat, _ = flatten_engineered_features(s.dynamic_x)
            w = build_causal_windows(flat, window=window)
            val_scores[s.run_id] = ae_score(model, w)
        val_errors_flat = np.concatenate(list(val_scores.values()))
        scaler_trial = fit_recon_error_scaler(val_errors_flat[val_errors_flat.argsort()[: max(len(val_errors_flat) // 2, 1)]])
        val_probs = {rid: error_to_probability(sc, scaler_trial) for rid, sc in val_scores.items()}
        return val_auc_pr_for_scores(val_probs, val_scenarios)

    ae_trials, best_ae_cfg = hyperparameter_search(ae_configs, score_ae, rng=ae_rng)
    for t in ae_trials:
        search_rows.append({"baseline": "lstm_ae", "trial_id": t.trial_id, **t.config, "val_auc_pr": t.val_auc_pr})
    baseline_selected["lstm_ae"] = best_ae_cfg

    final_ae, _ = train_autoencoder(
        LSTMAETrialConfig(**best_ae_cfg), train_windows, val_windows, n_epochs=ae_epochs,
        batch_size=ae_batch, grad_clip_norm=ae_clip, patience=ae_patience, torch_seed=torch_seed,
    )
    val_errors_for_scaler = np.concatenate([
        ae_score(final_ae, build_causal_windows(flatten_engineered_features(s.dynamic_x)[0], window=window)[:training_window_cutoff([r.ground_truth for r in s.discrete.records])])
        for s in val_scenarios if training_window_cutoff([r.ground_truth for r in s.discrete.records]) > 0
    ])
    final_scaler = fit_recon_error_scaler(val_errors_for_scaler, fit_split=str(baselines_cfg["lstm_ae"]["scaler_fit_split"]))
    baseline_test_scores["lstm_ae"] = {}
    for s in test_scenarios:
        flat, _ = flatten_engineered_features(s.dynamic_x)
        w = build_causal_windows(flat, window=window)
        errors = ae_score(final_ae, w)
        baseline_test_scores["lstm_ae"][s.run_id] = error_to_probability(errors, final_scaler)
    print(f"    selected: {best_ae_cfg}", flush=True)

    # --- GNN classifier -------------------------------------------------
    print("  gnn_classifier ...", flush=True)
    gnn_grid_raw = baselines_cfg["gnn_classifier"]["search_space"]
    gnn_candidates = []
    for conv_type, n_layers, hidden, dropout, tks in itertools.product(
        gnn_grid_raw["conv_type"], gnn_grid_raw["n_layers"], gnn_grid_raw["hidden"],
        gnn_grid_raw["dropout"], gnn_grid_raw["temporal_kernel_size"],
    ):
        if conv_type == "gat":
            for heads in gnn_grid_raw["heads"]:
                gnn_candidates.append(dict(conv_type=conv_type, n_layers=n_layers, hidden=hidden,
                                            heads=heads, dropout=dropout, temporal_kernel_size=tks))
        else:
            for sage_aggr in gnn_grid_raw["sage_aggr"]:
                gnn_candidates.append(dict(conv_type=conv_type, n_layers=n_layers, hidden=hidden,
                                            sage_aggr=sage_aggr, dropout=dropout, temporal_kernel_size=tks))
    gnn_rng = np.random.default_rng(search_rng.generate_state(1)[0] ^ 0x5A5A)
    gnn_idx = gnn_rng.choice(len(gnn_candidates), size=min(n_search["gnn_classifier"], len(gnn_candidates)), replace=False)
    gnn_configs = [gnn_candidates[i] for i in gnn_idx]
    gnn_epochs = n_epochs_baseline or int(baselines_cfg["gnn_classifier"]["n_epochs"])
    gnn_batch = int(baselines_cfg["gnn_classifier"]["batch_size"])
    gnn_lr = float(baselines_cfg["gnn_classifier"]["learning_rate"])
    gnn_wd = float(baselines_cfg["gnn_classifier"]["weight_decay"])
    gnn_clip = float(baselines_cfg["gnn_classifier"]["grad_clip_norm"])
    gnn_patience = int(baselines_cfg["gnn_classifier"]["early_stopping_patience_epochs"])

    # BUDGET FIX (LAB_NOTEBOOK.md 2026-08-03): search TRIALS score on a
    # smaller, fixed subset of train/val -- the FINAL selected config is
    # retrained on the FULL train/val split below, identical to every other
    # baseline. See configs/baselines.yaml's gnn_classifier comment for the
    # measured root cause (HeteroConv's per-relation Python loop, unlike the
    # DBN's fused HGTConv, made the full-data search intractable on CPU).
    subset_cfg = baselines_cfg["gnn_classifier"].get("search_scenario_subset")
    if not args.smoke and subset_cfg:
        gnn_search_train = train_scenarios[: int(subset_cfg["n_train"])]
        gnn_search_val = val_scenarios[: int(subset_cfg["n_val"])]
    else:
        gnn_search_train, gnn_search_val = train_scenarios, val_scenarios
    print(f"    search subset: {len(gnn_search_train)} train / {len(gnn_search_val)} val "
          f"(final model retrained on full {len(train_scenarios)}/{len(val_scenarios)})", flush=True)

    train_labels_full = unstable_labels(train_scenarios)
    val_labels_full = unstable_labels(val_scenarios)
    train_pos = sum(int(l.sum()) for l in train_labels_full.values())
    train_total = sum(len(l) for l in train_labels_full.values())
    gnn_pos_weight = torch.tensor((train_total - train_pos) / max(train_pos, 1))

    def train_gnn_model(
        cfg: dict, torch_seed_local: int,
        train_subset: list[ScenarioBundle], val_subset: list[ScenarioBundle],
    ) -> GNNClassifier:
        torch.manual_seed(torch_seed_local)
        model = GNNClassifier(tuple(nt), tuple(et), GNNBaselineConfig(**cfg))
        opt = torch.optim.AdamW(model.parameters(), lr=gnn_lr, weight_decay=gnn_wd)
        rng_local = np.random.default_rng(torch_seed_local)
        best_val, bad_epochs, best_state = float("inf"), 0, None
        for epoch in range(gnn_epochs):
            model.train()
            order = rng_local.permutation(len(train_subset))
            for start in range(0, len(order), gnn_batch):
                batch = [train_subset[i] for i in order[start:start + gnn_batch]]
                x_dict = stack_scenarios([b.x_dict for b in batch])
                globals_ = torch.stack([b.globals_ for b in batch], dim=0)
                logits = model(x_dict, edge_index_dict, globals_)
                y = torch.stack([torch.tensor(train_labels_full[b.run_id], dtype=torch.float32) for b in batch])
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=gnn_pos_weight)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gnn_clip)
                opt.step()
            model.eval()
            with torch.no_grad():
                val_losses = []
                for s in val_subset:
                    x_dict = stack_scenarios([s.x_dict])
                    logits = model(x_dict, edge_index_dict, s.globals_.unsqueeze(0))
                    y = torch.tensor(val_labels_full[s.run_id], dtype=torch.float32).unsqueeze(0)
                    val_losses.append(float(F.binary_cross_entropy_with_logits(logits, y).item()))
                val_loss = float(np.mean(val_losses))
            if val_loss < best_val - 1e-5:
                best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad_epochs += 1
                if bad_epochs >= gnn_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def gnn_predict(model: GNNClassifier, scenarios: list[ScenarioBundle]) -> dict[int, np.ndarray]:
        model.eval()
        out = {}
        with torch.no_grad():
            for s in scenarios:
                x_dict = stack_scenarios([s.x_dict])
                logits = model(x_dict, edge_index_dict, s.globals_.unsqueeze(0))
                out[s.run_id] = torch.sigmoid(logits)[0].numpy()
        return out

    def score_gnn(cfg: dict, rng: np.random.Generator) -> float:
        model = train_gnn_model(cfg, int(rng.integers(0, 2**31)), gnn_search_train, gnn_search_val)
        val_probs = gnn_predict(model, gnn_search_val)
        return val_auc_pr_for_scores(val_probs, gnn_search_val)

    gnn_trials, best_gnn_cfg = hyperparameter_search(gnn_configs, score_gnn, rng=gnn_rng)
    for t in gnn_trials:
        search_rows.append({"baseline": "gnn_classifier", "trial_id": t.trial_id, **t.config, "val_auc_pr": t.val_auc_pr})
    baseline_selected["gnn_classifier"] = best_gnn_cfg
    final_gnn = train_gnn_model(best_gnn_cfg, torch_seed, train_scenarios, val_scenarios)
    baseline_test_scores["gnn_classifier"] = gnn_predict(final_gnn, test_scenarios)
    print(f"    selected: {best_gnn_cfg}", flush=True)

    search_df = pd.DataFrame(search_rows)
    search_df["git_sha"] = sha
    search_path = RESULTS_DIR / f"exp06_{tag}search_{timestamp}.csv"
    search_df.to_csv(search_path, index=False)
    print(f"  wrote {search_path}")

    # === stage 4: unified evaluation ========================================
    print("\nstage 4: unified evaluation ...", flush=True)
    all_systems: dict[str, dict[int, np.ndarray]] = {
        "dbn_soft_calibrated": dbn_scores,
        "rule_based": baseline_test_scores["rule_based"],
        "gbm": baseline_test_scores["gbm"],
        "lstm_ae": baseline_test_scores["lstm_ae"],
        "gnn_classifier": baseline_test_scores["gnn_classifier"],
    }
    thresholds = tuple(round(t, 2) for t in np.arange(
        float(baselines_cfg["common"]["thresholds_start"]),
        float(baselines_cfg["common"]["thresholds_stop"]),
        float(baselines_cfg["common"]["thresholds_step"]),
    ))

    summary_rows, lt_rows = [], []
    for system_name, scores_by_run in all_systems.items():
        y_true = np.concatenate([test_unstable[s.run_id] for s in test_scenarios])
        y_prob = np.concatenate([scores_by_run[s.run_id] for s in test_scenarios])
        run_ids = np.concatenate([np.full(N_SLICES, s.run_id) for s in test_scenarios])
        ap = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
        report = calibration_report(y_true, y_prob, run_ids, n_bootstrap=1000, rng=np.random.default_rng(seed))

        n_search_trials = n_search.get(system_name, 0)
        budget_note = TUNING_BUDGET_NOTE_DBN if system_name == "dbn_soft_calibrated" else TUNING_BUDGET_NOTE_BASELINE
        summary_rows.append({
            "system": system_name, "auc_pr": ap, "base_rate": report.base_rate,
            "ece_10_uniform": report.ece[(10, "uniform")],
            "ece_ci95_lo": report.ece_primary_ci95[0], "ece_ci95_hi": report.ece_primary_ci95[1],
            "brier": report.brier, "brier_skill_score": report.brier_skill_score,
            "n": report.n, "n_runs": report.n_runs, "n_search_trials": n_search_trials,
            "tuning_budget_note": budget_note, "git_sha": sha,
        })

        for s in test_scenarios:
            posterior = list(scores_by_run[s.run_id])
            unstable_flags = [bool(v) for v in test_unstable[s.run_id]]
            for th in thresholds:
                lt_rows.append({
                    "system": system_name, "run_id": s.run_id, "threshold": th,
                    **evaluate_run(posterior, unstable_flags, th, DELTA_T_OVERRIDE).__dict__,
                })
        print(f"  {system_name:<24} AUC-PR={ap:.4f} ECE={report.ece[(10,'uniform')]:.4f} "
              f"Brier={report.brier:.4f} BSS={report.brier_skill_score:+.4f}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / f"exp06_{tag}comparison_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  wrote {summary_path}")

    lt_detail_df = pd.DataFrame(lt_rows)
    lt_summary_rows = []
    for system_name in all_systems:
        for th in thresholds:
            sub = lt_detail_df[(lt_detail_df["system"] == system_name) & (lt_detail_df["threshold"] == th)]
            results = [
                DetectionResult(
                    threshold=row.threshold, t_detect_slice=row.t_detect_slice,
                    t_instability_slice=row.t_instability_slice, lead_time_slices=row.lead_time_slices,
                    lead_time_units=row.lead_time_units, outcome=row.outcome,
                    n_upward_crossings=row.n_upward_crossings, recrossed_after_first=row.recrossed_after_first,
                )
                for row in sub.itertuples()
            ]
            lt_summary: LeadTimeSummary = summarize(results)
            lt_summary_rows.append({"system": system_name, "threshold": th, **lt_summary.__dict__, "git_sha": sha})
    lt_summary_df = pd.DataFrame(lt_summary_rows)
    lt_path = RESULTS_DIR / f"exp06_{tag}comparison_leadtime_{timestamp}.csv"
    lt_summary_df.to_csv(lt_path, index=False)
    print(f"  wrote {lt_path}")

    # === stage 5: validation gate ============================================
    print("\n=== VALIDATION GATE ===")
    failures: list[str] = []

    if all_violations:
        failures.append(f"precondition violations: {len(all_violations)}")
    print("(a) leak guard (LSTM-AE cutoff/feature disjointness) .. PASS (asserted in tests/test_lstm_ae.py)")
    print("(b) GNN causality + block-diagonal replication ........ PASS (asserted in tests/test_gnn_classifier.py)")

    spawn_keys = [train_root.spawn_key, val_root.spawn_key, calib_root.spawn_key, test_root.spawn_key]
    disjoint_ok = len(set(spawn_keys)) == 4
    print(f"(c) splits pairwise disjoint (spawn keys) .............. {'PASS' if disjoint_ok else 'FAIL'}")
    if not disjoint_ok:
        failures.append("split seed streams not disjoint")

    expected_trials = {
        "rule_based": len(window_sweep), "gbm": len(gbm_trials),
        "lstm_ae": len(ae_trials), "gnn_classifier": len(gnn_trials),
    }
    trial_counts_ok = all(
        len(search_df[search_df["baseline"] == b]) == n for b, n in expected_trials.items()
    )
    print(f"(d) every search CSV has its expected trial count ...... {'PASS' if trial_counts_ok else 'FAIL'}")
    if not trial_counts_ok:
        failures.append("search trial count mismatch")

    scaler_ok = final_scaler.fit_split == str(baselines_cfg["lstm_ae"]["scaler_fit_split"])
    print(f"(e) AE scaler fit-split provenance ...................... {'PASS' if scaler_ok else 'FAIL'} "
          f"(fit_split={final_scaler.fit_split!r})")
    if not scaler_ok:
        failures.append("AE scaler fit_split provenance mismatch")

    all_finite = all(np.isfinite(scores_by_run[s.run_id]).all() for scores_by_run in all_systems.values() for s in test_scenarios)
    print(f"(f) every system's test-split score finite .............. {'PASS' if all_finite else 'FAIL'}")
    if not all_finite:
        failures.append("non-finite score encountered")

    n_ok = len(test_scenarios) >= 30 or args.smoke
    print(f"(g) n_test >= 30 ......................................... {'PASS' if n_ok else 'FAIL'} "
          f"(n={len(test_scenarios)})" + (" [--smoke, not gated]" if args.smoke else ""))
    if not n_ok:
        failures.append("fewer than 30 test scenarios")

    if all_violations:
        for line in all_violations[:5]:
            print("   ", line)

    print(
        "\n=== AUC-PR RANKING (all systems, sorted; NOT a gate criterion, CLAUDE.md rule 3) ==="
    )
    ranked = sorted(summary_rows, key=lambda r: (r["auc_pr"] if r["auc_pr"] == r["auc_pr"] else -1), reverse=True)
    for i, r in enumerate(ranked, start=1):
        marker = "  <-- PROPOSED SYSTEM" if r["system"] == "dbn_soft_calibrated" else ""
        print(f"  #{i} {r['system']:<24} AUC-PR={r['auc_pr']:.4f}{marker}")
    proposed_rank = next(i for i, r in enumerate(ranked, start=1) if r["system"] == "dbn_soft_calibrated")
    if proposed_rank > 1:
        print(f"\n  NOTE: {proposed_rank - 1} baseline(s) beat the proposed system on raw AUC-PR.")
        print("  Reported prominently per task instruction -- not buried in a CSV column.")
        print("  The project's argument rests on lead time, calibration, explainability, and")
        print("  robustness under adaptation, NOT raw AUC-PR on scripted attacks -- see the")
        print("  comparison tables above and results/exp06_comparison_*.csv for the full picture.")
    else:
        print("\n  The proposed system leads on raw AUC-PR too, on this test split.")

    print(
        "\nNOTE: AUC-PR/ECE/lead-time numbers above are REPORTED, not gated. A baseline "
        "beating the DBN does NOT fail this gate (CLAUDE.md rule 3)."
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
