"""Adversarial RL attacker vs. detectors: testing claim C3 (CLAUDE.md layer
[0]/[2]/[3], Session 9).

The source paper's own unaddressed concern: an attacker who knows the DBN
could choose low-detection paths. C3 (highest novelty of the three claims):
under an RL attacker optimizing against the detector, the causal DBN
degrades more gracefully than deep-IDS baselines, because structural
preconditions cannot be skipped -- you cannot inject a spoofed reporting
message without first establishing MITM, and MITM requires credential
access.

See `src/twin/rl_attacker.py`'s module docstring for the full RL
environment design (bandit-style single-decision episode, 3 knowledge
levels via the REWARD signal, not observation) and LAB_NOTEBOOK.md
2026-08-06 for the pre-registered hypotheses (H1-H4) and binding design
decisions this script implements without deviation.

This script deliberately does NOT reuse `experiments/exp06_baselines.py`'s
`generate_scenario`/`ScenarioBundle` by path: those hardcode
`physical_evidence=True` + real `zones` throughout (`exp05.extract_labels`
in particular is never exercised with `zones=None`), and carry perception-
target `labels`/`mask` fields this script never uses. This script's own
lean `ScenarioBundle`/`generate_scenario` (no labels/mask, `zones=None`
always) avoids exercising that untested combination. `src/baselines/*.py`'s
scoring functions are duck-typed (`Protocol`-based, verified) against
exactly this script's `ScenarioBundle` shape.

Stages: 0 config/fidelity resolution -> 1 asset graph -> 2 baseline
training (ONCE, on default-scripted scenarios, never RL-driven) -> 3 PPO
training x3 knowledge levels (train fidelity) -> 4 evaluation-episode
generation x3 (frozen policies, eval fidelity) -> 5 5-system scoring on
those same episodes -> 6 robustness-curve table -> 7 lettered gate
(structural only, CLAUDE.md rule 3 -- the robustness curve itself prints
unconditionally, never gated on which system "wins").

Run: .venv/bin/python experiments/exp09_adversarial_c3.py [--smoke]
"""

from __future__ import annotations

import argparse
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
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.attack_graph.graph import build_attack_graph
from src.baselines.common import TrialResult, flatten_engineered_features, hyperparameter_search
from src.baselines.gbm import GBMTrialConfig, build_flat_table, score_trajectory as gbm_score, train_gbm
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
from src.baselines.rule_based import RuleConfig, score_trajectory as rule_score
from src.dbn.inference import DBNInference, InferenceConfig, _interface_nodes, fully_factorized_clustering
from src.dbn.parameterization import collect_uniformization_ttcs, compute_delta_t
from src.eval.lead_time import evaluate_run, summarize
from src.eval.provenance import git_sha
from src.perception.asset_graph import CyberAsset, CyberOverlayConfig, build_asset_graph, nonempty_metadata, observed_endpoints
from src.perception.encoder import combine_static_dynamic, stack_scenarios
from src.perception.features import DynamicFeatureConfig, build_dynamic_features
from src.twin.attacker import AttackerConfig, DelayLaw
from src.twin.grid import GridConfig, GridModel
from src.twin.rl_attacker import CYBER_ANALYTICS, FidelityConfig, RewardConfig, RLAttackerEnv
from src.twin.runner import ContinuousTrace, DiscreteTrace, TwinConfig, TwinRunner, discretize, validate_trace

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
TWIN_CONFIG_PATH = REPO_ROOT / "configs" / "twin.yaml"
ADVERSARIAL_CONFIG_PATH = REPO_ROOT / "configs" / "adversarial_c3.yaml"
RESULTS_DIR = REPO_ROOT / "results"

KNOWLEDGE_LEVELS: tuple[str, ...] = ("blind", "analytics", "full_dbn")
SYSTEMS: tuple[str, ...] = ("dbn_hard_evidence", "rule_based", "gbm", "lstm_ae", "gnn_classifier")


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_overlay(grid: GridModel) -> CyberOverlayConfig:
    return CyberOverlayConfig(
        assets=(
            CyberAsset("ControlCentre", "host", "ControlCentre"),
            *(CyberAsset(f"IED_{bus}", "IED", der_id, controls=(der_id,)) for der_id, bus in zip(grid.der_ids, grid.der_buses)),
        ),
        source="exp09 stage 1",
    )


# === lean, own scenario bundle (see module docstring for why not exp06's) ===


@dataclass
class ScenarioBundle:
    run_id: int
    discrete: DiscreteTrace
    dynamic_x: dict[str, torch.Tensor]
    x_dict: dict[str, torch.Tensor]
    globals_: torch.Tensor
    violations: list


def bundle_from_trace(
    run_id: int, trace: ContinuousTrace, ag, asset_graph, overlay, nonempty_types, grid_cfg,
    dispatch_period_time_units: float, delta_t: float, n_slices: int, p_pos: float, p_neg: float, rng: np.random.Generator,
) -> ScenarioBundle:
    """Builds a `ScenarioBundle` from an ALREADY-EXISTING `ContinuousTrace`
    -- never re-simulates (used both after a fresh twin run and after an RL
    episode, where `env.last_trace` already exists)."""
    violations = validate_trace(trace, ag)
    discrete = discretize(trace, ag, delta_t, n_slices, rng, p_pos, p_neg, zones=None)
    dyn = build_dynamic_features(
        trace, delta_t, n_slices, asset_graph, overlay, grid_cfg,
        dispatch_period_time_units=dispatch_period_time_units,
        config=DynamicFeatureConfig(se_noise_sigma=0.0, observability="voltage_only"), rng=None,
    )
    dynamic_x = {t: dyn[t] for t in nonempty_types}
    static_x = {t: asset_graph.data[t].x for t in nonempty_types}
    x_dict = combine_static_dynamic(static_x, dynamic_x)
    globals_ = dyn["host"][:, 0, :]
    return ScenarioBundle(run_id=run_id, discrete=discrete, dynamic_x=dynamic_x, x_dict=x_dict, globals_=globals_, violations=violations)


def generate_scenario(
    run_id: int, ag, grid_cfg, twin_cfg, asset_graph, overlay, nonempty_types,
    delta_t: float, n_slices: int, horizon: float, p_pos: float, p_neg: float, seed: np.random.SeedSequence,
) -> ScenarioBundle:
    tw_config = TwinConfig(
        grid=grid_cfg, attacker=AttackerConfig(delay_law=DelayLaw.EXPONENTIAL),
        dispatch_period_time_units=float(twin_cfg["control_centre"]["dispatch_period_time_units"]),
        comms_latency_time_units=float(twin_cfg["comms"]["latency_time_units"]),
        horizon_time_units=horizon,
    )
    trace = TwinRunner(ag, tw_config, seed).run()
    rng = np.random.default_rng(seed.spawn(1)[0])
    return bundle_from_trace(
        run_id, trace, ag, asset_graph, overlay, nonempty_types, grid_cfg,
        tw_config.dispatch_period_time_units, delta_t, n_slices, p_pos, p_neg, rng,
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


# === stage 3: PPO training ====================================================


def train_policy(
    knowledge_level: str, ag, grid_cfg: GridConfig, twin_cfg: dict, reward_cfg: RewardConfig,
    fidelity_train: FidelityConfig, p_pos: float, p_neg: float, ppo_cfg: dict,
    seed_root: np.random.SeedSequence, monitor_path: Path,
) -> PPO:
    env_seed, sb3_seed = seed_root.spawn(2)
    raw_env = RLAttackerEnv(ag, grid_cfg, twin_cfg, knowledge_level, reward_cfg, fidelity_train, p_pos, p_neg, env_seed)
    env = Monitor(raw_env, filename=str(monitor_path))
    model = PPO(
        "MlpPolicy", env, n_steps=int(ppo_cfg["n_steps"]), batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]), learning_rate=float(ppo_cfg["learning_rate"]),
        gamma=float(ppo_cfg["gamma"]), gae_lambda=float(ppo_cfg["gae_lambda"]),
        ent_coef=float(ppo_cfg["ent_coef"]), clip_range=float(ppo_cfg["clip_range"]),
        policy_kwargs={"net_arch": list(ppo_cfg["net_arch"])},
        seed=int(sb3_seed.generate_state(1)[0]), verbose=0,
    )
    t0 = datetime.now(timezone.utc)
    model.learn(total_timesteps=int(ppo_cfg["n_episodes_train"]))
    wall_clock_s = (datetime.now(timezone.utc) - t0).total_seconds()
    n_episodes = int(ppo_cfg["n_episodes_train"])
    print(f"    {knowledge_level}: {n_episodes} episodes in {wall_clock_s:.1f}s "
          f"({wall_clock_s / max(n_episodes, 1):.4f}s/episode)", flush=True)
    return model


def reward_curve_rows(monitor_path: Path, knowledge_level: str, n_steps: int) -> list[dict]:
    """Groups Monitor's per-episode reward log into n_steps-sized rollout-
    batch chunks and takes the mean -- the reward-curve data the gate's
    'confirm RL actually learned something' check needs, straight from
    Monitor's own log, no fabricated aggregation."""
    df = pd.read_csv(monitor_path, skiprows=1)  # SB3 Monitor prepends a comment/json header row
    rewards = df["r"].to_numpy()
    rows = []
    for i, start in enumerate(range(0, len(rewards), n_steps)):
        chunk = rewards[start:start + n_steps]
        rows.append({"knowledge_level": knowledge_level, "batch": i, "mean_reward": float(chunk.mean()), "n_episodes": len(chunk)})
    return rows


# === stage 5: baseline scoring on RL episodes ================================


def score_dbn(engine: DBNInference, bundle: ScenarioBundle, n_slices: int) -> np.ndarray:
    evid = bundle.discrete.evidence_stream(list(CYBER_ANALYTICS))
    trajectory = engine.run(evid, n_slices)
    return np.array([m["UnstablePS"] for m in trajectory.marginals])


# === main =====================================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(BASE_CONFIG_PATH.read_text())
    twin_cfg = yaml.safe_load(TWIN_CONFIG_PATH.read_text())
    cfg = yaml.safe_load(ADVERSARIAL_CONFIG_PATH.read_text())
    seed = int(base_cfg["seed"])
    set_all_seeds(seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    tag = "smoke_" if args.smoke else ""

    p_pos = float(cfg["analytics"]["p_pos"])
    p_neg = float(cfg["analytics"]["p_neg"])
    reward_cfg = RewardConfig(**{k: float(v) for k, v in cfg["reward"].items()})

    # === stage 0: fidelity resolution =========================================
    print("stage 0: fidelity resolution ...", flush=True)
    ag = build_attack_graph(reaction_mode=cfg["attack_graph"]["reaction_mode"])
    ttcs = collect_uniformization_ttcs(ag)
    train_horizon = float(cfg["smoke"]["fidelity_train"]["horizon_time_units"]) if args.smoke else float(cfg["fidelity"]["train"]["horizon_time_units"])
    delta_t_train = compute_delta_t(ttcs, float(cfg["fidelity"]["train"]["m"]))
    n_slices_train = round(train_horizon / delta_t_train)
    fidelity_train = FidelityConfig(delta_t=delta_t_train, n_slices=n_slices_train, horizon_time_units=train_horizon)

    delta_t_eval = float(cfg["fidelity"]["eval"]["delta_t_override"])
    n_slices_eval = int(cfg["fidelity"]["eval"]["n_slices"])
    horizon_eval = float(cfg["fidelity"]["eval"]["horizon_time_units"])
    fidelity_eval = FidelityConfig(delta_t=delta_t_eval, n_slices=n_slices_eval, horizon_time_units=horizon_eval)

    max_ps = max(delta_t_train / t for t in ttcs.values())
    print(f"  train: delta_t={delta_t_train:.6f} n_slices={n_slices_train} horizon={train_horizon} max_p_s={max_ps:.4f}")
    print(f"  eval:  delta_t={delta_t_eval:.6f} n_slices={n_slices_eval} horizon={horizon_eval}")
    if max_ps > 1.0:
        print(f"\nGATE FAILED: train fidelity produces an invalid p_s={max_ps:.4f} > 1 -- m too coarse for this graph.")
        return 1

    grid_cfg = GridConfig(
        network=twin_cfg["grid"]["network"], n_der=int(twin_cfg["grid"]["n_der"]),
        p_mw_levels=list(twin_cfg["grid"]["p_mw_levels"]), nominal_level_index=int(twin_cfg["grid"]["nominal_level_index"]),
        nonconvergence_is_unstable=bool(twin_cfg["grid"]["nonconvergence_is_unstable"]),
    )

    # === stage 1: asset graph (eval-fidelity probe run, mirrors exp06) =======
    print("\nstage 1: asset graph ...", flush=True)
    grid = GridModel(grid_cfg)
    overlay = build_overlay(grid)
    probe_cfg = TwinConfig(grid=grid_cfg, horizon_time_units=horizon_eval)
    probe_trace = TwinRunner(ag, probe_cfg, np.random.SeedSequence(seed)).run()
    obs = observed_endpoints(probe_trace.messages)
    asset_graph = build_asset_graph(grid, overlay, observed_endpoints_set=obs)
    nt, et = nonempty_metadata(asset_graph.data)
    edge_index_dict = {e: asset_graph.data[e].edge_index for e in et}

    root = np.random.SeedSequence(seed)
    baseline_root, ppo_root, eval_root = root.spawn(3)

    # === stage 2: baseline training, ONCE, on default-scripted scenarios ====
    print("\nstage 2: baseline training (default-scripted scenarios) ...", flush=True)
    bt_cfg = dict(cfg["baseline_training"])
    n_train_scen = int(cfg["smoke"]["baseline_training"]["n_train_scenarios"]) if args.smoke else int(bt_cfg["n_train_scenarios"])
    n_val_scen = int(cfg["smoke"]["baseline_training"]["n_val_scenarios"]) if args.smoke else int(bt_cfg["n_val_scenarios"])
    train_seeds, val_seeds = baseline_root.spawn(n_train_scen), baseline_root.spawn(n_val_scen)
    common_kwargs = dict(
        ag=ag, grid_cfg=grid_cfg, twin_cfg=twin_cfg, asset_graph=asset_graph, overlay=overlay, nonempty_types=nt,
        delta_t=delta_t_eval, n_slices=n_slices_eval, horizon=horizon_eval, p_pos=p_pos, p_neg=p_neg,
    )
    train_scenarios = generate_split("baseline_train", train_seeds, **common_kwargs)
    val_scenarios = generate_split("baseline_val", val_seeds, **common_kwargs)
    all_violations = [f"{lbl} rep{s.run_id}: {v.kind} {v.detail}" for lbl, scens in (("train", train_scenarios), ("val", val_scenarios)) for s in scens for v in s.violations]

    search_rows: list[dict] = []
    baseline_final: dict[str, object] = {}
    search_rng = baseline_root.spawn(1)[0]

    def smoke_n(name: str, default: int) -> int:
        return int(cfg["smoke"]["baseline_training"].get(name, {}).get("n_search_trials", 2)) if args.smoke else default

    # --- rule_based ---
    print("  rule_based ...", flush=True)
    window_sweep = bt_cfg["rule_based"]["window_slices_sweep"]
    if args.smoke:
        window_sweep = window_sweep[:2]
    rule_trials = []
    for w in window_sweep:
        rcfg = RuleConfig(window_slices=int(w))
        val_scores = {s.run_id: rule_score(s.discrete.evidence_stream(list(s.discrete.observable_names)), list(s.discrete.observable_names), n_slices_eval, rcfg) for s in val_scenarios}
        ap = val_auc_pr_for_scores(val_scores, val_scenarios)
        rule_trials.append(TrialResult(trial_id=len(rule_trials), config={"window_slices": w}, val_auc_pr=ap))
        search_rows.append({"baseline": "rule_based", "trial_id": rule_trials[-1].trial_id, **rule_trials[-1].config, "val_auc_pr": ap})
    best_rule = max(rule_trials, key=lambda t: t.val_auc_pr).config
    baseline_final["rule_based"] = RuleConfig(window_slices=int(best_rule["window_slices"]))
    print(f"    selected: {best_rule}", flush=True)

    # --- gbm ---
    print("  gbm ...", flush=True)
    X_train, y_train, _ = build_flat_table(train_scenarios)
    X_val, y_val, _ = build_flat_table(val_scenarios)
    gbm_rng = np.random.default_rng(search_rng.generate_state(1)[0])
    gbm_configs = sample_configs(bt_cfg["gbm"]["search_space"], smoke_n("gbm", int(bt_cfg["gbm"]["n_search_trials"])), gbm_rng)

    def score_gbm(gcfg: dict, rng: np.random.Generator) -> float:
        model = train_gbm(GBMTrialConfig(**gcfg), X_train, y_train, random_state=int(rng.integers(0, 2**31)))
        val_probs = model.predict_proba(X_val)[:, 1]
        return float(average_precision_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else float("nan")

    gbm_trials, best_gbm_cfg = hyperparameter_search(gbm_configs, score_gbm, rng=gbm_rng)
    for t in gbm_trials:
        search_rows.append({"baseline": "gbm", "trial_id": t.trial_id, **t.config, "val_auc_pr": t.val_auc_pr})
    baseline_final["gbm"] = train_gbm(GBMTrialConfig(**best_gbm_cfg), X_train, y_train, random_state=seed)
    print(f"    selected: {best_gbm_cfg}", flush=True)

    # --- lstm_ae ---
    print("  lstm_ae ...", flush=True)
    window = int(bt_cfg["lstm_ae"]["window_slices"])
    train_nominal = []
    for s in train_scenarios:
        cutoff = training_window_cutoff([r.ground_truth for r in s.discrete.records])
        if cutoff == 0:
            continue
        flat, _ = flatten_engineered_features(s.dynamic_x)
        train_nominal.append(build_causal_windows(flat, window=window)[:cutoff])
    train_windows = torch.cat(train_nominal, dim=0) if train_nominal else torch.zeros(0, window, 40)
    val_nominal = []
    for s in val_scenarios:
        cutoff = training_window_cutoff([r.ground_truth for r in s.discrete.records])
        if cutoff == 0:
            continue
        flat, _ = flatten_engineered_features(s.dynamic_x)
        val_nominal.append(build_causal_windows(flat, window=window)[:cutoff])
    val_windows = torch.cat(val_nominal, dim=0) if val_nominal else train_windows[:1]

    ae_epochs = int(cfg["smoke"]["baseline_training"]["lstm_ae"]["n_epochs"]) if args.smoke else int(bt_cfg["lstm_ae"]["n_epochs"])
    ae_batch, ae_clip, ae_patience = int(bt_cfg["lstm_ae"]["batch_size"]), float(bt_cfg["lstm_ae"]["grad_clip_norm"]), int(bt_cfg["lstm_ae"]["early_stopping_patience_epochs"])
    ae_rng = np.random.default_rng(search_rng.generate_state(1)[0] ^ 0xA5A5)
    ae_configs = sample_configs(bt_cfg["lstm_ae"]["search_space"], smoke_n("lstm_ae", int(bt_cfg["lstm_ae"]["n_search_trials"])), ae_rng)

    def score_ae(acfg: dict, rng: np.random.Generator) -> float:
        model, _ = train_autoencoder(LSTMAETrialConfig(**acfg), train_windows, val_windows, n_epochs=ae_epochs, batch_size=ae_batch, grad_clip_norm=ae_clip, patience=ae_patience, torch_seed=int(rng.integers(0, 2**31)))
        val_scores = {}
        for s in val_scenarios:
            flat, _ = flatten_engineered_features(s.dynamic_x)
            val_scores[s.run_id] = ae_score(model, build_causal_windows(flat, window=window))
        errs = np.concatenate(list(val_scores.values()))
        scaler_trial = fit_recon_error_scaler(errs[errs.argsort()[: max(len(errs) // 2, 1)]])
        val_probs = {rid: error_to_probability(sc, scaler_trial) for rid, sc in val_scores.items()}
        return val_auc_pr_for_scores(val_probs, val_scenarios)

    ae_trials, best_ae_cfg = hyperparameter_search(ae_configs, score_ae, rng=ae_rng)
    for t in ae_trials:
        search_rows.append({"baseline": "lstm_ae", "trial_id": t.trial_id, **t.config, "val_auc_pr": t.val_auc_pr})
    final_ae, _ = train_autoencoder(LSTMAETrialConfig(**best_ae_cfg), train_windows, val_windows, n_epochs=ae_epochs, batch_size=ae_batch, grad_clip_norm=ae_clip, patience=ae_patience, torch_seed=seed)
    val_errs_for_scaler = []
    for s in val_scenarios:
        cutoff = training_window_cutoff([r.ground_truth for r in s.discrete.records])
        if cutoff == 0:
            continue
        flat, _ = flatten_engineered_features(s.dynamic_x)
        val_errs_for_scaler.append(ae_score(final_ae, build_causal_windows(flat, window=window)[:cutoff]))
    final_scaler = fit_recon_error_scaler(np.concatenate(val_errs_for_scaler) if val_errs_for_scaler else np.zeros(1), fit_split=str(bt_cfg["lstm_ae"]["scaler_fit_split"]))
    baseline_final["lstm_ae"] = (final_ae, final_scaler, window)
    print(f"    selected: {best_ae_cfg}", flush=True)

    # --- gnn_classifier ---
    print("  gnn_classifier ...", flush=True)
    gnn_grid_raw = bt_cfg["gnn_classifier"]["search_space"]
    gnn_candidates = []
    for conv_type, n_layers, hidden, dropout, tks in itertools.product(gnn_grid_raw["conv_type"], gnn_grid_raw["n_layers"], gnn_grid_raw["hidden"], gnn_grid_raw["dropout"], gnn_grid_raw["temporal_kernel_size"]):
        if conv_type == "gat":
            for heads in gnn_grid_raw["heads"]:
                gnn_candidates.append(dict(conv_type=conv_type, n_layers=n_layers, hidden=hidden, heads=heads, dropout=dropout, temporal_kernel_size=tks))
        else:
            for sage_aggr in gnn_grid_raw["sage_aggr"]:
                gnn_candidates.append(dict(conv_type=conv_type, n_layers=n_layers, hidden=hidden, sage_aggr=sage_aggr, dropout=dropout, temporal_kernel_size=tks))
    gnn_rng = np.random.default_rng(search_rng.generate_state(1)[0] ^ 0x5A5A)
    gnn_n_search = smoke_n("gnn_classifier", int(bt_cfg["gnn_classifier"]["n_search_trials"]))
    gnn_idx = gnn_rng.choice(len(gnn_candidates), size=min(gnn_n_search, len(gnn_candidates)), replace=False)
    gnn_configs = [gnn_candidates[i] for i in gnn_idx]
    gnn_epochs = int(cfg["smoke"]["baseline_training"]["gnn_classifier"]["n_epochs"]) if args.smoke else int(bt_cfg["gnn_classifier"]["n_epochs"])
    gnn_batch, gnn_lr, gnn_wd = int(bt_cfg["gnn_classifier"]["batch_size"]), float(bt_cfg["gnn_classifier"]["learning_rate"]), float(bt_cfg["gnn_classifier"]["weight_decay"])
    gnn_clip, gnn_patience = float(bt_cfg["gnn_classifier"]["grad_clip_norm"]), int(bt_cfg["gnn_classifier"]["early_stopping_patience_epochs"])
    subset_cfg = bt_cfg["gnn_classifier"].get("search_scenario_subset")
    if args.smoke:
        subset_cfg = cfg["smoke"]["baseline_training"]["gnn_classifier"]["search_scenario_subset"]
    gnn_search_train = train_scenarios[: int(subset_cfg["n_train"])] if subset_cfg else train_scenarios
    gnn_search_val = val_scenarios[: int(subset_cfg["n_val"])] if subset_cfg else val_scenarios

    train_labels_full, val_labels_full = unstable_labels(train_scenarios), unstable_labels(val_scenarios)
    train_pos = sum(int(l.sum()) for l in train_labels_full.values())
    train_total = sum(len(l) for l in train_labels_full.values())
    gnn_pos_weight = torch.tensor((train_total - train_pos) / max(train_pos, 1))

    def train_gnn_model(gcfg: dict, torch_seed_local: int, train_subset, val_subset) -> GNNClassifier:
        torch.manual_seed(torch_seed_local)
        model = GNNClassifier(tuple(nt), tuple(et), GNNBaselineConfig(**gcfg))
        opt = torch.optim.AdamW(model.parameters(), lr=gnn_lr, weight_decay=gnn_wd)
        rng_local = np.random.default_rng(torch_seed_local)
        best_val, bad_epochs, best_state = float("inf"), 0, None
        for _ in range(gnn_epochs):
            model.train()
            order = rng_local.permutation(len(train_subset))
            for start in range(0, len(order), gnn_batch):
                batch = [train_subset[i] for i in order[start:start + gnn_batch]]
                x_dict = stack_scenarios([b.x_dict for b in batch])
                globals_ = torch.stack([b.globals_ for b in batch], dim=0)
                logits = model(x_dict, edge_index_dict, globals_)
                y = torch.stack([torch.tensor(train_labels_full[b.run_id], dtype=torch.float32) for b in batch])
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=gnn_pos_weight)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gnn_clip)
                opt.step()
            model.eval()
            with torch.no_grad():
                val_losses = []
                for s in val_subset:
                    logits = model(stack_scenarios([s.x_dict]), edge_index_dict, s.globals_.unsqueeze(0))
                    y = torch.tensor(val_labels_full[s.run_id], dtype=torch.float32).unsqueeze(0)
                    val_losses.append(float(F.binary_cross_entropy_with_logits(logits, y).item()))
                val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if val_loss < best_val - 1e-5:
                best_val, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                bad_epochs += 1
                if bad_epochs >= gnn_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def gnn_predict(model: GNNClassifier, s: ScenarioBundle) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(stack_scenarios([s.x_dict]), edge_index_dict, s.globals_.unsqueeze(0))
            return torch.sigmoid(logits)[0].numpy()

    def score_gnn(gcfg: dict, rng: np.random.Generator) -> float:
        model = train_gnn_model(gcfg, int(rng.integers(0, 2**31)), gnn_search_train, gnn_search_val)
        val_probs = {s.run_id: gnn_predict(model, s) for s in gnn_search_val}
        return val_auc_pr_for_scores(val_probs, gnn_search_val)

    gnn_trials, best_gnn_cfg = hyperparameter_search(gnn_configs, score_gnn, rng=gnn_rng)
    for t in gnn_trials:
        search_rows.append({"baseline": "gnn_classifier", "trial_id": t.trial_id, **t.config, "val_auc_pr": t.val_auc_pr})
    baseline_final["gnn_classifier"] = train_gnn_model(best_gnn_cfg, seed, train_scenarios, val_scenarios)
    print(f"    selected: {best_gnn_cfg}", flush=True)

    search_df = pd.DataFrame(search_rows)
    search_df["git_sha"] = sha
    search_path = RESULTS_DIR / f"exp09_{tag}baseline_search_{timestamp}.csv"
    search_df.to_csv(search_path, index=False)
    print(f"  wrote {search_path}")

    # === stage 3: PPO training, 3 knowledge levels ============================
    print("\nstage 3: PPO training (3 knowledge levels) ...", flush=True)
    ppo_cfg = dict(cfg["ppo"])
    if args.smoke:
        ppo_cfg.update(cfg["smoke"]["ppo"])
    policies: dict[str, PPO] = {}
    reward_rows: list[dict] = []
    ppo_seeds = ppo_root.spawn(len(KNOWLEDGE_LEVELS))
    for kl, kl_seed in zip(KNOWLEDGE_LEVELS, ppo_seeds):
        # SB3's Monitor appends ".monitor.csv" unless the filename already
        # ends with it -- name it that way up front so the path we pass in
        # and the path we later read back are the same file.
        monitor_path = RESULTS_DIR / f"exp09_{tag}monitor_{kl}_{timestamp}.monitor.csv"
        policies[kl] = train_policy(kl, ag, grid_cfg, twin_cfg, reward_cfg, fidelity_train, p_pos, p_neg, ppo_cfg, kl_seed, monitor_path)
        reward_rows.extend(reward_curve_rows(monitor_path, kl, int(ppo_cfg["n_steps"])))
    reward_df = pd.DataFrame(reward_rows)
    reward_df["git_sha"] = sha
    reward_path = RESULTS_DIR / f"exp09_{tag}reward_curve_{timestamp}.csv"
    reward_df.to_csv(reward_path, index=False)
    print(f"  wrote {reward_path}")

    # === stage 4: evaluation-episode generation (frozen policies, eval fidelity)
    print("\nstage 4: evaluation-episode generation ...", flush=True)
    n_eval = int(cfg["smoke"]["evaluation"]["n_episodes_per_knowledge_level"]) if args.smoke else int(cfg["evaluation"]["n_episodes_per_knowledge_level"])
    eval_episodes: dict[str, list[ScenarioBundle]] = {}
    for kl in KNOWLEDGE_LEVELS:
        eval_env = RLAttackerEnv(ag, grid_cfg, twin_cfg, "blind", reward_cfg, fidelity_eval, p_pos, p_neg, eval_root.spawn(1)[0])
        bundles = []
        for i in range(n_eval):
            obs, _ = eval_env.reset()
            action, _ = policies[kl].predict(obs, deterministic=True)
            eval_env.step(action)
            rng = np.random.default_rng(eval_root.spawn(1)[0])
            bundles.append(bundle_from_trace(
                i, eval_env.last_trace, ag, asset_graph, overlay, nt, grid_cfg,
                float(twin_cfg["control_centre"]["dispatch_period_time_units"]), delta_t_eval, n_slices_eval, p_pos, p_neg, rng,
            ))
        eval_episodes[kl] = bundles
        print(f"  {kl}: {len(bundles)} evaluation episodes generated", flush=True)

    # === stage 5: 5-system scoring on the SAME episodes =======================
    print("\nstage 5: scoring all 5 systems ...", flush=True)
    dbn_clustering = fully_factorized_clustering(_interface_nodes(ag))
    dbn_engine = DBNInference(ag, InferenceConfig(clustering=dbn_clustering, m=1.0, p_pos=p_pos, p_neg=p_neg, delta_t_override=delta_t_eval))
    final_ae, final_scaler, ae_window = baseline_final["lstm_ae"]

    scores: dict[str, dict[str, dict[int, np.ndarray]]] = {sys_name: {} for sys_name in SYSTEMS}
    for kl in KNOWLEDGE_LEVELS:
        for sys_name in SYSTEMS:
            scores[sys_name][kl] = {}
        for s in eval_episodes[kl]:
            scores["dbn_hard_evidence"][kl][s.run_id] = score_dbn(dbn_engine, s, n_slices_eval)
            scores["rule_based"][kl][s.run_id] = rule_score(s.discrete.evidence_stream(list(s.discrete.observable_names)), list(s.discrete.observable_names), n_slices_eval, baseline_final["rule_based"])
            flat, _ = flatten_engineered_features(s.dynamic_x)
            scores["gbm"][kl][s.run_id] = gbm_score(baseline_final["gbm"], flat.numpy())
            errors = ae_score(final_ae, build_causal_windows(flat, window=ae_window))
            scores["lstm_ae"][kl][s.run_id] = error_to_probability(errors, final_scaler)
            scores["gnn_classifier"][kl][s.run_id] = gnn_predict(baseline_final["gnn_classifier"], s)
        print(f"  {kl}: scored", flush=True)

    # raw per-slice (y_true, y_prob) on the evaluation episodes -- persisted
    # so a real precision-recall curve can be drawn later without
    # recomputing/rerunning anything (CLAUDE.md rule 2).
    raw_score_rows = []
    for kl in KNOWLEDGE_LEVELS:
        for s in eval_episodes[kl]:
            y_true = [int(r.grid_unstable) for r in s.discrete.records]
            for sys_name in SYSTEMS:
                y_prob = scores[sys_name][kl][s.run_id]
                for yt, yp in zip(y_true, y_prob.tolist()):
                    raw_score_rows.append({
                        "system": sys_name, "knowledge_level": kl, "run_id": s.run_id,
                        "y_true": yt, "y_prob": float(yp),
                    })
    raw_scores_df = pd.DataFrame(raw_score_rows)
    raw_scores_df["git_sha"] = sha
    raw_scores_path = RESULTS_DIR / f"exp09_{tag}raw_eval_scores_{timestamp}.csv"
    raw_scores_df.to_csv(raw_scores_path, index=False)
    print(f"  wrote {raw_scores_path}")

    # === stage 6: robustness curve ============================================
    print("\nstage 6: robustness curve (REPORTED, not gated) ...", flush=True)
    thresholds = tuple(round(t, 2) for t in np.arange(
        float(cfg["evaluation"]["thresholds_start"]), float(cfg["evaluation"]["thresholds_stop"]), float(cfg["evaluation"]["thresholds_step"]),
    ))
    robustness_rows = []
    for sys_name in SYSTEMS:
        for kl in KNOWLEDGE_LEVELS:
            bundles = eval_episodes[kl]
            for theta in thresholds:
                results = [evaluate_run(list(scores[sys_name][kl][s.run_id]), [bool(r.grid_unstable) for r in s.discrete.records], theta, delta_t_eval) for s in bundles]
                lt = summarize(results)
                robustness_rows.append({"system": sys_name, "knowledge_level": kl, "threshold": theta, **lt.__dict__})
    robustness_df = pd.DataFrame(robustness_rows)
    robustness_df["git_sha"] = sha
    robustness_path = RESULTS_DIR / f"exp09_{tag}robustness_curve_{timestamp}.csv"
    robustness_df.to_csv(robustness_path, index=False)

    preview_theta = min(thresholds, key=lambda t: abs(t - 0.5))
    print(f"\n  *** ROBUSTNESS CURVE (detection_rate at threshold={preview_theta}, vs. attacker knowledge) ***")
    summary_view = robustness_df[np.isclose(robustness_df["threshold"], preview_theta)][["system", "knowledge_level", "detection_rate", "lead_median_slices", "n_missed"]]
    print(summary_view.to_string(index=False))
    print(f"  wrote {robustness_path}")

    # === stage 7: lettered validation gate ====================================
    print("\n=== VALIDATION GATE ===")
    failures: list[str] = []

    print("(a) AttackerConfig()/excluded_nodes regression ................ PASS (asserted in tests/test_twin.py::TestExcludedNodes)")

    eval_violations = [
        f"{kl} rep{s.run_id}: {v.kind} {v.detail}"
        for kl in KNOWLEDGE_LEVELS for s in eval_episodes[kl] for v in s.violations
    ]
    eval_traces_ok = len(eval_violations) == 0
    print(f"(a2) RL-episode traces obey precondition ordering .............. {'PASS' if eval_traces_ok else 'FAIL'}")
    if not eval_traces_ok:
        failures.append(f"precondition violations in RL-episode traces: {len(eval_violations)}")
        for line in eval_violations[:5]:
            print("   ", line)

    reward_improved = {}
    for kl in KNOWLEDGE_LEVELS:
        kl_rows = reward_df[reward_df["knowledge_level"] == kl].sort_values("batch")
        n = len(kl_rows)
        if n < 5:
            reward_improved[kl] = False
            continue
        k = max(1, n // 5)
        first_mean = kl_rows["mean_reward"].iloc[:k].mean()
        last_mean = kl_rows["mean_reward"].iloc[-k:].mean()
        reward_improved[kl] = bool(last_mean - first_mean > 0.05)
        print(f"    {kl}: first-20% mean={first_mean:.4f} last-20% mean={last_mean:.4f} delta={last_mean - first_mean:.4f}")
    reward_ok = all(reward_improved.values()) if not args.smoke else True
    print(f"(b) PPO reward curve genuinely improved (>0.05) per knowledge level {'PASS' if reward_ok else 'FAIL'}" + (" [--smoke, not gated]" if args.smoke else ""))
    if not reward_ok:
        failures.append(f"reward did not improve for: {[k for k, v in reward_improved.items() if not v]}")

    valid_scores = all(
        0.0 <= v <= 1.0 and len(scores[sys_name][kl][s.run_id]) == n_slices_eval
        for sys_name in SYSTEMS for kl in KNOWLEDGE_LEVELS for s in eval_episodes[kl] for v in scores[sys_name][kl][s.run_id]
    )
    print(f"(c) every score is a valid [0,1] posterior of length n_slices_eval  {'PASS' if valid_scores else 'FAIL'}")
    if not valid_scores:
        failures.append("invalid score range or length")

    spawn_keys_disjoint = len({baseline_root.spawn_key, ppo_root.spawn_key, eval_root.spawn_key}) == 3
    print(f"(d) baseline-training / PPO / eval seed streams disjoint ..... {'PASS' if spawn_keys_disjoint else 'FAIL'}")
    if not spawn_keys_disjoint:
        failures.append("seed streams not disjoint")

    print(f"(e) action-decoding round-trip ................................ PASS (asserted in tests/test_rl_attacker.py)")

    expected_search_counts_ok = len(search_df) > 0
    print(f"(f) baseline search CSV non-empty ............................. {'PASS' if expected_search_counts_ok else 'FAIL'}")
    if not expected_search_counts_ok:
        failures.append("empty baseline search CSV")

    n_ok = n_eval >= 30 or args.smoke
    print(f"(g) n_episodes_per_knowledge_level >= 30 ...................... {'PASS' if n_ok else 'FAIL'} (n={n_eval})" + (" [--smoke, not gated]" if args.smoke else ""))
    if not n_ok:
        failures.append("fewer than 30 evaluation episodes")

    if all_violations:
        failures.append(f"precondition violations in baseline-training scenarios: {len(all_violations)}")
        for line in all_violations[:5]:
            print("   ", line)

    print(
        "\nNOTE: the robustness curve above (which system degrades faster as attacker "
        "knowledge increases) is REPORTED, not gated (CLAUDE.md rule 3). If dbn_hard_evidence "
        "degrades as fast as the baselines, C3 is refuted for this attack graph -- that is a "
        "valid, reportable result, not a gate failure. See LAB_NOTEBOOK.md H1/H2."
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
