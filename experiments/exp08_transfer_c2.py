"""Learned TTC parameterization: testing claim C2 (CLAUDE.md layer [2]/[3]).

The source paper hand-elicits every attack-step TTC (Table 3) from experts
with no stated derivation -- an admitted weakness this experiment directly
attacks. Claim C2: a model mapping (MITRE technique, asset context,
defensive posture, attacker capability) -> T_bar_s, trained on twin
executions, can match/beat those expert numbers AND transfer to attack
graphs it never saw fitted data for. Per the task's own framing: "transfer
is the entire claim -- same-graph fitting is not a contribution."

See LAB_NOTEBOOK.md 2026-08-05 for the full pre-registration (H1-H5) and the
binding design decisions this script implements without deviation:
  1. Detection-quality via DBN self-consistency (forward-sample a true
     trajectory from oracle CPTs, run EXISTING unmodified DBNInference with
     each arm's TTC-mutated graph, score via src/eval/metrics.py + lead_time.py).
  2. Attacker capability / defensive posture: multiplicative TTC scaling
     (src/twin/attacker.py's speed_multiplier / defense_slowdown_multiplier).
  3. Family graphs (src/attack_graph/family.py) are NEVER twin-executed;
     their ground-truth TTC uses the SAME closed-form multiplicative
     mechanism, applied outside the twin.
  4. Zero modification to src/dbn/parameterization.py or inference.py's
     dispatch -- integration via amortized.apply_ttc_predictions only.
  5. Amortized-model training data pools real twin rows with the 30 TRAIN
     family graphs' synthetic-label rows (accepted circularity risk, H3).

Stages: 0 config -> 1 twin TTC-measurement sweep (task 1) -> 2 family graph
generation (task 3) -> 3 amortized model training (task 2, pooled data) ->
4 zero-shot 3-arm evaluation on the 25 TEST graphs, no retraining (task 4)
-> 5 feature-transfer diagnostic (always run, never gated) -> 6 lettered
validation gate (structural correctness only, CLAUDE.md rule 3).

Run: .venv/bin/python experiments/exp08_transfer_c2.py [--smoke]
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml

from src.attack_graph.family import FamilyGeneratorConfig, family_graph_rows, generate_family
from src.attack_graph.graph import build_attack_graph, technique_table3_ttc
from src.dbn.compiler import compile_to_2tbn
from src.dbn.forward_sample import (
    forward_sample_evidence_stream,
    forward_sample_trajectory,
    validate_slice_trajectory,
)
from src.dbn.inference import DBNInference, InferenceConfig, _interface_nodes, fully_factorized_clustering
from src.dbn.parameterization import attach_cpds, collect_uniformization_ttcs, compute_delta_t
from src.eval.lead_time import evaluate_run, summarize
from src.eval.metrics import binary_kl, m_kl
from src.eval.provenance import git_sha
from src.parameterization.amortized import (
    AmortizedTrainConfig,
    apply_ttc_predictions,
    fit_ttc_amortized_model,
    known_techniques,
    predict_ttc_for_graph,
)
from src.twin.attacker import AttackerConfig, DelayLaw
from src.twin.grid import GridConfig
from src.twin.runner import TwinConfig, TwinRunner, validate_trace

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
TRANSFER_CONFIG_PATH = REPO_ROOT / "configs" / "transfer_c2.yaml"
RESULTS_DIR = REPO_ROOT / "results"

ROW_COLUMNS = ["technique", "asset_context", "defensive_posture", "attacker_capability", "true_ttc"]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# === stage 1: twin TTC-measurement sweep (task 1) ============================


def run_twin_sweep(sweep_cfg: dict, m: float, seed_sequence: np.random.SeedSequence) -> pd.DataFrame:
    ag = build_attack_graph()
    timed_nodes = {n: d["mitre_technique"] for n, d in ag.nodes(data=True) if d.get("ttc") is not None}

    grid_configs = sweep_cfg["grid_configs"]
    speed_multipliers = [float(s) for s in sweep_cfg["speed_multipliers"]]
    defense_multipliers = [float(d) for d in sweep_cfg["defense_slowdown_multipliers"]]
    n_seeds = int(sweep_cfg["n_seeds_per_config"])
    base_horizon = float(sweep_cfg["base_horizon_time_units"])
    safety = float(sweep_cfg["horizon_safety_factor"])
    # Per-COMBO horizon (base_horizon * safety * defense/speed for THAT
    # combo), not one global horizon sized for the single slowest combo
    # applied to every run -- a first full run used a single global
    # worst-case horizon (16x base_horizon) for every combo and took over
    # 40 minutes just for stage 1 before being killed; the fast combos
    # (speed>=1, defense<=1) never needed anywhere near that many events.
    rows: list[dict] = []
    n_missing = 0
    combo_seeds = seed_sequence.spawn(len(grid_configs) * len(speed_multipliers) * len(defense_multipliers) * n_seeds)
    seed_iter = iter(combo_seeds)

    for grid_idx, grid_point in enumerate(grid_configs):
        asset_context = grid_idx / max(len(grid_configs) - 1, 1)
        grid_cfg = GridConfig(n_der=int(grid_point["n_der"]), nominal_level_index=int(grid_point["nominal_level_index"]))
        for speed in speed_multipliers:
            for defense in defense_multipliers:
                horizon = base_horizon * safety * defense / speed
                for _ in range(n_seeds):
                    seed = next(seed_iter)
                    tw_cfg = TwinConfig(
                        grid=grid_cfg,
                        attacker=AttackerConfig(
                            delay_law=DelayLaw.EXPONENTIAL, speed_multiplier=speed, defense_slowdown_multiplier=defense,
                        ),
                        horizon_time_units=horizon,
                    )
                    trace = TwinRunner(ag, tw_cfg, seed).run()
                    violations = validate_trace(trace, ag)
                    if violations:
                        raise AssertionError(f"twin sweep precondition violation: {violations[:3]}")
                    for node, technique in timed_nodes.items():
                        if node not in trace.step_completion_times:
                            n_missing += 1
                            continue
                        realized = trace.step_completion_times[node] - trace.step_eligible_times[node]
                        rows.append({
                            "technique": technique, "asset_context": asset_context,
                            "defensive_posture": defense, "attacker_capability": speed,
                            "true_ttc": realized, "node": node, "n_der": grid_point["n_der"],
                            "nominal_level_index": grid_point["nominal_level_index"],
                        })

    min_horizon = base_horizon * safety * min(defense_multipliers) / max(speed_multipliers)
    max_horizon = base_horizon * safety * max(defense_multipliers) / min(speed_multipliers)
    print(f"  twin sweep: {len(rows)} realized-TTC rows, {n_missing} node-runs never completed "
          f"(per-combo horizon ranged {min_horizon:.1f}-{max_horizon:.1f})")
    return pd.DataFrame(rows)


# === stage 4 helpers ==========================================================


def _constant_prior_ttc(train_rows: pd.DataFrame) -> float:
    """Grand mean of realized_ttc / table3_ttc[technique] across ALL pooled
    training rows -- a single scalar multiplier on Table 3, carrying zero
    technique/context signal (the control arm)."""
    table3 = technique_table3_ttc()
    ratios = train_rows["true_ttc"] / train_rows["technique"].map(table3)
    return float(ratios.mean())


def _arm_ttc_predictions(
    ag, arm: str, model, normalizer, techniques: tuple[str, ...], constant_ratio: float,
) -> dict[str, float]:
    table3 = technique_table3_ttc()
    timed_nodes = {n: d for n, d in ag.nodes(data=True) if d.get("ttc") is not None}
    if arm == "amortized":
        return predict_ttc_for_graph(ag, model, normalizer, techniques)
    if arm == "table3":
        return {n: table3[d["mitre_technique"]] for n, d in timed_nodes.items()}
    if arm == "constant_prior":
        return {n: table3[d["mitre_technique"]] * constant_ratio for n, d in timed_nodes.items()}
    raise ValueError(f"unknown arm {arm!r}")


def _n_slices_for_graph(ag, m: float, multiple: float, cap: int) -> tuple[int, float]:
    ttcs = collect_uniformization_ttcs(ag)
    delta_t = compute_delta_t(ttcs, m)
    n_slices = int(np.ceil(multiple * max(ttcs.values()) / delta_t))
    return min(n_slices, cap), delta_t


# === main =====================================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(BASE_CONFIG_PATH.read_text())
    cfg = yaml.safe_load(TRANSFER_CONFIG_PATH.read_text())
    seed = int(base_cfg["seed"])
    m = float(base_cfg["discretization"]["m"])
    set_all_seeds(seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)
    tag = "smoke_" if args.smoke else ""

    sweep_cfg = cfg["smoke"]["twin_sweep"] if args.smoke else cfg["twin_sweep"]
    family_cfg = dict(cfg["family"])
    if args.smoke:
        family_cfg.update(cfg["smoke"]["family"])
    train_cfg = dict(cfg["training"])
    if args.smoke:
        train_cfg.update(cfg["smoke"]["training"])

    root = np.random.SeedSequence(seed)
    twin_root, family_root, model_root, oracle_root = root.spawn(4)

    # === stage 1: twin TTC-measurement sweep (task 1) ========================
    print("stage 1: twin TTC-measurement sweep ...", flush=True)
    twin_df = run_twin_sweep(sweep_cfg, m, twin_root)
    twin_df["git_sha"] = sha
    twin_path = RESULTS_DIR / f"exp08_{tag}twin_ttc_dataset_{timestamp}.csv"
    twin_df.to_csv(twin_path, index=False)
    print(f"  wrote {twin_path}")

    # === stage 2: family graph generation (task 3) ===========================
    print("\nstage 2: family graph generation ...", flush=True)
    family_config = FamilyGeneratorConfig(
        n_graphs=int(family_cfg["n_graphs"]), n_train=int(family_cfg["n_train"]),
        n_val=int(family_cfg["n_val"]), n_test=int(family_cfg["n_test"]),
        depth_range=tuple(family_cfg["depth_range"]), branching_factor_range=tuple(family_cfg["branching_factor_range"]),
        n_root_processes_range=tuple(family_cfg["n_root_processes_range"]),
        analytic_coverage_range=tuple(family_cfg["analytic_coverage_range"]),
        asset_context_range=tuple(family_cfg["asset_context_range"]),
        defensive_posture_range=tuple(family_cfg["defensive_posture_range"]),
        attacker_capability_range=tuple(family_cfg["attacker_capability_range"]),
    )
    graphs = generate_family(family_config, family_root)
    structural_failures = []
    for fg in graphs:
        try:
            dbn = compile_to_2tbn(fg.ag)
            dbn = attach_cpds(dbn, fg.ag, m=m, p_pos=1e-4, p_neg=1e-4)
            interface = _interface_nodes(fg.ag)
            engine = DBNInference(fg.ag, InferenceConfig(clustering=fully_factorized_clustering(interface), m=m, p_pos=1e-4, p_neg=1e-4))
            engine.run({}, T=2)
        except Exception as e:  # noqa: BLE001
            structural_failures.append((fg.graph_id, str(e)))
    print(f"  generated {len(graphs)} graphs ({family_config.n_train}/{family_config.n_val}/{family_config.n_test} "
          f"train/val/test); structural smoke-check failures: {len(structural_failures)}")

    family_rows_df = family_graph_rows(graphs)
    family_rows_df["git_sha"] = sha
    family_path = RESULTS_DIR / f"exp08_{tag}family_graph_nodes_{timestamp}.csv"
    family_rows_df.to_csv(family_path, index=False)
    print(f"  wrote {family_path}")

    # === stage 3: amortized model training (task 2, pooled data) =============
    print("\nstage 3: amortized model training ...", flush=True)
    family_train_rows = family_rows_df[family_rows_df["split"] == "train"][ROW_COLUMNS]
    family_val_rows = family_rows_df[family_rows_df["split"] == "val"][ROW_COLUMNS]
    twin_rows = twin_df[ROW_COLUMNS]
    pooled_train_rows = pd.concat([twin_rows, family_train_rows], ignore_index=True)
    print(f"  train rows: {len(twin_rows)} twin + {len(family_train_rows)} family_train = {len(pooled_train_rows)}")
    print(f"  val rows: {len(family_val_rows)} (family_val only)")

    techniques = known_techniques()
    amortized_cfg = AmortizedTrainConfig(
        embedding_dim=int(cfg["architecture"]["embedding_dim"]), hidden_dim=int(cfg["architecture"]["hidden_dim"]),
        learning_rate=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]),
        n_epochs=int(train_cfg["n_epochs"]), early_stopping_patience_epochs=int(train_cfg["early_stopping_patience_epochs"]),
        grad_clip_norm=float(train_cfg["grad_clip_norm"]), seed=int(model_root.generate_state(1)[0]),
    )
    model, normalizer, epoch_rows = fit_ttc_amortized_model(pooled_train_rows, family_val_rows, techniques, amortized_cfg)
    training_df = pd.DataFrame(epoch_rows)
    training_df["git_sha"] = sha
    training_path = RESULTS_DIR / f"exp08_{tag}amortized_training_{timestamp}.csv"
    training_df.to_csv(training_path, index=False)
    print(f"  wrote {training_path} ({len(epoch_rows)} epochs, final val_mae_log_ttc={epoch_rows[-1]['val_mae_log_ttc']:.4f})")

    pre_test_state = {k: v.clone() for k, v in model.state_dict().items()}
    constant_ratio = _constant_prior_ttc(pooled_train_rows)
    print(f"  constant-prior ratio (grand mean realized/table3): {constant_ratio:.4f}")

    # === stage 4: zero-shot 3-arm evaluation on 25 TEST graphs, no retraining =
    print("\nstage 4: zero-shot evaluation on test graphs ...", flush=True)
    p_pos = float(cfg["inference"]["p_pos"])
    p_neg = float(cfg["inference"]["p_neg"])
    thresholds = [float(t) for t in cfg["inference"]["detection_thresholds"]]
    multiple = float(family_cfg["n_slices_multiple_of_max_ttc"])
    cap = int(family_cfg["max_n_slices"])
    arms = ["amortized", "table3", "constant_prior"]

    test_graphs = [fg for fg in graphs if fg.split == "test"]
    transfer_rows: list[dict] = []
    trajectory_violation_graphs: list[str] = []
    oracle_seeds = oracle_root.spawn(len(test_graphs))
    results_by_arm_threshold: dict[tuple[str, float], list] = {(arm, t): [] for arm in arms for t in thresholds}

    for fg, oracle_seed in zip(test_graphs, oracle_seeds):
        n_slices, oracle_delta_t = _n_slices_for_graph(fg.ag, m, multiple, cap)
        rng_traj, rng_evid = [np.random.default_rng(s) for s in oracle_seed.spawn(2)]
        oracle_trajectory, _ = forward_sample_trajectory(fg.ag, m, n_slices, rng_traj)
        violations = validate_slice_trajectory(oracle_trajectory, fg.ag)
        if violations:
            trajectory_violation_graphs.append(fg.graph_id)
        evidence = forward_sample_evidence_stream(fg.ag, oracle_trajectory, p_pos, p_neg, rng_evid)

        goal_name = fg.spec.graph_id + "_goal"
        true_goal_flags = [bool(v) for v in oracle_trajectory[goal_name]]

        for arm in arms:
            predictions = _arm_ttc_predictions(fg.ag, arm, model, normalizer, techniques, constant_ratio)
            mutated = apply_ttc_predictions(fg.ag, predictions)
            dbn = compile_to_2tbn(mutated)
            dbn = attach_cpds(dbn, mutated, m=m, p_pos=p_pos, p_neg=p_neg)
            interface = _interface_nodes(mutated)
            arm_delta_t = compute_delta_t(collect_uniformization_ttcs(mutated), m)
            engine = DBNInference(mutated, InferenceConfig(clustering=fully_factorized_clustering(interface), m=m, p_pos=p_pos, p_neg=p_neg))
            arm_trajectory = engine.run(evidence, T=n_slices)
            posterior = [slice_marginals[goal_name] for slice_marginals in arm_trajectory.marginals]

            kl_by_slice = {t + 1: binary_kl(float(true_goal_flags[t]), posterior[t]) for t in range(n_slices)}
            m_kl_value, m_kl_argmax = m_kl(kl_by_slice)

            for theta in thresholds:
                result = evaluate_run(posterior, true_goal_flags, theta, arm_delta_t)
                results_by_arm_threshold[(arm, theta)].append(result)
                transfer_rows.append({
                    "graph_id": fg.graph_id, "arm": arm, "threshold": theta, "n_slices": n_slices,
                    "arm_delta_t": arm_delta_t, "oracle_delta_t": oracle_delta_t,
                    "m_kl": m_kl_value, "m_kl_argmax_slice": m_kl_argmax,
                    "outcome": result.outcome.value, "lead_time_slices": result.lead_time_slices,
                    "lead_time_units": result.lead_time_units,
                })
        print(f"  {fg.graph_id}: n_slices={n_slices} done", flush=True)

    transfer_df = pd.DataFrame(transfer_rows)
    transfer_df["git_sha"] = sha
    transfer_path = RESULTS_DIR / f"exp08_{tag}transfer_eval_{timestamp}.csv"
    transfer_df.to_csv(transfer_path, index=False)
    print(f"  wrote {transfer_path}")

    print("\n  *** per-arm summary across all test graphs/thresholds (REPORTED, not gated) ***")
    lead_time_rows = []
    for arm in arms:
        arm_rows = transfer_df[transfer_df["arm"] == arm]
        mean_mkl = arm_rows.drop_duplicates("graph_id")["m_kl"].mean()
        print(f"  {arm:<16} mean M_KL over test graphs = {mean_mkl:.4f}")
        for theta in thresholds:
            lt_summary = summarize(results_by_arm_threshold[(arm, theta)])
            lead_time_rows.append({"arm": arm, **lt_summary.__dict__})
            print(f"    theta={theta:<5} detection_rate={lt_summary.detection_rate} "
                  f"lead_median_slices={lt_summary.lead_median_slices} n_missed={lt_summary.n_missed}")
    lead_time_df = pd.DataFrame(lead_time_rows)
    lead_time_df["git_sha"] = sha
    lead_time_path = RESULTS_DIR / f"exp08_{tag}lead_time_summary_{timestamp}.csv"
    lead_time_df.to_csv(lead_time_path, index=False)
    print(f"  wrote {lead_time_path}")

    # === stage 5: feature-transfer diagnostic (always run, never gated) ======
    print("\nstage 5: feature-transfer diagnostic ...", flush=True)
    breakdown_rows: list[dict] = []
    test_node_rows = family_rows_df[family_rows_df["split"] == "test"].copy()
    node_predictions = []
    for fg in test_graphs:
        preds = predict_ttc_for_graph(fg.ag, model, normalizer, techniques)
        for node, pred in preds.items():
            node_predictions.append({"graph_id": fg.graph_id, "node": node, "predicted_ttc": pred})
    pred_df = pd.DataFrame(node_predictions)
    merged = test_node_rows.merge(pred_df, on=["graph_id", "node"], how="inner")
    merged["log_abs_error"] = (np.log(merged["predicted_ttc"]) - np.log(merged["true_ttc"])).abs()

    for technique, group in merged.groupby("technique"):
        breakdown_rows.append({"axis": "technique", "bucket": technique, "n": len(group), "mean_log_abs_error": group["log_abs_error"].mean()})
    for axis in ("asset_context", "defensive_posture", "attacker_capability"):
        try:
            merged[f"{axis}_quartile"] = pd.qcut(merged[axis], 4, labels=False, duplicates="drop")
        except ValueError:
            continue
        for q, group in merged.groupby(f"{axis}_quartile"):
            breakdown_rows.append({"axis": axis, "bucket": f"q{int(q)}", "n": len(group), "mean_log_abs_error": group["log_abs_error"].mean()})

    breakdown_df = pd.DataFrame(breakdown_rows)
    breakdown_df["git_sha"] = sha
    breakdown_path = RESULTS_DIR / f"exp08_{tag}transfer_error_breakdown_{timestamp}.csv"
    breakdown_df.to_csv(breakdown_path, index=False)
    print(breakdown_df.to_string(index=False))
    print(f"  wrote {breakdown_path}")

    # === stage 6: lettered validation gate ====================================
    print("\n=== VALIDATION GATE ===")
    failures: list[str] = []

    leak_ag = test_graphs[0].ag
    baseline_preds = predict_ttc_for_graph(leak_ag, model, normalizer, techniques)
    perturbed = leak_ag.copy()
    for node, data in perturbed.nodes(data=True):
        if data.get("ttc") is not None:
            data["ttc"] = data["ttc"] * 1000.0
    perturbed_preds = predict_ttc_for_graph(perturbed, model, normalizer, techniques)
    leak_ok = baseline_preds == perturbed_preds
    print(f"(a) leak-barrier re-check .................. {'PASS' if leak_ok else 'FAIL'}")
    if not leak_ok:
        failures.append("predict_ttc_for_graph output changed under ttc perturbation")

    split_ids = {s: {fg.graph_id for fg in graphs if fg.split == s} for s in ("train", "val", "test")}
    disjoint_ok = (
        len(split_ids["train"] & split_ids["val"]) == 0
        and len(split_ids["train"] & split_ids["test"]) == 0
        and len(split_ids["val"] & split_ids["test"]) == 0
        and len(split_ids["train"]) == family_config.n_train
        and len(split_ids["val"]) == family_config.n_val
        and len(split_ids["test"]) == family_config.n_test
    )
    print(f"(b) graph-level split disjointness .......... {'PASS' if disjoint_ok else 'FAIL'}")
    if not disjoint_ok:
        failures.append("family split not disjoint or wrong sizes")

    no_retrain_ok = all(torch.equal(pre_test_state[k], v) for k, v in model.state_dict().items())
    print(f"(c) no retraining occurred on test graphs ... {'PASS' if no_retrain_ok else 'FAIL'}")
    if not no_retrain_ok:
        failures.append("model weights changed after stage 3 (retraining leaked into stage 4)")

    delta_t_ok = bool((transfer_df["arm_delta_t"] > 0).all())
    print(f"(d) every arm's delta_t > 0 .................. {'PASS' if delta_t_ok else 'FAIL'}")
    if not delta_t_ok:
        failures.append("a non-positive delta_t was produced")

    expected_rows = len(test_graphs) * len(arms) * len(thresholds)
    csv_ok = len(transfer_df) == expected_rows and bool((transfer_df["git_sha"] == sha).all())
    print(f"(e) CSV row-count/provenance ................. {'PASS' if csv_ok else 'FAIL'} "
          f"(rows={len(transfer_df)}, expected={expected_rows})")
    if not csv_ok:
        failures.append("transfer_eval row count or provenance mismatch")

    traj_ok = len(trajectory_violation_graphs) == 0
    print(f"(f) oracle trajectories structurally valid ... {'PASS' if traj_ok else 'FAIL'} "
          f"(violations on: {trajectory_violation_graphs})")
    if not traj_ok:
        failures.append("validate_slice_trajectory found violations on oracle trajectory")

    structural_ok = len(structural_failures) == 0
    print(f"(g) every family graph compiles/infers ....... {'PASS' if structural_ok else 'FAIL'} "
          f"(failures: {structural_failures[:3]})")
    if not structural_ok:
        failures.append(f"{len(structural_failures)} family graphs failed structural smoke-check")

    print(
        "\nNOTE: mean M_KL / detection lead time per arm above are REPORTED, not gated. "
        "A C2 transfer failure is exactly as valid an experimental outcome as a success "
        "(CLAUDE.md rule 3) -- see LAB_NOTEBOOK.md's H1-H5 and the stage-5 diagnostic above "
        "for WHICH features (if any) failed to transfer."
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
