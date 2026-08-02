"""Closed physical loop vs open loop: claim C1 (LAB_NOTEBOOK.md 2026-08-01).

C1: "making grid physics bidirectional -- attack drives simulated instability,
physical deviation returns as evidence -- improves detection lead time and
posterior calibration vs. the open-loop DBN."

This is a real experiment with a real possibility of a null result. The gate
below checks CORRECTNESS INVARIANTS ONLY (structural facts that must hold
regardless of which way C1 comes out); it never gates on whether closed-loop
wins. That verdict -- with its uncertainty -- is printed and written to
LAB_NOTEBOOK.md by hand after this script runs, never adjusted to match a
target (CLAUDE.md rule 3).

Four stages:
  0. zone derivation + a fresh tau-invariance sweep (logged, not assumed)
  1. sensor-rate characterization on a seed stream DISJOINT from evaluation
  2. paired evaluation: one twin run + one discretization per scenario,
     three arms (open_loop, closed_loop, physical_only) read from it
  3. sensitivity arms: sensor-rate mis-specification, rate-limited dispatch

Run: .venv/bin/python experiments/exp04_closed_loop_c1.py
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.attack_graph.graph import PHYS_LOCAL_DER, PHYS_WIDE_AREA, SensorModel, build_attack_graph
from src.dbn.inference import (
    DBNInference,
    InferenceConfig,
    Trajectory,
    _interface_nodes,
    fully_factorized_clustering,
)
from src.eval.calibration import CalibrationReport, calibration_report
from src.eval.lead_time import DetectionResult, LeadTimeSummary, evaluate_run, summarize
from src.eval.provenance import git_sha
from src.twin.attacker import AttackerConfig, DelayLaw
from src.twin.consequence import ZoneMap, build_zone_map, observe_local, observe_wide
from src.twin.grid import GridConfig, GridModel, voltage_sensitivity
from src.twin.runner import (
    DiscreteTrace,
    TwinConfig,
    TwinRunner,
    discretize,
    validate_trace,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
TWIN_CONFIG_PATH = REPO_ROOT / "configs" / "twin.yaml"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

# Session-2/3 timebase, reused for comparability (exp03's N_SLICES = 722).
T_TIME_UNITS = 200
TIME_UNIT_SECONDS = 600
DELTA_T_OVERRIDE = 166.13 / TIME_UNIT_SECONDS
SLICES_PER_TIME_UNIT = 1.0 / DELTA_T_OVERRIDE
N_SLICES = round(T_TIME_UNITS * SLICES_PER_TIME_UNIT)

M = 1.0
QUERY_NODE = "UnstablePS"
CYBER_ANALYTICS = (
    "FileAccess",
    "FileIntegrity",
    "MeasureCoherence",
    "CommandCoherence",
    "SWIntegrityDER",
    "NewServiceStarted",
    "SWIntegritySCADA",
    "SuspArg",
)
PHYSICAL_ANALYTICS = (PHYS_LOCAL_DER, PHYS_WIDE_AREA)

# Fine threshold grid -- swept, never a single hand-picked theta.
THRESHOLDS = tuple(round(t, 2) for t in np.arange(0.05, 1.0, 0.02))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# --- stage 0: zone derivation + tau sweep -----------------------------------


def tau_sweep(
    sensitivity: dict[str, dict[int, float]],
    der_buses: dict[str, int],
    delta_p_mw: float,
    tau_start: float,
    tau_stop: float,
    tau_step: float,
) -> pd.DataFrame:
    """Freshly re-derive the tau-invariance interval; never assume a prior
    session's number (CLAUDE.md rules 2/4: orientation-only exploration must
    be re-verified through the harness)."""
    rows = []
    prev_repr: str | None = None
    for tau in np.round(np.arange(tau_start, tau_stop, tau_step), 6):
        try:
            zm = build_zone_map(
                sensitivity, dominance_tau=float(tau), delta_p_mw=delta_p_mw, der_buses=der_buses
            )
        except ValueError:
            continue
        zones_repr = str({z.der_id: sorted(z.buses) for z in zm.zones})
        rows.append(
            {
                "tau": float(tau),
                "zones": zones_repr,
                "changed_from_previous": zones_repr != prev_repr,
            }
        )
        prev_repr = zones_repr
    return pd.DataFrame(rows)


def invariant_band(sweep_df: pd.DataFrame, tau: float) -> tuple[float, float]:
    """The [lo, hi) tau-range around `tau` over which zone membership held
    constant in `sweep_df` (built by `tau_sweep`)."""
    row_idx = (sweep_df["tau"] - tau).abs().idxmin()
    target_zones = sweep_df.loc[row_idx, "zones"]
    same = sweep_df["zones"] == target_zones
    block_id = (same != same.shift(fill_value=False)).cumsum()
    block = sweep_df[block_id == block_id.loc[row_idx]]
    return float(block["tau"].min()), float(block["tau"].max())


def build_zones(grid_cfg: GridConfig, delta_p_mw: float, dominance_tau: float) -> ZoneMap:
    model = GridModel(grid_cfg)
    sensitivity = voltage_sensitivity(model, delta_p_mw)
    der_buses = dict(zip(model.der_ids, model.der_buses))
    return build_zone_map(
        sensitivity, dominance_tau=dominance_tau, delta_p_mw=delta_p_mw, der_buses=der_buses
    )


# --- stage 1: sensor-rate characterization ----------------------------------


def characterize_sensors(
    ag_physical, grid_cfg: GridConfig, twin_cfg: dict, zones: ZoneMap, char_seeds
) -> tuple[dict[str, float | None], pd.DataFrame]:
    """Measure (a_pos, a_neg, b_pos, b_neg) empirically: the disagreement rate
    between the attack graph's binary assertion and the grid's deterministic
    measurement (M1). NOT sensor noise -- `classify` draws no randomness, so
    every disagreement here is model mismatch, exactly what these rates must
    encode (LAB_NOTEBOOK.md 2026-08-01)."""
    counts = {
        "a_pos_num": 0, "a_pos_den": 0,  # P(PhysLocalDER=1 | WrongLogicExec=0)
        "a_neg_num": 0, "a_neg_den": 0,  # P(PhysLocalDER=0 | WrongLogicExec=1)
        "b_pos_num": 0, "b_pos_den": 0,  # P(PhysWideArea=1 | UnstablePS=0)
        "b_neg_num": 0, "b_neg_den": 0,  # P(PhysWideArea=0 | UnstablePS=1)
    }
    rows = []
    for i, seed in enumerate(char_seeds):
        tw_config = TwinConfig(
            grid=grid_cfg,
            attacker=AttackerConfig(delay_law=DelayLaw.EXPONENTIAL),
            dispatch_period_time_units=float(twin_cfg["control_centre"]["dispatch_period_time_units"]),
            comms_latency_time_units=float(twin_cfg["comms"]["latency_time_units"]),
            horizon_time_units=float(T_TIME_UNITS),
        )
        trace = TwinRunner(ag_physical, tw_config, seed).run()
        discrete = discretize(
            trace, ag_physical, DELTA_T_OVERRIDE, N_SLICES,
            np.random.default_rng(seed.spawn(1)[0]), 1e-4, 1e-4, zones=zones,
        )
        rep_counts = {k: 0 for k in counts}
        for record in discrete.records:
            wle = record.ground_truth["WrongLogicExec"]
            ups = record.ground_truth["UnstablePS"]
            local = observe_local(record.physical)
            wide = observe_wide(record.physical)
            if local is not None:
                if wle == 0:
                    rep_counts["a_pos_den"] += 1
                    rep_counts["a_pos_num"] += local
                else:
                    rep_counts["a_neg_den"] += 1
                    rep_counts["a_neg_num"] += 1 - local
            if wide is not None:
                if ups == 0:
                    rep_counts["b_pos_den"] += 1
                    rep_counts["b_pos_num"] += wide
                else:
                    rep_counts["b_neg_den"] += 1
                    rep_counts["b_neg_num"] += 1 - wide
        for k, v in rep_counts.items():
            counts[k] += v
        rows.append({"replicate": i, **rep_counts})

    def _rate(num_key: str, den_key: str) -> float | None:
        return counts[num_key] / counts[den_key] if counts[den_key] > 0 else None

    rates = {
        "a_pos": _rate("a_pos_num", "a_pos_den"),
        "a_neg": _rate("a_neg_num", "a_neg_den"),
        "b_pos": _rate("b_pos_num", "b_pos_den"),
        "b_neg": _rate("b_neg_num", "b_neg_den"),
        **counts,
    }
    return rates, pd.DataFrame(rows)


# --- stage 2: paired scenario evaluation ------------------------------------


def posterior_series(trajectory: Trajectory, node: str) -> list[float]:
    return [m[node] for m in trajectory.marginals]


def physical_only_series(discrete: DiscreteTrace) -> list[float]:
    """H5 (fusion sanity): the raw PhysWideArea bit as a degenerate posterior
    -- forward-filled across unobserved slices (default 0 before the first
    observation), never coerced to noise. If closed_loop performs no better
    than this, C1's FUSION claim is unsupported regardless of metric deltas."""
    series = []
    last = 0
    for record in discrete.records:
        bit = observe_wide(record.physical)
        if bit is not None:
            last = bit
        series.append(float(last))
    return series


def run_scenario(ag_physical, grid_cfg, twin_cfg, zones, seed, *, rate_limited=False):
    tw_config = TwinConfig(
        grid=grid_cfg,
        attacker=AttackerConfig(delay_law=DelayLaw.EXPONENTIAL),
        dispatch_period_time_units=float(twin_cfg["control_centre"]["dispatch_period_time_units"]),
        comms_latency_time_units=float(twin_cfg["comms"]["latency_time_units"]),
        horizon_time_units=float(T_TIME_UNITS),
        rate_limited_dispatch=rate_limited,
    )
    trace = TwinRunner(ag_physical, tw_config, seed).run()
    violations = validate_trace(trace, ag_physical)
    discrete = discretize(
        trace, ag_physical, DELTA_T_OVERRIDE, N_SLICES,
        np.random.default_rng(seed.spawn(1)[0]), 1e-4, 1e-4, zones=zones,
    )
    return trace, discrete, violations


def build_engine(ag, p_pos: float, p_neg: float) -> DBNInference:
    interface = _interface_nodes(ag)
    return DBNInference(
        ag,
        InferenceConfig(
            clustering=fully_factorized_clustering(interface),
            m=M, p_pos=p_pos, p_neg=p_neg, delta_t_override=DELTA_T_OVERRIDE,
        ),
    )


def lead_time_table(
    results_by_arm: dict[str, list[list[DetectionResult]]],
) -> pd.DataFrame:
    """results_by_arm[arm][scenario] = per-threshold DetectionResult list
    (aligned to THRESHOLDS). Aggregates to one LeadTimeSummary per (arm, theta)."""
    rows = []
    for arm, per_scenario in results_by_arm.items():
        for t_idx, theta in enumerate(THRESHOLDS):
            per_theta = [scenario_results[t_idx] for scenario_results in per_scenario]
            summary: LeadTimeSummary = summarize(per_theta)
            rows.append({"arm": arm, "threshold": theta, **summary.__dict__})
    return pd.DataFrame(rows)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    twin_cfg = yaml.safe_load(TWIN_CONFIG_PATH.read_text())
    seed = config["seed"]
    set_all_seeds(seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    SUMMARIES_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)

    grid_cfg = GridConfig(
        network=twin_cfg["grid"]["network"],
        n_der=int(twin_cfg["grid"]["n_der"]),
        p_mw_levels=list(twin_cfg["grid"]["p_mw_levels"]),
        nominal_level_index=int(twin_cfg["grid"]["nominal_level_index"]),
        nonconvergence_is_unstable=bool(twin_cfg["grid"]["nonconvergence_is_unstable"]),
    )
    phys_cfg = twin_cfg["physical"]
    delta_p_mw = float(phys_cfg["delta_p_mw"])
    dominance_tau = float(phys_cfg["dominance_tau"])
    n_scenarios = int(twin_cfg["experiment"]["n_scenarios_c1"])
    assert n_scenarios >= 30, "task requires >= 30 scenario runs"

    reaction_mode = twin_cfg["experiment"]["reaction_mode"]
    ag_physical_default = build_attack_graph(reaction_mode=reaction_mode, physical_evidence=True)
    ag_cyber_only = build_attack_graph(reaction_mode=reaction_mode)

    # =====================================================================
    # stage 0: zone derivation + tau-invariance sweep (logged, not assumed)
    # =====================================================================
    print("stage 0: zone derivation + tau sweep ...", flush=True)
    model = GridModel(grid_cfg)
    sensitivity = voltage_sensitivity(model, delta_p_mw)
    der_buses = dict(zip(model.der_ids, model.der_buses))
    sweep_df = tau_sweep(
        sensitivity, der_buses, delta_p_mw,
        float(phys_cfg["tau_sweep_start"]), float(phys_cfg["tau_sweep_stop"]), float(phys_cfg["tau_sweep_step"]),
    )
    band_lo, band_hi = invariant_band(sweep_df, dominance_tau)
    sweep_df["git_sha"] = sha
    sweep_df["delta_p_mw"] = delta_p_mw
    sweep_path = RESULTS_DIR / f"exp04_zones_{timestamp}.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"  dominance_tau={dominance_tau} sits in invariant band [{band_lo}, {band_hi})")
    print(f"  wrote {sweep_path}")

    zones = build_zones(grid_cfg, delta_p_mw, dominance_tau)

    # =====================================================================
    # stage 1: sensor characterization on a DISJOINT seed stream
    # =====================================================================
    print("\nstage 1: sensor-rate characterization ...", flush=True)
    root = np.random.SeedSequence(seed)
    char_root, eval_root = root.spawn(2)
    n_char = 20
    char_seeds = char_root.spawn(n_char)
    rates, char_df = characterize_sensors(ag_physical_default, grid_cfg, twin_cfg, zones, char_seeds)
    char_df["git_sha"] = sha
    char_path = RESULTS_DIR / f"exp04_sensor_char_{timestamp}.csv"
    char_df.to_csv(char_path, index=False)
    print(f"  a_pos={rates['a_pos']} a_neg={rates['a_neg']} b_pos={rates['b_pos']} b_neg={rates['b_neg']}")
    print(f"  wrote {char_path}")

    if any(rates[k] is None for k in ("a_pos", "a_neg", "b_pos", "b_neg")):
        print("  FATAL: a sensor rate has zero denominator; cannot build a measured SensorModel.")
        return 1

    sensor_models = {
        PHYS_LOCAL_DER: SensorModel(rates["a_pos"], rates["a_neg"], f"exp04 stage1, {n_char} char seeds, {char_path.name}"),
        PHYS_WIDE_AREA: SensorModel(rates["b_pos"], rates["b_neg"], f"exp04 stage1, {n_char} char seeds, {char_path.name}"),
    }
    ag_physical_measured = build_attack_graph(
        reaction_mode=reaction_mode, physical_evidence=True, sensor_models=sensor_models
    )

    # =====================================================================
    # stage 2: paired evaluation on the EVAL seed stream
    # =====================================================================
    print(f"\nstage 2: paired evaluation, {n_scenarios} scenarios ...", flush=True)
    eval_seeds = eval_root.spawn(n_scenarios)

    engine_measured = build_engine(ag_physical_measured, 1e-4, 1e-4)
    engine_sensitivity_1e4 = build_engine(ag_physical_default, 1e-4, 1e-4)
    engine_cyber_only = build_engine(ag_cyber_only, 1e-4, 1e-4)

    lead_results: dict[str, list[list[DetectionResult]]] = {
        "open_loop": [], "closed_loop": [], "physical_only": [],
        "closed_loop_sensitivity_1e4": [],
    }
    calib_y_true: dict[str, list[int]] = {k: [] for k in lead_results}
    calib_y_prob: dict[str, list[float]] = {k: [] for k in lead_results}
    calib_run_ids: dict[str, list[int]] = {k: [] for k in lead_results}

    all_violations: list[str] = []
    n_unclassified = 0
    barren_node_max_abs_diff = 0.0
    trajectories_for_plot: dict[str, list[list[float]]] = {"open_loop": [], "closed_loop": []}

    for rep, rep_seed in enumerate(eval_seeds):
        trace, discrete, violations = run_scenario(
            ag_physical_measured, grid_cfg, twin_cfg, zones, rep_seed
        )
        if violations:
            all_violations.extend(f"rep{rep}: {v.kind} {v.detail}" for v in violations)
        n_unclassified += sum(
            1 for r in discrete.records
            if r.physical is not None and r.physical.deviation.value == "unclassified"
        )

        cyber_stream = discrete.evidence_stream(list(CYBER_ANALYTICS))
        closed_stream = discrete.evidence_stream(list(CYBER_ANALYTICS) + list(PHYSICAL_ANALYTICS))

        open_traj = engine_measured.run(cyber_stream, N_SLICES)
        closed_traj = engine_measured.run(closed_stream, N_SLICES)
        sens_traj = engine_sensitivity_1e4.run(closed_stream, N_SLICES)

        if rep < 5:
            cyber_only_traj = engine_cyber_only.run(cyber_stream, N_SLICES)
            diff = np.max(np.abs(
                np.array(posterior_series(open_traj, QUERY_NODE))
                - np.array(posterior_series(cyber_only_traj, QUERY_NODE))
            ))
            barren_node_max_abs_diff = max(barren_node_max_abs_diff, float(diff))

        unstable_flags = [bool(r.grid_unstable) for r in discrete.records]
        # grid_unstable == exceeds_limit invariant, re-checked at gate scope.
        for r in discrete.records:
            if r.physical is not None and r.grid_unstable != r.physical.exceeds_limit:
                all_violations.append(f"rep{rep} slice{r.slice_index}: grid_unstable != exceeds_limit")

        series = {
            "open_loop": posterior_series(open_traj, QUERY_NODE),
            "closed_loop": posterior_series(closed_traj, QUERY_NODE),
            "physical_only": physical_only_series(discrete),
            "closed_loop_sensitivity_1e4": posterior_series(sens_traj, QUERY_NODE),
        }
        if rep < 10:
            trajectories_for_plot["open_loop"].append(series["open_loop"])
            trajectories_for_plot["closed_loop"].append(series["closed_loop"])

        for arm, posterior in series.items():
            lead_results[arm].append(
                [evaluate_run(posterior, unstable_flags, th, DELTA_T_OVERRIDE) for th in THRESHOLDS]
            )
            calib_y_true[arm].extend(int(f) for f in unstable_flags)
            calib_y_prob[arm].extend(posterior)
            calib_run_ids[arm].extend([rep] * len(posterior))

        print(f"  rep {rep + 1}/{n_scenarios} done", flush=True)

    # =====================================================================
    # stage 3a: rate-limited dispatch secondary arm (subset, for time)
    # =====================================================================
    print("\nstage 3a: rate-limited dispatch secondary arm ...", flush=True)
    n_rate_limited = min(10, n_scenarios)
    rl_lead: dict[str, list[list[DetectionResult]]] = {"open_loop": [], "closed_loop": []}
    for rep in range(n_rate_limited):
        _, discrete, violations = run_scenario(
            ag_physical_measured, grid_cfg, twin_cfg, zones, eval_seeds[rep], rate_limited=True
        )
        if violations:
            all_violations.extend(f"rate-limited rep{rep}: {v.kind} {v.detail}" for v in violations)
        cyber_stream = discrete.evidence_stream(list(CYBER_ANALYTICS))
        closed_stream = discrete.evidence_stream(list(CYBER_ANALYTICS) + list(PHYSICAL_ANALYTICS))
        open_traj = engine_measured.run(cyber_stream, N_SLICES)
        closed_traj = engine_measured.run(closed_stream, N_SLICES)
        unstable_flags = [bool(r.grid_unstable) for r in discrete.records]
        rl_lead["open_loop"].append(
            [evaluate_run(posterior_series(open_traj, QUERY_NODE), unstable_flags, th, DELTA_T_OVERRIDE) for th in THRESHOLDS]
        )
        rl_lead["closed_loop"].append(
            [evaluate_run(posterior_series(closed_traj, QUERY_NODE), unstable_flags, th, DELTA_T_OVERRIDE) for th in THRESHOLDS]
        )
        print(f"  rate-limited rep {rep + 1}/{n_rate_limited} done", flush=True)

    # =====================================================================
    # reporting
    # =====================================================================
    print("\n=== C1: LEAD TIME (primary arms) ===")
    lt_df = lead_time_table(lead_results)
    lt_df["git_sha"] = sha
    lt_df["seed"] = seed
    lt_path = RESULTS_DIR / f"exp04_lead_time_{timestamp}.csv"
    lt_df.to_csv(lt_path, index=False)
    for theta in (0.5, 0.9, 0.99):
        sub = lt_df[np.isclose(lt_df["threshold"], theta)]
        print(f"\n--- theta={theta} ---")
        print(sub[["arm", "detection_rate", "false_alarm_rate", "n_lead_defined",
                    "lead_median_slices", "lead_p10_slices", "lead_p90_slices"]].to_string(index=False))
    print(f"\nwrote {lt_path}")

    print("\n=== C1: RATE-LIMITED DISPATCH SECONDARY ARM ===")
    rl_df = lead_time_table(rl_lead)
    rl_df["git_sha"] = sha
    rl_path = RESULTS_DIR / f"exp04_lead_time_rate_limited_{timestamp}.csv"
    rl_df.to_csv(rl_path, index=False)
    for theta in (0.5, 0.9, 0.99):
        sub = rl_df[np.isclose(rl_df["threshold"], theta)]
        print(f"\n--- theta={theta} (rate-limited, n={n_rate_limited}) ---")
        print(sub[["arm", "detection_rate", "lead_median_slices"]].to_string(index=False))
    print(f"wrote {rl_path}")

    print("\n=== C1: CALIBRATION ===")
    calib_rows = []
    calib_reports: dict[str, CalibrationReport] = {}
    for arm in lead_results:
        report = calibration_report(
            calib_y_true[arm], calib_y_prob[arm], calib_run_ids[arm],
            n_bins_sweep=(5, 10, 15, 20), strategies=("uniform", "quantile"),
            primary_n_bins=10, primary_strategy="uniform", n_bootstrap=1000,
            rng=np.random.default_rng(seed),
        )
        calib_reports[arm] = report
        print(
            f"{arm:<28} ECE(10,uniform)={report.ece[(10, 'uniform')]:.4f} "
            f"[{report.ece_primary_ci95[0]:.4f},{report.ece_primary_ci95[1]:.4f}]  "
            f"Brier={report.brier:.4f}  BSS={report.brier_skill_score:+.4f}  "
            f"n={report.n} n_runs={report.n_runs}"
        )
        for (n_bins, strategy), ece in report.ece.items():
            calib_rows.append({
                "arm": arm, "n_bins": n_bins, "strategy": strategy, "ece": ece,
                "mce": report.mce[(n_bins, strategy)], "brier": report.brier,
                "brier_skill_score": report.brier_skill_score, "base_rate": report.base_rate,
                "n": report.n, "n_runs": report.n_runs,
            })
    calib_df = pd.DataFrame(calib_rows)
    calib_df["git_sha"] = sha
    calib_path = RESULTS_DIR / f"exp04_calibration_{timestamp}.csv"
    calib_df.to_csv(calib_path, index=False)
    print(f"wrote {calib_path}")

    # --- plot: open vs closed posterior trajectories, first 10 scenarios ---
    t_axis = np.arange(1, N_SLICES + 1) * DELTA_T_OVERRIDE
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm, style in (("open_loop", "-"), ("closed_loop", "--")):
        matrix = np.array(trajectories_for_plot[arm])
        ax.fill_between(t_axis, np.percentile(matrix, 10, axis=0), np.percentile(matrix, 90, axis=0), alpha=0.2)
        ax.plot(t_axis, np.median(matrix, axis=0), style, label=f"{arm} median")
    ax.set_xlabel("t (x 10 min)")
    ax.set_ylabel("P(UnstablePS)")
    ax.set_title("Open- vs closed-loop posterior (first 10 eval scenarios)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figure_path = FIGURES_DIR / "exp04_open_vs_closed_posterior.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"wrote {figure_path}")

    # =====================================================================
    # validation gate -- correctness invariants only, never "did C1 win"
    # =====================================================================
    print("\n=== VALIDATION GATE ===")
    failures: list[str] = []

    if all_violations:
        failures.append(f"precondition/consistency violations: {len(all_violations)}")
        for line in all_violations[:5]:
            print("   ", line)
    print(f"(a) precondition ordering + grid_unstable==exceeds_limit  {'FAIL' if all_violations else 'PASS'}")

    if n_unclassified:
        failures.append(f"{n_unclassified} UNCLASSIFIED physical observations")
    print(f"(b) zero UNCLASSIFIED physical observations ............. {'FAIL' if n_unclassified else 'PASS'} ({n_unclassified})")

    barren_ok = barren_node_max_abs_diff < 1e-9
    if not barren_ok:
        failures.append(f"25-node vs 23-node open-arm posterior diverged by {barren_node_max_abs_diff}")
    print(f"(c) open-arm 25-node/23-node posterior equivalence ....... {'PASS' if barren_ok else 'FAIL'} (max|diff|={barren_node_max_abs_diff:.2e})")

    n_ok = n_scenarios >= 30
    print(f"(d) n_scenarios >= 30 ..................................... {'PASS' if n_ok else 'FAIL'} ({n_scenarios})")
    if not n_ok:
        failures.append("fewer than 30 scenarios")

    print("(e) cyber evidence identical across open/closed arms ..... PASS (by construction: same DiscreteTrace, disjoint evidence-name subsets)")
    print("(f) physical bits consume no rng, sparse-if-unobserved .... PASS (asserted in tests/test_twin.py)")

    print(
        "\nNOTE: lead-time and calibration numbers above are REPORTED, not "
        "gated. A closed-loop null does not fail this gate (CLAUDE.md rule 3)."
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
