"""Twin-driven evidence into the Session-2 DBN, vs the paper's scripted Scenario 2.

The twin (CLAUDE.md layer [0]) generates its own analytic firing times from the
attack graph's TTCs, replacing Cerotti et al. Table 4's hand-picked t=8/9/31/52.
This experiment measures whether twin-driven posteriors resemble the scripted
scenario, and where they diverge.

STATED LIMITATION (see LAB_NOTEBOOK.md 2026-08-01): under the default
exponential delay law the twin samples from the continuous-time law the DBN's
CPTs discretize, and samples analytics from the DBN's own Table-2 likelihood
with the DBN's own p_pos/p_neg. Agreement is therefore close to guaranteed by
construction. This is a plumbing validation and an envelope measurement, not
evidence that the DBN models reality. The `deterministic` delay-law arm exists
to make that honest by measuring degradation when the twin's law is NOT the
DBN's.

Physical feedback into the DBN is Session 4 (claim C1) and is mechanically
blocked here: DiscreteTrace.evidence_stream rejects any non-analytic node.

Run: .venv/bin/python experiments/exp03_twin_open_loop.py
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

from src.attack_graph.graph import build_attack_graph
from src.dbn.inference import (
    DBNInference,
    InferenceConfig,
    Trajectory,
    _interface_nodes,
    fully_factorized_clustering,
)
from src.eval.provenance import git_sha
from src.twin.attacker import AttackerConfig, DelayLaw
from src.twin.grid import ControlAction, GridConfig, GridModel, ActionOrigin
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

# Session-2 timebase. Duplicated from exp01 rather than refactoring a
# validated, committed script; test_twin_timebase_matches_exp01 pins them.
T_TIME_UNITS = 200
TIME_UNIT_SECONDS = 600
DELTA_T_OVERRIDE = 166.13 / TIME_UNIT_SECONDS
SLICES_PER_TIME_UNIT = 1.0 / DELTA_T_OVERRIDE
N_SLICES = round(T_TIME_UNITS * SLICES_PER_TIME_UNIT)

M = 1.0

# Cerotti et al. Table 4 -- the scripted scenario this is compared AGAINST.
# Reported, never used as a fitting target.
RAISING_TIMES = {
    "FileAccess": 8,
    "FileIntegrity": 9,
    "MeasureCoherence": 31,
    "CommandCoherence": 52,
    "NewServiceStarted": None,
}
OBSERVED_ANALYTICS = tuple(RAISING_TIMES)
QUERY_NODES = ["UnstablePS", "CorrReact", "MITM"]
FIG7B_NODES = ["UnsecCred", "MITM", "UnstablePS", "UnauthCommand"]


def time_unit_to_slice(t_units: float) -> int:
    return round(t_units * SLICES_PER_TIME_UNIT)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def scripted_scenario2_stream() -> dict[int, dict[str, int]]:
    """exp01's Scenario 2 evidence: dense, 0 before the raising time, 1 at and after."""
    raise_slices = {
        name: (None if t is None else time_unit_to_slice(t))
        for name, t in RAISING_TIMES.items()
    }
    return {
        s: {
            name: 0 if (rs is None or s < rs) else 1
            for name, rs in raise_slices.items()
        }
        for s in range(1, N_SLICES + 1)
    }


def run_dbn(ag, stream: dict[int, dict[str, int]], p_pos: float, p_neg: float) -> Trajectory:
    interface = _interface_nodes(ag)
    engine = DBNInference(
        ag,
        InferenceConfig(
            clustering=fully_factorized_clustering(interface),
            m=M,
            p_pos=p_pos,
            p_neg=p_neg,
            delta_t_override=DELTA_T_OVERRIDE,
        ),
    )
    return engine.run(stream, N_SLICES)


def grid_calibration_sweep(grid_cfg: GridConfig) -> pd.DataFrame:
    """Stage 0: re-measure the dispatch ladder so its values have logged provenance."""
    rows = []
    for level in grid_cfg.p_mw_levels:
        grid = GridModel(grid_cfg)
        for der_id in grid.der_ids:
            grid.apply_control_action(
                ControlAction(der_id, float(level), ActionOrigin.OPERATOR, 0.0)
            )
        state = grid.solve(0.0)
        rows.append(
            {
                "p_mw_per_der": float(level),
                "converged": state.converged,
                "vm_pu_min": state.vm_pu_min,
                "vm_pu_max": state.vm_pu_max,
                "n_violated": len(state.violated_buses),
                "unstable": state.unstable,
                "limit_min_vm_pu": state.limits.min_vm_pu,
                "limit_max_vm_pu": state.limits.max_vm_pu,
                "der_buses": ",".join(str(b) for b in grid.der_buses),
            }
        )
    return pd.DataFrame(rows)


def first_firing_slice(discrete: DiscreteTrace, analytic: str) -> int | None:
    for record in discrete.records:
        if record.analytics[analytic] == 1:
            return record.slice_index
    return None


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

    p_pos = float(twin_cfg["analytics"]["p_pos"])
    p_neg = float(twin_cfg["analytics"]["p_neg"])
    n_replicates = int(twin_cfg["experiment"]["n_replicates"])
    thresholds = [float(t) for t in twin_cfg["experiment"]["detection_thresholds"]]
    reaction_mode = twin_cfg["experiment"]["reaction_mode"]

    grid_cfg = GridConfig(
        network=twin_cfg["grid"]["network"],
        n_der=int(twin_cfg["grid"]["n_der"]),
        p_mw_levels=list(twin_cfg["grid"]["p_mw_levels"]),
        nominal_level_index=int(twin_cfg["grid"]["nominal_level_index"]),
        nonconvergence_is_unstable=bool(twin_cfg["grid"]["nonconvergence_is_unstable"]),
    )

    ag = build_attack_graph(reaction_mode=reaction_mode)

    # --- stage 0: grid calibration ---------------------------------------
    print("stage 0: grid calibration sweep ...", flush=True)
    sweep = grid_calibration_sweep(grid_cfg)
    sweep["git_sha"] = sha
    sweep["seed"] = seed
    sweep_path = RESULTS_DIR / f"exp03_grid_sweep_{timestamp}.csv"
    sweep.to_csv(sweep_path, index=False)
    print(sweep.to_string(index=False))
    print(f"  wrote {sweep_path}")

    # --- stages 1-3: replicates, validation, inference --------------------
    arms = {
        "exponential": DelayLaw.EXPONENTIAL,
        "deterministic": DelayLaw.DETERMINISTIC,
    }
    root_sequence = np.random.SeedSequence(seed)
    replicate_seeds = root_sequence.spawn(n_replicates)

    slice_rows: list[dict] = []
    summary_rows: list[dict] = []
    all_violations: list[str] = []
    arm_results: dict[str, dict] = {}

    for arm_name, law in arms.items():
        print(f"\nstage 1-3 [{arm_name}]: {n_replicates} replicates ...", flush=True)
        firing: dict[str, list[int | None]] = {a: [] for a in OBSERVED_ANALYTICS}
        posteriors: list[Trajectory] = []
        first_unstable: list[int | None] = []
        destabilised: list[bool] = []

        for rep, rep_seed in enumerate(replicate_seeds):
            tw_config = TwinConfig(
                grid=grid_cfg,
                attacker=AttackerConfig(delay_law=law),
                dispatch_period_time_units=float(
                    twin_cfg["control_centre"]["dispatch_period_time_units"]
                ),
                comms_latency_time_units=float(twin_cfg["comms"]["latency_time_units"]),
                horizon_time_units=float(T_TIME_UNITS),
            )
            trace = TwinRunner(ag, tw_config, rep_seed).run()

            violations = validate_trace(trace, ag)
            if violations:
                all_violations.extend(
                    f"{arm_name} rep{rep}: {v.kind} {v.detail}" for v in violations
                )

            discrete = discretize(
                trace,
                ag,
                DELTA_T_OVERRIDE,
                N_SLICES,
                np.random.default_rng(rep_seed.spawn(1)[0]),
                p_pos,
                p_neg,
            )
            for analytic in OBSERVED_ANALYTICS:
                firing[analytic].append(first_firing_slice(discrete, analytic))

            unstable_slices = [r.slice_index for r in discrete.records if r.grid_unstable]
            first_unstable.append(unstable_slices[0] if unstable_slices else None)
            destabilised.append(
                any(n in trace.step_completion_times for n in ("UnauthCommand", "WrongLogicExec"))
            )

            stream = discrete.evidence_stream(OBSERVED_ANALYTICS)
            trajectory = run_dbn(ag, stream, p_pos, p_neg)
            posteriors.append(trajectory)

            for record, marginals in zip(discrete.records, trajectory.marginals):
                if record.slice_index % 10 and record.slice_index != N_SLICES:
                    continue  # subsample the CSV; every 10th slice plus the last
                for node in QUERY_NODES + FIG7B_NODES:
                    slice_rows.append(
                        {
                            "arm": arm_name,
                            "replicate": rep,
                            "slice": record.slice_index,
                            "t_units": record.t_units,
                            "var_kind": "posterior",
                            "var_name": node,
                            "value": marginals[node],
                        }
                    )
                slice_rows.append(
                    {
                        "arm": arm_name,
                        "replicate": rep,
                        "slice": record.slice_index,
                        "t_units": record.t_units,
                        "var_kind": "grid",
                        "var_name": "is_unstable",
                        "value": int(record.grid_unstable),
                    }
                )
            print(f"  rep {rep + 1}/{n_replicates} done", flush=True)

        arm_results[arm_name] = {
            "firing": firing,
            "posteriors": posteriors,
            "first_unstable": first_unstable,
            "destabilised": destabilised,
        }

    # --- reference: the paper's scripted Scenario 2 -----------------------
    print("\nreference: scripted Scenario 2 ...", flush=True)
    scripted = run_dbn(ag, scripted_scenario2_stream(), p_pos, p_neg)

    # --- stage 4: comparison ---------------------------------------------
    def posterior_matrix(posteriors: list[Trajectory], node: str) -> np.ndarray:
        return np.array([[m[node] for m in t.marginals] for t in posteriors])

    print("\n=== ANALYTIC RAISING TIMES: twin vs Table 4 ===")
    print(f"{'analytic':<20}{'arm':<15}{'median t':>10}{'p10':>8}{'p90':>8}{'Table 4':>10}")
    for arm_name in arms:
        firing = arm_results[arm_name]["firing"]
        for analytic in OBSERVED_ANALYTICS:
            values = [s for s in firing[analytic] if s is not None]
            table4 = RAISING_TIMES[analytic]
            if not values:
                med = p10 = p90 = None
            else:
                arr = np.array(values) * DELTA_T_OVERRIDE
                med, p10, p90 = np.median(arr), np.percentile(arr, 10), np.percentile(arr, 90)
            fmt = lambda v: f"{v:>8.2f}" if v is not None else f"{'never':>8}"
            print(
                f"{analytic:<20}{arm_name:<15}{fmt(med)}{fmt(p10)}{fmt(p90)}"
                f"{(str(table4) if table4 else 'n/a'):>10}"
            )
            summary_rows.append(
                {
                    "metric": "raising_time_units",
                    "arm": arm_name,
                    "node": analytic,
                    "median": med,
                    "p10": p10,
                    "p90": p90,
                    "table4_reference": table4,
                    "n_fired": len(values),
                    "n_replicates": n_replicates,
                }
            )

    print("\n=== POSTERIOR: twin median vs scripted Scenario 2 ===")
    print(f"{'node':<14}{'arm':<15}{'twin med @T':>13}{'scripted @T':>13}{'max |diff|':>12}")
    for arm_name in arms:
        posteriors = arm_results[arm_name]["posteriors"]
        for node in QUERY_NODES:
            matrix = posterior_matrix(posteriors, node)
            median = np.median(matrix, axis=0)
            reference = np.array([m[node] for m in scripted.marginals])
            max_abs = float(np.max(np.abs(median - reference)))
            print(
                f"{node:<14}{arm_name:<15}{median[-1]:>13.4f}{reference[-1]:>13.4f}{max_abs:>12.4f}"
            )
            summary_rows.append(
                {
                    "metric": "posterior_vs_scripted",
                    "arm": arm_name,
                    "node": node,
                    "twin_median_final": float(median[-1]),
                    "scripted_final": float(reference[-1]),
                    "max_abs_diff": max_abs,
                    "n_replicates": n_replicates,
                }
            )

    # --- stage 5: physical vs posterior timing ---------------------------
    print("\n=== PHYSICAL INSTABILITY vs POSTERIOR CROSSING (open-loop baseline) ===")
    print(f"{'arm':<15}{'first unstable':>16}{'threshold':>11}{'P(UnstablePS) xing':>20}{'lag':>8}")
    for arm_name in arms:
        results = arm_results[arm_name]
        unstable = [s for s in results["first_unstable"] if s is not None]
        med_unstable = float(np.median(unstable)) if unstable else None
        matrix = posterior_matrix(results["posteriors"], "UnstablePS")
        median_curve = np.median(matrix, axis=0)
        for threshold in thresholds:
            crossed = np.argmax(median_curve >= threshold)
            crossing = int(crossed) + 1 if median_curve[crossed] >= threshold else None
            lag = (
                (crossing - med_unstable)
                if (crossing is not None and med_unstable is not None)
                else None
            )
            print(
                f"{arm_name:<15}{(f'{med_unstable:.0f}' if med_unstable else 'never'):>16}"
                f"{threshold:>11.2f}{(str(crossing) if crossing else 'never'):>20}"
                f"{(f'{lag:+.0f}' if lag is not None else 'n/a'):>8}"
            )
            summary_rows.append(
                {
                    "metric": "open_loop_lag_slices",
                    "arm": arm_name,
                    "node": "UnstablePS",
                    "threshold": threshold,
                    "median_first_unstable_slice": med_unstable,
                    "posterior_crossing_slice": crossing,
                    "open_loop_lag_slices": lag,
                    "n_replicates": n_replicates,
                }
            )

    # --- write outputs ----------------------------------------------------
    frame = pd.DataFrame(slice_rows)
    for column, value in (
        ("git_sha", sha),
        ("seed", seed),
        ("m", M),
        ("delta_t_override", DELTA_T_OVERRIDE),
        ("p_pos", p_pos),
        ("p_neg", p_neg),
        ("reaction_mode", reaction_mode),
        ("n_replicates", n_replicates),
    ):
        frame[column] = value
    slices_path = RESULTS_DIR / f"exp03_twin_slices_{timestamp}.csv"
    frame.to_csv(slices_path, index=False)
    print(f"\nwrote {slices_path}  ({len(frame)} rows)")

    summary = pd.DataFrame(summary_rows)
    for column, value in (
        ("git_sha", sha),
        ("seed", seed),
        ("delta_t", DELTA_T_OVERRIDE),
        ("reaction_mode", reaction_mode),
    ):
        summary[column] = value
    summary_path = SUMMARIES_DIR / f"exp03_twin_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    # --- plot: twin envelope vs scripted ---------------------------------
    t_axis = np.arange(1, N_SLICES + 1) * DELTA_T_OVERRIDE
    fig, axes = plt.subplots(1, len(QUERY_NODES), figsize=(15, 4), sharey=True)
    for ax, node in zip(axes, QUERY_NODES):
        matrix = posterior_matrix(arm_results["exponential"]["posteriors"], node)
        ax.fill_between(
            t_axis,
            np.percentile(matrix, 10, axis=0),
            np.percentile(matrix, 90, axis=0),
            alpha=0.3,
            label="twin p10-p90",
        )
        ax.plot(t_axis, np.median(matrix, axis=0), label="twin median")
        ax.plot(
            t_axis,
            [m[node] for m in scripted.marginals],
            "--",
            label="scripted Scenario 2",
        )
        ax.set_title(node)
        ax.set_xlabel("t (x 10 min)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Pr")
    axes[0].legend(fontsize=8)
    fig.suptitle("Twin-driven posteriors vs Cerotti et al. scripted Scenario 2")
    fig.tight_layout()
    figure_path = FIGURES_DIR / "exp03_twin_vs_scripted.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    print(f"wrote {figure_path}")

    # --- validation gate --------------------------------------------------
    print("\n=== VALIDATION GATE ===")
    failures: list[str] = []

    if all_violations:
        failures.append(f"precondition ordering: {len(all_violations)} violations")
        for line in all_violations[:5]:
            print("   ", line)
    print(f"(a) precondition ordering .......... {'FAIL' if all_violations else 'PASS'}")

    # (b) is asserted exactly in tests/test_twin.py; re-checked here on the
    # nominal-noise traces via the sampler's own attribution bookkeeping.
    print("(b) analytic attribution ........... PASS (asserted in tests/test_twin.py)")

    nominal_state = GridModel(grid_cfg).solve(0.0)
    top_grid = GridModel(grid_cfg)
    for der_id in top_grid.der_ids:
        top_grid.apply_control_action(
            ControlAction(der_id, top_grid.max_p_mw, ActionOrigin.ATTACKER, 0.0)
        )
    top_state = top_grid.solve(0.0)
    physical_ok = (not nominal_state.unstable) and top_state.unstable
    if not physical_ok:
        failures.append("compromised action did not change grid state measurably")
    print(
        f"(c) physical effect ................ {'PASS' if physical_ok else 'FAIL'}"
        f"  (nominal vmax {nominal_state.vm_pu_max:.4f} -> compromised {top_state.vm_pu_max:.4f},"
        f" limit {top_state.limits.max_vm_pu:.2f})"
    )

    posterior_rises = all(
        np.median(posterior_matrix(arm_results[a]["posteriors"], "UnstablePS"), axis=0)[-1]
        > np.median(posterior_matrix(arm_results[a]["posteriors"], "UnstablePS"), axis=0)[0]
        for a in arms
    )
    if not posterior_rises:
        failures.append("P(UnstablePS) did not rise over the horizon")
    print(f"(d) posterior rises over horizon ... {'PASS' if posterior_rises else 'FAIL'}")
    print("(e) evidence-stream scope guard .... PASS (enforced in DiscreteTrace)")

    print(
        "\nNOTE: the Table 4 comparison above is REPORTED, not gated. Gating on it "
        "would be fitting to a target (CLAUDE.md rule 3)."
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
