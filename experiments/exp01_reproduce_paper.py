"""Reproduce Cerotti et al. Sec. IV, Scenarios 1 and 2.

Runs FF and EX clustered forward filtering over the paper's T=200 time-unit
horizon (~722 DBN slices at delta_t=166.13 s) under both
scenarios, logs per-timestep posteriors and latency to results/, computes
KL(EX||FF) and M_KL (Eq. 5) for {UnstablePS, CorrReact, MITM}, and produces
plots matching Figs. 5-8's node groupings and layout.

This is the project's validation gate (CLAUDE.md): the printed comparison
table at the end checks the actual computed numbers against the paper's
published values. No target is adjusted to fit the output.

Run: .venv/bin/python experiments/exp01_reproduce_paper.py
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.attack_graph.graph import build_attack_graph
from src.dbn.inference import (
    DBNInference,
    InferenceConfig,
    Trajectory,
    exact_clustering,
    fully_factorized_clustering,
)
from src.eval.provenance import git_sha
from src.eval.metrics import MemoryReport, binary_kl, m_kl, measure_memory

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

# Cerotti et al. Sec. IV's own experiment settings, not configs/base.yaml's
# scaffolding defaults (that file's horizon_T=100 is explicitly a placeholder).
#
# CRITICAL distinction: the paper's t axis is in TIME UNITS, not DBN slices.
# Fig. 6's x axis is labelled "t (x 10 min)" and Table 4's raising times are
# "(x 10 min)", i.e. Table 3's own time unit of 10 min = 600 s. A slice is
# delta_t = 166.13 s, so one time unit is 600/166.13 = 3.6117 slices and the
# T=200 horizon is ~722 slices, not 200.
#
# Confirmed against Table 5's own arithmetic: at m=1 the EX total is 157.87 s,
# and 157.87 / 722 = 0.219 s per slice, matching the paper's stated "for the EX
# clustering the average per-slice computation time is 0.22 seconds" (CL and FF
# likewise give 0.033/0.036, matching its "approximately 0.03 seconds"). Reading
# T=200 as slices instead would give 0.79 s/slice for EX and miss by ~4x.
T_TIME_UNITS = 200
TIME_UNIT_SECONDS = 600
M = 1.0
P_POS = 1e-4
P_NEG = 1e-4

# delta_t calibrated directly to Cerotti et al. Table 5's own published value
# at m=1 (166.13 seconds), converted to Table 3's time unit (10 min = 600 s),
# rather than computed via compute_delta_t(collect_uniformization_ttcs(ag), m).
# The 11-node sum gives a delta_t about 4x smaller than every one of Table 5's
# six published m-values implies (Sigma consistently ~=3.6117, not ~=14.6117);
# the exact TTC subset producing 3.6117 could not be uniquely determined from
# Sec. III-E's text (LAB_NOTEBOOK.md, 2026-07-31: a brute-force search over all
# 2^11 subsets found several structurally unrelated node-sets tied on the
# identical sum). Matching Table 5's own number directly sidesteps that
# ambiguity instead of guessing which subset to encode in
# collect_uniformization_ttcs. A first ad hoc check with this delta_t moved
# MITM@t=20 from 0.333 to 0.806 and made CorrReact hit exactly 0.7000 (its own
# fixed success probability) one slice after Table 4's raising time -- strong
# evidence this is the right delta_t scale.
DELTA_T_OVERRIDE = 166.13 / TIME_UNIT_SECONDS  # in time units

# Slices per time unit, and the total slice count for the T=200 horizon.
SLICES_PER_TIME_UNIT = 1.0 / DELTA_T_OVERRIDE
N_SLICES = round(T_TIME_UNITS * SLICES_PER_TIME_UNIT)


def slice_to_time_unit(slice_index: int) -> float:
    """Slice index (1-based) -> the paper's t axis, in units of 10 min."""
    return slice_index * DELTA_T_OVERRIDE


def time_unit_to_slice(t_units: float) -> int:
    """The paper's t (units of 10 min) -> nearest 1-based slice index."""
    return round(t_units * SLICES_PER_TIME_UNIT)


# Table 4: raising times for the 5 analytics the paper's Sec. IV-A states are
# "active" in this scenario, in TIME UNITS (the table's own "x 10 min").
# Resolved evidence semantics (see LAB_NOTEBOOK.md and the plan): each is
# observed at EVERY slice, 0 before its raising time and 1 at/after it,
# persisting once raised. This is not the only possible reading of "raising
# time" -- it is the one that reproduces Fig. 7a's ModifyProgram curve, which
# sits at exactly 0 for the whole horizon despite never appearing in Table 4.
# That is only explicable if NewServiceStarted (ModifyProgram's one analytic
# child) is continuously observed as negative, not merely unmentioned.
RAISING_TIMES = {
    "FileAccess": 8,
    "FileIntegrity": 9,
    "MeasureCoherence": 31,
    "CommandCoherence": 52,
    "NewServiceStarted": None,  # never fires: observed as 0 at every slice
}

QUERY_NODES = ["UnstablePS", "CorrReact", "MITM"]

# Node groupings replicated exactly from the paper's own Figs. 5a/5b, 7a/7b so
# the plots are directly comparable curve-for-curve.
FIG5A_NODES = ["ModAuthProc", "ModifyProgram", "SpoofRepMsg", "CorrReact", "ModCtrlLogic"]
FIG5B_NODES = ["UnsecCred", "MITM", "UnstablePS", "UnauthCommand"]
FIG7A_NODES = ["ModAuthProc", "SpoofRepMsg", "CorrReact", "ModifyProgram"]
FIG7B_NODES = ["UnsecCred", "MITM", "UnstablePS", "UnauthCommand"]


def scenario1_evidence_stream() -> dict[int, dict[str, int]]:
    """No evidence at all (Cerotti et al. Figs. 5, 6, 9)."""
    return {}


def scenario2_evidence_stream() -> dict[int, dict[str, int]]:
    """Table 4's raising times, continuous 0-before/1-at-and-after (Figs. 7, 8, 10).

    Keys are SLICE indices; Table 4's times are converted from time units.
    """
    raise_slices = {
        name: (None if raise_t is None else time_unit_to_slice(raise_t))
        for name, raise_t in RAISING_TIMES.items()
    }
    return {
        s: {
            name: 0 if (rs is None or s < rs) else 1
            for name, rs in raise_slices.items()
        }
        for s in range(1, N_SLICES + 1)
    }


def set_all_seeds(seed: int) -> None:
    # Forward filtering here is a deterministic computation (exact/mean-field
    # marginals given fixed CPTs and evidence) -- no sampling occurs, so the
    # seed has no numerical effect on this experiment's output. Set and
    # logged anyway for consistency with CLAUDE.md's seeding requirement and
    # so this script's logging conventions don't silently diverge from the
    # rest of the project when a later experiment does add sampling.
    random.seed(seed)
    np.random.seed(seed)


def run_config(
    ag, clustering_name: str, evidence_stream: dict[int, dict[str, int]]
) -> tuple[Trajectory, MemoryReport]:
    interface = sorted(n for n, d in ag.nodes(data=True) if d["self_loop"])
    clustering = (
        fully_factorized_clustering(interface)
        if clustering_name == "FF"
        else exact_clustering(interface)
    )
    config = InferenceConfig(
        clustering=clustering,
        m=M,
        p_pos=P_POS,
        p_neg=P_NEG,
        delta_t_override=DELTA_T_OVERRIDE,
    )
    engine = DBNInference(ag, config)

    def _run():
        return engine.run(evidence_stream, N_SLICES)

    trajectory, memory = measure_memory(_run)
    return trajectory, memory


def trajectory_to_frame(trajectory: Trajectory, scenario: str, clustering: str) -> pd.DataFrame:
    rows = []
    for slice_index, marginals in enumerate(trajectory.marginals, start=1):
        for node, p in marginals.items():
            rows.append(
                {
                    "scenario": scenario,
                    "clustering": clustering,
                    "slice": slice_index,
                    "t": slice_to_time_unit(slice_index),
                    "node": node,
                    "p_active": p,
                    "latency_s": trajectory.latencies_s[slice_index - 1],
                }
            )
    return pd.DataFrame(rows)


def compute_kl_trajectories(
    ex_trajectory: Trajectory, ff_trajectory: Trajectory
) -> dict[str, dict[int, float]]:
    """KL per node, keyed by slice index (Eq. 4's psi(t))."""
    kl: dict[str, dict[int, float]] = {node: {} for node in QUERY_NODES}
    for i in range(len(ex_trajectory.marginals)):
        for node in QUERY_NODES:
            p = ex_trajectory.marginals[i][node]
            q = ff_trajectory.marginals[i][node]
            kl[node][i + 1] = binary_kl(p, q)
    return kl


def plot_probabilities(trajectory: Trajectory, nodes: list[str], title: str, path: Path) -> None:
    n = len(trajectory.marginals)
    ts = np.array([slice_to_time_unit(i) for i in range(1, n + 1)])
    fig, ax = plt.subplots(figsize=(7, 5))
    for node in nodes:
        values = [trajectory.marginals[i][node] for i in range(n)]
        ax.plot(ts, values, linewidth=1.2, label=node)
    ax.set_xlabel("t (x 10 min)")
    ax.set_ylabel("Pr")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_kl(kl_trajectories: dict[str, dict[int, float]], title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(QUERY_NODES), figsize=(5 * len(QUERY_NODES), 4))
    for ax, node in zip(axes, QUERY_NODES):
        slices = sorted(kl_trajectories[node])
        ts = [slice_to_time_unit(s) for s in slices]
        values = [kl_trajectories[node][s] for s in slices]
        ax.plot(ts, values, linewidth=1.2)
        ax.set_title(f"EX vs FF [{node}]")
        ax.set_xlabel("t (x 10 min)")
        ax.set_ylabel("KL divergence")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    seed = config["seed"]
    set_all_seeds(seed)

    # Which reading of the control-centre reactions to run. The paper's own
    # figures disagree (Fig. 5a wants memoryless, Figs. 6c/8a want memory) --
    # see LAB_NOTEBOOK.md 2026-07-31. Selectable so both are reproducible.
    reaction_mode = "latched" if "--latched" in sys.argv else "memoryless"

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    SUMMARIES_DIR.mkdir(exist_ok=True)

    ag = build_attack_graph(reaction_mode=reaction_mode)
    print(f"reaction_mode: {reaction_mode}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)

    scenarios = {
        "scenario1": scenario1_evidence_stream(),
        "scenario2": scenario2_evidence_stream(),
    }

    trajectories: dict[tuple[str, str], Trajectory] = {}
    memory_reports = {}

    for scenario_name, evidence_stream in scenarios.items():
        for clustering_name in ["FF", "EX"]:
            print(f"running {scenario_name} / {clustering_name} ...", flush=True)
            trajectory, memory = run_config(ag, clustering_name, evidence_stream)
            trajectories[(scenario_name, clustering_name)] = trajectory
            memory_reports[(scenario_name, clustering_name)] = memory

            frame = trajectory_to_frame(trajectory, scenario_name, clustering_name)
            frame["git_sha"] = sha
            frame["seed"] = seed
            frame["m"] = M
            frame["delta_t_override"] = DELTA_T_OVERRIDE
            frame["p_pos"] = P_POS
            frame["p_neg"] = P_NEG
            csv_path = RESULTS_DIR / f"exp01_{scenario_name}_{clustering_name.lower()}_{reaction_mode}_{timestamp}.csv"
            frame.to_csv(csv_path, index=False)
            print(f"  wrote {csv_path}")

    # --- KL divergence and M_KL (Eq. 4, Eq. 5) ---
    summary_rows = []
    kl_by_scenario: dict[str, dict[str, dict[int, float]]] = {}
    for scenario_name in scenarios:
        ex_traj = trajectories[(scenario_name, "EX")]
        ff_traj = trajectories[(scenario_name, "FF")]
        kl = compute_kl_trajectories(ex_traj, ff_traj)
        kl_by_scenario[scenario_name] = kl
        for node in QUERY_NODES:
            value, argmax_t = m_kl(kl[node])
            summary_rows.append(
                {
                    "scenario": scenario_name,
                    "node": node,
                    "m_kl": value,
                    "argmax_t": argmax_t,
                    "git_sha": sha,
                    "seed": seed,
                    "m": M,
                    "p_pos": P_POS,
                    "p_neg": P_NEG,
                }
            )

    for (scenario_name, clustering_name), traj in trajectories.items():
        mean_latency = float(np.mean(traj.latencies_s))
        mem = memory_reports[(scenario_name, clustering_name)]
        summary_rows.append(
            {
                "scenario": scenario_name,
                "node": f"__{clustering_name}_perf__",
                "m_kl": mean_latency,  # reused column: mean per-slice latency (s)
                "argmax_t": mem.tracemalloc_peak_bytes,  # reused column: peak bytes
                "git_sha": sha,
                "seed": seed,
                "m": M,
                "p_pos": P_POS,
                "p_neg": P_NEG,
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    summary_path = SUMMARIES_DIR / f"exp01_summary_{reaction_mode}_{timestamp}.csv"
    summary_frame.to_csv(summary_path, index=False)
    print(f"\nwrote {summary_path}")

    # --- plots ---
    plot_probabilities(
        trajectories[("scenario1", "EX")], FIG5A_NODES,
        "Scenario 1 (no evidence), EX -- cf. Fig. 5a", FIGURES_DIR / f"exp01_{reaction_mode}_scenario1_probs_a.png",
    )
    plot_probabilities(
        trajectories[("scenario1", "EX")], FIG5B_NODES,
        "Scenario 1 (no evidence), EX -- cf. Fig. 5b", FIGURES_DIR / f"exp01_{reaction_mode}_scenario1_probs_b.png",
    )
    plot_kl(kl_by_scenario["scenario1"], "Scenario 1 KL(EX||FF) -- cf. Fig. 6",
            FIGURES_DIR / f"exp01_{reaction_mode}_scenario1_kl.png")

    plot_probabilities(
        trajectories[("scenario2", "EX")], FIG7A_NODES,
        "Scenario 2 (Table 4 evidence), EX -- cf. Fig. 7a", FIGURES_DIR / f"exp01_{reaction_mode}_scenario2_probs_a.png",
    )
    plot_probabilities(
        trajectories[("scenario2", "EX")], FIG7B_NODES,
        "Scenario 2 (Table 4 evidence), EX -- cf. Fig. 7b", FIGURES_DIR / f"exp01_{reaction_mode}_scenario2_probs_b.png",
    )
    plot_kl(kl_by_scenario["scenario2"], "Scenario 2 KL(EX||FF) -- cf. Fig. 8",
            FIGURES_DIR / f"exp01_{reaction_mode}_scenario2_kl.png")
    print(f"wrote plots to {FIGURES_DIR}")

    # --- validation gate ---
    # Targets are quoted at the paper's t (TIME UNITS), so convert to slices.
    ex2_marginals = trajectories[("scenario2", "EX")].marginals

    def at(t_units: float) -> dict[str, float]:
        return ex2_marginals[time_unit_to_slice(t_units) - 1]

    checks = [
        ("Scenario 2 @t=8: UnsecCred ~1.0",
         at(8)["UnsecCred"], 1.0, 0.05),
        ("Scenario 2 @t=20: MITM ~1.0",
         at(20)["MITM"], 1.0, 0.05),
        ("Scenario 2 @t=31: SpoofRepMsg ~1.0",
         at(31)["SpoofRepMsg"], 1.0, 0.02),
        ("Scenario 2 @t=31: CorrReact ~0.7",
         at(31)["CorrReact"], 0.7, 0.02),
        ("Scenario 2 @t=31: UnstablePS ~0.85",
         at(31)["UnstablePS"], 0.85, 0.02),
        ("Scenario 2 @t=52: UnauthCommand ~1.0",
         at(52)["UnauthCommand"], 1.0, 0.02),
        ("Scenario 2 @t=52: UnstablePS ~1.0",
         at(52)["UnstablePS"], 1.0, 0.02),
    ]

    unstable_kl_s1 = kl_by_scenario["scenario1"]["UnstablePS"]
    corr_kl_s1 = kl_by_scenario["scenario1"]["CorrReact"]
    mitm_kl_s1 = kl_by_scenario["scenario1"]["MITM"]
    unstable_kl_s2 = kl_by_scenario["scenario2"]["UnstablePS"]

    ff_latency = float(np.mean(trajectories[("scenario2", "FF")].latencies_s))

    print("\n=== VALIDATION GATE ===")
    print(f"{'check':<55}{'got':>12}{'target':>10}{'result':>10}")
    all_ok = True
    for label, got, target, tol in checks:
        ok = abs(got - target) <= tol
        all_ok &= ok
        print(f"{label:<55}{got:>12.4f}{target:>10.2f}{'PASS' if ok else 'FAIL':>10}")

    max_unstable_s1, argmax_unstable_s1 = m_kl(unstable_kl_s1)
    max_corr_s1, _ = m_kl(corr_kl_s1)
    max_mitm_s1, argmax_mitm_s1 = m_kl(mitm_kl_s1)
    max_unstable_s2, argmax_unstable_s2 = m_kl(unstable_kl_s2)

    print(f"\n{'KL check':<48}{'got':>12}{'at t':>9}{'target':>14}")
    for label, value, argmax_slice, target in [
        ("S1 UnstablePS KL peak (t~40-50)", max_unstable_s1, argmax_unstable_s1, "~2e-2"),
        ("S1 CorrReact KL plateau", max_corr_s1, None, "~0.11-0.12"),
        ("S1 MITM KL peak (t~10)", max_mitm_s1, argmax_mitm_s1, "~1.8e-2"),
        ("S2 UnstablePS KL peak (t~31)", max_unstable_s2, argmax_unstable_s2, "~1.15e-2"),
    ]:
        t_str = "" if argmax_slice is None else f"{slice_to_time_unit(argmax_slice):.1f}"
        print(f"{label:<48}{value:>12.4g}{t_str:>9}{target:>14}")

    print(f"\nhorizon: {T_TIME_UNITS} time units = {N_SLICES} slices "
          f"(delta_t = {DELTA_T_OVERRIDE * TIME_UNIT_SECONDS:.2f} s)")
    print(f"FF mean per-slice latency: {ff_latency:.4f}s (paper: ~0.03s)")

    for (scen, clus), mem in memory_reports.items():
        line = f"{scen}/{clus} tracemalloc peak: {mem.tracemalloc_peak_bytes/1e6:.3f} MB"
        if mem.rss_delta_bytes is not None:
            line += f"  (rss delta: {mem.rss_delta_bytes/1e6:.3f} MB)"
        print(line)

    print(f"\nProbability checks: {'ALL PASS' if all_ok else 'SOME FAILED -- see above'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
