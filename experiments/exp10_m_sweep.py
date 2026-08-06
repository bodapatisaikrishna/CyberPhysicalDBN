"""Discretization m sweep (Cerotti et al. Sec. IV-B; CLAUDE.md's own
"Reference numbers to reproduce in Phase 1" table names m in {1/3, 1}).

SCOPE NOTE, read before trusting any absolute number below: this does NOT
reproduce Cerotti et al. Table 5 / CLAUDE.md's reference-table MB/seconds
figures. Session 1 established (`experiments/exp01_reproduce_paper.py`
lines 67-80) that Eq. 3's general formula,
`compute_delta_t(collect_uniformization_ttcs(ag), m)`, yields a delta_t
about 4x smaller than every one of Table 5's own six published m-values,
and a brute-force search over all 2^11 TTC subsets found no unique subset
that reproduces Table 5's number. exp01 sidesteps this ambiguity by
hardcoding Table 5's own m=1 delta_t directly (166.13/600) rather than
deriving it from m -- which means m is functionally inert in exp01: no
experiment in this repo has ever actually varied m and observed its effect
on KL/latency/memory. This script does, using the general Eq. 3 formula
directly (the same mechanism `configs/adversarial_c3.yaml` already uses
successfully at m=0.22). It reports the QUALITATIVE m-dependence Section
IV-B describes -- does finer discretization cost more memory/latency; does
FF's approximation quality depend on m -- not a byte-exact replay of the
paper's own Table-5-calibrated numbers.

Scenario 1 (no evidence at all, Cerotti et al. Figs. 5/6/9) is used
throughout: this isolates the pure temporal-uniformization behavior from
any evidence-conditioning effect on VariableElimination's cost (evidence-
present slices are measurably cheaper than evidence-free ones -- see
LAB_NOTEBOOK.md 2026-08-06 exp09 plan notes -- which would confound a clean
m comparison).

Run: .venv/bin/python experiments/exp10_m_sweep.py
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
    Trajectory,
    exact_clustering,
    fully_factorized_clustering,
)
from src.dbn.parameterization import collect_uniformization_ttcs, compute_delta_t
from src.eval.metrics import MemoryReport, binary_kl, m_kl, measure_memory
from src.eval.provenance import git_sha

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
RESULTS_DIR = REPO_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

P_POS = 1e-4
P_NEG = 1e-4
T_TIME_UNITS = 200  # same horizon exp01 uses, for comparability across m
M_VALUES = (1.0 / 3.0, 1.0)  # exactly CLAUDE.md's reference-table pair
QUERY_NODES = ["UnstablePS", "CorrReact", "MITM"]


def set_all_seeds(seed: int) -> None:
    # Same rationale as exp01: forward filtering is deterministic given
    # fixed CPTs/evidence, no sampling occurs, seed has no numerical effect
    # on this experiment's output. Set and logged for consistency anyway.
    random.seed(seed)
    np.random.seed(seed)


def run_config(ag, clustering_name: str, m: float, delta_t: float, n_slices: int) -> tuple[Trajectory, MemoryReport]:
    interface = sorted(n for n, d in ag.nodes(data=True) if d["self_loop"])
    clustering = (
        fully_factorized_clustering(interface) if clustering_name == "FF" else exact_clustering(interface)
    )
    config = InferenceConfig(clustering=clustering, m=m, p_pos=P_POS, p_neg=P_NEG, delta_t_override=delta_t)
    engine = DBNInference(ag, config)

    def _run():
        return engine.run({}, n_slices)  # scenario1: no evidence

    trajectory, memory = measure_memory(_run)
    return trajectory, memory


def main() -> int:
    with open(CONFIG_PATH) as f:
        base_cfg = yaml.safe_load(f)
    seed = int(base_cfg["seed"])
    set_all_seeds(seed)

    RESULTS_DIR.mkdir(exist_ok=True)
    SUMMARIES_DIR.mkdir(exist_ok=True)

    ag = build_attack_graph(reaction_mode="memoryless")
    ttcs = collect_uniformization_ttcs(ag)
    min_ttc = min(ttcs.values())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = git_sha(REPO_ROOT)

    print(f"min TTC across {len(ttcs)} timed nodes: {min_ttc:.6f} time units")

    perf_rows = []
    trajectories: dict[tuple[float, str], Trajectory] = {}
    invalid_cpt: list[str] = []

    for m in M_VALUES:
        delta_t = compute_delta_t(ttcs, m)
        n_slices = round(T_TIME_UNITS / delta_t)
        max_p_s = delta_t / min_ttc
        print(f"\nm={m:.6f} delta_t={delta_t:.6f} n_slices={n_slices} max_p_s={max_p_s:.4f}", flush=True)
        if max_p_s > 1.0:
            invalid_cpt.append(f"m={m}: max_p_s={max_p_s:.4f} > 1.0")
            print(f"  SKIPPING m={m}: would produce an invalid CPT (p_s > 1)")
            continue

        for clustering_name in ["FF", "EX"]:
            print(f"  running {clustering_name} ...", flush=True)
            trajectory, memory = run_config(ag, clustering_name, m, delta_t, n_slices)
            trajectories[(m, clustering_name)] = trajectory
            mean_latency = float(np.mean(trajectory.latencies_s))
            perf_rows.append(
                {
                    "m": m,
                    "clustering": clustering_name,
                    "delta_t": delta_t,
                    "n_slices": n_slices,
                    "max_p_s": max_p_s,
                    "mean_latency_s": mean_latency,
                    "peak_tracemalloc_bytes": memory.tracemalloc_peak_bytes,
                    "rss_delta_bytes": memory.rss_delta_bytes,
                    "git_sha": sha,
                    "seed": seed,
                    "p_pos": P_POS,
                    "p_neg": P_NEG,
                }
            )
            print(
                f"    mean_latency={mean_latency:.4f}s "
                f"peak_tracemalloc={memory.tracemalloc_peak_bytes / 1e6:.2f}MB "
                f"rss_delta={(memory.rss_delta_bytes or 0) / 1e6:.2f}MB"
            )

    perf_df = pd.DataFrame(perf_rows)
    perf_path = RESULTS_DIR / f"exp10_m_sweep_perf_{timestamp}.csv"
    perf_df.to_csv(perf_path, index=False)
    print(f"\nwrote {perf_path}")

    # --- KL(EX||FF) per m, mirroring exp01's compute_kl_trajectories/m_kl ---
    kl_rows = []
    summary_rows = []
    for m in M_VALUES:
        if (m, "EX") not in trajectories or (m, "FF") not in trajectories:
            continue
        ex_traj = trajectories[(m, "EX")]
        ff_traj = trajectories[(m, "FF")]
        n = len(ex_traj.marginals)
        for node in QUERY_NODES:
            kl_by_slice = {
                i + 1: binary_kl(ex_traj.marginals[i][node], ff_traj.marginals[i][node]) for i in range(n)
            }
            value, argmax_slice = m_kl(kl_by_slice)
            row = {
                "m": m, "node": node, "m_kl": value, "argmax_t": argmax_slice,
                "git_sha": sha, "seed": seed, "p_pos": P_POS, "p_neg": P_NEG,
            }
            kl_rows.append(row)
            summary_rows.append(row)

    for row in perf_rows:
        summary_rows.append(
            {
                "m": row["m"], "node": f"__{row['clustering']}_perf__",
                "m_kl": row["mean_latency_s"], "argmax_t": row["peak_tracemalloc_bytes"],
                "git_sha": sha, "seed": seed, "p_pos": P_POS, "p_neg": P_NEG,
            }
        )

    kl_df = pd.DataFrame(kl_rows)
    kl_path = RESULTS_DIR / f"exp10_m_sweep_kl_{timestamp}.csv"
    kl_df.to_csv(kl_path, index=False)
    print(f"wrote {kl_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = SUMMARIES_DIR / f"exp10_m_sweep_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    # === validation gate ======================================================
    print("\n=== VALIDATION GATE ===")
    failures: list[str] = []

    if invalid_cpt:
        failures.append(f"invalid CPT for m values: {invalid_cpt}")
    print(f"(a) every swept m produces a valid CPT (max_p_s<=1) " f"{'PASS' if not invalid_cpt else 'FAIL: ' + str(invalid_cpt)}")

    all_finite = all(np.isfinite(r["m_kl"]) for r in kl_rows) and all(
        np.isfinite(r["mean_latency_s"]) and np.isfinite(r["peak_tracemalloc_bytes"]) for r in perf_rows
    )
    if not all_finite:
        failures.append("non-finite KL/latency/memory value found")
    print(f"(b) every KL/latency/memory value is finite ......... {'PASS' if all_finite else 'FAIL'}")

    ex_latencies = {row["m"]: row["mean_latency_s"] for row in perf_rows if row["clustering"] == "EX"}
    ex_memory = {row["m"]: row["peak_tracemalloc_bytes"] for row in perf_rows if row["clustering"] == "EX"}
    monotonic_latency = len(ex_latencies) < 2 or ex_latencies[M_VALUES[1]] >= ex_latencies[M_VALUES[0]]
    monotonic_memory = len(ex_memory) < 2 or ex_memory[M_VALUES[1]] >= ex_memory[M_VALUES[0]]
    print(
        f"(c) EX latency/memory increase with m (1/3 -> 1), REPORTED not gated: "
        f"latency {'monotonic' if monotonic_latency else 'NON-MONOTONIC (real finding, see LAB_NOTEBOOK)'}, "
        f"memory {'monotonic' if monotonic_memory else 'NON-MONOTONIC (real finding, see LAB_NOTEBOOK)'}"
    )

    print(
        "\nNOTE: this experiment does not reproduce Table 5's absolute "
        "MB/seconds figures (documented ~4x delta_t-scale gap, see module "
        "docstring). The m-dependence trend above is the deliverable, not "
        "the absolute numbers."
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
