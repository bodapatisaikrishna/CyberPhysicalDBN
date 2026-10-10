"""Cross-experiment consolidation (Session 10, task 3): read every relevant
results/*.csv already on disk and write tidy, per-ablation-axis tables to a
new results/summary/, plus two cross-experiment figures. Every output row
is copied UNMODIFIED from an existing logged CSV (no recomputation of any
metric, per CLAUDE.md rule 2) with two columns added: `source_file` (which
exact CSV the row came from) and `ablation_axis`.

Source-file selection: for experiments with more than one non-smoke run on
disk (exp01: 3 runs; exp04: 2 runs), the CANONICAL file is the one
LAB_NOTEBOOK.md actually cites (hardcoded below, verified against the
notebook text) -- NOT simply "the newest," since this session's own
reproducibility check (scripts/verify_reproducibility.py) creates a fresh
exp01 run as a verification byproduct that must NOT silently become the
new "canonical" source. exp10/exp11 (this session's own new experiments)
have exactly one real non-smoke run each, selected via newest-non-smoke
glob (there is no pre-existing canonical citation to preserve yet).

Run: .venv/bin/python scripts/build_summary_tables.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
SUMMARIES_DIR = RESULTS_DIR / "summaries"
SUMMARY_DIR = RESULTS_DIR / "summary"
SUMMARY_FIGURES_DIR = SUMMARY_DIR / "figures"

TS_PATTERN = re.compile(r"(\d{8}T\d{6}Z)")


def newest_nonsmoke(directory: Path, pattern: str) -> Path | None:
    candidates = [p for p in directory.glob(pattern) if "smoke" not in p.name]
    if not candidates:
        return None
    def ts_key(p: Path) -> str:
        m = TS_PATTERN.search(p.name)
        return m.group(1) if m else ""
    return max(candidates, key=ts_key)


# --- canonical sources, hardcoded where a specific historical run is the
# citation of record (verified against LAB_NOTEBOOK.md) -----------------
EXP13_STATE_TS = "20260928T130418Z"
EXP13_PHYSICAL_TS = "20261003T093637Z"

CANONICAL = {
    "exp01_summary": SUMMARIES_DIR / "exp01_summary_20260731T164052Z.csv",
    "exp04_lead_time": RESULTS_DIR / "exp04_lead_time_20261010T073122Z.csv",
    "exp04_calibration": RESULTS_DIR / "exp04_calibration_20261010T073122Z.csv",
    "exp05_dbn_lead_time": RESULTS_DIR / "exp05_dbn_lead_time_20261010T154000Z.csv",
    "exp05_dbn_calibration": RESULTS_DIR / "exp05_dbn_calibration_20261010T154000Z.csv",
    "exp08_transfer_eval": RESULTS_DIR / "exp08_transfer_eval_20261010T100012Z.csv",
    "exp08_lead_time_summary": RESULTS_DIR / "exp08_lead_time_summary_20261010T100012Z.csv",
    # 2026-08-10: repointed to the post-float32-fix rerun
    # (results/exp09_rerun_20260810T091530Z.log), confirmed to reproduce
    # the original 20260806T070130Z run's numbers exactly (see
    # LAB_NOTEBOOK.md's 2026-08-10 entry) -- this is the more current
    # citation now that both runs are on record, not a value correction.
    # 2026-10-10: exp05/08/09/12 repointed to current-code reruns
    # (results/expNN_rerun_20261010T*.log). exp08, exp09 and the exp12
    # baseline arm reproduce their earlier pinned runs exactly; exp05 and the
    # exp12 zone_aux arm changed (earlier runs were from uncommitted trees) --
    # see LAB_NOTEBOOK.md 2026-10-10. exp05 is pinned to its second current-code
    # run (154000Z, which wrote the exp05 reliability figures); perception
    # training is not bit-reproducible, so its soft-evidence arms vary by run.
    "exp09_robustness_curve": RESULTS_DIR / "exp09_robustness_curve_20261010T075400Z.csv",
    "exp09_reward_curve": RESULTS_DIR / "exp09_reward_curve_20261010T075400Z.csv",
    "exp12_cluster_assignment": RESULTS_DIR / "exp12_cluster_assignment_baseline_20261010T101722Z.csv",
    "exp12_observable_kl": RESULTS_DIR / "exp12_observable_kl_baseline_20261010T101722Z.csv",
    "exp12_posterior_kl": RESULTS_DIR / "exp12_posterior_kl_baseline_20261010T101722Z.csv",
    "exp12_zone_aux_cluster_assignment": RESULTS_DIR / "exp12_cluster_assignment_zone_aux_20261010T101722Z.csv",
    "exp12_zone_aux_observable_kl": RESULTS_DIR / "exp12_observable_kl_zone_aux_20261010T101722Z.csv",
    "exp12_zone_aux_posterior_kl": RESULTS_DIR / "exp12_posterior_kl_zone_aux_20261010T101722Z.csv",
}

AXES: dict[str, list[tuple[str, Path | None]]] = {
    "open_vs_closed": [
        ("lead_time", CANONICAL["exp04_lead_time"]),
        ("calibration", CANONICAL["exp04_calibration"]),
    ],
    "expert_vs_learned_ttc": [
        ("transfer_eval", CANONICAL["exp08_transfer_eval"]),
        ("lead_time_summary", CANONICAL["exp08_lead_time_summary"]),
    ],
    "evidence_calibration": [
        ("lead_time", CANONICAL["exp05_dbn_lead_time"]),
        ("calibration", CANONICAL["exp05_dbn_calibration"]),
    ],
    "ex_vs_ff": [
        ("summary", CANONICAL["exp01_summary"]),
    ],
    "adversarial_robustness": [
        ("robustness_curve", CANONICAL["exp09_robustness_curve"]),
        ("reward_curve", CANONICAL["exp09_reward_curve"]),
    ],
    "gnn_vs_mlp": [
        ("perception_metrics", newest_nonsmoke(RESULTS_DIR, "exp11_perception_metrics_*.csv")),
        ("lead_time", newest_nonsmoke(RESULTS_DIR, "exp11_dbn_lead_time_*.csv")),
        ("calibration", newest_nonsmoke(RESULTS_DIR, "exp11_dbn_calibration_*.csv")),
    ],
    "m_sweep": [
        ("perf", newest_nonsmoke(RESULTS_DIR, "exp10_m_sweep_perf_*.csv")),
        ("kl", newest_nonsmoke(RESULTS_DIR, "exp10_m_sweep_kl_*.csv")),
    ],
    # state-view run (3 scenarios, 02 truncated to ~3 h); pinned so the later
    # physical-view run (a different feature view) never replaces it
    "sherlock_full": [
        (m, RESULTS_DIR / f"exp13_{m}_{EXP13_STATE_TS}.csv")
        for m in ("inventory", "anomaly_metrics", "attack_type_summary", "cross_network", "transfer_matrix")
    ],
    # 02-Semiurban full 12 h from raw physical.zip (LAB_NOTEBOOK.md 2026-10-03)
    "sherlock_physical": [
        (m, RESULTS_DIR / f"exp13_{m}_{EXP13_PHYSICAL_TS}.csv")
        for m in ("inventory", "anomaly_metrics", "attack_type_summary", "cross_network", "transfer_matrix")
    ],
    # 2026-10-03 improvement attempts on 02-Semiurban physical view (pinned run timestamps)
    "sherlock_component": [
        ("anomaly_metrics", RESULTS_DIR / "exp14_anomaly_metrics_20261003T122426Z.csv"),
        ("events", RESULTS_DIR / "exp14_events_20261003T122426Z.csv"),
        ("criteria", RESULTS_DIR / "exp14_criteria_20261003T122426Z.csv"),
    ],
    "sherlock_rolling_baseline": [
        ("anomaly_metrics", RESULTS_DIR / "exp15_anomaly_metrics_20261003T123459Z.csv"),
        ("events", RESULTS_DIR / "exp15_events_20261003T123459Z.csv"),
        ("criteria", RESULTS_DIR / "exp15_criteria_20261003T123459Z.csv"),
    ],
    "sherlock_supervised": [
        ("pooled", RESULTS_DIR / "exp16_pooled_20261003T153317Z.csv"),
        ("events", RESULTS_DIR / "exp16_events_20261003T153317Z.csv"),
        ("criteria", RESULTS_DIR / "exp16_criteria_20261003T153317Z.csv"),
    ],
    # 2026-10-05: network-capture view + uniform-grid cadence sensitivity (pinned)
    "sherlock_network": [
        (m, RESULTS_DIR / f"exp17_{m}_20261005T161136Z.csv")
        for m in ("anomaly_metrics", "events", "attack_type_summary", "criteria", "grid_info")
    ],
    # 2026-10-09: out-of-sample replication on 01-Basic (confirmatory) + exploratory rerun on 02 (pinned)
    "sherlock_replication_01basic": [
        (m, RESULTS_DIR / f"exp18_{m}_20261009T150348Z.csv")
        for m in ("anomaly_metrics", "events", "attack_type_summary", "criteria", "grid_info")
    ],
    "sherlock_fixes_02semiurban_exploratory": [
        (m, RESULTS_DIR / f"exp18_{m}_20261009T150422Z.csv")
        for m in ("anomaly_metrics", "attack_type_summary", "criteria")
    ],
    # 2026-10-09: scenario-agnostic detector, leave-one-scenario-out; 03-Rural confirmatory (pinned)
    "sherlock_universal_loso": [
        (m, RESULTS_DIR / f"exp19_{m}_20261009T163931Z.csv")
        for m in ("anomaly_metrics", "events", "attack_type_summary", "criteria")
    ],
    "gnn_cluster_vs_heuristic": [
        ("cluster_assignment", CANONICAL["exp12_cluster_assignment"]),
        ("observable_kl", CANONICAL["exp12_observable_kl"]),
        ("posterior_kl", CANONICAL["exp12_posterior_kl"]),
    ],
    "gnn_cluster_vs_heuristic_zone_aux": [
        ("cluster_assignment", CANONICAL["exp12_zone_aux_cluster_assignment"]),
        ("observable_kl", CANONICAL["exp12_zone_aux_observable_kl"]),
        ("posterior_kl", CANONICAL["exp12_zone_aux_posterior_kl"]),
    ],
}


def write_axis_tables() -> list[str]:
    missing = []
    for axis, metrics in AXES.items():
        for metric_name, path in metrics:
            dest = SUMMARY_DIR / f"{axis}_{metric_name}.csv"
            if path is None or not path.exists():
                missing.append(f"{axis}/{metric_name}: source file not found ({path})")
                continue
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                missing.append(f"{axis}/{metric_name}: {path.name} has no rows")
                continue
            df["source_file"] = path.name
            df["ablation_axis"] = axis
            df.to_csv(dest, index=False)
            print(f"  wrote {dest}  (from {path.name}, {len(df)} rows)")
    return missing


def figure_m_sweep() -> None:
    path = newest_nonsmoke(RESULTS_DIR, "exp10_m_sweep_perf_*.csv")
    if path is None:
        print("  skipping m_sweep figure: no exp10 perf CSV found")
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for clustering, group in df.groupby("clustering"):
        group = group.sort_values("m")
        axes[0].plot(group["m"], group["mean_latency_s"], marker="o", label=clustering)
        axes[1].plot(group["m"], group["peak_tracemalloc_bytes"] / 1e6, marker="o", label=clustering)
    axes[0].set_xlabel("m")
    axes[0].set_ylabel("mean per-slice latency (s)")
    axes[0].set_title("Latency vs. m")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("m")
    axes[1].set_ylabel("peak tracemalloc (MB)")
    axes[1].set_title("Memory vs. m")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Discretization m sweep (source: {path.name})")
    fig.tight_layout()
    out = SUMMARY_FIGURES_DIR / "m_sweep_latency_memory.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def figure_gnn_vs_mlp() -> None:
    path = newest_nonsmoke(RESULTS_DIR, "exp11_perception_metrics_*.csv")
    if path is None:
        print("  skipping gnn_vs_mlp figure: no exp11 perception metrics CSV found")
        return
    df = pd.read_csv(path)
    targets = sorted(df["target"].unique())
    arms = sorted(df["arm"].unique())
    x = range(len(targets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, arm in enumerate(arms):
        vals = [df[(df["target"] == t) & (df["arm"] == arm)]["auc_pr"].iloc[0] for t in targets]
        offset = (i - (len(arms) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], vals, width=width, label=arm)
    ax.set_xticks(list(x))
    ax.set_xticklabels(targets, rotation=20, ha="right")
    ax.set_ylabel("AUC-PR")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"GNN vs. per-asset MLP perception encoder\nsource: {path.name}", fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = SUMMARY_FIGURES_DIR / "gnn_vs_mlp_auc_pr.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> int:
    SUMMARY_DIR.mkdir(exist_ok=True)
    SUMMARY_FIGURES_DIR.mkdir(exist_ok=True)

    print("writing per-axis tables ...")
    missing = write_axis_tables()

    print("\nwriting figures ...")
    figure_m_sweep()
    figure_gnn_vs_mlp()

    print("\n=== SUMMARY ===")
    if missing:
        print("MISSING source files (table not written for these):")
        for m in missing:
            print(f"  - {m}")
        return 1
    print("All axis tables and figures written to results/summary/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
