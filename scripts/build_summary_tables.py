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
CANONICAL = {
    "exp01_summary": SUMMARIES_DIR / "exp01_summary_20260731T164052Z.csv",
    "exp04_lead_time": RESULTS_DIR / "exp04_lead_time_20260802T042212Z.csv",
    "exp04_calibration": RESULTS_DIR / "exp04_calibration_20260802T042212Z.csv",
    "exp05_dbn_lead_time": RESULTS_DIR / "exp05_dbn_lead_time_20260802T185037Z.csv",
    "exp05_dbn_calibration": RESULTS_DIR / "exp05_dbn_calibration_20260802T185037Z.csv",
    "exp08_transfer_eval": RESULTS_DIR / "exp08_transfer_eval_20260806T044635Z.csv",
    "exp08_lead_time_summary": RESULTS_DIR / "exp08_lead_time_summary_20260806T044635Z.csv",
    # 2026-08-10: repointed to the post-float32-fix rerun
    # (results/exp09_rerun_20260810T091530Z.log), confirmed to reproduce
    # the original 20260806T070130Z run's numbers exactly (see
    # LAB_NOTEBOOK.md's 2026-08-10 entry) -- this is the more current
    # citation now that both runs are on record, not a value correction.
    "exp09_robustness_curve": RESULTS_DIR / "exp09_robustness_curve_20260810T091537Z.csv",
    "exp09_reward_curve": RESULTS_DIR / "exp09_reward_curve_20260810T091537Z.csv",
    "exp12_cluster_assignment": RESULTS_DIR / "exp12_cluster_assignment_20260813T104934Z.csv",
    "exp12_observable_kl": RESULTS_DIR / "exp12_observable_kl_20260813T104934Z.csv",
    "exp12_posterior_kl": RESULTS_DIR / "exp12_posterior_kl_20260813T104934Z.csv",
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
    "gnn_cluster_vs_heuristic": [
        ("cluster_assignment", CANONICAL["exp12_cluster_assignment"]),
        ("observable_kl", CANONICAL["exp12_observable_kl"]),
        ("posterior_kl", CANONICAL["exp12_posterior_kl"]),
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
            df = pd.read_csv(path)
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
    ax.set_title(f"GNN vs. per-asset MLP perception encoder (source: {path.name})")
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
