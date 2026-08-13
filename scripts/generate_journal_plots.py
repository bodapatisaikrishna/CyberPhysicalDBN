"""Publication-style figures for experiments that never generated plots
(exp06, exp07, exp08, exp09, exp10-KL, exp12). Every figure is built
DIRECTLY from an already-logged results/*.csv (no recomputation of any
metric, per CLAUDE.md rule 2) -- this script only reads and draws.

Source-file selection reuses scripts/build_summary_tables.py's own
CANONICAL dict and newest_nonsmoke() helper, so a figure always points at
the same run LAB_NOTEBOOK.md/build_summary_tables.py already cite.

Run: .venv/bin/python scripts/generate_journal_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from build_summary_tables import CANONICAL, RESULTS_DIR, newest_nonsmoke  # noqa: E402

FIGURES_DIR = RESULTS_DIR / "figures"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 9,
    "figure.titlesize": 10,
})


def closest_threshold(thresholds: pd.Series, target: float = 0.5) -> float:
    """exp09's own headline-threshold convention: nearest grid point to 0.5."""
    return float(min(thresholds.unique(), key=lambda t: abs(t - target)))


def savefig(fig: plt.Figure, name: str) -> None:
    out = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------- exp06 ---

def exp06_plots() -> None:
    summary_path = newest_nonsmoke(RESULTS_DIR, "exp06_comparison_summary_*.csv")
    leadtime_path = newest_nonsmoke(RESULTS_DIR, "exp06_comparison_leadtime_*.csv")
    if summary_path is None or leadtime_path is None:
        print("  skipping exp06: source CSVs not found")
        return

    df = pd.read_csv(summary_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(df["system"], df["auc_pr"], color="#5285e0")
    axes[0].set_ylabel("AUC-PR")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Detection quality")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(df["system"], df["ece_10_uniform"], color="#e0a02a")
    axes[1].set_ylabel("ECE (10-bin, uniform)")
    axes[1].set_title("Calibration error")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle(f"Proposed system vs. external baselines\nsource: {summary_path.name}")
    savefig(fig, "exp06_auc_pr_and_ece.png")

    lt = pd.read_csv(leadtime_path)
    theta = closest_threshold(lt["threshold"])
    row = lt[np.isclose(lt["threshold"], theta)].sort_values("lead_mean_slices")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(row["system"], row["lead_mean_slices"], color="#3fb37f")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("mean lead time (slices, +=early)")
    ax.set_title(f"Detection lead time at threshold={theta:.2f}\nsource: {leadtime_path.name}")
    ax.tick_params(axis="x", rotation=30)
    savefig(fig, "exp06_leadtime_by_system.png")


# ---------------------------------------------------------------- exp07 ---

def exp07_plots() -> None:
    train_path = newest_nonsmoke(RESULTS_DIR, "exp07_training_*.csv")
    transfer_path = newest_nonsmoke(RESULTS_DIR, "exp07_transfer_*.csv")
    if train_path is None or transfer_path is None:
        print("  skipping exp07: source CSVs not found")
        return

    tr = pd.read_csv(train_path)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(tr["epoch"], tr["train_loss"], label="train_loss", color="#5285e0")
    ax.plot(tr["epoch"], tr["val_loss"], label="val_loss", color="#e05252")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(f"Sherlock perception-layer training\nsource: {train_path.name}")
    ax.legend()
    savefig(fig, "exp07_training_curve.png")

    tf = pd.read_csv(transfer_path)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(tf["direction"], tf["auc_pr"], color="#8a63d2")
    ax.set_ylabel("AUC-PR")
    ax.set_ylim(0, 1.05)
    for i, (_, r) in enumerate(tf.iterrows()):
        ax.text(i, r["auc_pr"] + 0.02, f"n={int(r['n_eval_scenarios'])}", ha="center", fontsize=8)
    ax.set_title(f"Twin ↔ Sherlock transfer\nsource: {transfer_path.name}")
    savefig(fig, "exp07_transfer_auc_pr.png")


# ---------------------------------------------------------------- exp08 ---

def exp08_plots() -> None:
    train_path = newest_nonsmoke(RESULTS_DIR, "exp08_amortized_training_*.csv")
    transfer_path = CANONICAL["exp08_transfer_eval"]
    leadtime_path = CANONICAL["exp08_lead_time_summary"]

    if train_path is not None and train_path.exists():
        tr = pd.read_csv(train_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].plot(tr["epoch"], tr["train_loss"], color="#5285e0")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("train_loss")
        axes[0].set_title("Amortized TTC model training loss")
        axes[1].plot(tr["epoch"], tr["val_mae_log_ttc"], color="#e05252")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("val MAE (log TTC)")
        axes[1].set_title("Validation error")
        fig.suptitle(f"Amortized TTC model\nsource: {train_path.name}")
        savefig(fig, "exp08_amortized_training_curve.png")
    else:
        print("  skipping exp08 training curve: source CSV not found")

    if transfer_path.exists():
        te = pd.read_csv(transfer_path)
        arms = sorted(te["arm"].unique())
        data = [te[te["arm"] == a]["m_kl"].to_numpy() for a in arms]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(data, tick_labels=arms, showfliers=True)
        ax.set_ylabel("M_KL (per graph/threshold row)")
        ax.set_title(f"Amortized vs. expert vs. constant-prior TTC\nsource: {transfer_path.name}")
        savefig(fig, "exp08_transfer_kl_by_arm.png")
    else:
        print("  skipping exp08 transfer KL plot: source CSV not found")

    if leadtime_path.exists():
        lt = pd.read_csv(leadtime_path)
        theta = closest_threshold(lt["threshold"])
        row = lt[np.isclose(lt["threshold"], theta)]
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.bar(row["arm"], row["detection_rate"], color="#3fb37f")
        ax.set_ylabel("detection rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"C2 transfer detection rate at threshold={theta:.2f}\nsource: {leadtime_path.name}")
        savefig(fig, "exp08_leadtime_detection_rate.png")
    else:
        print("  skipping exp08 leadtime plot: source CSV not found")


# ---------------------------------------------------------------- exp09 ---

def exp09_plots() -> None:
    rc_path = CANONICAL["exp09_robustness_curve"]
    rw_path = CANONICAL["exp09_reward_curve"]

    if rc_path.exists():
        rc = pd.read_csv(rc_path)
        theta = closest_threshold(rc["threshold"])
        sub = rc[np.isclose(rc["threshold"], theta)]
        order = ["blind", "analytics", "full_dbn"]
        fig, ax = plt.subplots(figsize=(8, 5))
        for system, group in sub.groupby("system"):
            group = group.set_index("knowledge_level").reindex(order)
            ax.plot(order, group["lead_mean_slices"], marker="o", label=system)
        ax.set_xlabel("attacker knowledge level")
        ax.set_ylabel("mean lead time (slices)")
        ax.set_title(f"Robustness curve: lead time vs. attacker knowledge (threshold={theta:.2f})\nsource: {rc_path.name}")
        ax.legend()
        savefig(fig, "exp09_robustness_curve.png")
    else:
        print("  skipping exp09 robustness curve: source CSV not found")

    if rw_path.exists():
        rw = pd.read_csv(rw_path)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for kl, group in rw.groupby("knowledge_level"):
            group = group.sort_values("batch")
            ax.plot(group["batch"], group["mean_reward"], label=kl)
        ax.set_xlabel("PPO training batch")
        ax.set_ylabel("mean episode reward")
        ax.set_title(f"RL attacker training\nsource: {rw_path.name}")
        ax.legend()
        savefig(fig, "exp09_reward_curve.png")
    else:
        print("  skipping exp09 reward curve: source CSV not found")


# ---------------------------------------------------------------- exp10 ---

def exp10_kl_plot() -> None:
    path = newest_nonsmoke(RESULTS_DIR, "exp10_m_sweep_kl_*.csv")
    if path is None:
        print("  skipping exp10 KL plot: source CSV not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for node, group in df.groupby("node"):
        group = group.sort_values("m")
        ax.plot(group["m"], group["m_kl"], marker="o", label=node)
    ax.set_xlabel("m (discretization parameter)")
    ax.set_ylabel("M_KL(EX‖FF)")
    ax.set_yscale("log")
    ax.set_title(f"KL(EX‖FF) vs. m, per node\nsource: {path.name}")
    ax.legend()
    savefig(fig, "exp10_kl_vs_m.png")


# ---------------------------------------------------------------- exp12 ---

def exp12_plots() -> None:
    post_path = CANONICAL["exp12_posterior_kl"]
    obs_path = CANONICAL["exp12_observable_kl"]
    clu_path = CANONICAL["exp12_cluster_assignment"]

    if post_path.exists():
        pk = pd.read_csv(post_path).sort_values("scenario")
        colors = ["#e05252" if v > 5 else ("#e0a02a" if v > 0.05 else "#3fb37f") for v in pk["m_kl"]]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(pk["scenario"].astype(str), pk["m_kl"], color=colors)
        ax.axhline(pk["m_kl"].mean(), color="black", linestyle="--", linewidth=1,
                   label=f"mean = {pk['m_kl'].mean():.3f}")
        ax.set_xlabel("test scenario")
        ax.set_ylabel("M_KL(UnstablePS): heuristic ‖ GNN zoning")
        ax.set_title(f"Downstream posterior KL, per scenario -- bimodal\nsource: {post_path.name}")
        ax.tick_params(axis="x", labelsize=7, rotation=90)
        ax.legend()
        savefig(fig, "exp12_posterior_kl_by_scenario.png")
    else:
        print("  skipping exp12 posterior KL plot: source CSV not found")

    if obs_path.exists():
        ob = pd.read_csv(obs_path)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.bar(ob["target"], ob["m_kl"], color="#5285e0")
        ax.set_ylabel("M_KL: heuristic ‖ GNN zoning")
        ax.set_title(f"Observable-level KL\nsource: {obs_path.name}")
        savefig(fig, "exp12_observable_kl.png")
    else:
        print("  skipping exp12 observable KL plot: source CSV not found")

    if clu_path.exists():
        cl = pd.read_csv(clu_path)
        heur_labels = sorted(cl["heuristic_label"].unique())
        gnn_labels = sorted(cl["gnn_cluster"].unique())
        mat = np.zeros((len(heur_labels), len(gnn_labels)), dtype=int)
        for i, h in enumerate(heur_labels):
            for j, g in enumerate(gnn_labels):
                mat[i, j] = int(((cl["heuristic_label"] == h) & (cl["gnn_cluster"] == g)).sum())
        fig, ax = plt.subplots(figsize=(7, 4.8))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(gnn_labels)))
        ax.set_xticklabels([f"gnn_cluster_{g}" for g in gnn_labels])
        ax.set_yticks(range(len(heur_labels)))
        ax.set_yticklabels(heur_labels)
        for i in range(len(heur_labels)):
            for j in range(len(gnn_labels)):
                ax.text(j, i, mat[i, j], ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
        ax.set_title(f"Bus-count overlap: heuristic zone × GNN cluster\n(source: {clu_path.name})", fontsize=9)
        fig.colorbar(im, ax=ax, label="bus count")
        savefig(fig, "exp12_cluster_overlap_heatmap.png")
    else:
        print("  skipping exp12 cluster heatmap: source CSV not found")


def main() -> int:
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    print("exp06 ..."); exp06_plots()
    print("exp07 ..."); exp07_plots()
    print("exp08 ..."); exp08_plots()
    print("exp09 ..."); exp09_plots()
    print("exp10 ..."); exp10_kl_plot()
    print("exp12 ..."); exp12_plots()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
