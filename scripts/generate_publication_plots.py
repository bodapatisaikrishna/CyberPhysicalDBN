"""Round 2 of publication figures: full threshold-sweep curves (not single-
theta bars), the exp12 zoning comparison drawn on the actual feeder topology,
an illustrative system-architecture diagram, and a combined C1/C2/C3 claims
summary. Every DATA figure reads an existing results/*.csv unmodified (no
recomputation, CLAUDE.md rule 2); the architecture diagram is illustrative
schematic, not a data plot, and is drawn from CLAUDE.md's own Layer 0-6
description, not from any experiment output.

Run: .venv/bin/python scripts/generate_publication_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandapower.networks as pn
import pandas as pd
from matplotlib.patches import FancyArrow, FancyBboxPatch
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

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


def savefig(fig: plt.Figure, name: str) -> None:
    out = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------- threshold sweeps ---

def exp06_leadtime_sweep() -> None:
    path = newest_nonsmoke(RESULTS_DIR, "exp06_comparison_leadtime_*.csv")
    if path is None:
        print("  skipping exp06 sweep: source CSV not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for system, group in df.groupby("system"):
        group = group.sort_values("threshold")
        ax.plot(group["threshold"], group["lead_mean_slices"], marker=".", label=system)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("detection threshold θ")
    ax.set_ylabel("mean lead time (slices, +=early)")
    ax.set_title(f"Lead time vs. threshold, full sweep\nsource: {path.name}")
    ax.legend()
    savefig(fig, "exp06_leadtime_vs_threshold.png")


def exp08_leadtime_sweep() -> None:
    path = CANONICAL["exp08_lead_time_summary"]
    if not path.exists():
        print("  skipping exp08 sweep: source CSV not found")
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for arm, group in df.groupby("arm"):
        group = group.sort_values("threshold")
        axes[0].plot(group["threshold"], group["detection_rate"], marker="o", label=arm)
        axes[1].plot(group["threshold"], group["lead_mean_slices"], marker="o", label=arm)
    axes[0].set_xlabel("threshold θ")
    axes[0].set_ylabel("detection rate")
    axes[0].set_title("Detection rate vs. threshold")
    axes[0].legend()
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("threshold θ")
    axes[1].set_ylabel("mean lead time (slices)")
    axes[1].set_title("Lead time vs. threshold")
    fig.suptitle(f"C2 transfer: amortized vs. expert (table3) vs. constant-prior TTC\nsource: {path.name}")
    savefig(fig, "exp08_leadtime_vs_threshold.png")


def exp09_robustness_sweep() -> None:
    path = CANONICAL["exp09_robustness_curve"]
    if not path.exists():
        print("  skipping exp09 sweep: source CSV not found")
        return
    df = pd.read_csv(path)
    order = ["blind", "analytics", "full_dbn"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, kl in zip(axes, order):
        sub = df[df["knowledge_level"] == kl]
        for system, group in sub.groupby("system"):
            group = group.sort_values("threshold")
            ax.plot(group["threshold"], group["lead_mean_slices"], marker=".", label=system)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("threshold θ")
        ax.set_title(f"attacker knowledge: {kl}")
    axes[0].set_ylabel("mean lead time (slices)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"C3 robustness: lead time vs. threshold, per attacker-knowledge level\nsource: {path.name}")
    savefig(fig, "exp09_robustness_full_sweep.png")


# -------------------------------------------------------- PR curves -----

def exp06_pr_curve() -> None:
    path = newest_nonsmoke(RESULTS_DIR, "exp06_raw_test_scores_*.csv")
    if path is None:
        print("  skipping exp06 PR curve: source CSV not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(7, 6))
    for system, group in df.groupby("system"):
        ap = average_precision_score(group["y_true"], group["y_prob"])
        PrecisionRecallDisplay.from_predictions(group["y_true"], group["y_prob"],
                                                 name=system, ax=ax)
    ax.set_title(f"Precision-recall curves, test split\nsource: {path.name}")
    savefig(fig, "exp06_pr_curve.png")


def exp09_pr_curve() -> None:
    path = newest_nonsmoke(RESULTS_DIR, "exp09_raw_eval_scores_*.csv")
    if path is None:
        print("  skipping exp09 PR curve: source CSV not found")
        return
    df = pd.read_csv(path)
    order = ["blind", "analytics", "full_dbn"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    for ax, kl in zip(axes, order):
        sub = df[df["knowledge_level"] == kl]
        for system, group in sub.groupby("system"):
            ap = average_precision_score(group["y_true"], group["y_prob"])
            PrecisionRecallDisplay.from_predictions(group["y_true"], group["y_prob"],
                                                     name=system, ax=ax)
        ax.set_title(f"attacker knowledge: {kl}")
        ax.legend(fontsize=6, loc="lower left")
    fig.suptitle(f"Precision-recall curves, evaluation episodes, per attacker-knowledge level\nsource: {path.name}")
    savefig(fig, "exp09_pr_curve.png")


def exp07_pr_curve() -> None:
    path = newest_nonsmoke(RESULTS_DIR, "exp07_raw_test_scores_*.csv")
    if path is None:
        print("  skipping exp07 PR curve: source CSV not found")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for stage, group in df.groupby("stage"):
        if len(group["y_true"].unique()) < 2:
            print(f"  skipping exp07 PR curve for stage={stage}: single-class y_true (base_rate degenerate)")
            continue
        ap = average_precision_score(group["y_true"], group["y_prob"])
        PrecisionRecallDisplay.from_predictions(group["y_true"], group["y_prob"],
                                                 name=stage, ax=ax)
    ax.set_title(f"Precision-recall curve, Sherlock test split\nsource: {path.name}")
    savefig(fig, "exp07_pr_curve.png")


# ----------------------------------------------------- exp12 spatial map -

def exp12_spatial_zone_map() -> None:
    clu_path = CANONICAL["exp12_cluster_assignment"]
    if not clu_path.exists():
        print("  skipping exp12 spatial map: source CSV not found")
        return
    cl = pd.read_csv(clu_path).set_index("bus")

    net = pn.case33bw()
    G = nx.Graph()
    G.add_nodes_from(net.bus.index.tolist())
    for _, row in net.line[["from_bus", "to_bus"]].iterrows():
        G.add_edge(int(row["from_bus"]), int(row["to_bus"]))
    if len(net.trafo):
        for _, row in net.trafo[["hv_bus", "lv_bus"]].iterrows():
            G.add_edge(int(row["hv_bus"]), int(row["lv_bus"]))
    # kamada_kawai on the real electrical topology graph -- a standard,
    # deterministic layout algorithm, not fabricated coordinates (case33bw
    # ships no bus_geodata).
    pos = nx.kamada_kawai_layout(G)

    heur_labels = sorted(cl["heuristic_label"].unique())
    heur_cmap = {lbl: plt.cm.tab10(i) for i, lbl in enumerate(heur_labels)}
    gnn_labels = sorted(cl["gnn_cluster"].unique())
    gnn_cmap = {lbl: plt.cm.tab10(i) for i, lbl in enumerate(gnn_labels)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    heur_colors = [heur_cmap[cl.loc[n, "heuristic_label"]] for n in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=axes[0], edge_color="#999999", width=1)
    nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=heur_colors, node_size=180)
    nx.draw_networkx_labels(G, pos, ax=axes[0], font_size=6)
    axes[0].set_title("Heuristic ZoneMap (threshold rule)")
    axes[0].axis("off")

    gnn_colors = [gnn_cmap[cl.loc[n, "gnn_cluster"]] for n in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=axes[1], edge_color="#999999", width=1)
    nx.draw_networkx_nodes(G, pos, ax=axes[1], node_color=gnn_colors, node_size=180)
    nx.draw_networkx_labels(G, pos, ax=axes[1], font_size=6)
    axes[1].set_title("GNN KMeans(k=2) cluster")
    axes[1].axis("off")

    handles0 = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=heur_cmap[l],
                            markersize=9, label=str(l)) for l in heur_labels]
    axes[0].legend(handles=handles0, loc="lower left", fontsize=7, title="heuristic label")
    handles1 = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=gnn_cmap[l],
                            markersize=9, label=str(l)) for l in gnn_labels]
    axes[1].legend(handles=handles1, loc="lower left", fontsize=7, title="gnn cluster")

    fig.suptitle(f"Zoning comparison on the real feeder topology (case33bw, kamada-kawai layout)\n"
                 f"source: {clu_path.name}")
    savefig(fig, "exp12_spatial_zone_map.png")


# --------------------------------------------------- architecture diagram -

def architecture_diagram() -> None:
    """Illustrative schematic of CLAUDE.md's Layer 0-6 pipeline. Not a data
    plot -- no experiment output is read here."""
    layers = [
        ("[0] Digital twin", "pandapower grid + abstracted comms\n+ attacker/defender agents", "#cfe3f7"),
        ("[1] Perception", "Heterogeneous GNN + temporal encoder\n-> calibrated soft evidence", "#d7f0d7"),
        ("[2] Parameterization", "GNN/hypernetwork: (technique, context)\n-> TTC -> p_s (uniformization)", "#fdeecb"),
        ("[3] DBN causal core", "2TBN + FF/BK inference\nposteriors, causal paths", "#f6d6d6"),
        ("[4] Physical consequence", "compromised actions -> grid state\n-> measured instability (closed loop)", "#e3d6f6"),
        ("[5] Decision (stretch)", "DBN posterior as POMDP belief\n-> RL defense policy", "#f0f0f0"),
        ("[6] Explanation (opt.)", "max-posterior causal path\n-> LLM analyst narrative", "#f0f0f0"),
    ]
    fig, ax = plt.subplots(figsize=(9, 12))
    box_h = 1.5
    spacing = 2.1
    n = len(layers)
    for i, (title, desc, color) in enumerate(layers):
        y = (n - i) * spacing
        box = FancyBboxPatch((0.5, y - box_h / 2), 6, box_h,
                              boxstyle="round,pad=0.05,rounding_size=0.08",
                              linewidth=1.2, edgecolor="black", facecolor=color)
        ax.add_patch(box)
        ax.text(0.8, y + 0.35, title, fontsize=12, fontweight="bold", va="center")
        ax.text(0.8, y - 0.35, desc, fontsize=9, va="center")
        if i < n - 1:
            gap_top = y - box_h / 2 - 0.08
            gap_bottom = y - spacing + box_h / 2 + 0.08
            ax.add_patch(FancyArrow(3.5, gap_top, 0, gap_bottom - gap_top,
                                     width=0.015, head_width=0.18, head_length=0.15,
                                     color="black", length_includes_head=True))
    layer4_y = (n - 4) * spacing
    ax.annotate("attack drives physics; physical deviation\nreturns as evidence\n(the closed loop, layer 4 -> layer 3)",
                xy=(6.6, layer4_y), fontsize=9, style="italic",
                bbox=dict(boxstyle="round", fc="#fff8dc", ec="gray"))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, (n + 0.6) * spacing)
    ax.axis("off")
    ax.set_title("System architecture (CLAUDE.md layers 0-6)\nillustrative schematic, not an experiment output", fontsize=11)
    savefig(fig, "architecture_diagram.png")


# ---------------------------------------------------- claims summary -----

def claims_summary() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # C1: open vs. closed lead time, threshold sweep
    p1 = CANONICAL["exp04_lead_time"]
    if p1.exists():
        df1 = pd.read_csv(p1)
        for arm, group in df1.groupby("arm"):
            group = group.sort_values("threshold")
            axes[0].plot(group["threshold"], group["lead_mean_slices"], marker=".", label=arm)
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("threshold θ")
        axes[0].set_ylabel("mean lead time (slices)")
        axes[0].legend(fontsize=7)
        axes[0].set_title("C1: closing the physical loop")
    else:
        axes[0].text(0.5, 0.5, "exp04 lead_time CSV\nnot found", ha="center", va="center")

    # C2: transfer detection rate by arm, threshold sweep
    p2 = CANONICAL["exp08_lead_time_summary"]
    if p2.exists():
        df2 = pd.read_csv(p2)
        for arm, group in df2.groupby("arm"):
            group = group.sort_values("threshold")
            axes[1].plot(group["threshold"], group["detection_rate"], marker="o", label=arm)
        axes[1].set_xlabel("threshold θ")
        axes[1].set_ylabel("detection rate (held-out test graphs)")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend(fontsize=7)
        axes[1].set_title("C2: learned TTC transfer")
    else:
        axes[1].text(0.5, 0.5, "exp08 lead_time_summary CSV\nnot found", ha="center", va="center")

    # C3: robustness at headline threshold
    p3 = CANONICAL["exp09_robustness_curve"]
    if p3.exists():
        df3 = pd.read_csv(p3)
        theta = float(min(df3["threshold"].unique(), key=lambda t: abs(t - 0.5)))
        sub = df3[np.isclose(df3["threshold"], theta)]
        order = ["blind", "analytics", "full_dbn"]
        for system, group in sub.groupby("system"):
            group = group.set_index("knowledge_level").reindex(order)
            axes[2].plot(order, group["lead_mean_slices"], marker="o", label=system)
        axes[2].set_xlabel("attacker knowledge level")
        axes[2].set_ylabel(f"mean lead time (slices), θ={theta:.2f}")
        axes[2].legend(fontsize=7)
        axes[2].set_title("C3: adversarial robustness")
    else:
        axes[2].text(0.5, 0.5, "exp09 robustness_curve CSV\nnot found", ha="center", va="center")

    fig.suptitle("Three falsifiable claims -- headline evidence\n"
                  f"sources: {p1.name}, {p2.name}, {p3.name}", fontsize=9)
    savefig(fig, "claims_c1_c2_c3_summary.png")


def main() -> int:
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    print("threshold sweeps ...")
    exp06_leadtime_sweep()
    exp08_leadtime_sweep()
    exp09_robustness_sweep()
    print("precision-recall curves ...")
    exp06_pr_curve()
    exp09_pr_curve()
    exp07_pr_curve()
    print("exp12 spatial zone map ...")
    exp12_spatial_zone_map()
    print("architecture diagram ...")
    architecture_diagram()
    print("claims summary ...")
    claims_summary()
    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
