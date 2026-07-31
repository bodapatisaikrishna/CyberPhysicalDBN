"""Validation-gate summary for the attack graph, 2TBN compile and CPT generation.

Reports counts measured from the built objects. It does not assert that they
look correct; comparing them against the source paper is a human judgement.

Run:  .venv/bin/python scripts/build_and_report.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attack_graph.graph import build_attack_graph, undetermined_fields
from src.dbn.compiler import ANTERIOR, ULTERIOR, compile_to_2tbn
from src.dbn.parameterization import (
    attach_cpds,
    collect_uniformization_ttcs,
    compute_delta_t,
    compute_ps,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"

# Analytic error rates are a property of the monitoring scenario, not of the
# attack graph, so they are not in base.yaml yet. Cerotti et al. Sec. IV uses
# these values; they are passed explicitly here so the summary states its own
# inputs rather than reading a default from source.
P_POS = 1e-4
P_NEG = 1e-4


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    m = config["discretization"]["m"]

    ag = build_attack_graph()
    dbn = compile_to_2tbn(ag)
    dbn = attach_cpds(dbn, ag, m=m, p_pos=P_POS, p_neg=P_NEG)

    print("=== ATTACK GRAPH (Cerotti et al. Fig. 2) ===")
    print(f"nodes: {ag.number_of_nodes()}")
    for node_type, count in sorted(
        Counter(d["node_type"] for _, d in ag.nodes(data=True)).items()
    ):
        print(f"  {node_type}: {count}")
    print(f"self-looping (persistent) nodes: "
          f"{sum(1 for _, d in ag.nodes(data=True) if d['self_loop'])}")
    print(f"gated nodes: "
          f"{sorted(n for n, d in ag.nodes(data=True) if d['gate'] is not None)}")

    print(f"\nedges: {ag.number_of_edges()}")
    for edge_type, count in sorted(
        Counter(d["edge_type"] for _, _, d in ag.edges(data=True)).items()
    ):
        print(f"  {edge_type}: {count}")
    self_loops = sum(1 for u, v in ag.edges() if u == v)
    print(f"  (of which self-loops: {self_loops})")
    for inter_slice, count in sorted(
        Counter(d["inter_slice"] for _, _, d in ag.edges(data=True)).items()
    ):
        label = "inter-slice" if inter_slice else "intra-slice"
        print(f"  {label}: {count}")

    print("\n=== COMPILED 2TBN ===")
    print(f"nodes: {dbn.number_of_nodes()}")
    print(f"  anterior (slice 0): {sum(1 for n in dbn.nodes() if n[1] == ANTERIOR)}")
    print(f"  ulterior (slice 1): {sum(1 for n in dbn.nodes() if n[1] == ULTERIOR)}")
    print(f"edges: {dbn.number_of_edges()}")
    print(f"  anterior-layer intra-slice arcs: "
          f"{sum(1 for u, v in dbn.edges() if u[1] == ANTERIOR and v[1] == ANTERIOR)}"
          f"  (canonical form requires 0)")

    print("\n=== PARAMETERIZATION (Eq. 3) ===")
    ttcs = collect_uniformization_ttcs(ag)
    delta_t = compute_delta_t(ttcs, m)
    print(f"m: {m}")
    print(f"TTCs entering Eq. 3: {len(ttcs)}")
    print(f"delta_t: {delta_t!r}")
    print(f"p_pos: {P_POS}   p_neg: {P_NEG}")
    print("per-node p_s:")
    for name in sorted(ttcs):
        print(f"  {name}: {compute_ps(ttcs[name], delta_t)!r}")

    cpds = dbn.get_cpds()
    print(f"\nCPTs generated: {len(cpds)}")
    print(f"  total table entries: {sum(cpd.get_values().size for cpd in cpds)}")

    print("\n=== VALUES NOT DETERMINABLE FROM THE SOURCE PAPER ===")
    for field, reason in undetermined_fields().items():
        print(f"  {field}: {reason}")
    print(
        "  anterior-layer priors: Tables 1-3 specify the transition model only; "
        "no initial distribution is published, so none is generated."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
