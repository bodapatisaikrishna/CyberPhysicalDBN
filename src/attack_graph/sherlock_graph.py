"""Attack graph for the Sherlock dataset's attacks (exp20; LAB_NOTEBOOK.md 2026-10-16).

Sherlock (Wagner et al., CODASPY 2025, Table 2) contains only the FINAL phase of each attack -- the part
that acts on the grid -- so this graph has no credential-theft stage, unlike Cerotti et al. Fig. 2:

    ARPSpoof (T0830) --> BlockReporting (T0804)     arp-spoof-dos: interrupts the MTU connection
              |------> SpoofReporting (T0856)       drift-off, control-and-freeze: forged measurements
              |------> UnauthCommand  (T0855) <---- RogueClient (T0848)   industroyer: second IEC-104 client
    AttackImpact = OR(BlockReporting, SpoofReporting, UnauthCommand)     -- the query node

Each attack step has one analytic (soft-evidence) child; UnauthCommand has two (command traffic and the
grid's physical reaction). Node and edge attributes follow `src/attack_graph/graph.py`, so the graph
compiles and runs through the unchanged `src/dbn` pipeline.

UNCERTAIN (stated in the pre-registration): Table 2 says drift-off and control-and-freeze act "as MitM";
that the MitM is set up by ARP spoofing is an assumption, not something the dataset paper states.

Transition parameters are NOT set here: `build_sherlock_attack_graph` takes them as arguments (learned
per fold from labelled data, see `step_transition_rates`), so no expert number is baked in.
"""

from __future__ import annotations

from typing import Mapping

import networkx as nx
import numpy as np

from src.attack_graph.graph import SensorModel

ROOTS: tuple[str, ...] = ("ARPSpoof", "RogueClient")
STEPS: tuple[str, ...] = ("ARPSpoof", "RogueClient", "BlockReporting", "SpoofReporting", "UnauthCommand")
GOAL = "AttackImpact"
MITRE: dict[str, tuple[str, str, str]] = {  # technique id, tactic, technique name (MITRE ATT&CK for ICS)
    "ARPSpoof": ("T0830", "Collection", "Adversary-in-the-Middle"),
    "RogueClient": ("T0848", "Initial Access", "Rogue Master"),
    "BlockReporting": ("T0804", "Inhibit Response Function", "Block Reporting Message"),
    "SpoofReporting": ("T0856", "Evasion", "Spoof Reporting Message"),
    "UnauthCommand": ("T0855", "Impair Process Control", "Unauthorized Command Message"),
}
STEP_PARENTS: dict[str, tuple[str, ...]] = {
    "BlockReporting": ("ARPSpoof",), "SpoofReporting": ("ARPSpoof",), "UnauthCommand": ("ARPSpoof", "RogueClient"),
}
# analytic -> (attack step it observes, feature group: (feature name, sign)); sign -1 = a DROP is anomalous
ANALYTICS: dict[str, tuple[str, tuple[tuple[str, int], ...]]] = {
    "ARPAnomaly": ("ARPSpoof", (("n_arp", 1), ("n_arp_reply", 1), ("arp_ip_multi_mac", 1))),
    "NewClient": ("RogueClient", (("n_distinct_ip_pairs", 1), ("n_distinct_src_ip", 1), ("n_syn", 1))),
    "ReportingGap": ("BlockReporting", (("n_iec104_pkts", -1), ("n_iec104_i", -1), ("n_rst", 1))),
    "MeasureAnomaly": ("SpoofReporting", (("n_iec104_measurements", 1), ("n_iec104_unchanged_measurements", 1),
                                          ("iec104_payload_bytes", 1))),
    "CommandBurst": ("UnauthCommand", (("n_iec104_commands", 1),)),
    "GridEvent": ("UnauthCommand", (("n_switch_changes", 1), ("nonfinite_total", 1))),
}
# Sherlock attack family (catalog description prefix) -> attack steps active during it
FAMILY_STEPS: dict[str, frozenset[str]] = {
    "arp-spoof": frozenset({"ARPSpoof", "BlockReporting"}),
    "drift-off": frozenset({"ARPSpoof", "SpoofReporting"}),
    "control-and-freeze": frozenset({"ARPSpoof", "SpoofReporting", "UnauthCommand"}),
    "industroyer": frozenset({"RogueClient", "UnauthCommand"}),
}


def step_labels(families: list[str | None]) -> dict[str, np.ndarray]:
    """Per-slice 0/1 activity of every attack step, from per-slice attack family (None = no attack)."""
    out = {s: np.zeros(len(families), dtype=np.int8) for s in STEPS}
    for t, fam in enumerate(families):
        if fam is None:
            continue
        if fam not in FAMILY_STEPS:
            raise ValueError(f"unknown Sherlock attack family {fam!r}")
        for s in FAMILY_STEPS[fam]:
            out[s][t] = 1
    return out


def step_transition_rates(seqs: list[dict[str, np.ndarray]]) -> dict[str, tuple[float, float]]:
    """Maximum-likelihood (p_s, p_end) per attack step from labelled step-activity sequences.

    p_s   = onsets / slices where the step was inactive and its precondition held at t-1
            (roots: precondition always holds; children: some parent active at t-1, matching the
            inter-slice parent arcs the compiler builds between persistent nodes)
    p_end = ends / slices where the step was active at t-1 and its precondition still held at t
    A zero count would give p = 0, which forbids the event outright; such a rate is floored at
    1 / (eligible + 1) and the floor is reported by the caller (never silent).
    """
    rates: dict[str, tuple[float, float]] = {}
    for s in STEPS:
        on = el_on = end = el_end = 0
        for seq in seqs:
            x = seq[s].astype(bool)
            pre = np.ones_like(x) if s in ROOTS else np.logical_or.reduce([seq[p].astype(bool) for p in STEP_PARENTS[s]])
            prev, cur, pre_prev, pre_cur = x[:-1], x[1:], pre[:-1], pre[1:]
            elig_on = ~prev & pre_prev
            el_on += int(elig_on.sum()); on += int((elig_on & cur).sum())
            elig_end = prev & pre_cur
            el_end += int(elig_end.sum()); end += int((elig_end & ~cur).sum())
        p_s = on / el_on if on else 1.0 / (el_on + 1)
        p_end = end / el_end if end else 1.0 / (el_end + 1)
        rates[s] = (p_s, p_end)
    return rates


def build_sherlock_attack_graph(rates: Mapping[str, tuple[float, float]], *, bin_s: float,
                                analytic_error: tuple[float, float], source: str) -> nx.DiGraph:
    """The exp20 graph. `rates[step] = (p_s, p_end)` per slice; each step's `ttc` is set to
    bin_s / p_s so that Eq. 3's p_s = delta_t / T_bar with delta_t = bin_s returns p_s exactly.
    `analytic_error = (p_pos, p_neg)` for every analytic, with `source` saying where it came from."""
    g = nx.DiGraph()
    for s in STEPS:
        tid, tactic, tech = MITRE[s]
        p_s, p_end = rates[s]
        g.add_node(s, name=s, node_type="attack_step", ttc=bin_s / p_s, p_end=p_end, self_loop=True, gate=None,
                   mitre_technique_id=tid, mitre_matrix="ICS", mitre_tactic=tactic, mitre_technique=tech)
    g.add_node(GOAL, name=GOAL, node_type="goal", ttc=None, self_loop=False, gate="OR", mitre_technique_id=None,
               mitre_matrix=None, mitre_tactic=None, mitre_technique=None)
    sm = SensorModel(p_pos=analytic_error[0], p_neg=analytic_error[1], source=source)
    for a, (step, _) in ANALYTICS.items():
        g.add_node(a, name=a, node_type="analytic", ttc=None, self_loop=False, gate=None, mitre_technique_id=None,
                   mitre_matrix=None, mitre_tactic=None, mitre_technique=None, sensor_model=sm,
                   observable_kind="physical" if a == "GridEvent" else "cyber")
        g.add_edge(step, a, edge_type="triggers_analytic")
    for child, parents in STEP_PARENTS.items():
        for p in parents:
            g.add_edge(p, child, edge_type="precondition")
    for p in ("BlockReporting", "SpoofReporting", "UnauthCommand"):
        g.add_edge(p, GOAL, edge_type="precondition")
    for s in STEPS:
        g.add_edge(s, s, edge_type="precondition")
    for u, v, d in g.edges(data=True):
        d["inter_slice"] = g.nodes[u]["self_loop"] and g.nodes[v]["self_loop"]
    return g
