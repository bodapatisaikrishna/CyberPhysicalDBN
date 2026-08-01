"""Figure-2 attack graph as a NetworkX DiGraph.

Implements the attack graph of Cerotti et al., "Dynamic Bayesian Networks for
the Detection and Analysis of Cyber Attacks to Power Systems," IEEE Access 13
(2025), Figure 2, with MITRE tactic/technique labels from Figures 1, 3 and 4
and mean times-to-completion from Table 3.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Literal

import networkx as nx

NodeType = Literal["attack_step", "analytic", "reaction", "goal", "latch"]
GateType = Literal["AND", "OR"]
EdgeType = Literal["precondition", "triggers_analytic"]


def _attack_step(
    ttc: Fraction,
    matrix: str,
    tactic: str,
    technique: str,
) -> dict[str, Any]:
    return {
        "node_type": "attack_step",
        "ttc": ttc,
        "self_loop": True,
        "gate": None,
        "mitre_matrix": matrix,
        "mitre_tactic": tactic,
        "mitre_technique": technique,
    }


def _analytic() -> dict[str, Any]:
    return {
        "node_type": "analytic",
        "ttc": None,
        "self_loop": False,
        "gate": None,
        "mitre_matrix": None,
        "mitre_tactic": None,
        "mitre_technique": None,
    }


# Mean TTCs are Table 3 (time unit: 10 min), stored as exact Fractions because
# the paper gives 1/3 and 1/2 exactly.
#
# Deviation from Table 3's presentation: Table 3 lists CorrReact,
# WrongLogicExec, CredAccess and UnstablePS with a completion time of 0. Those
# are not measured times — they mark nodes that are not parameterized by
# uniformization. They are stored here as ttc=None rather than 0.0, because a
# stored 0 would enter compute_delta_t's sum of 1/T_bar as a division by zero,
# and because 0 reads as a measured value rather than "not applicable."
_NODE_DEFS: dict[str, dict[str, Any]] = {
    # --- Centre: MITM chain (paper Figure 1) ---
    "UnsecCred1": _attack_step(
        Fraction(1, 3), "ENT", "Credential Access", "Unsecured Credentials"
    ),
    "UnsecCred2": _attack_step(
        Fraction(1, 3), "ENT", "Credential Access", "Unsecured Credentials"
    ),
    "UnsecCred": _attack_step(
        Fraction(1, 3), "ENT", "Credential Access", "Unsecured Credentials"
    ),
    "ModAuthProc1": _attack_step(
        Fraction(1, 2), "ENT", "Credential Access", "Modify authentication process"
    ),
    "ModAuthProc": _attack_step(
        Fraction(1, 2), "ENT", "Credential Access", "Modify authentication process"
    ),
    "CredAccess": {
        "node_type": "attack_step",
        "ttc": None,
        "self_loop": False,
        "gate": "AND",
        "mitre_matrix": None,
        "mitre_tactic": "Credential Access",
        "mitre_technique": None,
    },
    "MITM": _attack_step(
        Fraction(2), "ICS", "Collection", "Man-in-the-middle"
    ),
    "SpoofRepMsg": _attack_step(
        Fraction(15), "ICS", "Impair process control", "Spoof Reporting Message"
    ),
    "UnauthCommand": _attack_step(
        Fraction(40), "ICS", "Impair process control", "Unauthorized Command Message"
    ),
    # --- Left: Stuxnet-style control-logic manipulation (paper Figure 3) ---
    "ModCtrlLogic": _attack_step(
        Fraction(50), "ICS", "Impact", "Manipulation of Control"
    ),
    # --- Right: rogue ICS service on SCADA controller (paper Figure 4) ---
    "ModifyProgram": _attack_step(
        Fraction(2), "ICS", "Persistence", "Modify Program"
    ),
    "Masquerade": _attack_step(
        Fraction(2), "ICS", "Evasion", "Masquerading"
    ),
    # --- Control-centre reactions. Not attack steps (paper Sec. IV): their
    # probabilities express defence efficacy and are given directly, not
    # derived from a TTC via uniformization.
    #
    # self_loop=False, matching Table 3's TTC of 0: like the AND/OR gates,
    # a reaction resolves within the slice rather than racing to complete,
    # so it neither persists nor crosses a slice boundary. Fig. 2 does draw
    # self-loops on these two nodes and its caption lists them inside
    # BK-clusters, which points the other way -- see build_reaction_cpt and
    # LAB_NOTEBOOK.md (2026-07-31) for why the numerical evidence (Fig. 5a
    # has CorrReact plateau at exactly 0.7, tracking 0.7 x P(SpoofRepMsg))
    # is treated as decisive here. This is a documented deviation from the
    # drawn figure, not an oversight. ---
    "CorrReact": {
        "node_type": "reaction",
        "ttc": None,
        "self_loop": False,
        "gate": None,
        "mitre_matrix": None,
        "mitre_tactic": None,
        "mitre_technique": None,
        "fixed_success_prob": 0.7,
    },
    "WrongLogicExec": {
        "node_type": "reaction",
        "ttc": None,
        "self_loop": False,
        "gate": None,
        "mitre_matrix": None,
        "mitre_tactic": None,
        "mitre_technique": None,
        "fixed_success_prob": 0.8,
    },
    # --- Goal ---
    "UnstablePS": {
        "node_type": "goal",
        "ttc": None,
        "self_loop": False,
        "gate": "OR",
        "mitre_matrix": None,
        "mitre_tactic": None,
        "mitre_technique": None,
    },
    # --- Analytics (untimed evidence nodes) ---
    "FileAccess": _analytic(),
    "FileIntegrity": _analytic(),
    "MeasureCoherence": _analytic(),
    "CommandCoherence": _analytic(),
    "SWIntegrityDER": _analytic(),
    "NewServiceStarted": _analytic(),
    "SWIntegritySCADA": _analytic(),
    "SuspArg": _analytic(),
}

_PRECONDITION_EDGES: list[tuple[str, str]] = [
    ("UnsecCred1", "UnsecCred2"),
    ("UnsecCred2", "UnsecCred"),
    ("ModAuthProc1", "ModAuthProc"),
    ("UnsecCred", "CredAccess"),
    ("ModAuthProc", "CredAccess"),
    ("CredAccess", "MITM"),
    ("MITM", "SpoofRepMsg"),
    ("MITM", "UnauthCommand"),
    ("SpoofRepMsg", "CorrReact"),
    ("ModCtrlLogic", "WrongLogicExec"),
    ("WrongLogicExec", "UnstablePS"),
    ("CorrReact", "UnstablePS"),
    ("UnauthCommand", "UnstablePS"),
    ("ModifyProgram", "Masquerade"),
    ("Masquerade", "UnauthCommand"),
]

_TRIGGERS_ANALYTIC_EDGES: list[tuple[str, str]] = [
    ("UnsecCred", "FileAccess"),
    ("ModAuthProc", "FileIntegrity"),
    ("SpoofRepMsg", "MeasureCoherence"),
    ("UnauthCommand", "CommandCoherence"),
    ("ModCtrlLogic", "SWIntegrityDER"),
    ("ModifyProgram", "NewServiceStarted"),
    ("ModifyProgram", "SWIntegritySCADA"),
    ("Masquerade", "SuspArg"),
]


ReactionMode = Literal["memoryless", "latched"]

# Suffix for the auxiliary latch node accompanying each reaction in "latched"
# mode. Kept as a constant so downstream code can identify latches without
# string-matching a literal.
LATCH_SUFFIX = "__Seen"


def build_attack_graph(reaction_mode: ReactionMode = "memoryless") -> nx.DiGraph:
    """Build the attack graph of Cerotti et al. Figure 2.

    Nodes carry name, node_type, mitre_technique_id, mitre_matrix, mitre_tactic,
    mitre_technique, ttc, self_loop and gate. Edges carry edge_type and
    inter_slice.

    Self-loops model attack-step persistence (Sec. III-A: "if an attack step has
    been carried out at time t, in the future time slice we want to preserve the
    information that it occurred"). Analytics are untimed and carry none.

    inter_slice is derived here rather than in the compiler: an edge crosses a
    time slice exactly when both of its endpoints persist, i.e. both self-loop.

    reaction_mode selects between two readings of the control-centre reactions
    (CorrReact, WrongLogicExec) that the source paper's own figures disagree on
    (see LAB_NOTEBOOK.md 2026-07-31):

      "memoryless" -- a reaction is redrawn every slice from its precondition,
          so its marginal is p_fixed * P(precondition). Plateaus at exactly 0.7
          for CorrReact, reproducing Fig. 5a, but carries no state, so FF and
          EX converge and KL(EX||FF) decays to 0 -- contradicting Fig. 6c,
          which states the FF divergence "is not able to converge to the exact
          solution (it stabilizes just below 0.12)".

      "latched" -- the control centre gets exactly ONE chance to react, at the
          moment the precondition first holds, succeeding with p_fixed; that
          outcome then persists. Also plateaus at p_fixed (Fig. 5a) but carries
          state across slices, so FF's independence assumption produces a
          permanent error (Fig. 6c, Fig. 8a).

    "latched" requires an auxiliary latch node per reaction and is therefore a
    structural addition not drawn in Fig. 2. It is needed because the one-shot
    condition is "precondition holds now AND did not hold last slice", i.e. a
    dependency on t-2, which a 2TBN's Markov-order-1 transition model cannot
    express without carrying that extra slice of memory in a node.
    """
    graph = nx.DiGraph()

    for name, attrs in _NODE_DEFS.items():
        graph.add_node(name, name=name, mitre_technique_id=None, **attrs)

    for source, target in _PRECONDITION_EDGES:
        graph.add_edge(source, target, edge_type="precondition")

    for source, target in _TRIGGERS_ANALYTIC_EDGES:
        graph.add_edge(source, target, edge_type="triggers_analytic")

    if reaction_mode == "latched":
        _apply_latched_reactions(graph)

    for name, attrs in graph.nodes(data=True):
        if attrs["self_loop"]:
            graph.add_edge(name, name, edge_type="precondition")

    for source, target, data in graph.edges(data=True):
        data["inter_slice"] = (
            graph.nodes[source]["self_loop"] and graph.nodes[target]["self_loop"]
        )

    return graph


def _apply_latched_reactions(graph: nx.DiGraph) -> None:
    """Convert each reaction into a one-shot latched reaction, in place.

    For a reaction R with precondition P, adds `R__Seen`: a persistent node
    that latches to 1 once P has held, and makes R itself persistent with
    parents {P, R__Seen}. R can then only fire in the single slice where P
    holds but R__Seen has not yet latched.
    """
    reactions = [n for n, d in graph.nodes(data=True) if d["node_type"] == "reaction"]

    for reaction in reactions:
        parents = [
            p
            for p in graph.predecessors(reaction)
            if graph.edges[p, reaction]["edge_type"] == "precondition" and p != reaction
        ]
        if len(parents) != 1:
            raise ValueError(
                f"latched reaction {reaction!r} expects exactly one precondition, "
                f"got {parents}"
            )
        (precondition,) = parents

        latch = f"{reaction}{LATCH_SUFFIX}"
        graph.add_node(
            latch,
            name=latch,
            mitre_technique_id=None,
            node_type="latch",
            ttc=None,
            self_loop=True,
            gate=None,
            mitre_matrix=None,
            mitre_tactic=None,
            mitre_technique=None,
            latch_for=reaction,
        )
        graph.add_edge(precondition, latch, edge_type="precondition")
        graph.add_edge(latch, reaction, edge_type="precondition")

        graph.nodes[reaction]["self_loop"] = True
        graph.nodes[reaction]["latch"] = latch


def undetermined_fields() -> dict[str, str]:
    """Node attributes left None because the source paper does not supply them.

    Required by CLAUDE.md rule 1: a value that was never published does not get
    invented. Anything not listed here is populated from the paper.
    """
    return {
        "mitre_technique_id": (
            "Figures 1, 3 and 4 name the MITRE matrix, tactic and technique but "
            "never give numeric ATT&CK technique IDs (e.g. T1557), so no ID can "
            "be assigned without inventing one."
        )
    }
