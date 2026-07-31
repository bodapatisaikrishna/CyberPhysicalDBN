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

NodeType = Literal["attack_step", "analytic", "reaction", "goal"]
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
    # derived from a TTC via uniformization. ---
    "CorrReact": {
        "node_type": "reaction",
        "ttc": None,
        "self_loop": True,
        "gate": None,
        "mitre_matrix": None,
        "mitre_tactic": None,
        "mitre_technique": None,
        "fixed_success_prob": 0.7,
    },
    "WrongLogicExec": {
        "node_type": "reaction",
        "ttc": None,
        "self_loop": True,
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


def build_attack_graph() -> nx.DiGraph:
    """Build the attack graph of Cerotti et al. Figure 2.

    Nodes carry name, node_type, mitre_technique_id, mitre_matrix, mitre_tactic,
    mitre_technique, ttc, self_loop and gate. Edges carry edge_type and
    inter_slice.

    Self-loops model attack-step persistence (Sec. III-A: "if an attack step has
    been carried out at time t, in the future time slice we want to preserve the
    information that it occurred"). Analytics are untimed and carry none.

    inter_slice is derived here rather than in the compiler: an edge crosses a
    time slice exactly when both of its endpoints persist, i.e. both self-loop.
    """
    graph = nx.DiGraph()

    for name, attrs in _NODE_DEFS.items():
        graph.add_node(name, name=name, mitre_technique_id=None, **attrs)

    for source, target in _PRECONDITION_EDGES:
        graph.add_edge(source, target, edge_type="precondition")

    for source, target in _TRIGGERS_ANALYTIC_EDGES:
        graph.add_edge(source, target, edge_type="triggers_analytic")

    for name, attrs in _NODE_DEFS.items():
        if attrs["self_loop"]:
            graph.add_edge(name, name, edge_type="precondition")

    for source, target, data in graph.edges(data=True):
        data["inter_slice"] = (
            graph.nodes[source]["self_loop"] and graph.nodes[target]["self_loop"]
        )

    return graph


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
