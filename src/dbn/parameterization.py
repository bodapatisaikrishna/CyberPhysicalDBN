"""Uniformization and CPT construction.

Implements Cerotti et al. Eq. 3 (Sec. III-E) and the CPT structures of Table 1
(attack step), Table 2 (analytic) and the deterministic AND/OR gates of Fig. 2.
"""

from __future__ import annotations

import itertools
from typing import Literal

import networkx as nx
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from src.dbn.compiler import ANTERIOR, ULTERIOR, DBNNode

GateType = Literal["AND", "OR"]


def collect_uniformization_ttcs(ag: nx.DiGraph) -> dict[str, float]:
    """Mean completion times entering Eq. 3's denominator.

    Only nodes carrying a TTC participate. The control-centre reactions and the
    AND/OR gates are excluded: they are not attack steps racing to completion,
    so they do not belong in the sum of competing rates (Cerotti et al. Sec. IV).
    """
    return {
        name: float(data["ttc"])
        for name, data in ag.nodes(data=True)
        if data["ttc"] is not None
    }


def compute_delta_t(ttcs: dict[str, float], m: float) -> float:
    """Discretization step size (Cerotti et al. Eq. 3, Sec. III-E).

        delta_t = 1 / ( m * sum_i (1 / T_bar_i) )

    m sets the accuracy of the approximation: larger m gives a finer step.
    """
    if m <= 0:
        raise ValueError(f"m must be positive, got {m}")
    if not ttcs:
        raise ValueError("at least one TTC is required to compute delta_t")
    for name, ttc in ttcs.items():
        if ttc <= 0:
            raise ValueError(f"TTC for {name!r} must be positive, got {ttc}")

    return 1.0 / (m * sum(1.0 / ttc for ttc in ttcs.values()))


def compute_ps(ttc: float, delta_t: float) -> float:
    """Per-step attack success probability (Cerotti et al. Eq. 3, Sec. III-E).

        p_s = delta_t / T_bar_s
    """
    if ttc <= 0:
        raise ValueError(f"TTC must be positive, got {ttc}")
    return delta_t / ttc


def _parent_slice(ag: nx.DiGraph, parent: str, node: str) -> int:
    return ANTERIOR if ag.edges[parent, node]["inter_slice"] else ULTERIOR


def precondition_parents(ag: nx.DiGraph, node: str) -> list[str]:
    """Predecessors of `node` excluding its own persistence self-loop."""
    return sorted(p for p in ag.predecessors(node) if p != node)


def build_attack_step_cpt(
    node: str,
    parents: list[str],
    p_s: float,
    ag: nx.DiGraph,
) -> TabularCPD:
    """CPT of an attack step or reaction (Cerotti et al. Table 1, generalized).

    Table 1 gives the one-parent case (SpoofRepMsg given MITM and itself). Its
    eight rows decompose into three rules applied in this precedence order:

      1. precondition unsatisfied -> P(active) = 0   (Table 1 rows 1-4)
      2. else already active      -> P(active) = 1   (Table 1 rows 7-8)
      3. else                     -> P(active) = p_s (Table 1 rows 5-6)

    The precondition is the OR over the node's parents, so the rules do not
    depend on the number of parents and generalize to 2^(N+1) columns.

    Rule 1 outranking rule 2 is what Table 1 rows 3-4 state: with MITM inactive,
    SpoofRepMsg goes to 0 with probability 1 even when it was active. That
    region is unreachable from an inactive initial state (a child only activates
    while its parent is active, and parents persist), so it is a don't-care the
    paper filled with the forced-0 convention. Reproduced as published.

    Root nodes have no parents and take the precondition as vacuously satisfied.
    Otherwise no root could ever activate and the whole graph would stay inert,
    contradicting Table 3 assigning them finite TTCs. This is inference from the
    model's intent; Table 1 only shows the one-parent case.

    Evidence is ordered parents-then-self, matching Table 1's column order under
    pgmpy's product ordering over evidence.
    """
    ordered_parents = sorted(parents)
    evidence: list[DBNNode] = [
        (parent, _parent_slice(ag, parent, node)) for parent in ordered_parents
    ] + [(node, ANTERIOR)]

    prob_inactive: list[float] = []
    prob_active: list[float] = []
    for states in itertools.product([0, 1], repeat=len(ordered_parents) + 1):
        *parent_states, self_previously_active = states
        precondition_met = any(parent_states) if ordered_parents else True

        if not precondition_met:
            p_activate = 0.0
        elif self_previously_active:
            p_activate = 1.0
        else:
            p_activate = p_s

        prob_inactive.append(1.0 - p_activate)
        prob_active.append(p_activate)

    return TabularCPD(
        (node, ULTERIOR),
        2,
        [prob_inactive, prob_active],
        evidence=evidence,
        evidence_card=[2] * len(evidence),
    )


def build_analytic_cpt(
    node: str,
    parent: str,
    p_pos: float,
    p_neg: float,
    ag: nx.DiGraph,
) -> TabularCPD:
    """CPT of an analytic (Cerotti et al. Table 2).

    Analytics are untimed: no self-loop, no temporal arc, so the CPT depends
    only on its technique parent within the same slice. p_pos is the false
    positive rate (alarm with no attack step), p_neg the false negative rate
    (attack step goes undetected).
    """
    evidence: list[DBNNode] = [(parent, _parent_slice(ag, parent, node))]
    return TabularCPD(
        (node, ULTERIOR),
        2,
        [[1.0 - p_pos, p_neg], [p_pos, 1.0 - p_neg]],
        evidence=evidence,
        evidence_card=[2],
    )


def build_gate_cpt(
    node: str,
    parents: list[str],
    gate_type: GateType,
    ag: nx.DiGraph,
) -> TabularCPD:
    """CPT of a deterministic AND/OR gate (Cerotti et al. Fig. 2).

    CredAccess is AND-gated over its two credential-theft branches; UnstablePS
    is OR-gated over the three paths reaching it. Table 3 lists both with a
    completion time of 0, i.e. they resolve within the slice rather than racing
    to complete, so they carry no p_s and depend on their parents at the same
    slice.
    """
    ordered_parents = sorted(parents)
    combine = all if gate_type == "AND" else any
    evidence: list[DBNNode] = [
        (parent, _parent_slice(ag, parent, node)) for parent in ordered_parents
    ]

    prob_inactive: list[float] = []
    prob_active: list[float] = []
    for states in itertools.product([0, 1], repeat=len(ordered_parents)):
        active = combine(states)
        prob_inactive.append(0.0 if active else 1.0)
        prob_active.append(1.0 if active else 0.0)

    return TabularCPD(
        (node, ULTERIOR),
        2,
        [prob_inactive, prob_active],
        evidence=evidence,
        evidence_card=[2] * len(evidence),
    )


def attach_cpds(
    dbn: DiscreteBayesianNetwork,
    ag: nx.DiGraph,
    m: float,
    p_pos: float,
    p_neg: float,
) -> DiscreteBayesianNetwork:
    """Build and attach every ulterior-layer CPD.

    Anterior-layer priors are deliberately not generated: Cerotti et al. Tables
    1-3 specify the transition model only, and the paper states no initial
    distribution. Inventing one would violate CLAUDE.md rule 1. The model is
    therefore not check_model()-valid yet; that is deferred to the inference
    phase, where the prior becomes an explicit, logged configuration choice.
    """
    delta_t = compute_delta_t(collect_uniformization_ttcs(ag), m)

    cpds: list[TabularCPD] = []
    for node, data in ag.nodes(data=True):
        parents = precondition_parents(ag, node)

        if data["node_type"] == "analytic":
            (parent,) = parents
            cpds.append(build_analytic_cpt(node, parent, p_pos, p_neg, ag))
        elif data["gate"] is not None:
            cpds.append(build_gate_cpt(node, parents, data["gate"], ag))
        elif data["node_type"] == "reaction":
            cpds.append(
                build_attack_step_cpt(node, parents, data["fixed_success_prob"], ag)
            )
        else:
            p_s = compute_ps(float(data["ttc"]), delta_t)
            cpds.append(build_attack_step_cpt(node, parents, p_s, ag))

    dbn.add_cpds(*cpds)
    return dbn
