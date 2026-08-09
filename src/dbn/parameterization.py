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

    Raises if the result exceeds 1. Eq. 3's uniformization is only valid while
    delta_t <= min_i(T_bar_i); past that the "probability" is not one, and the
    resulting CPT column would carry a negative P(inactive) = 1 - p_s. pgmpy
    does reject that downstream ("CPD values must be non-negative"), but from a
    call site several frames away and with no hint of the real cause, which is
    always the same: m is too small for this graph's fastest TTC. Callers that
    sweep m (experiments/exp09, exp10) pre-check this themselves; this makes
    the invariant hold for every caller instead of by convention.
    """
    if ttc <= 0:
        raise ValueError(f"TTC must be positive, got {ttc}")
    p_s = delta_t / ttc
    if p_s > 1.0:
        raise ValueError(
            f"p_s = delta_t/TTC = {delta_t}/{ttc} = {p_s:.6g} > 1, which is not a "
            "probability: Eq. 3's uniformization requires delta_t <= the smallest "
            "TTC in the graph. Increase m (or lower delta_t_override)."
        )
    return p_s


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


def build_reaction_cpt(
    node: str,
    parents: list[str],
    success_prob: float,
    ag: nx.DiGraph,
) -> TabularCPD:
    """CPT of a control-centre reaction (CorrReact, WrongLogicExec).

    Reactions are NOT attack steps and do not use Table 1's persistence rule.
    Cerotti et al. Sec. IV: they "are reactions of the control center ... the
    parameters we choose for these nodes reflect the efficacy of these
    defenses; we allow CorrReact to succeed with probability 0.7 and
    WrongLogicExec with probability 0.8", and Table 3 lists both with a
    completion time of 0 -- they resolve within a slice rather than racing to
    complete, exactly like the AND/OR gates.

    So the rule is simply:

        P(active) = success_prob   if the precondition holds
        P(active) = 0              otherwise

    with NO dependence on the node's own previous state. Applying Table 1's
    persistence rule here instead (already-active forces 1) would make the
    reaction converge to 1 as its precondition saturates, which contradicts
    the paper's own Fig. 5a: CorrReact plateaus at exactly 0.7 there, and
    tracks 0.7 x P(SpoofRepMsg) to three decimals at t=20, 30, 50 and 200.
    That reading also reproduces the paper's stated Scenario 2 value of
    exactly 0.7 for CorrReact once MeasureCoherence forces SpoofRepMsg to 1.

    Structurally this makes a reaction behave exactly like an AND/OR gate: a
    TTC of 0 means it resolves within the slice rather than over time, so it
    carries no self-loop and reads its parent at the SAME slice. That is what
    lets CorrReact reach 0.7 at the very slice MeasureCoherence forces
    SpoofRepMsg to 1, matching the paper's Scenario 2 description of both
    changing at t=31 together. Note a tension with Fig. 2, which draws
    self-loops on both reaction nodes and lists them inside BK-clusters;
    Table 3's TTC=0 and the numerical evidence above are treated as decisive
    over the drawn arc, and this is flagged in LAB_NOTEBOOK.md rather than
    resolved silently.
    """
    ordered_parents = sorted(parents)
    evidence: list[DBNNode] = [
        (parent, _parent_slice(ag, parent, node)) for parent in ordered_parents
    ]

    prob_inactive: list[float] = []
    prob_active: list[float] = []
    for states in itertools.product([0, 1], repeat=len(ordered_parents)):
        precondition_met = any(states) if ordered_parents else True
        p_activate = success_prob if precondition_met else 0.0
        prob_inactive.append(1.0 - p_activate)
        prob_active.append(p_activate)

    return TabularCPD(
        (node, ULTERIOR),
        2,
        [prob_inactive, prob_active],
        evidence=evidence,
        evidence_card=[2] * len(evidence),
    )


def build_latch_cpt(node: str, precondition: str, ag: nx.DiGraph) -> TabularCPD:
    """CPT of a reaction's auxiliary latch (see graph.build_attack_graph).

    Deterministic: the latch is 1 once its precondition has held at least once.

        latch(t) = 1  iff  latch(t-1) = 1  or  precondition(t-1) = 1

    This carries the extra slice of memory that lets the accompanying reaction
    fire only on its FIRST opportunity -- a t-2 dependency a 2TBN cannot
    express in the reaction node alone.
    """
    evidence: list[DBNNode] = [
        (precondition, _parent_slice(ag, precondition, node)),
        (node, ANTERIOR),
    ]

    prob_inactive: list[float] = []
    prob_active: list[float] = []
    for precondition_state, self_previously_latched in itertools.product([0, 1], repeat=2):
        latched = bool(precondition_state or self_previously_latched)
        prob_inactive.append(0.0 if latched else 1.0)
        prob_active.append(1.0 if latched else 0.0)

    return TabularCPD(
        (node, ULTERIOR),
        2,
        [prob_inactive, prob_active],
        evidence=evidence,
        evidence_card=[2, 2],
    )


def build_latched_reaction_cpt(
    node: str,
    precondition: str,
    latch: str,
    success_prob: float,
    ag: nx.DiGraph,
) -> TabularCPD:
    """CPT of a one-shot latched reaction (Cerotti et al. Sec. IV, 0.7 / 0.8).

    The control centre gets exactly ONE chance to react, in the slice where its
    precondition first holds, succeeding with `success_prob`; that outcome then
    persists. Rules, in precedence order:

      1. already reacted            -> 1  (persistence)
      2. precondition not yet held  -> 0
      3. latch already set          -> 0  (the one chance has been used, and
                                           rule 1 did not fire, so it failed)
      4. otherwise (first chance)   -> success_prob

    The marginal is therefore success_prob * P(precondition ever held), which
    plateaus at exactly 0.7 for CorrReact (matching Fig. 5a) while carrying
    state across slices (so FF's independence assumption incurs a permanent
    error, matching Fig. 6c's non-converging divergence).

    Evidence order is (latch, precondition, self), all at the anterior layer.
    """
    evidence: list[DBNNode] = [
        (latch, ANTERIOR),
        (precondition, _parent_slice(ag, precondition, node)),
        (node, ANTERIOR),
    ]

    prob_inactive: list[float] = []
    prob_active: list[float] = []
    for latch_set, precondition_state, self_previously_active in itertools.product(
        [0, 1], repeat=3
    ):
        if self_previously_active:
            p_activate = 1.0
        elif not precondition_state:
            p_activate = 0.0
        elif latch_set:
            p_activate = 0.0
        else:
            p_activate = success_prob
        prob_inactive.append(1.0 - p_activate)
        prob_active.append(p_activate)

    return TabularCPD(
        (node, ULTERIOR),
        2,
        [prob_inactive, prob_active],
        evidence=evidence,
        evidence_card=[2, 2, 2],
    )


def analytic_error_rates(ag: nx.DiGraph, node: str, p_pos: float, p_neg: float) -> tuple[float, float]:
    """(p_pos, p_neg) for `node`, using its per-node SensorModel if it has one.

    Shared by `build_analytic_cpt` (here) and `src/twin/runner.py::discretize`'s
    graph-structure inspection, so the one place a node's error rates are
    declared (`ag.nodes[node]["sensor_model"]`, set in
    `src.attack_graph.graph.build_attack_graph`) is the only place they can be
    read from -- the CPT and the twin's own bookkeeping cannot drift apart.

    All 8 Session-1 analytics have `sensor_model=None` and fall through to the
    passed-in global rates unchanged, so this is a no-op for them: existing
    CPTs stay byte-identical. A voltage measurement (Session 4's PhysLocalDER/
    PhysWideArea) is not a 1e-4-false-positive cyber detector -- see
    LAB_NOTEBOOK.md 2026-08-01 on why a_pos/a_neg there encode model mismatch,
    not sensor noise, and must be measured rather than reused from Table 2.
    """
    sensor_model = ag.nodes[node].get("sensor_model")
    if sensor_model is None:
        return p_pos, p_neg
    return sensor_model.p_pos, sensor_model.p_neg


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
    (attack step goes undetected) -- unless `node` carries a per-node
    SensorModel override (`analytic_error_rates`), in which case its own
    rates are used instead of the two passed in.
    """
    p_pos, p_neg = analytic_error_rates(ag, node, p_pos, p_neg)
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
    delta_t_override: float | None = None,
) -> DiscreteBayesianNetwork:
    """Build and attach every ulterior-layer CPD.

    Anterior-layer priors are deliberately not generated: Cerotti et al. Tables
    1-3 specify the transition model only, and the paper states no initial
    distribution. Inventing one would violate CLAUDE.md rule 1. The model is
    therefore not check_model()-valid yet; that is deferred to the inference
    phase, where the prior becomes an explicit, logged configuration choice.

    delta_t_override, when given, replaces compute_delta_t(collect_
    uniformization_ttcs(ag), m) outright and m is not used. This is not a
    correction to Eq. 3 -- compute_delta_t is unit-tested against hand
    arithmetic and is correct as a formula. It is a documented uncertainty
    in what collect_uniformization_ttcs should sum over for THIS paper's
    specific graph: summing all 11 timed attack-step TTCs gives a delta_t
    about 4x smaller than the value Table 5 (Cerotti et al.) itself publishes,
    consistently across all six of Table 5's m values, and the exact TTC
    subset that reproduces Table 5's number could not be uniquely determined
    from Sec. III-E's text (see LAB_NOTEBOOK.md, 2026-07-31 entry). Passing
    delta_t_override=166.13/600 reproduces Table 5's m=1 row directly rather
    than guessing at which subset to change collect_uniformization_ttcs to.
    """
    delta_t = (
        delta_t_override
        if delta_t_override is not None
        else compute_delta_t(collect_uniformization_ttcs(ag), m)
    )

    cpds: list[TabularCPD] = []
    for node, data in ag.nodes(data=True):
        parents = precondition_parents(ag, node)

        if data["node_type"] == "analytic":
            (parent,) = parents
            cpds.append(build_analytic_cpt(node, parent, p_pos, p_neg, ag))
        elif data["gate"] is not None:
            cpds.append(build_gate_cpt(node, parents, data["gate"], ag))
        elif data["node_type"] == "latch":
            (parent,) = parents
            cpds.append(build_latch_cpt(node, parent, ag))
        elif data["node_type"] == "reaction":
            if "latch" in data:
                latch = data["latch"]
                (precondition,) = [p for p in parents if p != latch]
                cpds.append(
                    build_latched_reaction_cpt(
                        node, precondition, latch, data["fixed_success_prob"], ag
                    )
                )
            else:
                cpds.append(
                    build_reaction_cpt(node, parents, data["fixed_success_prob"], ag)
                )
        else:
            p_s = compute_ps(float(data["ttc"]), delta_t)
            cpds.append(build_attack_step_cpt(node, parents, p_s, ag))

    dbn.add_cpds(*cpds)
    return dbn
