"""Boyen-Koller-style clustered forward filtering over the compiled 2TBN.

Implements Cerotti et al. Sec. III-B/III-C (interface nodes, BK clustering) and
Sec. IV's FF (fully-factorized) and EX (exact, single cluster) configurations.

Engine: pgmpy VariableElimination, driven by swapping the anterior-layer prior
CPD(s) each step, rather than a hand-written tensor contraction. Two reasons.
First, `parameterization.attach_cpds`'s transition CPTs (tested 18/18 last
session) are reused completely unmodified as the one source of truth for
transition probabilities. Second, and more important: it was tempting to assume
each ulterior interface node's value depends only on anterior-layer parents,
making the 13 interface nodes conditionally independent given a fixed anterior
assignment. That is false for MITM: its only parent is CredAccess, which does
not self-loop, so CredAccess(t) -> MITM(t) is intra-slice
(see compiler.compile_to_2tbn and
tests/test_parameterization.py::test_mitm_depends_on_credaccess_within_slice).
MITM is correlated with UnsecCred/ModAuthProc through the same-slice CredAccess
gate, not only through history. A hand-written contraction that assumed
per-node independence would have silently produced KL(EX||FF)=0 for MITM at
every t, masking exactly the effect Cerotti et al. Fig. 6e reports. Routing
every step through pgmpy's exact elimination algorithm removes this whole class
of bug: the only approximation FF/EX differ on is how the anterior-layer prior
is factorized between steps, never how a single step's transition math is
computed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from src.dbn.compiler import ANTERIOR, ULTERIOR, compile_to_2tbn
from src.dbn.parameterization import attach_cpds

Clustering = frozenset[frozenset[str]]
BeliefState = dict[frozenset[str], np.ndarray]


def _interface_nodes(ag: nx.DiGraph) -> list[str]:
    """The 13 self-looping (persistent) nodes -- the only ones whose value
    crosses a time slice, and so the only ones a filtering approximation needs
    to track jointly or independently between steps."""
    return sorted(n for n, d in ag.nodes(data=True) if d["self_loop"])


def fully_factorized_clustering(interface_nodes: list[str]) -> Clustering:
    """FF: every interface node in its own cluster (Cerotti et al. Sec. IV)."""
    return frozenset(frozenset([n]) for n in interface_nodes)


def exact_clustering(interface_nodes: list[str]) -> Clustering:
    """EX: a single cluster containing every interface node (Sec. IV)."""
    return frozenset({frozenset(interface_nodes)})


def _validate_clustering(clustering: Clustering, interface_nodes: list[str]) -> None:
    covered: set[str] = set()
    for cluster in clustering:
        overlap = covered & cluster
        if overlap:
            raise ValueError(f"clusters are not disjoint, shared nodes: {overlap}")
        covered |= cluster
    if covered != set(interface_nodes):
        raise ValueError(
            "clustering must partition exactly the interface nodes; "
            f"missing={set(interface_nodes) - covered}, "
            f"extra={covered - set(interface_nodes)}"
        )


@dataclass(frozen=True)
class InferenceConfig:
    clustering: Clustering
    m: float
    p_pos: float
    p_neg: float
    # Overrides Eq. 3's computed delta_t (m and collect_uniformization_ttcs's
    # 11-node sum) with an explicit value when set. Exists because Table 5
    # (Cerotti et al.) publishes delta_t directly, and its own value implies
    # Sigma_i(1/T_bar_i) ~= 3.6117 -- about 4x smaller than the 11-node sum --
    # consistently across all six published m values. Which TTC subset
    # produces that Sigma could not be uniquely determined from Sec. III-E's
    # text (see LAB_NOTEBOOK.md, 2026-07-31 entry: a brute-force search over
    # all 2^11 subsets found several structurally unrelated node-sets tied on
    # the identical sum). Matching Table 5's own published delta_t sidesteps
    # that ambiguity rather than guessing at it. m is not used when this is
    # set, since delta_t depends only on the product m*Sigma.
    delta_t_override: float | None = None


@dataclass
class StepResult:
    marginals: dict[str, float]  # P(node=1) at this ulterior slice, all 23 nodes
    next_belief: BeliefState
    latency_s: float


@dataclass
class Trajectory:
    marginals: list[dict[str, float]] = field(default_factory=list)  # index = t
    latencies_s: list[float] = field(default_factory=list)


def _cluster_prior_node(cluster: frozenset[str]) -> tuple[str, int]:
    """Stable auxiliary-node name for a multi-member cluster's joint prior."""
    return (f"__cluster_prior__:{','.join(sorted(cluster))}", ANTERIOR)


class DBNInference:
    """Clustered forward filter over the Cerotti et al. attack-graph 2TBN.

    `clustering` partitions the 13 interface nodes into mutually-independent
    blocks (Cerotti et al.'s BK-clusters). A block of size 1 is a plain
    independent-marginal prior. A block of size k>1 is represented as one
    auxiliary categorical node of 2^k states plus k deterministic decoder CPDs
    (bit i of the state index -> the i-th node in sorted(cluster), i=0 most
    significant) -- this is how EX's single 13-node, 8192-state cluster is
    represented, and it generalizes to any future intermediate clustering (e.g.
    CL) without a different code path.
    """

    def __init__(self, ag: nx.DiGraph, config: InferenceConfig):
        self.ag = ag
        self.config = config
        self.interface_nodes = _interface_nodes(ag)
        _validate_clustering(config.clustering, self.interface_nodes)
        self.report_nodes = sorted(ag.nodes())

        dbn = compile_to_2tbn(ag)
        self._multi_clusters = [c for c in config.clustering if len(c) > 1]
        for cluster in self._multi_clusters:
            prior_node = _cluster_prior_node(cluster)
            for name in sorted(cluster):
                dbn.add_edge(prior_node, (name, ANTERIOR))

        self.dbn: DiscreteBayesianNetwork = attach_cpds(
            dbn,
            ag,
            m=config.m,
            p_pos=config.p_pos,
            p_neg=config.p_neg,
            delta_t_override=config.delta_t_override,
        )

        for cluster in self._multi_clusters:
            self.dbn.add_cpds(*self._decoder_cpds(cluster))

    def _decoder_cpds(self, cluster: frozenset[str]) -> list[TabularCPD]:
        members = sorted(cluster)
        n_states = 2 ** len(members)
        prior_node = _cluster_prior_node(cluster)
        state_index = np.arange(n_states)
        decoders = []
        for i, name in enumerate(members):
            bit = (state_index >> (len(members) - 1 - i)) & 1
            values = np.array([1 - bit, bit], dtype=float)
            decoders.append(
                TabularCPD(
                    (name, ANTERIOR),
                    2,
                    values,
                    evidence=[prior_node],
                    evidence_card=[n_states],
                )
            )
        return decoders

    def initial_belief(self) -> BeliefState:
        """Point mass at 'every interface node inactive'.

        Evidenced directly by the paper's own Figs. 5 and 7: every technique
        curve starts at Pr=0 at t=0. Not an invented default.
        """
        belief: BeliefState = {}
        for cluster in self.config.clustering:
            arr = np.zeros((2,) * len(cluster))
            arr[(0,) * len(cluster)] = 1.0
            belief[cluster] = arr
        return belief

    def _attach_belief_as_prior(self, belief: BeliefState) -> None:
        cpds = []
        for cluster, arr in belief.items():
            if len(cluster) == 1:
                (name,) = cluster
                p1 = float(arr[1])
                cpds.append(TabularCPD((name, ANTERIOR), 2, [[1 - p1], [p1]]))
            else:
                prior_node = _cluster_prior_node(cluster)
                flat = arr.reshape(-1, 1)
                n_states = flat.shape[0]
                cpds.append(TabularCPD(prior_node, n_states, flat))
        self.dbn.add_cpds(*cpds)

    def step(self, belief: BeliefState, evidence: dict[str, int]) -> StepResult:
        self._attach_belief_as_prior(belief)
        infer = VariableElimination(self.dbn)
        ve_evidence = {(name, ULTERIOR): value for name, value in evidence.items()}

        # pgmpy disallows querying a variable that is also given as evidence:
        # an evidenced node's marginal is trivially the observed value itself.
        query_nodes = [n for n in self.report_nodes if n not in evidence]

        t0 = time.perf_counter()
        report = infer.query(
            [(n, ULTERIOR) for n in query_nodes],
            evidence=ve_evidence,
            joint=False,
            show_progress=False,
        )

        next_belief: BeliefState = {}
        for cluster in self.config.clustering:
            if len(cluster) == 1:
                (name,) = cluster
                p1 = float(report[(name, ULTERIOR)].values[1])
                next_belief[cluster] = np.array([1 - p1, p1])
            else:
                members = sorted(cluster)
                joint = infer.query(
                    [(n, ULTERIOR) for n in members],
                    evidence=ve_evidence,
                    joint=True,
                    show_progress=False,
                )
                wanted = [(n, ULTERIOR) for n in members]
                if joint.variables != wanted:
                    # pgmpy has empirically preserved the requested variable
                    # order in every check performed for this model, but
                    # nothing in the API contract guarantees it, so correct
                    # for it explicitly rather than assume it silently.
                    axis_order = [joint.variables.index(v) for v in wanted]
                    values = np.moveaxis(
                        np.asarray(joint.values), axis_order, range(len(wanted))
                    )
                else:
                    values = np.asarray(joint.values)
                next_belief[cluster] = values
        latency_s = time.perf_counter() - t0

        marginals = {
            name: float(evidence[name]) if name in evidence
            else float(report[(name, ULTERIOR)].values[1])
            for name in self.report_nodes
        }
        return StepResult(marginals=marginals, next_belief=next_belief, latency_s=latency_s)

    def run(self, evidence_stream: dict[int, dict[str, int]], T: int) -> Trajectory:
        """Run T steps (t=1..T), applying `evidence_stream.get(t, {})` at each."""
        trajectory = Trajectory()
        belief = self.initial_belief()
        for t in range(1, T + 1):
            result = self.step(belief, evidence_stream.get(t, {}))
            trajectory.marginals.append(result.marginals)
            trajectory.latencies_s.append(result.latency_s)
            belief = result.next_belief
        return trajectory
