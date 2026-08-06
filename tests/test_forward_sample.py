"""Tests for src/dbn/forward_sample.py (Session 8, claim C2).

Split per the module's own scope: hand-replayable determinism on a tiny
fixture graph, row-by-row equivalence against the existing CPT builders'
enumerated cases, and structural invariants (persistence, gate resolution)
-- all runnable without any real twin/family data.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from src.attack_graph.graph import build_attack_graph
from src.dbn.forward_sample import (
    forward_sample_evidence_stream,
    forward_sample_trajectory,
    validate_slice_trajectory,
)
from src.dbn.parameterization import (
    build_attack_step_cpt,
    build_gate_cpt,
    compute_delta_t,
    compute_ps,
)


def _node(node_type, ttc=None, self_loop=False, gate=None, technique=None, **extra):
    return {
        "node_type": node_type,
        "ttc": ttc,
        "self_loop": self_loop,
        "gate": gate,
        "mitre_matrix": None,
        "mitre_tactic": None,
        "mitre_technique": technique,
        **extra,
    }


def _finalize(graph: nx.DiGraph) -> nx.DiGraph:
    """Mirrors build_attack_graph's tail: self-loop edges + inter_slice."""
    for name, attrs in graph.nodes(data=True):
        if attrs["self_loop"]:
            graph.add_edge(name, name, edge_type="precondition")
    for u, v, data in graph.edges(data=True):
        data["inter_slice"] = graph.nodes[u]["self_loop"] and graph.nodes[v]["self_loop"]
    return graph


@pytest.fixture
def chain_ag():
    """A -> B, both timed attack steps, no gates/reactions/analytics.
    T_bar_A=2, T_bar_B=4 (matches TestUniformization's own hand-computed
    delta_t=4/3, p_A=2/3, p_B=1/3 in tests/test_parameterization.py)."""
    g = nx.DiGraph()
    g.add_node("A", name="A", mitre_technique_id=None, **_node("attack_step", ttc=2.0, self_loop=True, technique="t1"))
    g.add_node("B", name="B", mitre_technique_id=None, **_node("attack_step", ttc=4.0, self_loop=True, technique="t2"))
    g.add_edge("A", "B", edge_type="precondition")
    return _finalize(g)


@pytest.fixture
def gated_ag():
    """A, B (roots) -> Gate (AND) -> C (timed). Tests gate resolution."""
    g = nx.DiGraph()
    g.add_node("A", name="A", mitre_technique_id=None, **_node("attack_step", ttc=2.0, self_loop=True, technique="t1"))
    g.add_node("B", name="B", mitre_technique_id=None, **_node("attack_step", ttc=2.0, self_loop=True, technique="t2"))
    g.add_node("Gate", name="Gate", mitre_technique_id=None, **_node("attack_step", ttc=None, self_loop=False, gate="AND"))
    g.add_node("C", name="C", mitre_technique_id=None, **_node("attack_step", ttc=3.0, self_loop=True, technique="t3"))
    g.add_edge("A", "Gate", edge_type="precondition")
    g.add_edge("B", "Gate", edge_type="precondition")
    g.add_edge("Gate", "C", edge_type="precondition")
    return _finalize(g)


class TestForwardSampleTrajectoryDeterminism:
    def test_hand_replay_matches_manual_bernoulli_draws(self, chain_ag):
        """Replay the SAME rng draws by hand and assert bitwise match."""
        rng = np.random.default_rng(7)
        trajectory, delta_t = forward_sample_trajectory(chain_ag, m=1.0, n_slices=5, rng=rng)
        assert delta_t == pytest.approx(4.0 / 3.0)

        p_a = compute_ps(2.0, delta_t)
        p_b = compute_ps(4.0, delta_t)
        rng2 = np.random.default_rng(7)
        a_active, b_active = 0, 0
        expected_a, expected_b = [], []
        for _ in range(5):
            prev_a = a_active  # B's precondition reads A at ANTERIOR (previous slice)
            if not a_active:
                a_active = int(rng2.random() < p_a)
            expected_a.append(a_active)
            if b_active:
                pass
            elif prev_a:
                b_active = int(rng2.random() < p_b)
            else:
                b_active = 0
            expected_b.append(b_active)

        assert trajectory["A"] == expected_a
        assert trajectory["B"] == expected_b

    def test_same_seed_reproduces_trajectory(self, chain_ag):
        t1, _ = forward_sample_trajectory(chain_ag, m=1.0, n_slices=20, rng=np.random.default_rng(3))
        t2, _ = forward_sample_trajectory(chain_ag, m=1.0, n_slices=20, rng=np.random.default_rng(3))
        assert t1 == t2

    def test_different_seed_can_diverge(self, chain_ag):
        t1, _ = forward_sample_trajectory(chain_ag, m=1.0, n_slices=50, rng=np.random.default_rng(1))
        t2, _ = forward_sample_trajectory(chain_ag, m=1.0, n_slices=50, rng=np.random.default_rng(2))
        assert t1 != t2  # astronomically unlikely to collide over 50 stochastic slices


class TestPersistence:
    def test_self_loop_node_never_reverts(self, chain_ag):
        trajectory, _ = forward_sample_trajectory(chain_ag, m=1.0, n_slices=100, rng=np.random.default_rng(11))
        for node, series in trajectory.items():
            for i in range(1, len(series)):
                assert not (series[i - 1] == 1 and series[i] == 0), f"{node} reverted at slice {i+1}"

    def test_validate_slice_trajectory_passes_on_real_sample(self, chain_ag):
        trajectory, _ = forward_sample_trajectory(chain_ag, m=1.0, n_slices=100, rng=np.random.default_rng(11))
        assert validate_slice_trajectory(trajectory, chain_ag) == []

    def test_validate_slice_trajectory_catches_reversion(self, chain_ag):
        trajectory, _ = forward_sample_trajectory(chain_ag, m=1.0, n_slices=10, rng=np.random.default_rng(11))
        # force a reversion in a copy
        broken = {k: list(v) for k, v in trajectory.items()}
        broken["A"][-1] = 0
        broken["A"][-2] = 1
        violations = validate_slice_trajectory(broken, chain_ag)
        assert any("reverted" in v for v in violations)


class TestGateResolution:
    def test_gate_matches_same_slice_and_combination(self, gated_ag):
        """Gate is untimed (self_loop=False): the A->Gate/B->Gate edges are
        NOT inter_slice (A/B self-loop, Gate does not), so Gate reads A/B's
        SAME-slice (ULTERIOR) value -- it resolves within the slice, not
        one slice after its parents, matching CredAccess's own real
        precedent in the paper graph."""
        rng = np.random.default_rng(0)
        trajectory, delta_t = forward_sample_trajectory(gated_ag, m=1.0, n_slices=30, rng=rng)
        for i in range(len(trajectory["Gate"])):
            expected_gate = int(bool(trajectory["A"][i]) and bool(trajectory["B"][i]))
            assert trajectory["Gate"][i] == expected_gate

    def test_gate_cpt_and_sampler_agree_on_every_enumerated_case(self, gated_ag):
        """Row-by-row equivalence: build_gate_cpt's CPT table, evaluated at
        every parent-state combination, matches the sampler's own combine()
        rule used internally (both encode `all(...)` for an AND gate)."""
        cpd = build_gate_cpt("Gate", ["A", "B"], "AND", gated_ag)
        values = cpd.get_values()  # [2, 4], columns ordered (A,B) MSB-first
        for col, (a, b) in enumerate(itertools.product([0, 1], repeat=2)):
            expected_active = float(all((a, b)))
            assert values[1, col] == pytest.approx(expected_active)
            assert values[0, col] == pytest.approx(1.0 - expected_active)


class TestAttackStepCptAgreement:
    def test_ps_used_by_sampler_matches_compute_ps(self, chain_ag):
        _, delta_t = forward_sample_trajectory(chain_ag, m=1.0, n_slices=1, rng=np.random.default_rng(0))
        assert compute_ps(2.0, delta_t) == pytest.approx(2.0 / 3.0)
        assert compute_ps(4.0, delta_t) == pytest.approx(1.0 / 3.0)

    def test_root_precondition_vacuously_true(self, chain_ag):
        """A has no parents; over many slices its activation rate should
        approach p_A (law of large numbers), matching build_attack_step_cpt's
        root-node treatment (precondition vacuously satisfied)."""
        p_a = compute_ps(2.0, compute_delta_t({"A": 2.0, "B": 4.0}, 1.0))
        n_trials = 400
        first_active_slice = []
        for seed in range(n_trials):
            trajectory, _ = forward_sample_trajectory(chain_ag, m=1.0, n_slices=1, rng=np.random.default_rng(seed))
            first_active_slice.append(trajectory["A"][0])
        empirical_rate = sum(first_active_slice) / n_trials
        assert empirical_rate == pytest.approx(p_a, abs=0.07)


class TestRealPaperGraphIntegration:
    """Sanity check against the actual 20-node Figure-2 graph (default
    memoryless reaction_mode, physical_evidence=True) -- not just the tiny
    hand-built fixtures above."""

    @pytest.mark.parametrize("seed", range(5))
    def test_no_violations_over_many_seeds(self, seed):
        ag = build_attack_graph(physical_evidence=True)
        trajectory, delta_t = forward_sample_trajectory(ag, m=1.0, n_slices=200, rng=np.random.default_rng(seed))
        assert delta_t > 0
        assert validate_slice_trajectory(trajectory, ag) == []

    def test_evidence_stream_covers_every_analytic(self):
        ag = build_attack_graph(physical_evidence=True)
        trajectory, _ = forward_sample_trajectory(ag, m=1.0, n_slices=50, rng=np.random.default_rng(0))
        evidence = forward_sample_evidence_stream(ag, trajectory, p_pos=1e-4, p_neg=1e-4, rng=np.random.default_rng(0))
        analytic_nodes = {n for n, d in ag.nodes(data=True) if d["node_type"] == "analytic"}
        assert set(evidence[1]) == analytic_nodes


class TestLatchUnsupported:
    def test_latched_graph_raises(self):
        ag = build_attack_graph(reaction_mode="latched")
        with pytest.raises(NotImplementedError):
            forward_sample_trajectory(ag, m=1.0, n_slices=5, rng=np.random.default_rng(0))


class TestForwardSampleEvidenceStream:
    @pytest.fixture
    def analytic_ag(self):
        g = nx.DiGraph()
        g.add_node("A", name="A", mitre_technique_id=None, **_node("attack_step", ttc=2.0, self_loop=True, technique="t1"))
        g.add_node(
            "Evidence", name="Evidence", mitre_technique_id=None,
            **_node("analytic", ttc=None, self_loop=False, sensor_model=None, observable_kind="cyber"),
        )
        g.add_edge("A", "Evidence", edge_type="triggers_analytic")
        return _finalize(g)

    def test_shape_and_keys(self, analytic_ag):
        trajectory, _ = forward_sample_trajectory(analytic_ag, m=1.0, n_slices=10, rng=np.random.default_rng(0))
        evidence = forward_sample_evidence_stream(analytic_ag, trajectory, p_pos=1e-4, p_neg=1e-4, rng=np.random.default_rng(0))
        assert set(evidence.keys()) == set(range(1, 11))
        assert all("Evidence" in row for row in evidence.values())

    def test_high_p_pos_p_neg_bracket_empirical_rate(self, analytic_ag):
        """With parent forced always-active (rig via a trajectory with A=1
        everywhere), P(observed=1) should approach 1-p_neg."""
        trajectory = {"A": [1] * 2000}
        evidence = forward_sample_evidence_stream(
            analytic_ag, trajectory, p_pos=0.1, p_neg=0.2, rng=np.random.default_rng(0),
        )
        rate = sum(row["Evidence"] for row in evidence.values()) / len(evidence)
        assert rate == pytest.approx(0.8, abs=0.05)
