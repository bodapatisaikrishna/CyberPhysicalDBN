"""exp20: episode-end extension of the attack-step CPT and the Sherlock attack graph."""

import numpy as np
import pytest

from src.attack_graph.graph import build_attack_graph
from src.attack_graph.sherlock_graph import (ANALYTICS, GOAL, STEPS, build_sherlock_attack_graph, step_labels,
                                             step_transition_rates)
from src.dbn.compiler import compile_to_2tbn
from src.dbn.inference import DBNInference, InferenceConfig, _interface_nodes, exact_clustering
from src.dbn.parameterization import build_attack_step_cpt


def test_p_end_default_reproduces_persistence_exactly():
    ag = build_attack_graph()
    a = build_attack_step_cpt("MITM", ["CredAccess"], 0.3, ag)
    b = build_attack_step_cpt("MITM", ["CredAccess"], 0.3, ag, p_end=0.0)
    assert np.array_equal(a.get_values(), b.get_values())


def test_p_end_rows_hand_computed():
    ag = build_attack_graph()
    v = build_attack_step_cpt("MITM", ["CredAccess"], 0.3, ag, p_end=0.25).get_values()
    # columns ordered (CredAccess, MITM_prev) = 00, 01, 10, 11; row 1 = P(active)
    assert v[1].tolist() == pytest.approx([0.0, 0.0, 0.3, 0.75])
    assert np.allclose(v.sum(axis=0), 1.0)


def test_p_end_out_of_range_raises():
    with pytest.raises(ValueError):
        build_attack_step_cpt("MITM", ["CredAccess"], 0.3, build_attack_graph(), p_end=1.5)


def test_step_labels_follow_family_signatures():
    lab = step_labels([None, "industroyer", "drift-off", "control-and-freeze", "arp-spoof"])
    assert lab["RogueClient"].tolist() == [0, 1, 0, 0, 0]
    assert lab["ARPSpoof"].tolist() == [0, 0, 1, 1, 1]
    assert lab["UnauthCommand"].tolist() == [0, 1, 0, 1, 0]
    assert lab["SpoofReporting"].tolist() == [0, 0, 1, 1, 0]
    assert lab["BlockReporting"].tolist() == [0, 0, 0, 0, 1]
    with pytest.raises(ValueError):
        step_labels(["not-a-family"])


def test_transition_rates_hand_computed():
    # one sequence, industroyer active at slices 2-4 of 8
    seq = step_labels([None, None, "industroyer", "industroyer", "industroyer", None, None, None])
    r = step_transition_rates([seq])
    # RogueClient (root): inactive-at-(t-1) transitions = t-1 in {0,1,4..6}... eligible prev-inactive pairs:
    # (0,1),(1,2),(5,6),(6,7) -> 4 eligible, 1 onset; active-prev pairs (2,3),(3,4),(4,5) -> 1 end
    assert r["RogueClient"] == pytest.approx((1 / 4, 1 / 3))
    # UnauthCommand: precondition = RogueClient at t-1; eligible onsets: prev inactive & parent active at t-1
    # -> pair (2,3)? no: Unauth already active at 2. Eligible = none -> floored at 1/(0+1) = 1.0
    assert r["UnauthCommand"][0] == pytest.approx(1.0)


def test_sherlock_graph_compiles_and_runs():
    rates = {s: (1e-3, 1e-2) for s in STEPS}
    g = build_sherlock_attack_graph(rates, bin_s=2.0, analytic_error=(0.01, 0.01), source="test")
    assert sorted(_interface_nodes(g)) == sorted(STEPS)
    assert set(ANALYTICS) <= set(g.nodes) and g.nodes[GOAL]["gate"] == "OR"
    compile_to_2tbn(g)
    cfg = InferenceConfig(clustering=exact_clustering(_interface_nodes(g)), m=1.0, p_pos=0.01, p_neg=0.01,
                          delta_t_override=2.0)
    inf = DBNInference(g, cfg)
    b = inf.initial_belief()
    r = inf.step(b, {"ARPAnomaly": 1, "MeasureAnomaly": 1})
    r = inf.step(r.next_belief, {"ARPAnomaly": 1, "MeasureAnomaly": 1})
    assert r.marginals["ARPSpoof"] > 0.5
    assert 0.0 <= r.marginals[GOAL] <= 1.0
