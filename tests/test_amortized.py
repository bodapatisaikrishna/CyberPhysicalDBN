"""Tests for src/parameterization/amortized.py (Session 8, claim C2)."""

from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

from src.attack_graph.graph import technique_table3_ttc
from src.parameterization.amortized import (
    AmortizedTrainConfig,
    ContextNormalizer,
    TTCAmortizedModel,
    apply_ttc_predictions,
    fit_context_normalizer,
    fit_ttc_amortized_model,
    known_techniques,
    predict_ttc_for_graph,
)


def _fixture_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(
        "A", name="A", node_type="attack_step", ttc=2.0, self_loop=True, gate=None,
        mitre_technique="Man-in-the-middle", asset_context=0.5, defensive_posture=1.0, attacker_capability=1.0,
    )
    g.add_node(
        "B", name="B", node_type="attack_step", ttc=15.0, self_loop=True, gate=None,
        mitre_technique="Spoof Reporting Message", asset_context=0.2, defensive_posture=2.0, attacker_capability=0.5,
    )
    g.add_node(
        "Goal", name="Goal", node_type="goal", ttc=None, self_loop=False, gate="OR",
        mitre_technique=None,
    )
    g.add_edge("A", "Goal", edge_type="precondition")
    g.add_edge("B", "Goal", edge_type="precondition")
    # Mirrors build_attack_graph's tail: self-loop edges for persistent
    # nodes + derived inter_slice on every edge.
    for name, attrs in g.nodes(data=True):
        if attrs["self_loop"]:
            g.add_edge(name, name, edge_type="precondition")
    for u, v, data in g.edges(data=True):
        data["inter_slice"] = g.nodes[u]["self_loop"] and g.nodes[v]["self_loop"]
    return g


def _tiny_rows(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    techniques = list(known_techniques())
    rows = []
    for i in range(n):
        t = techniques[i % len(techniques)]
        table3 = technique_table3_ttc()[t]
        defense = float(rng.uniform(0.5, 4.0))
        capability = float(rng.uniform(0.25, 4.0))
        rows.append({
            "technique": t,
            "asset_context": float(rng.uniform(0.0, 1.0)),
            "defensive_posture": defense,
            "attacker_capability": capability,
            "true_ttc": table3 * defense / capability,
        })
    return pd.DataFrame(rows)


class TestKnownTechniques:
    def test_matches_technique_table3_ttc_keys(self):
        assert set(known_techniques()) == set(technique_table3_ttc())

    def test_exactly_eight(self):
        assert len(known_techniques()) == 8

    def test_sorted_deterministic(self):
        assert known_techniques() == tuple(sorted(known_techniques()))


class TestContextNormalizer:
    def test_fit_on_train_zero_means_after_transform(self):
        rows = _tiny_rows(50, seed=0)
        normalizer = fit_context_normalizer(rows)
        transformed = normalizer.transform(rows[["asset_context", "defensive_posture", "attacker_capability"]].to_numpy())
        assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-6)

    def test_transform_rejects_wrong_width(self):
        normalizer = ContextNormalizer(mean=np.zeros(3), std=np.ones(3))
        with pytest.raises(ValueError):
            normalizer.transform(np.zeros((5, 2)))

    def test_fit_ttc_amortized_model_never_refits_on_val(self):
        """Structural leak guard: fit_ttc_amortized_model's signature has
        no parameter shaped like a combined/test dataset -- only
        train_rows/val_rows/techniques/config, mirroring
        tests/test_lstm_ae.py's disjointness-by-signature pattern."""
        sig = inspect.signature(fit_ttc_amortized_model)
        assert set(sig.parameters) == {"train_rows", "val_rows", "techniques", "config"}
        for name in sig.parameters:
            assert "test" not in name


class TestApplyTtcPredictions:
    def test_correct_predictions_applied(self):
        ag = _fixture_graph()
        mutated = apply_ttc_predictions(ag, {"A": 1.5, "B": 20.0})
        assert mutated.nodes["A"]["ttc"] == pytest.approx(1.5)
        assert mutated.nodes["B"]["ttc"] == pytest.approx(20.0)
        # original untouched
        assert ag.nodes["A"]["ttc"] == pytest.approx(2.0)

    def test_raises_on_missing_node(self):
        ag = _fixture_graph()
        with pytest.raises(ValueError, match="missing"):
            apply_ttc_predictions(ag, {"A": 1.5})

    def test_raises_on_extra_node(self):
        ag = _fixture_graph()
        with pytest.raises(ValueError, match="extra"):
            apply_ttc_predictions(ag, {"A": 1.5, "B": 20.0, "Ghost": 1.0})

    def test_goal_node_never_needs_a_prediction(self):
        """Goal has ttc=None -- not in the timed-node set at all."""
        ag = _fixture_graph()
        mutated = apply_ttc_predictions(ag, {"A": 1.5, "B": 20.0})
        assert mutated.nodes["Goal"]["ttc"] is None


class TestPredictTtcForGraphLeakBarrier:
    @pytest.fixture
    def trained(self):
        train_rows = _tiny_rows(80, seed=1)
        val_rows = _tiny_rows(20, seed=2)
        config = AmortizedTrainConfig(
            embedding_dim=4, hidden_dim=8, learning_rate=1e-2, weight_decay=1e-4,
            n_epochs=20, early_stopping_patience_epochs=20, grad_clip_norm=1.0, seed=0,
        )
        model, normalizer, epoch_rows = fit_ttc_amortized_model(train_rows, val_rows, known_techniques(), config)
        return model, normalizer, epoch_rows

    def test_prediction_signature_never_reads_ttc(self):
        sig = inspect.signature(predict_ttc_for_graph)
        assert set(sig.parameters) == {"ag", "model", "normalizer", "techniques"}

    def test_predictions_invariant_to_ttc_perturbation(self, trained):
        """Load-bearing: perturbing every node's ttc x1000, leaving
        technique/context untouched, must leave predict_ttc_for_graph's
        output EXACTLY unchanged -- proves the prediction path structurally
        cannot read ttc."""
        model, normalizer, _ = trained
        ag = _fixture_graph()
        baseline = predict_ttc_for_graph(ag, model, normalizer, known_techniques())

        perturbed = ag.copy()
        for node, data in perturbed.nodes(data=True):
            if data.get("ttc") is not None:
                data["ttc"] = data["ttc"] * 1000.0
        again = predict_ttc_for_graph(perturbed, model, normalizer, known_techniques())

        assert baseline == again

    def test_returns_only_timed_nodes(self, trained):
        model, normalizer, _ = trained
        ag = _fixture_graph()
        preds = predict_ttc_for_graph(ag, model, normalizer, known_techniques())
        assert set(preds) == {"A", "B"}

    def test_predictions_are_positive(self, trained):
        model, normalizer, _ = trained
        ag = _fixture_graph()
        preds = predict_ttc_for_graph(ag, model, normalizer, known_techniques())
        assert all(v > 0 for v in preds.values())

    def test_unknown_technique_raises(self, trained):
        model, normalizer, _ = trained
        ag = _fixture_graph()
        ag.nodes["A"]["mitre_technique"] = "Not A Real Technique"
        with pytest.raises(ValueError, match="unknown technique"):
            predict_ttc_for_graph(ag, model, normalizer, known_techniques())


class TestEndToEndFit:
    def test_train_loss_decreases_and_val_mae_finite(self):
        train_rows = _tiny_rows(100, seed=5)
        val_rows = _tiny_rows(20, seed=6)
        config = AmortizedTrainConfig(
            embedding_dim=4, hidden_dim=8, learning_rate=1e-2, weight_decay=1e-4,
            n_epochs=100, early_stopping_patience_epochs=100, grad_clip_norm=1.0, seed=0,
        )
        model, normalizer, rows = fit_ttc_amortized_model(train_rows, val_rows, known_techniques(), config)
        assert rows[-1]["train_loss"] < rows[0]["train_loss"]
        assert np.isfinite(rows[-1]["val_mae_log_ttc"])

    def test_apply_predictions_then_compiles_valid_cpds(self):
        """End-to-end sanity: predictions -> apply_ttc_predictions ->
        attach_cpds (existing, unmodified) succeeds without error."""
        from src.dbn.parameterization import attach_cpds
        from src.dbn.compiler import compile_to_2tbn

        ag = _fixture_graph()
        train_rows = _tiny_rows(60, seed=3)
        val_rows = _tiny_rows(15, seed=4)
        config = AmortizedTrainConfig(
            embedding_dim=4, hidden_dim=8, learning_rate=1e-2, weight_decay=1e-4,
            n_epochs=30, early_stopping_patience_epochs=30, grad_clip_norm=1.0, seed=0,
        )
        model, normalizer, _ = fit_ttc_amortized_model(train_rows, val_rows, known_techniques(), config)
        preds = predict_ttc_for_graph(ag, model, normalizer, known_techniques())
        mutated = apply_ttc_predictions(ag, preds)

        dbn = compile_to_2tbn(mutated)
        dbn = attach_cpds(dbn, mutated, m=1.0, p_pos=1e-4, p_neg=1e-4)
        assert len(dbn.get_cpds()) == 3
