"""Tests for src/baselines/common.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.baselines.common import (
    ENGINEERED_AGGS,
    ENGINEERED_TYPES,
    BaselineResult,
    TrialResult,
    aggregate_node_type,
    flatten_engineered_features,
    hyperparameter_search,
)


class TestAggregateNodeType:
    def test_hand_computed_two_node_mean_and_max(self):
        # [S=2, N=2, F=1]: slice0 = [1.0, 3.0], slice1 = [5.0, 2.0]
        x = torch.tensor([[[1.0], [3.0]], [[5.0], [2.0]]])
        out = aggregate_node_type(x, aggs=("mean", "max"))
        assert out.shape == (2, 2)  # F=1 * 2 aggs
        torch.testing.assert_close(out[:, 0], torch.tensor([2.0, 3.5]))  # mean
        torch.testing.assert_close(out[:, 1], torch.tensor([3.0, 5.0]))  # max

    def test_single_node_mean_equals_max_no_duplication(self):
        """N=1 (host): mean and max coincide -- must emit ONE copy of F
        columns, not two identical copies (that would silently double-weight
        host relative to every other type in a flat concatenation)."""
        x = torch.tensor([[[7.0, 9.0]]])  # [S=1, N=1, F=2]
        out = aggregate_node_type(x, aggs=("mean", "max"))
        assert out.shape == (1, 2)
        torch.testing.assert_close(out, torch.tensor([[7.0, 9.0]]))

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError):
            aggregate_node_type(torch.randn(3, 4))

    def test_rejects_unknown_agg(self):
        with pytest.raises(ValueError):
            aggregate_node_type(torch.randn(2, 3, 1), aggs=("bogus",))


class TestFlattenEngineeredFeatures:
    def test_shape_and_names_length_match(self):
        S = 5
        dynamic_x = {
            "bus": torch.randn(S, 33, 4),
            "IED": torch.randn(S, 2, 10),
            "host": torch.randn(S, 1, 6),
            "DER": torch.randn(S, 2, 3),
        }
        flat, names = flatten_engineered_features(dynamic_x)
        assert flat.shape == (S, 40)
        assert len(names) == 40
        assert len(set(names)) == 40  # every column name unique

    def test_column_names_reflect_source_and_aggregation(self):
        S = 3
        dynamic_x = {
            "bus": torch.randn(S, 33, 4), "IED": torch.randn(S, 2, 10),
            "host": torch.randn(S, 1, 6), "DER": torch.randn(S, 2, 3),
        }
        _, names = flatten_engineered_features(dynamic_x)
        assert "bus.vm_pu.mean" in names
        assert "bus.vm_pu.max" in names
        assert "host.dispatch_phase.value" in names  # single-node -> "value", not mean/max
        assert not any(n.startswith("host.") and n.endswith(".max") for n in names)

    def test_excludes_static_only_types_by_construction(self):
        """Only ENGINEERED_TYPES are consumed; a caller passing zero-width
        line/transformer/RTU/relay tensors would break aggregate_node_type
        (F=0), so flatten_engineered_features must never touch them."""
        assert set(ENGINEERED_TYPES) == {"bus", "IED", "host", "DER"}

    def test_no_nan_or_inf_on_real_feature_output(self):
        """Integration smoke: run against src.perception.features' real
        output shape contract (dtype/finiteness), without regenerating a
        full twin scenario -- a synthetic but correctly-shaped tensor set
        exercises the same code path."""
        S = 10
        dynamic_x = {
            "bus": torch.rand(S, 33, 4), "IED": torch.rand(S, 2, 10),
            "host": torch.rand(S, 1, 6), "DER": torch.rand(S, 2, 3),
        }
        flat, _ = flatten_engineered_features(dynamic_x)
        assert torch.isfinite(flat).all()


class TestHyperparameterSearch:
    def test_runs_every_trial_and_returns_all(self):
        configs = [{"x": i} for i in range(5)]

        def score(config, rng):
            return float(config["x"])

        trials, best = hyperparameter_search(configs, score, rng=np.random.default_rng(0))
        assert len(trials) == 5
        assert [t.trial_id for t in trials] == [0, 1, 2, 3, 4]
        assert best == {"x": 4}

    def test_best_selected_by_max_val_auc_pr(self):
        configs = [{"score": 0.3}, {"score": 0.9}, {"score": 0.1}]

        def score(config, rng):
            return config["score"]

        trials, best = hyperparameter_search(configs, score, rng=np.random.default_rng(1))
        assert best == {"score": 0.9}
        assert max(t.val_auc_pr for t in trials) == pytest.approx(0.9)

    def test_rng_threaded_not_reseeded_per_trial(self):
        """A single rng advances across trials rather than being reset --
        verified by checking successive draws differ (a re-seeded generator
        would repeat the same first draw every trial)."""
        seen = []

        def score(config, rng):
            seen.append(rng.random())
            return 0.0

        hyperparameter_search([{}] * 3, score, rng=np.random.default_rng(2))
        assert len(set(seen)) == 3  # all three draws distinct

    def test_rejects_empty_configs(self):
        with pytest.raises(ValueError):
            hyperparameter_search([], lambda c, r: 0.0, rng=np.random.default_rng(0))

    def test_trial_result_and_baseline_result_are_frozen(self):
        t = TrialResult(trial_id=0, config={}, val_auc_pr=0.5)
        with pytest.raises(Exception):
            t.trial_id = 1  # type: ignore[misc]
        r = BaselineResult(name="x", run_id=0, score=np.zeros(3))
        with pytest.raises(Exception):
            r.name = "y"  # type: ignore[misc]
