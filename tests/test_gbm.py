"""Tests for src/baselines/gbm.py."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from src.baselines.gbm import GBMTrialConfig, build_flat_table, score_trajectory, train_gbm


def _fake_scenario(run_id: int, n_slices: int, unstable_pattern: list[int]):
    """A minimal duck-typed stand-in for exp06's scenario bundle: only
    `.run_id`, `.discrete.records[i].grid_unstable`, `.dynamic_x` are
    touched by build_flat_table."""
    records = [SimpleNamespace(grid_unstable=bool(v)) for v in unstable_pattern]
    discrete = SimpleNamespace(records=records)
    dynamic_x = {
        "bus": torch.rand(n_slices, 33, 4),
        "IED": torch.rand(n_slices, 2, 10),
        "host": torch.rand(n_slices, 1, 6),
        "DER": torch.rand(n_slices, 2, 3),
    }
    return SimpleNamespace(run_id=run_id, discrete=discrete, dynamic_x=dynamic_x)


class TestBuildFlatTable:
    def test_shape_and_label_alignment(self):
        s0 = _fake_scenario(0, 4, [0, 0, 1, 1])
        s1 = _fake_scenario(1, 3, [0, 1, 0])
        X, y, run_ids = build_flat_table([s0, s1])
        assert X.shape == (7, 40)
        np.testing.assert_array_equal(y, [0, 0, 1, 1, 0, 1, 0])
        np.testing.assert_array_equal(run_ids, [0, 0, 0, 0, 1, 1, 1])

    def test_rejects_slice_count_mismatch(self):
        s = _fake_scenario(0, 4, [0, 0, 1])  # 3 labels, 4 feature rows
        with pytest.raises(ValueError, match="disagree"):
            build_flat_table([s])

    def test_single_scenario(self):
        s = _fake_scenario(5, 2, [1, 1])
        X, y, run_ids = build_flat_table([s])
        assert X.shape == (2, 40)
        np.testing.assert_array_equal(y, [1, 1])
        np.testing.assert_array_equal(run_ids, [5, 5])


class TestTrainGBM:
    def _data(self):
        rng = np.random.default_rng(0)
        X = rng.random((200, 40))
        y = (rng.random(200) < 0.15).astype(np.int64)  # imbalanced
        return X, y

    def test_class_weight_balanced_is_passed(self):
        config = GBMTrialConfig(
            max_iter=20, max_depth=3, learning_rate=0.1, l2_regularization=0.0, max_leaf_nodes=15
        )
        X, y = self._data()
        model = train_gbm(config, X, y, random_state=0)
        assert model.class_weight == "balanced"

    def test_returns_fitted_classifier(self):
        config = GBMTrialConfig(
            max_iter=20, max_depth=None, learning_rate=0.1, l2_regularization=0.0, max_leaf_nodes=31
        )
        X, y = self._data()
        model = train_gbm(config, X, y, random_state=0)
        assert isinstance(model, HistGradientBoostingClassifier)
        preds = model.predict_proba(X)
        assert preds.shape == (200, 2)

    def test_deterministic_given_random_state(self):
        config = GBMTrialConfig(
            max_iter=10, max_depth=3, learning_rate=0.1, l2_regularization=0.1, max_leaf_nodes=15
        )
        X, y = self._data()
        m1 = train_gbm(config, X, y, random_state=42)
        m2 = train_gbm(config, X, y, random_state=42)
        np.testing.assert_array_equal(m1.predict_proba(X), m2.predict_proba(X))

    def test_class_weight_param_exists_on_installed_sklearn(self):
        """Guards the plan's own load-bearing verification: if a future
        sklearn upgrade drops this param, this test fails loudly instead of
        train_gbm silently falling back to an unweighted fit."""
        sig = inspect.signature(HistGradientBoostingClassifier.__init__)
        assert "class_weight" in sig.parameters


class TestScoreTrajectory:
    def test_output_is_probability_column(self):
        rng = np.random.default_rng(1)
        X_train = rng.random((100, 40))
        y_train = (rng.random(100) < 0.3).astype(np.int64)
        config = GBMTrialConfig(
            max_iter=20, max_depth=3, learning_rate=0.1, l2_regularization=0.0, max_leaf_nodes=15
        )
        model = train_gbm(config, X_train, y_train, random_state=0)
        scores = score_trajectory(model, rng.random((10, 40)))
        assert scores.shape == (10,)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()
