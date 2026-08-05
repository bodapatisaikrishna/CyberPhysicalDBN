"""Gradient-boosted trees on engineered features (CLAUDE.md's `src/baselines/`:
"GBM"). `sklearn.ensemble.HistGradientBoostingClassifier` -- already pinned
(scikit-learn==1.9.0), has a native `class_weight` param (verified via
`inspect.signature` before this module was written), so no manual
oversampling/reweighting hack is needed for the imbalanced `grid_unstable`
label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.baselines.common import flatten_engineered_features
from src.twin.runner import DiscreteTrace


class _ScenarioLike(Protocol):
    """Duck-typed so this module has no import-time dependency on
    `experiments/exp06_baselines.py`'s scenario-bundle type. Anything with
    these three attributes works -- exercised directly with a synthetic
    stand-in in `tests/test_gbm.py`."""

    run_id: int
    discrete: DiscreteTrace
    dynamic_x: dict  # str -> torch.Tensor, build_dynamic_features' RAW output


@dataclass(frozen=True)
class GBMTrialConfig:
    max_iter: int
    max_depth: int | None
    learning_rate: float
    l2_regularization: float
    max_leaf_nodes: int


def build_flat_table(
    scenarios: Sequence[_ScenarioLike],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per `(scenario, slice)`. `y` is `grid_unstable` (measured,
    never `ground_truth["UnstablePS"]`, per this session's stated rule).

    Returns `(X [n_rows, 40], y [n_rows], run_ids [n_rows])`.
    """
    X_parts, y_parts, run_id_parts = [], [], []
    for scenario in scenarios:
        flat, _ = flatten_engineered_features(scenario.dynamic_x)
        X_parts.append(flat.numpy())
        y = np.array([int(r.grid_unstable) for r in scenario.discrete.records], dtype=np.int64)
        if len(y) != flat.shape[0]:
            raise ValueError(
                f"run {scenario.run_id}: {len(y)} labels vs {flat.shape[0]} feature "
                "rows -- discrete.records and dynamic_x disagree on slice count"
            )
        y_parts.append(y)
        run_id_parts.append(np.full(len(y), scenario.run_id, dtype=np.int64))
    return np.concatenate(X_parts), np.concatenate(y_parts), np.concatenate(run_id_parts)


def train_gbm(
    config: GBMTrialConfig, X_train: np.ndarray, y_train: np.ndarray, *, random_state: int
) -> HistGradientBoostingClassifier:
    """`class_weight="balanced"`: `grid_unstable` is expected to be a
    low-base-rate label (matching the DBN's own base-rate finding in
    Session 5), and `HistGradientBoostingClassifier` supports
    `class_weight` natively."""
    model = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        l2_regularization=config.l2_regularization,
        max_leaf_nodes=config.max_leaf_nodes,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def score_trajectory(model: HistGradientBoostingClassifier, X_scenario: np.ndarray) -> np.ndarray:
    """`model.predict_proba(X_scenario)[:, 1]` -- a genuine fitted
    probability, unlike the rule-based baseline's raw count or the AE's
    post-hoc sigmoid transform."""
    return model.predict_proba(X_scenario)[:, 1]
