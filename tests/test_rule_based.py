"""Tests for src/baselines/rule_based.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.baselines.rule_based import RuleConfig, score_trajectory

NAMES = ("A", "B", "C")


class TestRuleConfig:
    def test_rejects_nonpositive_window(self):
        with pytest.raises(ValueError):
            RuleConfig(window_slices=0)
        with pytest.raises(ValueError):
            RuleConfig(window_slices=-1)

    def test_accepts_window_one(self):
        RuleConfig(window_slices=1)


class TestScoreTrajectory:
    def test_hand_computed_window_one(self):
        # slice 1: A fires. slice 2: nothing. slice 3: B and C fire.
        stream = {
            1: {"A": 1, "B": 0, "C": 0},
            2: {"A": 0, "B": 0, "C": 0},
            3: {"A": 0, "B": 1, "C": 1},
        }
        scores = score_trajectory(stream, NAMES, 3, RuleConfig(window_slices=1))
        np.testing.assert_allclose(scores, [1 / 3, 0.0, 2 / 3])

    def test_hand_computed_window_two_carries_forward_one_slice(self):
        stream = {
            1: {"A": 1, "B": 0, "C": 0},
            2: {"A": 0, "B": 0, "C": 0},
            3: {"A": 0, "B": 1, "C": 1},
        }
        scores = score_trajectory(stream, NAMES, 3, RuleConfig(window_slices=2))
        # slice1: window=[1] -> A -> 1/3
        # slice2: window=[1,2] -> A (from slice1) -> 1/3
        # slice3: window=[2,3] -> B,C -> 2/3
        np.testing.assert_allclose(scores, [1 / 3, 1 / 3, 2 / 3])

    def test_persistent_dense_bit_not_recounted_across_window(self):
        """A cyber analytic that is DENSE (1 at every slice from its trigger
        onward) must not inflate the score just because the window is wide
        -- it is one distinct signature, firing the whole time, not many."""
        stream = {s: {"A": 1, "B": 0, "C": 0} for s in range(1, 11)}
        scores = score_trajectory(stream, NAMES, 10, RuleConfig(window_slices=10))
        np.testing.assert_allclose(scores, [1 / 3] * 10)

    def test_sparse_omission_does_not_override_earlier_firing_in_window(self):
        """A physical observable's key can be OMITTED in some slices
        (unobserved) -- an omission must not erase a ==1 seen earlier in
        the same window."""
        stream = {
            1: {"A": 1, "B": 0, "C": 0},
            2: {"B": 0, "C": 0},  # A omitted this slice (unobserved)
        }
        scores = score_trajectory(stream, NAMES, 2, RuleConfig(window_slices=2))
        assert scores[1] == pytest.approx(1 / 3)  # A still counts at slice 2

    def test_larger_window_never_decreases_score(self):
        """Monotonicity: a wider trailing window can only add firings it
        might have missed, never remove one."""
        rng = np.random.default_rng(0)
        stream = {
            s: {n: int(rng.random() < 0.1) for n in NAMES} for s in range(1, 51)
        }
        small = score_trajectory(stream, NAMES, 50, RuleConfig(window_slices=1))
        large = score_trajectory(stream, NAMES, 50, RuleConfig(window_slices=20))
        assert (large >= small - 1e-12).all()

    def test_score_in_unit_interval(self):
        rng = np.random.default_rng(1)
        stream = {
            s: {n: int(rng.random() < 0.3) for n in NAMES} for s in range(1, 21)
        }
        scores = score_trajectory(stream, NAMES, 20, RuleConfig(window_slices=5))
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    def test_output_length_matches_n_slices(self):
        stream = {s: {"A": 0, "B": 0, "C": 0} for s in range(1, 8)}
        scores = score_trajectory(stream, NAMES, 7, RuleConfig(window_slices=3))
        assert scores.shape == (7,)

    def test_rejects_unknown_observable_name(self):
        stream = {1: {"A": 1}}
        with pytest.raises(ValueError, match="Z"):
            score_trajectory(stream, ("A", "Z"), 1, RuleConfig(window_slices=1))

    def test_rejects_empty_observable_names(self):
        with pytest.raises(ValueError):
            score_trajectory({1: {"A": 1}}, (), 1, RuleConfig(window_slices=1))

    def test_rejects_nonpositive_n_slices(self):
        with pytest.raises(ValueError):
            score_trajectory({}, NAMES, 0, RuleConfig(window_slices=1))

    def test_zero_denominator_impossible_given_nonempty_names(self):
        """Non-vacuous: at least one all-zero and one all-firing case both
        stay within [0,1] and are exact."""
        stream = {1: {"A": 0, "B": 0, "C": 0}}
        scores = score_trajectory(stream, NAMES, 1, RuleConfig(window_slices=1))
        assert scores[0] == pytest.approx(0.0)
        stream_all = {1: {"A": 1, "B": 1, "C": 1}}
        scores_all = score_trajectory(stream_all, NAMES, 1, RuleConfig(window_slices=1))
        assert scores_all[0] == pytest.approx(1.0)
