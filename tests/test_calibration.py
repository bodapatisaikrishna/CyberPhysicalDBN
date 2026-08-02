"""Tests for src/eval/calibration.py.

Written before experiments/exp04 uses it, matching the lead-time module's
protocol: this is the code that decides whether a reported calibration
improvement is real.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.eval.calibration import (
    brier_score,
    brier_skill_score,
    calibration_report,
    expected_calibration_error,
)


class TestExpectedCalibrationError:
    def test_hand_computed_two_bin(self):
        """4 samples, 2 uniform bins [0,0.5) and [0.5,1.0].

        bin0: y_prob in [0,0.5) -> {0.1, 0.2}, y_true {0, 0}. mean_pred=0.15,
              empirical=0.0, |gap|=0.15, weight=2/4.
        bin1: y_prob in [0.5,1.0] -> {0.8, 0.9}, y_true {1, 1}. mean_pred=0.85,
              empirical=1.0, |gap|=0.15, weight=2/4.
        ECE = 0.5*0.15 + 0.5*0.15 = 0.15
        """
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        ece, bins = expected_calibration_error(y_true, y_prob, n_bins=2, strategy="uniform")
        assert ece == pytest.approx(0.15)
        assert len(bins) == 2
        assert bins[0].count == 2
        assert bins[0].mean_predicted == pytest.approx(0.15)
        assert bins[0].empirical_rate == pytest.approx(0.0)
        assert bins[1].mean_predicted == pytest.approx(0.85)
        assert bins[1].empirical_rate == pytest.approx(1.0)

    def test_perfect_calibration_gives_zero(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.0, 1.0, 0.0, 1.0]
        ece, _ = expected_calibration_error(y_true, y_prob, n_bins=5, strategy="uniform")
        assert ece == pytest.approx(0.0, abs=1e-12)

    def test_empty_bins_carry_none_not_imputed(self):
        # All mass in [0.9,1.0] with n_bins=5 -> 4 empty bins.
        y_true = [1, 1, 1]
        y_prob = [0.95, 0.96, 0.97]
        _, bins = expected_calibration_error(y_true, y_prob, n_bins=5, strategy="uniform")
        empty = [b for b in bins if b.count == 0]
        nonempty = [b for b in bins if b.count > 0]
        assert len(empty) == 4
        assert len(nonempty) == 1
        for b in empty:
            assert b.mean_predicted is None
            assert b.empirical_rate is None

    def test_last_bin_is_closed_on_both_ends(self):
        """A probability of exactly 1.0 must land in the top bin, not be lost."""
        y_true = [1]
        y_prob = [1.0]
        _, bins = expected_calibration_error(y_true, y_prob, n_bins=4, strategy="uniform")
        assert bins[-1].count == 1

    def test_agrees_with_sklearn_calibration_curve(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, 300)
        y_prob = rng.random(300)

        _, bins = expected_calibration_error(y_true, y_prob, n_bins=5, strategy="uniform")
        sk_frac_pos, sk_mean_pred = calibration_curve(y_true, y_prob, n_bins=5, strategy="uniform")

        nonempty = [b for b in bins if b.count > 0]
        assert len(nonempty) == len(sk_frac_pos)
        for mine, sk_rate, sk_pred in zip(nonempty, sk_frac_pos, sk_mean_pred):
            assert mine.empirical_rate == pytest.approx(sk_rate)
            assert mine.mean_predicted == pytest.approx(sk_pred)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            expected_calibration_error([1, 0], [0.5], n_bins=5, strategy="uniform")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            expected_calibration_error([], [], n_bins=5, strategy="uniform")


class TestBrierScore:
    def test_agrees_with_sklearn(self):
        rng = np.random.default_rng(4)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.random(200)
        assert brier_score(y_true, y_prob) == pytest.approx(
            brier_score_loss(y_true, y_prob)
        )

    def test_perfect_predictions_score_zero(self):
        assert brier_score([0, 1, 0, 1], [0.0, 1.0, 0.0, 1.0]) == pytest.approx(0.0)


class TestBrierSkillScore:
    def test_base_rate_constant_predictor_scores_zero(self):
        """The pitfall, demonstrated: a predictor outputting the base rate at
        every sample is by definition the reference, so BSS == 0 for it --
        even though a naive ECE on this same predictor is also ~0, which is
        exactly the degenerate case BSS exists to catch.
        """
        rng = np.random.default_rng(5)
        y_true = rng.integers(0, 2, 500)
        base_rate = float(y_true.mean())
        y_prob = [base_rate] * len(y_true)

        bss = brier_skill_score(y_true, y_prob)
        ece, _ = expected_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform")

        assert bss == pytest.approx(0.0, abs=1e-9)
        assert ece < 0.05  # "well calibrated" by the naive metric alone

    def test_perfect_predictor_has_positive_skill(self):
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_prob = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        assert brier_skill_score(y_true, y_prob) == pytest.approx(1.0)

    def test_degenerate_constant_ground_truth_returns_zero_not_nan(self):
        assert brier_skill_score([1, 1, 1], [0.5, 0.9, 0.99]) == pytest.approx(0.0)


class TestCalibrationReport:
    def test_sweeps_full_grid_and_declares_primary(self):
        rng = np.random.default_rng(6)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.random(200)
        run_ids = np.repeat(np.arange(20), 10)

        report = calibration_report(
            y_true, y_prob, run_ids,
            n_bins_sweep=(5, 10, 15, 20), strategies=("uniform", "quantile"),
            primary_n_bins=10, primary_strategy="uniform", n_bootstrap=50,
            rng=np.random.default_rng(0),
        )
        assert set(report.ece.keys()) == {
            (nb, s) for nb in (5, 10, 15, 20) for s in ("uniform", "quantile")
        }
        assert report.ece[(10, "uniform")] == pytest.approx(
            expected_calibration_error(y_true, y_prob, n_bins=10, strategy="uniform")[0]
        )
        assert report.n == 200
        assert report.n_runs == 20

    def test_bootstrap_ci_is_run_level_not_slice_level(self):
        """Resampling by run, not slice: a report built from few highly
        autocorrelated runs should show visible CI width even with many
        pooled slices, because n_runs (not n) governs the bootstrap.
        """
        rng = np.random.default_rng(7)
        n_runs = 4
        slices_per_run = 200
        y_true, y_prob, run_ids = [], [], []
        for r in range(n_runs):
            base = rng.random()  # each run has a systematically different bias
            y_true.extend(rng.integers(0, 2, slices_per_run).tolist())
            y_prob.extend(np.clip(rng.normal(base, 0.05, slices_per_run), 0, 1).tolist())
            run_ids.extend([r] * slices_per_run)

        report = calibration_report(
            y_true, y_prob, run_ids, n_bootstrap=200, rng=np.random.default_rng(1)
        )
        assert report.n == n_runs * slices_per_run
        assert report.n_runs == n_runs
        lo, hi = report.ece_primary_ci95
        assert lo >= 0.0
        assert hi > lo  # a real interval, not a degenerate point

    def test_required_fields_present_so_n_cannot_be_quoted_without_n_runs(self):
        y_true = [0, 1] * 10
        y_prob = [0.1, 0.9] * 10
        run_ids = list(range(20))
        report = calibration_report(y_true, y_prob, run_ids, n_bootstrap=20)
        assert hasattr(report, "n") and hasattr(report, "n_runs")

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            calibration_report([0, 1], [0.5], [0, 1])
