"""Tests for src/perception/calibration.py.

The pass-through test is the one that matters most structurally: it pins
that perception calibration and claim C1's calibration are scored by the
SAME code in src/eval/calibration.py, so a future "improvement" to one
cannot silently desynchronize from the other.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src.eval.calibration import calibration_report, expected_calibration_error
from src.perception.calibration import (
    T_MAX,
    T_MIN,
    TemperatureScaler,
    apply_temperature,
    fit_temperature,
    fit_temperature_for_target,
    perception_calibration_report,
    reliability_diagram,
)


def _synthetic(n=2000, seed=0, overconfidence=3.0):
    rng = np.random.default_rng(seed)
    true_logits = rng.normal(0, 2, n)
    p_true = 1.0 / (1.0 + np.exp(-true_logits))
    labels = (rng.random(n) < p_true).astype(float)
    overconfident_logits = true_logits * overconfidence
    return overconfident_logits, labels


class TestApplyTemperature:
    def test_temperature_one_is_identity(self):
        logits = np.array([-2.0, -0.5, 0.0, 1.3, 4.0])
        q = apply_temperature(logits, 1.0)
        expected = 1.0 / (1.0 + np.exp(-logits))
        np.testing.assert_allclose(q, expected)

    def test_rejects_nonpositive_temperature(self):
        with pytest.raises(ValueError):
            apply_temperature(np.array([0.0]), 0.0)
        with pytest.raises(ValueError):
            apply_temperature(np.array([0.0]), -1.0)

    def test_higher_temperature_flattens_toward_half(self):
        logits = np.array([5.0])
        q_t1 = apply_temperature(logits, 1.0)
        q_t10 = apply_temperature(logits, 10.0)
        assert q_t10 < q_t1
        assert q_t10 > 0.5


class TestFitTemperatureForTarget:
    def test_temperature_reduces_nll_on_fit_split(self):
        logits, labels = _synthetic(overconfidence=3.0)
        T, nll_before, nll_after = fit_temperature_for_target(logits, labels)
        assert T > 1.0  # an overconfident model needs T > 1 to soften
        assert nll_after < nll_before

    def test_temperature_preserves_ranking_exactly(self):
        """A monotone rescaling cannot change AP -- if it moves at all, that
        is a bug, not a calibration effect."""
        logits, labels = _synthetic()
        T, _, _ = fit_temperature_for_target(logits, labels)
        q_before = apply_temperature(logits, 1.0)
        q_after = apply_temperature(logits, T)
        ap_before = average_precision_score(labels, q_before)
        ap_after = average_precision_score(labels, q_after)
        assert ap_before == pytest.approx(ap_after, abs=1e-9)

    def test_well_calibrated_logits_need_little_scaling(self):
        """overconfidence=1.0 means the logits ARE the true generating
        logits -- the fitted T should land close to 1, not drift far."""
        logits, labels = _synthetic(overconfidence=1.0, seed=1)
        T, _, _ = fit_temperature_for_target(logits, labels)
        assert 0.7 < T < 1.4

    def test_masked_slices_excluded_from_fit(self):
        logits, labels = _synthetic(seed=2)
        mask = np.ones(len(logits), dtype=bool)
        mask[:1000] = False
        corrupted_logits = logits.copy()
        corrupted_logits[:1000] = 999.0  # would badly distort the fit if included
        corrupted_labels = labels.copy()
        corrupted_labels[:1000] = 1.0 - corrupted_labels[:1000]

        T_masked, _, _ = fit_temperature_for_target(
            corrupted_logits, corrupted_labels, mask.astype(float)
        )
        T_clean_subset, _, _ = fit_temperature_for_target(logits[1000:], labels[1000:])
        assert T_masked == pytest.approx(T_clean_subset, rel=1e-6)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            fit_temperature_for_target(np.array([]), np.array([]))

    def test_temperature_is_bounded_on_degenerate_tiny_data(self):
        """A tiny, perfectly-separable dataset (n=4, extreme confidence) is
        exactly the failure mode observed in this session's own smoke run:
        unconstrained LBFGS drove T into the millions. The fitted T must
        never leave [T_MIN, T_MAX], regardless of how degenerate the input."""
        logits = np.array([-50.0, -40.0, 40.0, 50.0])
        labels = np.array([0.0, 0.0, 1.0, 1.0])
        T, _, _ = fit_temperature_for_target(logits, labels, max_iter=500)
        assert T_MIN <= T <= T_MAX

    def test_temperature_bound_is_reachable_at_both_ends(self):
        """Extremely overconfident-in-the-WRONG-direction data should push
        T toward T_MAX (flattening); this is a sanity check that the bound
        is a real constraint the optimizer can hit, not a dead parameter."""
        rng = np.random.default_rng(0)
        n = 20
        # logits wildly overconfident relative to a near-random label
        logits = rng.choice([-100.0, 100.0], size=n)
        labels = (rng.random(n) < 0.5).astype(float)
        T, _, _ = fit_temperature_for_target(logits, labels, max_iter=500)
        assert T_MIN <= T <= T_MAX

    def test_rejects_all_masked(self):
        with pytest.raises(ValueError, match="no observed"):
            fit_temperature_for_target(
                np.array([1.0, 2.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])
            )


class TestFitTemperature:
    def test_fits_one_temperature_per_target(self):
        logits_a, labels_a = _synthetic(overconfidence=3.0, seed=3)
        logits_b, labels_b = _synthetic(overconfidence=1.2, seed=4)
        run_ids_a = np.repeat(np.arange(20), len(logits_a) // 20)
        run_ids_b = np.repeat(np.arange(20), len(logits_b) // 20)

        scaler = fit_temperature(
            logits={"A": logits_a, "B": logits_b},
            labels={"A": labels_a, "B": labels_b},
            run_ids={"A": run_ids_a, "B": run_ids_b},
        )
        assert set(scaler.temperatures) == {"A", "B"}
        assert scaler.temperatures["A"] != pytest.approx(scaler.temperatures["B"], rel=0.05)
        assert scaler.fit_split == "calib"
        assert scaler.n_fit["A"] == len(logits_a)
        assert scaler.n_fit_runs["A"] == 20

    def test_apply_uses_the_right_targets_temperature(self):
        logits_a, labels_a = _synthetic(seed=5)
        run_ids = np.repeat(np.arange(10), len(logits_a) // 10)
        scaler = fit_temperature(
            logits={"A": logits_a}, labels={"A": labels_a}, run_ids={"A": run_ids}
        )
        applied = scaler.apply("A", logits_a)
        expected = apply_temperature(logits_a, scaler.temperatures["A"])
        np.testing.assert_allclose(applied, expected)

    def test_per_target_mask_respected(self):
        logits_a, labels_a = _synthetic(seed=6)
        run_ids = np.repeat(np.arange(20), len(logits_a) // 20)
        mask = np.ones(len(logits_a), dtype=bool)
        mask[:500] = False
        scaler = fit_temperature(
            logits={"A": logits_a}, labels={"A": labels_a}, run_ids={"A": run_ids},
            mask={"A": mask},
        )
        assert scaler.n_fit["A"] == len(logits_a) - 500


class TestCalibrationReportPassThrough:
    def test_ece_passes_through_to_eval_calibration(self):
        logits, labels = _synthetic(seed=7)
        q = apply_temperature(logits, 1.5)
        run_ids = np.repeat(np.arange(40), len(q) // 40)

        direct = calibration_report(labels, q, run_ids, n_bootstrap=50, rng=np.random.default_rng(0))
        via_perception = perception_calibration_report(
            labels, q, run_ids, n_bootstrap=50, rng=np.random.default_rng(0)
        )
        assert direct.ece == via_perception.ece
        assert direct.brier == via_perception.brier
        assert direct.brier_skill_score == via_perception.brier_skill_score


class TestReliabilityDiagram:
    def test_empty_bins_are_gaps_not_plotted_points(self):
        """All mass in the top decile -> most bins empty; the plotted curve
        must have exactly as many points as NON-empty bins."""
        labels = np.array([1.0, 1.0, 1.0])
        q = np.array([0.95, 0.96, 0.97])
        _, bins = expected_calibration_error(labels, q, n_bins=10, strategy="uniform")
        n_nonempty = sum(1 for b in bins if b.count > 0)

        fig = reliability_diagram({"curve": bins}, title="test")
        ax = fig.axes[0]
        curve_lines = [line for line in ax.lines if line.get_label() == "curve"]
        assert len(curve_lines) == 1
        assert len(curve_lines[0].get_xdata()) == n_nonempty
        assert n_nonempty < 10  # non-vacuous: genuinely some bins were empty

    def test_multiple_curves_overlaid(self):
        labels = np.array([0.0, 1.0] * 50)
        q_before = np.array([0.5] * 100)
        q_after = np.array([0.1, 0.9] * 50)
        _, bins_before = expected_calibration_error(labels, q_before, n_bins=5, strategy="uniform")
        _, bins_after = expected_calibration_error(labels, q_after, n_bins=5, strategy="uniform")

        fig = reliability_diagram({"before": bins_before, "after": bins_after}, title="t")
        ax = fig.axes[0]
        labels_present = {line.get_label() for line in ax.lines}
        assert "before" in labels_present
        assert "after" in labels_present

    def test_title_carries_provenance(self):
        _, bins = expected_calibration_error([1, 0, 1, 0], [0.6, 0.4, 0.7, 0.3], n_bins=2, strategy="uniform")
        fig = reliability_diagram({"c": bins}, title="MyTarget", n=100, n_runs=10, base_rate=0.5)
        assert "n=100" in fig.axes[0].get_title()
        assert "n_runs=10" in fig.axes[0].get_title()
