"""Tests for src/eval/lead_time.py, written before experiments/exp04 runs it.

CLAUDE.md: "a post-hoc metric definition is a reviewable flaw." These tests
pin the metric's behavior against hand-constructed cases BEFORE any real
posterior trajectory is scored against it.
"""

from __future__ import annotations

import pytest

from src.eval.lead_time import (
    DetectionOutcome,
    count_upward_crossings,
    evaluate_run,
    first_crossing,
    first_instability,
    summarize,
    sweep_thresholds,
)

DELTA_T = 166.13 / 600


class TestFirstCrossing:
    def test_uses_greater_equal_so_theta_one_is_attainable(self):
        assert first_crossing([0.0, 0.5, 1.0], 1.0) == 3

    def test_returns_first_slice_not_last(self):
        assert first_crossing([0.0, 0.6, 0.7, 0.9], 0.5) == 2

    def test_none_when_never_crossed(self):
        assert first_crossing([0.0, 0.3, 0.4], 0.5) is None

    def test_one_based_indexing(self):
        assert first_crossing([1.0], 0.5) == 1


class TestCountUpwardCrossings:
    def test_single_crossing(self):
        assert count_upward_crossings([0.0, 0.3, 0.6, 0.8], 0.5) == 1

    def test_multiple_crossings(self):
        # below, above, below, above -> 2 upward crossings
        assert count_upward_crossings([0.1, 0.9, 0.1, 0.9], 0.5) == 2

    def test_zero_when_never_crossed(self):
        assert count_upward_crossings([0.1, 0.2, 0.3], 0.5) == 0

    def test_starts_above_threshold(self):
        assert count_upward_crossings([0.9, 0.9, 0.1], 0.5) == 1


class TestEvaluateRun:
    def test_detected_before_requires_strictly_earlier_detection(self):
        posterior = [0.0, 0.9, 0.9, 0.9]
        unstable = [False, False, False, True]
        result = evaluate_run(posterior, unstable, threshold=0.5, delta_t=DELTA_T)
        assert result.outcome is DetectionOutcome.DETECTED_BEFORE
        assert result.t_detect_slice == 2
        assert result.t_instability_slice == 4
        assert result.lead_time_slices == 2
        assert result.lead_time_units == pytest.approx(2 * DELTA_T)

    def test_tied_crossing_is_detected_after_not_before(self):
        """lead == 0 -- simultaneous detection is not a warning.

        This is the modal case given the twin's zero-duration precursor
        (LAB_NOTEBOOK.md 2026-08-01, M2), so getting this backwards would
        silently inflate the DETECTED_BEFORE count.
        """
        posterior = [0.0, 0.0, 0.9]
        unstable = [False, False, True]
        result = evaluate_run(posterior, unstable, threshold=0.5, delta_t=DELTA_T)
        assert result.outcome is DetectionOutcome.DETECTED_AFTER
        assert result.lead_time_slices == 0

    def test_detected_after_when_instability_precedes_detection(self):
        posterior = [0.0, 0.2, 0.6]
        unstable = [False, True, True]
        result = evaluate_run(posterior, unstable, threshold=0.5, delta_t=DELTA_T)
        assert result.outcome is DetectionOutcome.DETECTED_AFTER
        assert result.t_detect_slice == 3
        assert result.t_instability_slice == 2
        assert result.lead_time_slices == -1

    def test_missed_when_instability_but_never_crosses(self):
        posterior = [0.0, 0.1, 0.2]
        unstable = [False, False, True]
        result = evaluate_run(posterior, unstable, threshold=0.9, delta_t=DELTA_T)
        assert result.outcome is DetectionOutcome.MISSED
        assert result.lead_time_slices is None
        assert result.lead_time_units is None

    def test_false_alarm_when_crosses_but_never_unstable(self):
        posterior = [0.0, 0.9, 0.9]
        unstable = [False, False, False]
        result = evaluate_run(posterior, unstable, threshold=0.5, delta_t=DELTA_T)
        assert result.outcome is DetectionOutcome.FALSE_ALARM
        assert result.lead_time_slices is None

    def test_no_event_when_neither_happens(self):
        posterior = [0.0, 0.1, 0.2]
        unstable = [False, False, False]
        result = evaluate_run(posterior, unstable, threshold=0.9, delta_t=DELTA_T)
        assert result.outcome is DetectionOutcome.NO_EVENT
        assert result.lead_time_slices is None

    def test_multiple_crossings_use_first_but_are_counted(self):
        posterior = [0.9, 0.1, 0.9, 0.1]
        unstable = [False, False, False, True]
        result = evaluate_run(posterior, unstable, threshold=0.5, delta_t=DELTA_T)
        assert result.t_detect_slice == 1  # the FIRST crossing, not the last
        assert result.n_upward_crossings == 2
        assert result.recrossed_after_first is True

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            evaluate_run([0.1, 0.2], [False], threshold=0.5, delta_t=DELTA_T)


class TestSweepThresholds:
    def test_sweeps_every_threshold(self):
        posterior = [0.0, 0.6, 0.95]
        unstable = [False, False, True]
        results = sweep_thresholds(posterior, unstable, [0.5, 0.9, 0.99], DELTA_T)
        assert [r.threshold for r in results] == [0.5, 0.9, 0.99]
        assert results[0].outcome is DetectionOutcome.DETECTED_BEFORE  # crosses at 2
        assert results[2].outcome is DetectionOutcome.MISSED  # 0.95 < 0.99, never crosses


class TestSummarize:
    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            summarize([])

    def test_rejects_mixed_thresholds(self):
        a = evaluate_run([0.9], [True], threshold=0.5, delta_t=DELTA_T)
        b = evaluate_run([0.9], [True], threshold=0.9, delta_t=DELTA_T)
        with pytest.raises(ValueError):
            summarize([a, b])

    def test_n_lead_defined_excludes_missed_and_false_alarm(self):
        """The field that keeps a mean lead from being quoted without its
        denominator. MISSED/FALSE_ALARM/NO_EVENT runs must not silently
        contribute a lead of 0 -- they must be absent from the lead stats,
        and n_lead_defined must say how many runs that leaves.
        """
        before = evaluate_run([0.0, 0.9], [False, True], threshold=0.5, delta_t=DELTA_T)
        missed = evaluate_run([0.0, 0.1], [False, True], threshold=0.5, delta_t=DELTA_T)
        false_alarm = evaluate_run([0.9, 0.9], [False, False], threshold=0.5, delta_t=DELTA_T)

        with_missed = summarize([before, missed, false_alarm])
        without_missed = summarize([before])

        assert with_missed.n_runs == 3
        assert with_missed.n_lead_defined == 1
        assert without_missed.n_lead_defined == 1
        # Lead statistics themselves are identical -- MISSED/FALSE_ALARM runs
        # contribute nothing to them, only to the denominators/counts.
        assert with_missed.lead_median_slices == without_missed.lead_median_slices
        assert with_missed.lead_mean_slices == without_missed.lead_mean_slices

    def test_detection_rate_none_when_no_before_after_or_missed_runs(self):
        """Only FALSE_ALARM/NO_EVENT present -> nothing to compute a
        detection rate FROM, so it must be None, not a spurious 0.0/0."""
        false_alarm = evaluate_run([0.9], [False], threshold=0.5, delta_t=DELTA_T)
        no_event = evaluate_run([0.1], [False], threshold=0.5, delta_t=DELTA_T)
        summary = summarize([false_alarm, no_event])
        assert summary.detection_rate is None
        assert summary.false_alarm_rate == pytest.approx(0.5)

    def test_false_alarm_rate_none_when_no_false_alarm_or_no_event_runs(self):
        """Only BEFORE/AFTER/MISSED present -> nothing to compute a
        false-alarm rate FROM, so it must be None."""
        before = evaluate_run([0.0, 0.9], [False, True], threshold=0.5, delta_t=DELTA_T)
        missed = evaluate_run([0.0, 0.1], [False, True], threshold=0.5, delta_t=DELTA_T)
        summary = summarize([before, missed])
        assert summary.false_alarm_rate is None
        assert summary.detection_rate == pytest.approx(0.5)

    def test_lead_stats_none_when_no_run_has_a_defined_lead(self):
        only_no_event = evaluate_run([0.1], [False], threshold=0.9, delta_t=DELTA_T)
        summary = summarize([only_no_event])
        assert summary.n_lead_defined == 0
        assert summary.lead_median_slices is None
        assert summary.lead_mean_slices is None

    def test_detection_and_false_alarm_rates_hand_computed(self):
        results = [
            evaluate_run([0.0, 0.9], [False, True], threshold=0.5, delta_t=DELTA_T),  # BEFORE
            evaluate_run([0.9, 0.9], [True, True], threshold=0.5, delta_t=DELTA_T),  # AFTER (tie)
            evaluate_run([0.0, 0.1], [False, True], threshold=0.5, delta_t=DELTA_T),  # MISSED
            evaluate_run([0.9, 0.9], [False, False], threshold=0.5, delta_t=DELTA_T),  # FALSE_ALARM
            evaluate_run([0.1, 0.1], [False, False], threshold=0.5, delta_t=DELTA_T),  # NO_EVENT
        ]
        summary = summarize(results)
        # detected=(BEFORE+AFTER)=2, denom=(detected+MISSED)=3
        assert summary.detection_rate == pytest.approx(2 / 3)
        # false_alarm=1, denom=(FALSE_ALARM+NO_EVENT)=2
        assert summary.false_alarm_rate == pytest.approx(1 / 2)
        assert summary.n_lead_defined == 2

    def test_median_p10_p90_hand_computed(self):
        # Five DETECTED_BEFORE runs, each detecting at slice 1 (posterior[0]
        # = 0.9) with instability at slice lead+1 -> lead_time_slices == lead.
        results = []
        for lead in [1, 2, 3, 4, 5]:
            posterior = [0.9] + [0.0] * lead
            unstable = [False] * lead + [True]
            results.append(evaluate_run(posterior, unstable, threshold=0.5, delta_t=DELTA_T))
        summary = summarize(results)
        leads = sorted(r.lead_time_slices for r in results)
        assert leads == [1, 2, 3, 4, 5]
        assert summary.lead_median_slices == 3.0
        assert summary.lead_mean_slices == pytest.approx(3.0)
