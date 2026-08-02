"""Detection lead time (claim C1's headline metric).

Written and tested BEFORE experiments/exp04_closed_loop_c1.py runs, per the
task instruction: a post-hoc metric definition is a reviewable flaw.

    t_detect      = first slice where P(UnstablePS) crosses threshold theta
    t_instability = first slice where the twin's measured GridState is unstable
    lead_time     = t_instability - t_detect

Five explicit outcomes, not three, because a bare mean over "the runs where
both happened" silently discards the runs that would tell you the metric is
failing (a missed detection, a standing false alarm) -- exactly the
survivorship bias that would make a null result look better than it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class DetectionOutcome(Enum):
    DETECTED_BEFORE = "detected_before"  # lead > 0, strictly
    DETECTED_AFTER = "detected_after"  # lead <= 0, including a tied crossing
    MISSED = "missed"  # instability occurred, posterior never crossed
    FALSE_ALARM = "false_alarm"  # crossed, instability never occurred
    NO_EVENT = "no_event"  # neither happened


@dataclass(frozen=True)
class DetectionResult:
    threshold: float
    t_detect_slice: int | None
    t_instability_slice: int | None
    lead_time_slices: int | None  # None unless outcome in {BEFORE, AFTER}
    lead_time_units: float | None
    outcome: DetectionOutcome
    n_upward_crossings: int
    recrossed_after_first: bool


@dataclass(frozen=True)
class LeadTimeSummary:
    threshold: float
    n_runs: int
    n_detected_before: int
    n_detected_after: int
    n_missed: int
    n_false_alarm: int
    n_no_event: int
    detection_rate: float | None  # None on a zero denominator, never 0.0
    false_alarm_rate: float | None
    n_lead_defined: int  # REQUIRED: the denominator a mean lead is silent about
    lead_median_slices: float | None
    lead_p10_slices: float | None
    lead_p90_slices: float | None
    lead_mean_slices: float | None


def first_crossing(posterior: Sequence[float], threshold: float) -> int | None:
    """First 1-based slice index where posterior[i] >= threshold.

    `>=`, not `>`, so theta=1.0 is attainable rather than structurally
    unreachable.
    """
    for i, p in enumerate(posterior, start=1):
        if p >= threshold:
            return i
    return None


def count_upward_crossings(posterior: Sequence[float], threshold: float) -> int:
    """Number of slices where posterior crosses from below threshold to >= threshold."""
    count = 0
    below = True
    for p in posterior:
        above = p >= threshold
        if above and below:
            count += 1
        below = not above
    return count


def first_instability(unstable_flags: Sequence[bool]) -> int | None:
    for i, flag in enumerate(unstable_flags, start=1):
        if flag:
            return i
    return None


def evaluate_run(
    posterior: Sequence[float],
    unstable_flags: Sequence[bool],
    threshold: float,
    delta_t: float,
) -> DetectionResult:
    """Classify one scenario run at one threshold.

    `lead == 0` (a tied crossing) is DETECTED_AFTER, not DETECTED_BEFORE:
    BEFORE requires strictly positive lead, because detecting simultaneously
    with the consequence is not a warning. Given the twin's zero-duration
    precursor window (LAB_NOTEBOOK.md 2026-08-01, M2), ties are the modal
    case here, not a rare corner case, so this choice materially shapes the
    reported detection_rate.
    """
    if len(posterior) != len(unstable_flags):
        raise ValueError(
            f"posterior ({len(posterior)}) and unstable_flags "
            f"({len(unstable_flags)}) must be the same length"
        )

    t_detect = first_crossing(posterior, threshold)
    t_instability = first_instability(unstable_flags)
    n_crossings = count_upward_crossings(posterior, threshold)
    recrossed = n_crossings > 1

    if t_instability is None and t_detect is None:
        outcome = DetectionOutcome.NO_EVENT
    elif t_instability is None:
        outcome = DetectionOutcome.FALSE_ALARM
    elif t_detect is None:
        outcome = DetectionOutcome.MISSED
    elif t_detect < t_instability:
        outcome = DetectionOutcome.DETECTED_BEFORE
    else:
        outcome = DetectionOutcome.DETECTED_AFTER

    lead_slices: int | None = None
    lead_units: float | None = None
    if outcome in (DetectionOutcome.DETECTED_BEFORE, DetectionOutcome.DETECTED_AFTER):
        lead_slices = t_instability - t_detect
        lead_units = lead_slices * delta_t

    return DetectionResult(
        threshold=threshold,
        t_detect_slice=t_detect,
        t_instability_slice=t_instability,
        lead_time_slices=lead_slices,
        lead_time_units=lead_units,
        outcome=outcome,
        n_upward_crossings=n_crossings,
        recrossed_after_first=recrossed,
    )


def sweep_thresholds(
    posterior: Sequence[float],
    unstable_flags: Sequence[bool],
    thresholds: Sequence[float],
    delta_t: float,
) -> list[DetectionResult]:
    return [evaluate_run(posterior, unstable_flags, t, delta_t) for t in thresholds]


def summarize(results: Sequence[DetectionResult]) -> LeadTimeSummary:
    """Aggregate DetectionResults for ONE threshold across many runs.

    Caller is responsible for passing results that all share the same
    `threshold` (not enforced here to keep this a pure aggregation step); a
    mismatched set raises.
    """
    if not results:
        raise ValueError("results must not be empty")
    thresholds = {r.threshold for r in results}
    if len(thresholds) != 1:
        raise ValueError(f"summarize expects one threshold, got {thresholds}")
    threshold = next(iter(thresholds))

    by_outcome: dict[DetectionOutcome, list[DetectionResult]] = {o: [] for o in DetectionOutcome}
    for r in results:
        by_outcome[r.outcome].append(r)

    n_before = len(by_outcome[DetectionOutcome.DETECTED_BEFORE])
    n_after = len(by_outcome[DetectionOutcome.DETECTED_AFTER])
    n_missed = len(by_outcome[DetectionOutcome.MISSED])
    n_false_alarm = len(by_outcome[DetectionOutcome.FALSE_ALARM])
    n_no_event = len(by_outcome[DetectionOutcome.NO_EVENT])

    detected = n_before + n_after
    detection_denom = detected + n_missed
    detection_rate = detected / detection_denom if detection_denom > 0 else None

    alarm_denom = n_false_alarm + n_no_event
    false_alarm_rate = n_false_alarm / alarm_denom if alarm_denom > 0 else None

    leads = [
        r.lead_time_slices
        for r in by_outcome[DetectionOutcome.DETECTED_BEFORE]
        + by_outcome[DetectionOutcome.DETECTED_AFTER]
    ]
    n_lead_defined = len(leads)

    def _pct(values: list[int], q: float) -> float | None:
        if not values:
            return None
        s = sorted(values)
        idx = min(int(q * (len(s) - 1)), len(s) - 1)
        return float(s[idx])

    return LeadTimeSummary(
        threshold=threshold,
        n_runs=len(results),
        n_detected_before=n_before,
        n_detected_after=n_after,
        n_missed=n_missed,
        n_false_alarm=n_false_alarm,
        n_no_event=n_no_event,
        detection_rate=detection_rate,
        false_alarm_rate=false_alarm_rate,
        n_lead_defined=n_lead_defined,
        lead_median_slices=_pct(leads, 0.5),
        lead_p10_slices=_pct(leads, 0.1),
        lead_p90_slices=_pct(leads, 0.9),
        lead_mean_slices=(sum(leads) / len(leads)) if leads else None,
    )
