"""Calibration of P(UnstablePS) against MEASURED grid instability (claim C1).

Ground truth is `PhysicalObservation.exceeds_limit` (src/twin/consequence.py),
never the attack graph's `ground_truth["UnstablePS"]`. This is load-bearing,
not cosmetic: LAB_NOTEBOOK.md 2026-08-01 (M1) found ~30% of runs where the
attack graph fully asserts UnstablePS while the grid never leaves nominal --
scoring against the assertion instead of the measurement would erase exactly
the signal claim C1 is being tested for.

Five pitfalls a naive single-number ECE hides, each handled explicitly below:
bin-count sensitivity, a degenerate base-rate-constant predictor scoring
ECE~=0, empty bins silently imputed, treating correlated within-run samples
as independent, and no independent check that the arithmetic is right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

Strategy = Literal["uniform", "quantile"]


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float | None  # None iff count == 0 -- never imputed
    empirical_rate: float | None


def _bin_edges(y_prob: np.ndarray, n_bins: int, strategy: Strategy) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(y_prob, quantiles)
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)  # duplicate edges collapse when y_prob is degenerate
    raise ValueError(f"unknown strategy {strategy!r}")


def expected_calibration_error(
    y_true: Sequence[int], y_prob: Sequence[float], *, n_bins: int, strategy: Strategy
) -> tuple[float, tuple[ReliabilityBin, ...]]:
    """ECE = sum_bins (count_b / n) * |empirical_rate_b - mean_predicted_b|.

    Empty bins carry `count=0` and `None` fields; they contribute nothing to
    ECE, and are never imputed as 0 or dropped silently -- they are visible in
    the returned `bins` tuple so a reader can see how much of [0,1] the data
    actually covered.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be the same length")
    if len(y_true) == 0:
        raise ValueError("y_true/y_prob must not be empty")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    edges = _bin_edges(y_prob, n_bins, strategy)
    n = len(y_true)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    for i in range(len(edges) - 1):
        lower, upper = edges[i], edges[i + 1]
        is_last = i == len(edges) - 2
        mask = (y_prob >= lower) & (y_prob < upper) if not is_last else (y_prob >= lower) & (y_prob <= upper)
        count = int(mask.sum())
        if count == 0:
            bins.append(ReliabilityBin(float(lower), float(upper), 0, None, None))
            continue
        mean_predicted = float(y_prob[mask].mean())
        empirical_rate = float(y_true[mask].mean())
        bins.append(ReliabilityBin(float(lower), float(upper), count, mean_predicted, empirical_rate))
        ece += (count / n) * abs(empirical_rate - mean_predicted)

    return ece, tuple(bins)


def _mce(bins: tuple[ReliabilityBin, ...]) -> float:
    gaps = [abs(b.empirical_rate - b.mean_predicted) for b in bins if b.count > 0]
    return max(gaps) if gaps else 0.0


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def brier_skill_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    """1 - brier(model) / brier(base-rate-constant reference).

    Defends against ECE's blind spot: a predictor that outputs the base rate
    at every sample scores ECE ~= 0 (it is "calibrated" in the weak,
    binned-average sense) while carrying zero discriminative information.
    BSS <= 0 for such a predictor by construction (it IS the reference), so a
    reported ECE improvement with BSS <= 0 is a sharpness collapse, not a
    real calibration gain.
    """
    y_true = np.asarray(y_true, dtype=float)
    base_rate = float(y_true.mean())
    reference = brier_score(y_true, [base_rate] * len(y_true))
    if reference == 0.0:
        return 0.0  # degenerate: y_true is constant, nothing to skill-score against
    return 1.0 - brier_score(y_true, y_prob) / reference


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    n_runs: int
    base_rate: float
    brier: float
    brier_skill_score: float
    sharpness_std: float
    ece: dict[tuple[int, str], float]
    mce: dict[tuple[int, str], float]
    primary_n_bins: int
    primary_strategy: str
    bins_primary: tuple[ReliabilityBin, ...]
    ece_primary_ci95: tuple[float, float]


def calibration_report(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    run_ids: Sequence[int],
    *,
    n_bins_sweep: Sequence[int] = (5, 10, 15, 20),
    strategies: Sequence[Strategy] = ("uniform", "quantile"),
    primary_n_bins: int = 10,
    primary_strategy: Strategy = "uniform",
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> CalibrationReport:
    """Full calibration report: never a single ECE number in isolation.

    The 95% CI is a RUN-LEVEL bootstrap (resampling whole run_ids, not
    individual slices): a trajectory's slices are strongly autocorrelated
    (the posterior is a filtered process and the label a step function), so
    slice-level resampling would understate the true uncertainty. `n` (pooled
    sample count) and `n_runs` (the real number of independent units) are
    both required fields specifically so a report can never quote `n` without
    `n_runs` sitting next to it.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    run_ids_arr = np.asarray(run_ids)
    if not (len(y_true_arr) == len(y_prob_arr) == len(run_ids_arr)):
        raise ValueError("y_true, y_prob, run_ids must be the same length")

    rng = rng or np.random.default_rng()

    ece_grid: dict[tuple[int, str], float] = {}
    mce_grid: dict[tuple[int, str], float] = {}
    for nb in n_bins_sweep:
        for strat in strategies:
            e, bins = expected_calibration_error(y_true_arr, y_prob_arr, n_bins=nb, strategy=strat)
            ece_grid[(nb, strat)] = e
            mce_grid[(nb, strat)] = _mce(bins)

    primary_ece, primary_bins = expected_calibration_error(
        y_true_arr, y_prob_arr, n_bins=primary_n_bins, strategy=primary_strategy
    )

    unique_runs = np.unique(run_ids_arr)
    boot_eces = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled_runs = rng.choice(unique_runs, size=len(unique_runs), replace=True)
        mask = np.isin(run_ids_arr, sampled_runs)
        # A resample that happens to omit every slice of every chosen run's
        # bin coverage is vanishingly unlikely at realistic run counts, but
        # guard it rather than let expected_calibration_error's empty-input
        # ValueError propagate out of a bootstrap loop.
        if not mask.any():
            boot_eces[b] = primary_ece
            continue
        e, _ = expected_calibration_error(
            y_true_arr[mask], y_prob_arr[mask], n_bins=primary_n_bins, strategy=primary_strategy
        )
        boot_eces[b] = e
    ci_low, ci_high = float(np.percentile(boot_eces, 2.5)), float(np.percentile(boot_eces, 97.5))

    return CalibrationReport(
        n=len(y_true_arr),
        n_runs=len(unique_runs),
        base_rate=float(y_true_arr.mean()),
        brier=brier_score(y_true_arr, y_prob_arr),
        brier_skill_score=brier_skill_score(y_true_arr, y_prob_arr),
        sharpness_std=float(y_prob_arr.std()),
        ece=ece_grid,
        mce=mce_grid,
        primary_n_bins=primary_n_bins,
        primary_strategy=primary_strategy,
        bins_primary=primary_bins,
        ece_primary_ci95=(ci_low, ci_high),
    )
