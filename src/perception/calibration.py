"""Temperature scaling + reliability diagrams for the perception layer
(CLAUDE.md layer [1]).

REUSES `src/eval/calibration.py` (Session 4) for everything that module
already does correctly -- ECE with its bin sweep, Brier/BSS, the run-level
bootstrap CI, `ReliabilityBin`'s empty-bin-is-`None` discipline. This module
adds only what that one does not have: fitting a per-target temperature and
drawing the reliability-diagram plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import expit

from src.eval.calibration import CalibrationReport, ReliabilityBin, calibration_report

# Box constraint on the fitted temperature (log-space, projected LBFGS).
# With a small or heavily-imbalanced calib split (e.g. PhysLocalDER's
# base_rate ~= 0.003), the unconstrained BCE-with-logits surface is nearly
# monotone in log_T and LBFGS drives it toward +/-infinity -- observed
# directly in this session (fitted T in the millions on a 2-scenario smoke
# split). Never report a temperature that large: it is not a meaningful
# calibration, it is the optimizer running off the edge of an
# underdetermined fit. [T_MIN, T_MAX] bounds both directions generously
# (T=20 already flattens any confident logit close to 0.5) without
# constraining a genuine, well-determined fit on a real-sized split.
T_MIN = 0.05
T_MAX = 20.0

__all__ = [
    "TemperatureScaler",
    "apply_temperature",
    "fit_temperature_for_target",
    "fit_temperature",
    "perception_calibration_report",
    "reliability_diagram",
]


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled sigmoid, via scipy's stable `expit`.

    NOT `1/(1+exp(-z))`: dividing by a small temperature amplifies the logit
    (T_MIN=0.05 is a 20x amplification, and it is genuinely reachable -- exp11's
    real run hit exactly that bound on PhysLocalDER), so a confident logit of
    -40 becomes z=-800, where `np.exp(-z)` overflows even float64 and emits a
    RuntimeWarning.

    What this does and does not buy, stated precisely because the difference
    matters: `expit` is the numerically stable formulation and removes the
    spurious overflow warning, and it is exact over a much wider range of z.
    It does NOT make the extreme tail nonzero -- for z below about -745,
    float64 simply has no representable value between 0 and the true
    probability, so the result still saturates to exactly 0.0 (verified, not
    assumed). Callers that cannot tolerate an exact 0/1 must clip: the DBN
    soft-evidence path already does, via `SoftEvidenceConfig.eps`. The
    metric path (AUC-PR/ECE/Brier) tolerates it -- measured impact of the
    saturated-vs-exact difference is ~1e-7 on AUC-PR.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return expit(np.asarray(logits, dtype=float) / temperature)


@dataclass(frozen=True)
class TemperatureScaler:
    """One temperature PER TARGET (not shared): the four perception targets
    have wildly different base rates and directions of miscalibration, so a
    single shared T would average them into meaninglessness. `fit_split` is
    recorded in the artifact itself so it cannot be misremembered later --
    ECE computed with this scaler must be reported on `test`, never on
    `fit_split`, or the number is circular."""

    temperatures: Mapping[str, float]
    fit_split: str
    n_fit: Mapping[str, int]
    n_fit_runs: Mapping[str, int]
    fit_nll_before: Mapping[str, float]
    fit_nll_after: Mapping[str, float]

    def apply(self, target: str, logits: np.ndarray) -> np.ndarray:
        return apply_temperature(logits, self.temperatures[target])


def _masked_bce_with_logits(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def fit_temperature_for_target(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    max_iter: int = 200,
) -> tuple[float, float, float]:
    """`(temperature, nll_before, nll_after)`. `T = exp(log_T)` for
    positivity by construction (never clamped, never able to go negative);
    LBFGS minimizes masked BCE-with-logits w.r.t. `log_T` alone -- `logits`
    and `labels` are frozen constants, this is a 1-parameter fit.

    `mask` (`bool`-like, same shape as `labels`): `False` slices are excluded
    from the loss entirely -- never coerced to a label of 0, matching
    `evidence_stream`'s existing sparse-if-unobserved discipline for the two
    physical targets (a slice with no grid solve yet has no label to fit
    against).
    """
    if len(logits) == 0:
        raise ValueError("logits must not be empty")
    logits_t = torch.tensor(np.asarray(logits, dtype=np.float64))
    labels_t = torch.tensor(np.asarray(labels, dtype=np.float64))
    mask_t = (
        torch.ones_like(labels_t) if mask is None
        else torch.tensor(np.asarray(mask, dtype=np.float64))
    )
    if float(mask_t.sum()) == 0.0:
        raise ValueError("no observed (unmasked) labels to fit a temperature on")

    log_bounds = (float(np.log(T_MIN)), float(np.log(T_MAX)))
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    nll_before = float(_masked_bce_with_logits(logits_t, labels_t, mask_t).item())

    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        # Projected LBFGS: clamp BEFORE evaluating the loss, so a step that
        # would leave the box is pulled back onto its boundary rather than
        # the optimizer chasing an unconstrained monotone surface to
        # +/-infinity (the failure mode this bound exists to prevent).
        with torch.no_grad():
            log_T.clamp_(*log_bounds)
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = _masked_bce_with_logits(logits_t / T, labels_t, mask_t)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        log_T.clamp_(*log_bounds)
        T_final = float(torch.exp(log_T).item())
        nll_after = float(_masked_bce_with_logits(logits_t / torch.exp(log_T), labels_t, mask_t).item())
    return T_final, nll_before, nll_after


def fit_temperature(
    logits: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    run_ids: Mapping[str, np.ndarray],
    mask: Mapping[str, np.ndarray] | None = None,
    *,
    fit_split: str = "calib",
    max_iter: int = 200,
) -> TemperatureScaler:
    """Fit one `TemperatureScaler` covering every target in `logits`. Caller
    is responsible for passing the CALIB split only (this function has no
    way to enforce that from array data alone) -- `fit_split` records the
    caller's claim in the artifact for downstream auditing."""
    temps, n_fit, n_fit_runs, nll_before, nll_after = {}, {}, {}, {}, {}
    for target, target_logits in logits.items():
        target_mask = (
            np.asarray(mask[target], dtype=bool) if mask and target in mask
            else np.ones(len(target_logits), dtype=bool)
        )
        T, before, after = fit_temperature_for_target(
            target_logits, labels[target], target_mask.astype(float), max_iter=max_iter
        )
        if T <= T_MIN * 1.01 or T >= T_MAX * 0.99:
            print(
                f"  WARNING: temperature for {target!r} hit its bound (T={T:.4g}, "
                f"n_fit={int(target_mask.sum())}) -- likely an underdetermined fit "
                "(too few calib slices or too extreme a base rate), not a "
                "meaningful calibration. Reported as-is, not silently trusted."
            )
        temps[target] = T
        n_fit[target] = int(target_mask.sum())
        n_fit_runs[target] = int(len(np.unique(np.asarray(run_ids[target])[target_mask])))
        nll_before[target] = before
        nll_after[target] = after
    return TemperatureScaler(
        temperatures=temps, fit_split=fit_split, n_fit=n_fit, n_fit_runs=n_fit_runs,
        fit_nll_before=nll_before, fit_nll_after=nll_after,
    )


def perception_calibration_report(
    y_true: Sequence[int], y_prob: Sequence[float], run_ids: Sequence[int], **kwargs
) -> CalibrationReport:
    """Thin pass-through to `src.eval.calibration.calibration_report`, so
    perception's calibration and claim C1's are scored by literally the same
    code -- a future improvement to one cannot silently desynchronize from
    the other (`test_ece_passes_through_to_eval_calibration` pins equality
    against a direct call)."""
    return calibration_report(y_true, y_prob, run_ids, **kwargs)


def reliability_diagram(
    bins_by_label: Mapping[str, Sequence[ReliabilityBin]],
    *,
    title: str,
    n: int | None = None,
    n_runs: int | None = None,
    base_rate: float | None = None,
    ax: "plt.Axes | None" = None,
) -> "plt.Figure":
    """One or more overlaid reliability curves (typically `{"before": ...,
    "after": ...}`), each drawn only over its NON-EMPTY bins -- `ReliabilityBin`
    with `count == 0` carries `mean_predicted=None`, `empirical_rate=None`
    (src/eval/calibration.py refuses to impute them); this function respects
    that and draws a GAP there, never a point plotted at (x, 0).
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
    else:
        fig = ax.figure

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
    for label, bins in bins_by_label.items():
        xs = [b.mean_predicted for b in bins if b.count > 0]
        ys = [b.empirical_rate for b in bins if b.count > 0]
        if xs:
            ax.plot(xs, ys, marker="o", markersize=4, label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("empirical positive rate")
    subtitle_bits = []
    if n is not None:
        subtitle_bits.append(f"n={n}")
    if n_runs is not None:
        subtitle_bits.append(f"n_runs={n_runs}")
    if base_rate is not None:
        subtitle_bits.append(f"base_rate={base_rate:.3f}")
    full_title = title if not subtitle_bits else f"{title}\n({', '.join(subtitle_bits)})"
    ax.set_title(full_title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.2)
    if own_fig:
        fig.tight_layout()
    return fig
