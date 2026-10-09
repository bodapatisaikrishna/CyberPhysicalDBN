"""Label-free multivariate detectors over the per-component Sherlock view
(LAB_NOTEBOOK.md 2026-10-03, "component view"). Everything is fit on attack-free
data only; no label, no test statistic, ever enters a fit.

Pipeline (all choices fixed a priori in the config, none searched on test labels):
  1. NaN fill: each column's NaN -> that column's median over the clean FIT
     chunk (the nonfinite-count columns carry the NaN information explicitly).
     A column that is NaN throughout the fit chunk is filled with 0.0 and is
     constant -> standardizes to 0.
  2. Standardize with `fit_standardizer` (mean/std on the fit chunk, |z| clipped).
  3. PCA (principal subspace) on the standardized fit chunk; k = smallest number
     of components reaching `explained_variance_target`, capped at `max_components`.
Scores per snapshot:
  pca_spe          squared residual outside the principal subspace (the classical
                   Q statistic / squared prediction error)
  pca_t2           Hotelling T^2 inside the subspace
  max_abs_z        largest |z| over columns -- catches one component moving
  top_k_mean_abs_z mean of the `top_k` largest |z| -- robust version of the above
They are standard process-monitoring statistics, not tuned detectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.perception.sherlock_anomaly import FeatureStandardizer, apply_standardizer, fit_standardizer


@dataclass(frozen=True)
class ComponentModel:
    fill: torch.Tensor  # [F] per-column NaN fill (clean-fit median)
    standardizer: FeatureStandardizer
    components: torch.Tensor  # [F, k] principal axes (orthonormal columns)
    eigenvalues: torch.Tensor  # [k] variances along those axes
    explained_variance_ratio: float  # fraction of fit variance captured by the k axes
    n_columns: int
    n_constant_in_fit: int  # columns whose fit std hit the eps floor


def fill_nonfinite(x: torch.Tensor, fill: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(x), x, fill.expand_as(x))


def fit_component_model(
    x_fit: torch.Tensor, *, eps: float, z_clip: float, explained_variance_target: float,
    max_components: int, fit_split: str,
) -> ComponentModel:
    """Fit on `x_fit: [S, F]` (raw, may hold NaN) from attack-free data."""
    if not 0.0 < explained_variance_target <= 1.0:
        raise ValueError("explained_variance_target must be in (0, 1]")
    finite = torch.isfinite(x_fit)
    x_nan = torch.where(finite, x_fit, torch.full_like(x_fit, float("nan"))).numpy()
    with np.errstate(all="ignore"):
        med = np.nanmedian(x_nan, axis=0)
    fill = torch.from_numpy(np.where(np.isfinite(med), med, 0.0).astype(np.float32))
    x_filled = fill_nonfinite(x_fit, fill)
    std = fit_standardizer(x_filled, eps=eps, z_clip=z_clip, fit_split=fit_split)
    z = apply_standardizer(x_filled, std).to(torch.float64)
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    evals, evecs = evals.flip(0).clamp_min(0.0), evecs.flip(1)
    total = float(evals.sum())
    if total <= 0.0:
        raise ValueError("fit chunk has zero total variance after standardizing")
    cum = torch.cumsum(evals, 0) / total
    k = int(min(int((cum < explained_variance_target).sum().item()) + 1, max_components, len(evals)))
    return ComponentModel(
        fill=fill, standardizer=std,
        components=evecs[:, :k].to(torch.float32), eigenvalues=evals[:k].to(torch.float32),
        explained_variance_ratio=float(cum[k - 1]), n_columns=int(x_fit.shape[1]),
        n_constant_in_fit=int((std.std <= eps * (1.0 + 1e-6)).sum().item()),
    )


def standardized(model: ComponentModel, x_raw: torch.Tensor) -> torch.Tensor:
    """Fill NaN with the fit medians, standardize with the fit statistics (|z| clipped)."""
    return apply_standardizer(fill_nonfinite(x_raw, model.fill), model.standardizer)


def component_scores(model: ComponentModel, x_raw: torch.Tensor, top_k: int, batch: int = 4096) -> dict[str, np.ndarray]:
    """`{detector: [S] float32}` for a whole file, batched to bound memory."""
    out: dict[str, list[np.ndarray]] = {"pca_spe": [], "pca_t2": [], "max_abs_z": [], "top_k_mean_abs_z": []}
    inv = (1.0 / model.eigenvalues.clamp_min(1e-12)).unsqueeze(0)
    for a in range(0, x_raw.shape[0], batch):
        z = standardized(model, x_raw[a:a + batch])
        proj = z @ model.components
        resid = z - proj @ model.components.T
        out["pca_spe"].append((resid ** 2).sum(dim=1).numpy())
        out["pca_t2"].append(((proj ** 2) * inv).sum(dim=1).numpy())
        az = z.abs()
        out["max_abs_z"].append(az.max(dim=1).values.numpy())
        out["top_k_mean_abs_z"].append(az.topk(min(top_k, az.shape[1]), dim=1).values.mean(dim=1).numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


# --- causal rolling baseline (exp15) ----------------------------------------


def causal_rolling_robust_z(
    x: torch.Tensor, *, window: int, min_history: int, scale_floor: torch.Tensor | None, z_clip: float,
    chunk_cols: int = 400, self_floor: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score each snapshot against the file's OWN trailing baseline, per column.

    m_t = median of x over snapshots [t-window, t-1]             (strictly past)
    r_u = |x_u - m_u|
    s_t = median of r over snapshots [t-window, t-1]             (causal MAD of residuals)
    z_t = (x_t - m_t) / (1.4826 * max(s_t, floor, 1e-6)), clipped to +-z_clip

    `self_floor=True` (exp19) replaces the supplied floor by the run's OWN expanding median of
    its past scales s_1..s_{t-1} (causal; needs no clean data of this network).
    Otherwise `floor` (per column) is supplied by the caller from CLEAN TRAIN data only
    (`median_t s_t` there), so a column that is quiet in the trailing window is not
    blown up by a near-zero scale. Rows with fewer than `min_history` finite past
    values in a column get z = 0 (no baseline yet). NaN inputs give NaN baselines
    where the window has too few finite values -> z = 0, never imputed with a level.
    No future snapshot and no label is ever used.

    Returns `(z [S,F], s_median [F])`; `s_median` is the column-wise median of s_t over
    this file's rows with a defined scale -- used to FIT the floor on clean train."""
    import pandas as pd

    S, F = x.shape
    z_out = torch.zeros((S, F), dtype=torch.float32)
    s_med = torch.zeros(F, dtype=torch.float32)
    xn = x.numpy()
    for a in range(0, F, chunk_cols):
        df = pd.DataFrame(xn[:, a:a + chunk_cols].astype(np.float64))
        past = df.shift(1)
        m = past.rolling(window, min_periods=min_history).median()
        r = (df - m).abs()
        s = r.shift(1).rolling(window, min_periods=min_history).median()
        s_arr = s.to_numpy()
        with np.errstate(all="ignore"):
            s_med[a:a + chunk_cols] = torch.from_numpy(np.nan_to_num(np.nanmedian(s_arr, axis=0), nan=0.0).astype(np.float32))
        if self_floor:
            fl = s.shift(1).expanding(min_periods=1).median().to_numpy()
            fl = np.nan_to_num(fl, nan=0.0)
        else:
            fl = np.zeros(s_arr.shape[1], dtype=np.float64) if scale_floor is None else scale_floor[a:a + chunk_cols].numpy().astype(np.float64)
        scale = 1.4826 * np.maximum(np.nan_to_num(s_arr, nan=0.0), np.maximum(fl, 1e-6))
        z = (df.to_numpy() - m.to_numpy()) / scale
        z = np.where(np.isfinite(z), z, 0.0)
        z_out[:, a:a + chunk_cols] = torch.from_numpy(np.clip(z, -z_clip, z_clip).astype(np.float32))
    return z_out, s_med
