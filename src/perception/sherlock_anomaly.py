"""Unsupervised anomaly-detection helpers for the real Sherlock scenarios
(LAB_NOTEBOOK.md 2026-09-28, exp13).

WHY THIS EXISTS. Sherlock's `01-Basic` and `02-Semiurban` ship an ATTACK-FREE
train split and an attack-bearing test split by the dataset authors' design.
A supervised classifier has no positive to learn from in that setting (exp07's
constant predictor: AUC-PR == base rate). The setting the dataset is built
for is anomaly detection: fit on clean data only, score the attack file. This
module supplies the small numerical pieces that setting needs, on top of the
EXISTING, tested LSTM autoencoder (`src.baselines.lstm_ae`, reused unchanged):

  * per-feature standardization fit on one clean chunk only (the aggregate
    Sherlock features span ~7 orders of magnitude -- voltage ~1 pu vs. load
    power ~1e6 W -- so an unstandardized autoencoder is dominated by the
    largest-scale column);
  * memory-flat causal windowing/scoring: `lstm_ae.score_trajectory` builds the
    full `[S, W, F]` window tensor at once, which is ~2 GB at S = 800k; here
    windows are built per batch from a padded copy of the `[S, F]` sequence;
  * a trivial reference detector (mean |z| over the standardized features), so
    "the autoencoder adds something" is checked rather than assumed;
  * contiguous attack-event runs from the per-slice event ids.

Nothing here reads a label as a feature: every function takes feature tensors
only (event ids are consumed solely by `event_runs`, after scoring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from src.baselines.lstm_ae import LSTMAutoencoder, reconstruction_error_last_step


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: torch.Tensor  # [F]
    std: torch.Tensor  # [F], floored at `eps`
    fit_split: str  # provenance, mirrors ReconErrorScaler / TemperatureScaler
    z_clip: float | None = None  # |z| is clipped here after standardizing (None = no clipping)


def fit_standardizer(
    x: torch.Tensor, *, eps: float = 1e-6, fit_split: str, z_clip: float | None = None
) -> FeatureStandardizer:
    """Per-feature mean / population-std over `x: [S, F]`. Fit on ONE clean
    chunk (never on any split that contains an attack). A constant column gets
    `std = eps` rather than 0, so it standardizes to exactly 0 instead of
    dividing by zero.

    `z_clip` exists because that eps floor is a NUMERICAL guard, not a scale:
    a feature that is constant in the fit data but moves later (exp13's
    preview: a transformer tap position, 0 for 82% of a clean recording, then
    127.5) would otherwise standardize to ~1e8 and, through a mean or a
    squared error, set the whole detector's threshold. Clipping caps any one
    feature's influence at `z_clip` clean standard deviations."""
    if x.dim() != 2 or x.shape[0] == 0:
        raise ValueError(f"expected non-empty [S,F], got shape {tuple(x.shape)}")
    if z_clip is not None and z_clip <= 0:
        raise ValueError(f"z_clip must be positive, got {z_clip}")
    x64 = x.to(torch.float64)
    return FeatureStandardizer(
        mean=x64.mean(dim=0).to(torch.float32),
        std=x64.std(dim=0, unbiased=False).clamp_min(eps).to(torch.float32),
        fit_split=fit_split,
        z_clip=z_clip,
    )


def apply_standardizer(x: torch.Tensor, s: FeatureStandardizer) -> torch.Tensor:
    z = (x - s.mean) / s.std
    return z if s.z_clip is None else z.clamp(-s.z_clip, s.z_clip)


def blocked_split(n: int, block_slices: int, pattern: Sequence[str]) -> dict[str, list[tuple[int, int]]]:
    """Cut `[0, n)` into consecutive `block_slices`-long blocks (the last one
    may be shorter) and deal them out cyclically following `pattern`, e.g.
    `("fit","fit","fit","val","calib")` -> `{"fit": [(0,B),(B,2B),...], ...}`.
    Unlike a chronological split this puts EVERY split in every operating
    regime a single long recording passes through (exp13's preview: a clean
    file whose transformer tap changes once, ~82% of the way in). Ranges are
    disjoint by construction and cover `[0, n)` exactly. Adjacent blocks of
    different splits are temporally correlated -- a stated limitation, not
    hidden: clean-vs-clean chunks are an easier comparison than a genuinely
    later period."""
    if block_slices < 1 or n < 1 or not pattern:
        raise ValueError("block_slices, n and pattern must all be non-empty/positive")
    out: dict[str, list[tuple[int, int]]] = {name: [] for name in dict.fromkeys(pattern)}
    for k, start in enumerate(range(0, n, block_slices)):
        out[pattern[k % len(pattern)]].append((start, min(start + block_slices, n)))
    return out


def endpoints_in_ranges(ranges: Sequence[tuple[int, int]], stride: int, min_end: int = 0) -> torch.Tensor:
    """Every `stride`-th slice index inside each `[a, b)` range (indices `<
    min_end` dropped -- e.g. `WINDOW - 1`, so training windows are fully real,
    never zero-padded), sorted ascending."""
    parts = [torch.arange(max(a, min_end), b, stride) for a, b in ranges if b > max(a, min_end)]
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long)


def score_ae_ranges(
    model: LSTMAutoencoder, x_std: torch.Tensor, window: int, ranges: Sequence[tuple[int, int]], *, batch_size: int = 4096,
) -> np.ndarray:
    """`score_ae_batched` over several `[a, b)` ranges of the SAME standardized
    sequence (each window still built from true preceding context), results
    concatenated in range order."""
    parts = [score_ae_batched(model, x_std, window, batch_size=batch_size, start=a, end=b) for a, b in ranges]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _padded(x_std: torch.Tensor, window: int) -> torch.Tensor:
    """`[W-1+S, F]`: `W-1` zero rows on the left, matching
    `lstm_ae.build_causal_windows`'s left-zero-pad convention exactly."""
    return torch.cat([torch.zeros(window - 1, x_std.shape[1], dtype=x_std.dtype), x_std], dim=0)


def windows_ending_at(padded: torch.Tensor, endpoints: torch.Tensor, window: int) -> torch.Tensor:
    """`[len(endpoints), W, F]`: window `k` is the `window` slices ending at
    (and including) slice `endpoints[k]`. Row `endpoints[k] + W - 1` of the
    padded sequence is slice `endpoints[k]`, so the window is
    `padded[endpoints[k] : endpoints[k] + W]`."""
    idx = endpoints.unsqueeze(1) + torch.arange(window).unsqueeze(0)  # [B, W]
    return padded[idx]


def causal_windows_for_endpoints(x_std: torch.Tensor, endpoints: torch.Tensor, window: int) -> torch.Tensor:
    """Public one-call form of `windows_ending_at(_padded(...))` for callers
    that need windows at a strided subset of slices (training corpora)."""
    return windows_ending_at(_padded(x_std, window), endpoints, window)


def score_ae_batched(
    model: LSTMAutoencoder, x_std: torch.Tensor, window: int, *, batch_size: int = 4096,
    start: int = 0, end: int | None = None,
) -> np.ndarray:
    """Raw last-step reconstruction error for slices `[start, end)` of the
    FULL standardized sequence `x_std: [S, F]`, each window built from true
    preceding context (not zero-padded at `start` -- only at the file start).
    Numerically identical to
    `lstm_ae.score_trajectory(model, build_causal_windows(x_std, window))[start:end]`
    (`tests/test_sherlock_anomaly.py`), without materializing `[S, W, F]`."""
    end = x_std.shape[0] if end is None else end
    padded = _padded(x_std, window)
    out = np.empty(end - start, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for a in range(start, end, batch_size):
            b = min(a + batch_size, end)
            w = windows_ending_at(padded, torch.arange(a, b), window)
            out[a - start:b - start] = reconstruction_error_last_step(model(w), w).numpy()
    return out


def zscore_mean_abs(x_std: torch.Tensor) -> np.ndarray:
    """Reference detector: mean over features of |z|. `[S, F] -> [S]`, float32.
    Needs no training beyond the standardizer, so it bounds what a model with
    no temporal or cross-feature structure achieves on the same inputs."""
    return x_std.abs().mean(dim=1).numpy()


def event_runs(event_ids: Sequence[str | None]) -> list[tuple[str, int, int]]:
    """Maximal contiguous runs of the SAME non-None event id ->
    `[(event_id, start, end_exclusive), ...]`. Two adjacent runs with
    different ids are two events; a `None` slice always ends a run."""
    runs: list[tuple[str, int, int]] = []
    cur: str | None = None
    start = 0
    for i, e in enumerate(event_ids):
        if e != cur:
            if cur is not None:
                runs.append((cur, start, i))
            cur, start = e, i
    if cur is not None:
        runs.append((cur, start, len(event_ids)))
    return runs
