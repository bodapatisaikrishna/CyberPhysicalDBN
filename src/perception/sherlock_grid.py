"""Uniform-grid alignment of the Sherlock physical snapshots and network-capture bins
(exp17). Everything here is causal and uses no label."""

from __future__ import annotations

import numpy as np


def grid_end_times(t0: float, n_bins: int, bin_s: float) -> np.ndarray:
    """End time of each bin `[t0 + k*bin_s, t0 + (k+1)*bin_s)`."""
    return t0 + bin_s * (np.arange(n_bins) + 1)


def locf_index(snapshot_ts: np.ndarray, t_end: np.ndarray) -> np.ndarray:
    """For each grid end time, the index of the latest snapshot with `ts <= t_end`
    (last observation carried forward). Grid times before the first snapshot map to
    index 0 (the earliest available snapshot -- reported by the caller via
    `n_before_first`). Requires sorted `snapshot_ts`."""
    idx = np.searchsorted(snapshot_ts, t_end, side="right") - 1
    return np.clip(idx, 0, len(snapshot_ts) - 1)


def n_before_first(snapshot_ts: np.ndarray, t_end: np.ndarray) -> int:
    return int((t_end < snapshot_ts[0]).sum())


def max_locf_staleness(snapshot_ts: np.ndarray, t_end: np.ndarray) -> float:
    """Largest age (s) of the carried-forward snapshot at any grid point -- the cost of LOCF."""
    idx = locf_index(snapshot_ts, t_end)
    return float(np.max(t_end - snapshot_ts[idx]))
