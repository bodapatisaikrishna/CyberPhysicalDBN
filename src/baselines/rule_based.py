"""Signature-based IDS proxy (CLAUDE.md's `src/baselines/`: "rule-based").

Consumes EXACTLY what the DBN's `hard` ablation arm consumes --
`discrete.evidence_stream(list(discrete.observable_names))`, the same call
`experiments/exp05_perception.py` uses for `hard_stream_full`. This is the
fairness contract for this baseline specifically: a legacy correlation rule
sees the same detector bits the hard-evidence DBN arm sees, nothing more (no
soft/calibrated perception, no graph structure, no temporal model beyond a
plain trailing window).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RuleConfig:
    window_slices: int

    def __post_init__(self) -> None:
        if self.window_slices < 1:
            raise ValueError(f"window_slices must be >= 1, got {self.window_slices}")


def score_trajectory(
    evidence_stream: Mapping[int, Mapping[str, int]],
    observable_names: Sequence[str],
    n_slices: int,
    config: RuleConfig,
) -> np.ndarray:
    """`score[s] = |{name in observable_names : name fired (==1) at least
    once in the trailing window [s-window+1, s]}| / len(observable_names)`,
    for `s = 1..n_slices` (1-based, matching every other per-slice
    convention in this repo), returned as a 0-based numpy array of length
    `n_slices` (`score[i]` corresponds to `slice_index = i+1`).

    Counts DISTINCT signatures, not raw 1-counts: cyber analytics are DENSE
    (0 before their trigger, 1 at/after -- `discretize()`'s own docstring),
    so summing raw 1s across the window would just re-count the same
    persistent bit every slice, which is not what a correlation-rule
    "how many distinct detectors are currently firing" means. Physical
    observables are SPARSE (a slice's entry can omit a name entirely when
    unobserved); an omission in one slice never overrides a `==1` seen in
    another slice within the same window.

    Already in `[0, 1]`, used AS-IS -- no calibration transform. A legacy
    count-rule has no probabilistic semantics, and if it turns out
    miscalibrated against measured `grid_unstable`, that is a reportable
    finding about this baseline, not something to "fix" by post-hoc scaling
    (which would just be temperature-scaling a rule count into looking like
    a posterior it never was).

    Raises on any name in `observable_names` that never appears in ANY
    slice's evidence dict -- mirrors `DiscreteTrace.evidence_stream`'s own
    scope guard, so a typo'd or stale observable name fails loudly here too
    rather than silently scoring as "never fired."
    """
    if n_slices < 1:
        raise ValueError(f"n_slices must be >= 1, got {n_slices}")
    if not observable_names:
        raise ValueError("observable_names must not be empty")

    known_names: set[str] = set()
    for entry in evidence_stream.values():
        known_names.update(entry.keys())
    unknown = [n for n in observable_names if n not in known_names]
    if unknown:
        raise ValueError(
            f"observable name(s) {unknown} never appear in evidence_stream; "
            "refusing to silently score them as never-fired"
        )

    fired_at: dict[str, set[int]] = {name: set() for name in observable_names}
    for slice_index, entry in evidence_stream.items():
        for name in observable_names:
            if entry.get(name) == 1:
                fired_at[name].add(slice_index)

    denom = float(len(observable_names))
    scores = np.zeros(n_slices, dtype=np.float64)
    for s in range(1, n_slices + 1):
        window_start = max(1, s - config.window_slices + 1)
        window = range(window_start, s + 1)
        distinct_firing = sum(
            1 for name in observable_names if fired_at[name] & set(window)
        )
        scores[s - 1] = distinct_firing / denom
    return scores
