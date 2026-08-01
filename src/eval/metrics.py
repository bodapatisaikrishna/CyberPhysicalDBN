"""Accuracy and resource-usage metrics (Cerotti et al. Eq. 2, Eq. 4, Eq. 5, Sec. IV-C).

Every query node in this project (UnstablePS, CorrReact, MITM, and every other
node in the compiled graph) is binary, so the general KL divergence sum over a
sample space (Eq. 2) reduces to a two-term Bernoulli comparison at each node
and timestep.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
import tracemalloc
from dataclasses import dataclass
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def binary_kl(p1: float, q1: float, eps: float = 1e-12) -> float:
    """D_KL(Bernoulli(p1) || Bernoulli(q1)) (Cerotti et al. Eq. 2, specialized).

        D_KL(P||Q) = sum_x P(x) log(P(x)/Q(x))

    with x in {0, 1}, P(1)=p1, Q(1)=q1. A P(x)=0 term contributes 0 by the
    standard convention (x log x -> 0 as x -> 0), so p1 is never clipped. Only
    q1 is clipped, and only away from exactly 0 or 1, to avoid a division by
    zero or log(0) when P(x)>0 but Q(x)=0 -- formally +inf, but under this
    model's structure an exact 0 or 1 in q1 should only arise at a hard
    precondition/gate boundary (Table 1 rows 1-4, 7-8; the AND/OR gate CPTs),
    where p1 should be at that same hard 0/1 and the clip is a no-op in
    practice. eps=1e-12 is far below the smallest KL magnitude this project
    reports (~1e-6, Fig. 6d/10b), so it cannot mask a real result; it exists
    only to keep the +inf case finite for logging.

    A clip that actually changes the result (p1 not itself near 0/1) is logged
    rather than silently absorbed (CLAUDE.md rule 5: state uncertainty), since
    it signals the two distributions disagree on which states are reachable at
    all -- worth seeing, not hiding.
    """
    if not (0.0 <= p1 <= 1.0):
        raise ValueError(f"p1 must be a probability, got {p1}")
    if not (0.0 <= q1 <= 1.0):
        raise ValueError(f"q1 must be a probability, got {q1}")

    p0, q0 = 1.0 - p1, 1.0 - q1
    # Only clip a q away from exactly 0 -- that is the only value that makes
    # p/q blow up. Never clip away from 1: q=1 exactly is a perfectly
    # ordinary, harmless value for the *other* state's probability.
    q1_clipped = q1 if q1 > eps else eps
    q0_clipped = q0 if q0 > eps else eps

    if (q1 <= eps and p1 > eps) or (q0 <= eps and p0 > eps):
        logger.warning(
            "binary_kl: clipping triggered with p1=%r q1=%r (non-trivial mass "
            "on a state the other distribution treats as impossible)",
            p1,
            q1,
        )

    total = 0.0
    if p1 > 0.0:
        total += p1 * math.log(p1 / q1_clipped)
    if p0 > 0.0:
        total += p0 * math.log(p0 / q0_clipped)
    return total


def m_kl(kl_trajectory: dict[int, float]) -> tuple[float, int]:
    """Eq. 5: M_KL = max_{t in [0,T]} psi(t). Returns (value, argmax t)."""
    if not kl_trajectory:
        raise ValueError("kl_trajectory must not be empty")
    best_t = max(kl_trajectory, key=lambda t: kl_trajectory[t])
    return kl_trajectory[best_t], best_t


@contextlib.contextmanager
def measure_step(latencies: list[float]):
    """Append one wall-clock duration (seconds) to `latencies` per use."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        latencies.append(time.perf_counter() - t0)


@dataclass
class MemoryReport:
    tracemalloc_peak_bytes: int
    rss_delta_bytes: int | None  # None if psutil is unavailable


def measure_memory(fn: Callable[[], T]) -> tuple[T, MemoryReport]:
    """Run `fn`, reporting peak Python-object allocation during the call.

    tracemalloc is the primary measurement: it tracks Python-level object
    allocation, which is the closest available analogue to the paper's own
    measurement method (MATLAB's `whos()`, scoped to workspace variables). Like
    `whos()`, this is an explicitly-stated LOWER BOUND on the process's true
    memory footprint -- it can undercount allocations made by C extensions
    (including some numpy buffer allocations, depending on the numpy build and
    allocator) that don't go through Python's own allocator.

    psutil's RSS delta (whole-process resident memory before/after `fn`) is
    reported as a secondary cross-check when psutil is installed. RSS captures
    more (interpreter and C-extension overhead) but is noisier at short
    timescales and not scoped to `fn` alone, so it is not the primary number.
    """
    try:
        import psutil

        process = psutil.Process()
        rss_before = process.memory_info().rss
    except ImportError:
        process = None
        rss_before = None

    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    rss_delta = None
    if process is not None:
        rss_delta = process.memory_info().rss - rss_before

    return result, MemoryReport(tracemalloc_peak_bytes=peak, rss_delta_bytes=rss_delta)
