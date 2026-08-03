"""Virtual (likelihood) evidence for the 2TBN, with this repo's tuple names.

Session 5 (LAB_NOTEBOOK.md 2026-08-02): replaces the fictional hard `p_pos =
p_neg = 1e-4` evidence on selected analytic nodes with a calibrated learned
likelihood `q = P(analytic=1 | telemetry)` from `src/perception`, entered as
Pearl's virtual evidence rather than a hard 0/1 bit.

pgmpy 1.1.2 already implements Pearl's construction natively
(`VariableElimination.query(virtual_evidence=[...])`, verified numerically
correct against hand computation before this module was written) but it is
unusable here: `pgmpy/inference/base.py:276` does `new_var = "__" + var`,
which requires a STRING variable name, and every DBN variable in this model is
a tuple `(name, slice)` (`src/dbn/compiler.py`). Confirmed to raise
`TypeError` on this repo's model. This module reimplements the identical
construction with tuple names -- `tests/test_soft_evidence.py::
test_matches_pgmpy_native_on_isomorphic_string_named_model` proves the two are
numerically indistinguishable on an isomorphic string-named copy of the same
model (verified to 0.000e+00 max absolute difference in the session's own
pre-check across 7 likelihood cases, including near-degenerate extremes).

The construction (Pearl 1988; pgmpy's own reference: Mrad et al., "Uncertain
evidence in Bayesian networks," IPMU 2012): for a node X with likelihood
L(x) = P(telemetry | X=x), add a binary child V with P(V=0|X=x) = L(x), then
condition on V=0. By Bayes, posterior(X=x) is then proportional to
prior(X=x) * L(x) -- exactly virtual evidence. The construction is
SCALE-INVARIANT: only the ratio L(1)/L(0) matters, so L need not be
normalized (verified in the pre-check: (0.2,0.8) and (0.1,0.4) give identical
posteriors).

Two footguns in pgmpy's native path, both sidestepped structurally rather than
guarded against:
  1. `_virtual_evidence` calls `self.__init__(bn)`, mutating the inference
     object in place on every query -- and `src/dbn/inference.py`'s `step()`
     issues a SECOND query (the `joint=True` multi-member-cluster case) that
     would silently re-augment an already-augmented network. This module
     never calls pgmpy's `virtual_evidence=` path at all, so this is moot.
  2. Because we never call it, our child nodes must be added ONCE, at
     `DBNInference.__init__` time (see inference.py), with only their CPD
     VALUES changing per step -- mirroring exactly how `_attach_belief_as_prior`
     already swaps the anterior-layer prior CPD in place every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

from pgmpy.factors.discrete import TabularCPD

from src.dbn.compiler import ULTERIOR

SOFT_CHILD_PREFIX = "__soft__"

LikelihoodMode = Literal["naive", "prior_corrected", "dbn_prior_corrected"]


def soft_child_node(node: str, time_slice: int = ULTERIOR) -> tuple[str, int]:
    """The auxiliary virtual-evidence child's name: (f'__soft__{node}', slice).

    A tuple, matching every other node name in this model. pgmpy's native
    construction hardcodes string concatenation (`"__" + var`,
    pgmpy/inference/base.py:276) and cannot express this.
    """
    return (f"{SOFT_CHILD_PREFIX}{node}", time_slice)


def soft_child_cpd(node: str, l0: float, l1: float, *, time_slice: int = ULTERIOR) -> TabularCPD:
    """P(child=0 | node=x) = L(x), P(child=1 | node=x) = 1 - L(x).

    Byte-for-byte the same table pgmpy's native path builds
    (pgmpy/inference/base.py:278, `vstack((cpd.values, 1 - cpd.values))`).
    Conditioning on child=0 then gives posterior(x) proportional to
    prior(x) * L(x) -- Pearl's virtual evidence. `l0`/`l1` need not sum to any
    particular total per state (the construction is scale-invariant); they
    are typically produced already-normalized by
    `likelihood_from_probability` so this CPD is a valid distribution.
    """
    return TabularCPD(
        variable=soft_child_node(node, time_slice),
        variable_card=2,
        values=[[l0, l1], [1.0 - l0, 1.0 - l1]],
        evidence=[(node, time_slice)],
        evidence_card=[2],
    )


def uniform_soft_child_cpd(node: str, *, time_slice: int = ULTERIOR) -> TabularCPD:
    """L(0) = L(1) = 0.5: an exact no-op (verified in the pre-check -- the
    posterior with this CPD attached and child=0 conditioned on is bit-
    identical to omitting the node from evidence entirely). Used as the
    placeholder CPD for every soft-evidence target on a slice where no
    learned likelihood is supplied, so that likelihood value can never
    persist from a previous slice (see `DBNInference.step`)."""
    return soft_child_cpd(node, 0.5, 0.5, time_slice=time_slice)


def likelihood_from_probability(
    q: float,
    *,
    mode: LikelihoodMode = "prior_corrected",
    base_rate: float | None = None,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Convert a calibrated classifier's P(A=1|telemetry) into a likelihood
    ratio L(1):L(0) suitable for `soft_child_cpd`.

    A classifier emits q = P(A=1|telemetry). Pearl's construction needs
    L(x) proportional to P(telemetry|A=x). By Bayes:

        L(1)/L(0) = [q / pi] / [(1-q) / (1-pi)]

    where `pi = P(A=1)` is the classifier's OWN implicit training prior (its
    positive base rate). This is the correction that matters:

    - `"naive"`: L = (1-q, q), ratio q/(1-q). Exactly correct only when
      pi = 1/2. For any other base rate, it double-counts a prior the DBN's
      own forward filter has already computed from the transition CPTs and
      evidence history. Measured in this session's LAB_NOTEBOOK pre-check at
      pi=0.12, q=0.60: naive gives a fused P(parent=1)=0.447 vs. the
      corrected 0.855 -- a 0.41 divergence from what looks like a cosmetic
      normalization choice. Kept as a named, LOGGED ablation arm (exp05's
      `soft_calibrated_naive_lik`) specifically to measure this damage.
    - `"prior_corrected"` (the DEFAULT): divides out the classifier's own
      base rate, computed once on the training split and required (never
      defaulted -- CLAUDE.md rule 1: an unmeasured base rate is a fabricated
      number, not a default).
    - `"dbn_prior_corrected"`: the theoretically exact version -- divide by
      the DBN's OWN current time-varying prior P(A=1|e_{1:s-1}) instead of a
      constant pi. `prior_corrected`'s constant pi is itself an approximation
      of this. Implemented and tested but requires the caller to have
      already computed that prior (one extra VE query per slice, ~2x
      inference cost per soft-evidenced slice) -- this function only
      performs the arithmetic; `base_rate` for this mode should be that
      per-slice prior rather than a constant, supplied by the caller.

    Returns (l0, l1) normalized to sum to 1 -- cosmetic (the construction is
    scale-invariant), kept so these CPDs are literally comparable to what
    pgmpy's native path would produce for a valid distribution.

    `q` is clipped to `[eps, 1-eps]` before use: an unclipped q=0 or q=1,
    combined with the deterministic zero entries already present in
    `build_gate_cpt`'s AND/OR tables, can zero an entire joint and make
    pgmpy return NaN or raise. Clipping bounds the damage; callers are
    expected to count and log clip events (exp05 does, via
    `n_q_clipped_this_slice`).
    """
    if not 0.0 < eps < 0.5:
        raise ValueError(f"eps must be in (0, 0.5), got {eps}")
    q = min(max(q, eps), 1.0 - eps)

    if mode == "naive":
        l0, l1 = 1.0 - q, q
    elif mode in ("prior_corrected", "dbn_prior_corrected"):
        if base_rate is None:
            raise ValueError(
                f"mode={mode!r} requires a measured base_rate; a default here "
                "would be a fabricated number (CLAUDE.md rule 1)"
            )
        pi = min(max(base_rate, eps), 1.0 - eps)
        # ratio l1:l0 = (q/pi) : ((1-q)/(1-pi))
        r1 = q / pi
        r0 = (1.0 - q) / (1.0 - pi)
        total = r0 + r1
        l0, l1 = r0 / total, r1 / total
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return l0, l1


@dataclass(frozen=True)
class SoftEvidenceConfig:
    """Which analytic nodes receive learned virtual evidence, and how their
    q is converted to a likelihood.

    `base_rates` is REQUIRED (not defaulted) for any mode other than
    `"naive"` -- `likelihood_from_probability` enforces this per-call, but
    the config validates it eagerly so a missing measurement fails at
    construction time, not mid-experiment.
    """

    targets: tuple[str, ...]
    mode: LikelihoodMode = "prior_corrected"
    base_rates: Mapping[str, float] | None = None
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("SoftEvidenceConfig.targets must not be empty")
        if self.mode != "naive":
            missing = set(self.targets) - set(self.base_rates or {})
            if missing:
                raise ValueError(
                    f"mode={self.mode!r} requires a measured base_rate for every "
                    f"target; missing for {sorted(missing)}. A default would be a "
                    "fabricated number (CLAUDE.md rule 1)."
                )
