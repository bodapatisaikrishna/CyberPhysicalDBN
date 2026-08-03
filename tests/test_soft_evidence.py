"""Tests for src/dbn/soft_evidence.py, written before it is wired into
src/dbn/inference.py (Session 5 sequencing: this piece is independent of the
GNN and the most likely to surface a blocker, so it is validated in
isolation first). Integration invariants that require a live `DBNInference`
(both query call sites, no cross-step persistence, mutation-freedom, the
existing hard-evidence path staying bit-identical) live in
tests/test_inference.py, added alongside the inference.py wiring.

The headline test proves our tuple-named virtual-evidence construction is
numerically indistinguishable from pgmpy's own native (string-named-only)
`virtual_evidence=` path, which is the entire justification for
reimplementing it rather than trusting it blind.
"""

from __future__ import annotations

import itertools

import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from src.dbn.compiler import ULTERIOR
from src.dbn.soft_evidence import (
    SoftEvidenceConfig,
    likelihood_from_probability,
    soft_child_cpd,
    soft_child_node,
    uniform_soft_child_cpd,
)


def _p1(result, variable) -> float:
    factor = result[variable] if isinstance(result, dict) else result
    return float(factor.values[1])


# A small chain PARENT -> ANALYTIC, with the repo's real analytic CPT shape
# ([[1-p_pos, p_neg],[p_pos, 1-p_neg]]) and a non-trivial prior, used
# throughout as the reference model.
PARENT_PRIOR = 0.65
P_POS, P_NEG = 1e-4, 1e-4


def _tuple_model() -> DiscreteBayesianNetwork:
    """This repo's convention: every node is (name, ULTERIOR)."""
    P, A = ("Parent", ULTERIOR), ("Analytic", ULTERIOR)
    m = DiscreteBayesianNetwork([(P, A)])
    m.add_cpds(
        TabularCPD(P, 2, [[1 - PARENT_PRIOR], [PARENT_PRIOR]]),
        TabularCPD(
            A, 2, [[1 - P_POS, P_NEG], [P_POS, 1 - P_NEG]],
            evidence=[P], evidence_card=[2],
        ),
    )
    return m


def _string_model() -> DiscreteBayesianNetwork:
    """An isomorphic copy with pgmpy's required plain-string names, used only
    to interrogate pgmpy's own native `virtual_evidence=` path as an oracle."""
    m = DiscreteBayesianNetwork([("Parent", "Analytic")])
    m.add_cpds(
        TabularCPD("Parent", 2, [[1 - PARENT_PRIOR], [PARENT_PRIOR]]),
        TabularCPD(
            "Analytic", 2, [[1 - P_POS, P_NEG], [P_POS, 1 - P_NEG]],
            evidence=["Parent"], evidence_card=[2],
        ),
    )
    return m


def _ours_soft_posterior(l0: float, l1: float, *, query_node: str = "Parent") -> float:
    """Our tuple-named construction: add the child once, condition on it."""
    m = _tuple_model()
    # Soft evidence is always attached to "Analytic"; query_node selects
    # whether we then read the posterior off Analytic itself or off its
    # parent (propagation), per caller.
    m.add_edge(("Analytic", ULTERIOR), soft_child_node("Analytic"))
    m.add_cpds(soft_child_cpd("Analytic", l0, l1))
    infer = VariableElimination(m)
    r = infer.query(
        [(query_node, ULTERIOR)],
        evidence={soft_child_node("Analytic"): 0},
        joint=False, show_progress=False,
    )
    return _p1(r, (query_node, ULTERIOR))


def _native_soft_posterior(l0: float, l1: float, *, query_node: str = "Parent") -> float:
    """pgmpy's own native virtual_evidence path, on the STRING model, as the
    independent oracle. A fresh VariableElimination per call, with exactly
    ONE query, deliberately avoids pgmpy's in-place re-augmentation
    (`_virtual_evidence` calls `self.__init__(bn)`; base.py:276/289)."""
    m = _string_model()
    L = TabularCPD("Analytic", 2, [[l0], [l1]])
    infer = VariableElimination(m)
    r = infer.query([query_node], virtual_evidence=[L], joint=False, show_progress=False)
    return _p1(r, query_node)


class TestMatchesPgmpyNative:
    """The headline: proves our reimplementation is not just plausible but
    numerically identical to pgmpy's own construction."""

    @pytest.mark.parametrize(
        "q",
        [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    )
    def test_matches_pgmpy_native_on_isomorphic_string_named_model(self, q):
        l0, l1 = 1.0 - q, q  # an arbitrary, not-necessarily-normalized-nicely L
        ours = _ours_soft_posterior(l0, l1, query_node="Parent")
        native = _native_soft_posterior(l0, l1, query_node="Parent")
        assert ours == pytest.approx(native, abs=1e-12)

    @pytest.mark.parametrize("q", [0.05, 0.4, 0.6, 0.95])
    def test_matches_pgmpy_native_querying_the_evidenced_node_itself(self, q):
        """Virtual evidence on the CHILD's parent (Analytic here is the node
        carrying the soft evidence's parent role -- i.e. query the node one
        hop from the soft-evidenced variable) must also match."""
        l0, l1 = 1.0 - q, q
        ours = _ours_soft_posterior(l0, l1, query_node="Analytic")
        native = _native_soft_posterior(l0, l1, query_node="Analytic")
        assert ours == pytest.approx(native, abs=1e-12)


class TestConstructionProperties:
    def test_uniform_likelihood_is_a_noop(self):
        m = _tuple_model()
        infer_plain = VariableElimination(m)
        baseline = _p1(
            infer_plain.query([("Parent", ULTERIOR)], joint=False, show_progress=False),
            ("Parent", ULTERIOR),
        )

        m2 = _tuple_model()
        m2.add_edge(("Analytic", ULTERIOR), soft_child_node("Analytic"))
        m2.add_cpds(uniform_soft_child_cpd("Analytic"))
        with_soft = _p1(
            VariableElimination(m2).query(
                [("Parent", ULTERIOR)],
                evidence={soft_child_node("Analytic"): 0},
                joint=False, show_progress=False,
            ),
            ("Parent", ULTERIOR),
        )
        assert with_soft == pytest.approx(baseline, abs=1e-12)

    def test_degenerate_likelihood_equals_hard_evidence(self):
        """Degenerate virtual evidence ON Analytic (L close to (0,1), i.e.
        near-certain Analytic=1), observed from elsewhere in the network
        (Parent's posterior), must reproduce what HARD evidence {Analytic: 1}
        would give Parent -- verified via the query_node='Parent' path so the
        soft evidence and the comparison target are the same physical
        quantity, not two different setups."""
        eps = 1e-9
        soft = _ours_soft_posterior(eps, 1.0 - eps, query_node="Parent")

        m = _tuple_model()
        hard = _p1(
            VariableElimination(m).query(
                [("Parent", ULTERIOR)], evidence={("Analytic", ULTERIOR): 1},
                joint=False, show_progress=False,
            ),
            ("Parent", ULTERIOR),
        )
        assert soft == pytest.approx(hard, abs=1e-6)

    def test_degenerate_likelihood_zero_equals_hard_evidence_zero(self):
        eps = 1e-9
        soft = _ours_soft_posterior(1.0 - eps, eps, query_node="Parent")
        m = _tuple_model()
        hard = _p1(
            VariableElimination(m).query(
                [("Parent", ULTERIOR)], evidence={("Analytic", ULTERIOR): 0},
                joint=False, show_progress=False,
            ),
            ("Parent", ULTERIOR),
        )
        assert soft == pytest.approx(hard, abs=1e-6)

    def test_scale_invariance(self):
        """Only the ratio L(1)/L(0) matters -- (0.2,0.8) and (0.1,0.4) share
        the same ratio and must give identical posteriors."""
        a = _ours_soft_posterior(0.2, 0.8, query_node="Parent")
        b = _ours_soft_posterior(0.1, 0.4, query_node="Parent")
        assert a == pytest.approx(b, abs=1e-12)

    def test_soft_child_node_is_a_tuple(self):
        node = soft_child_node("Analytic")
        assert node == ("__soft__Analytic", ULTERIOR)

    def test_soft_child_cpd_shape_and_normalization(self):
        cpd = soft_child_cpd("Analytic", 0.3, 0.9)
        values = cpd.get_values()
        assert values.shape == (2, 2)
        # column 0 = P(child|Analytic=0) = [l0, 1-l0]; column 1 = [l1, 1-l1]
        assert values[0, 0] == pytest.approx(0.3)
        assert values[1, 0] == pytest.approx(0.7)
        assert values[0, 1] == pytest.approx(0.9)
        assert values[1, 1] == pytest.approx(0.1)
        import numpy as np
        np.testing.assert_allclose(values.sum(axis=0), [1.0, 1.0])


class TestLikelihoodFromProbability:
    def test_prior_corrected_hand_computed(self):
        """q=0.9, pi=0.1: r1 = 0.9/0.1 = 9, r0 = 0.1/0.9 = 0.111...,
        l1 = 9 / (9 + 0.111...) = 0.987804878..."""
        l0, l1 = likelihood_from_probability(0.9, mode="prior_corrected", base_rate=0.1)
        assert l1 == pytest.approx(0.9878048780487805, abs=1e-12)
        assert l0 == pytest.approx(1.0 - 0.9878048780487805, abs=1e-12)

    def test_naive_hand_computed(self):
        l0, l1 = likelihood_from_probability(0.9, mode="naive")
        assert (l0, l1) == pytest.approx((0.1, 0.9))

    def test_naive_equals_prior_corrected_at_base_rate_half(self):
        naive = likelihood_from_probability(0.73, mode="naive")
        corrected = likelihood_from_probability(0.73, mode="prior_corrected", base_rate=0.5)
        assert naive == pytest.approx(corrected, abs=1e-12)

    def test_naive_differs_from_prior_corrected_otherwise(self):
        naive = likelihood_from_probability(0.6, mode="naive")
        corrected = likelihood_from_probability(0.6, mode="prior_corrected", base_rate=0.12)
        assert naive != pytest.approx(corrected, abs=1e-6)

    def test_prior_correction_materially_changes_fused_posterior(self):
        """The qualitative divergence from the LAB_NOTEBOOK pre-check (there
        measured at pi=0.12, q=0.60: naive fused posterior ~0.447 vs.
        corrected ~0.855, on a Parent prior of 0.35 -- a different constant
        from this file's PARENT_PRIOR=0.65 fixture, so the exact figures are
        not reproduced bit-for-bit here, only the qualitative claim: prior
        correction is not a cosmetic normalization tweak, it materially
        moves the fused posterior)."""
        naive_l = likelihood_from_probability(0.60, mode="naive")
        corrected_l = likelihood_from_probability(0.60, mode="prior_corrected", base_rate=0.12)

        naive_post = _ours_soft_posterior(*naive_l, query_node="Parent")
        corrected_post = _ours_soft_posterior(*corrected_l, query_node="Parent")

        assert abs(naive_post - corrected_post) > 0.2

    def test_rejects_missing_base_rate_for_prior_corrected(self):
        with pytest.raises(ValueError, match="base_rate"):
            likelihood_from_probability(0.5, mode="prior_corrected")

    def test_rejects_missing_base_rate_for_dbn_prior_corrected(self):
        with pytest.raises(ValueError, match="base_rate"):
            likelihood_from_probability(0.5, mode="dbn_prior_corrected")

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            likelihood_from_probability(0.5, mode="bogus")  # type: ignore[arg-type]

    def test_clips_extreme_q(self):
        l0, l1 = likelihood_from_probability(0.0, mode="naive", eps=1e-6)
        assert l0 == pytest.approx(1.0 - 1e-6)
        assert l1 == pytest.approx(1e-6)
        l0, l1 = likelihood_from_probability(1.0, mode="naive", eps=1e-6)
        assert l0 == pytest.approx(1e-6)
        assert l1 == pytest.approx(1.0 - 1e-6)

    def test_rejects_bad_eps(self):
        with pytest.raises(ValueError):
            likelihood_from_probability(0.5, mode="naive", eps=0.0)
        with pytest.raises(ValueError):
            likelihood_from_probability(0.5, mode="naive", eps=0.5)

    def test_output_always_normalized(self):
        for q, mode, base_rate in itertools.product(
            [0.01, 0.3, 0.5, 0.7, 0.99], ["naive", "prior_corrected"], [None, 0.2]
        ):
            if mode == "prior_corrected" and base_rate is None:
                continue
            l0, l1 = likelihood_from_probability(q, mode=mode, base_rate=base_rate)
            assert l0 + l1 == pytest.approx(1.0, abs=1e-10)


class TestSoftEvidenceConfig:
    def test_requires_nonempty_targets(self):
        with pytest.raises(ValueError):
            SoftEvidenceConfig(targets=())

    def test_naive_mode_does_not_require_base_rates(self):
        cfg = SoftEvidenceConfig(targets=("Analytic",), mode="naive")
        assert cfg.base_rates is None

    def test_prior_corrected_requires_base_rates(self):
        with pytest.raises(ValueError, match="base_rate"):
            SoftEvidenceConfig(targets=("Analytic",), mode="prior_corrected")

    def test_prior_corrected_requires_base_rate_for_every_target(self):
        with pytest.raises(ValueError, match="MeasureCoherence"):
            SoftEvidenceConfig(
                targets=("CommandCoherence", "MeasureCoherence"),
                mode="prior_corrected",
                base_rates={"CommandCoherence": 0.1},
            )

    def test_prior_corrected_accepts_complete_base_rates(self):
        cfg = SoftEvidenceConfig(
            targets=("CommandCoherence", "MeasureCoherence"),
            mode="prior_corrected",
            base_rates={"CommandCoherence": 0.1, "MeasureCoherence": 0.2},
        )
        assert cfg.mode == "prior_corrected"

    def test_default_mode_is_prior_corrected(self):
        cfg = SoftEvidenceConfig(targets=("Analytic",), base_rates={"Analytic": 0.3})
        assert cfg.mode == "prior_corrected"
