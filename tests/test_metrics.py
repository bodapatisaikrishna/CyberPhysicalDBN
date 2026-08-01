"""Tests for KL divergence and M_KL (Cerotti et al. Eq. 2, Eq. 4, Eq. 5)."""

from __future__ import annotations

import math

import pytest

from src.eval.metrics import binary_kl, m_kl


class TestBinaryKL:
    """Eq. 2 specialized to a Bernoulli pair.

    The whole KL half of this project's validation gate rests on this
    function, so it is checked against hand arithmetic rather than only
    against itself.
    """

    def test_matches_hand_computation(self):
        # D_KL(Bern(0.7) || Bern(0.5)) = 0.7*ln(0.7/0.5) + 0.3*ln(0.3/0.5)
        expected = 0.7 * math.log(0.7 / 0.5) + 0.3 * math.log(0.3 / 0.5)
        assert binary_kl(0.7, 0.5) == pytest.approx(expected, rel=1e-12)

    def test_matches_hand_computation_extreme(self):
        expected = 0.9 * math.log(0.9 / 0.1) + 0.1 * math.log(0.1 / 0.9)
        assert binary_kl(0.9, 0.1) == pytest.approx(expected, rel=1e-12)

    def test_zero_when_identical(self):
        assert binary_kl(0.42, 0.42) == 0.0

    def test_is_asymmetric(self):
        """KL is not a metric; Eq. 4 fixes P=EX and Q=FF, so order matters.

        Guards the argument order in the experiment scripts: swapping EX and
        FF would silently produce a different, wrong number rather than an
        error.
        """
        assert binary_kl(0.7, 0.5) != pytest.approx(binary_kl(0.5, 0.7))

    def test_nonnegative(self):
        for p in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for q in [0.05, 0.5, 0.95]:
                assert binary_kl(p, q) >= 0.0

    def test_p_zero_contributes_nothing(self):
        """0 log(0/q) = 0 by convention, so p=0 needs no clipping."""
        assert binary_kl(0.0, 0.5) == pytest.approx(math.log(1.0 / 0.5))

    def test_rejects_non_probabilities(self):
        with pytest.raises(ValueError):
            binary_kl(1.5, 0.5)
        with pytest.raises(ValueError):
            binary_kl(0.5, -0.1)

    def test_q_at_boundary_stays_finite(self):
        """P>0 where Q=0 is formally +inf; clipped so a run can still report."""
        value = binary_kl(0.5, 0.0)
        assert math.isfinite(value)
        assert value > 0.0


class TestMKL:
    """Eq. 5: M_KL = max over t of psi(t)."""

    def test_returns_max_and_argmax(self):
        assert m_kl({0: 0.1, 5: 0.9, 10: 0.3}) == (0.9, 5)

    def test_single_point(self):
        assert m_kl({7: 0.25}) == (0.25, 7)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            m_kl({})
