"""Tests for DualFloat: cycles and days normalization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, assume, settings
from hypothesis.strategies import composite, floats, integers

from jaxpint.dual_float import DualFloat

_EPS64 = np.finfo(np.float64).eps


def ulp_tol(expected):
    """2 ULP of float64 at the magnitude of `expected`."""
    mag = float(np.max(np.abs(np.asarray(expected))))
    return 2 * _EPS64 * max(mag, 1.0)


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@composite
def dual_floats_cycles(draw):
    """Generate a DualFloat normalized with cycles()."""
    int_part = float(draw(integers(min_value=-1_000_000, max_value=1_000_000)))
    frac_part = draw(floats(min_value=-0.5, max_value=0.5, exclude_max=True,
                            allow_nan=False, allow_infinity=False))
    return DualFloat.cycles(jnp.array(int_part), jnp.array(frac_part))


@composite
def dual_floats_days(draw):
    """Generate a DualFloat normalized with days()."""
    int_part = float(draw(integers(min_value=50000, max_value=70000)))
    frac_part = draw(floats(min_value=0.0, max_value=1.0, exclude_max=True,
                            allow_nan=False, allow_infinity=False))
    return DualFloat.days(jnp.array(int_part), jnp.array(frac_part))


# ===========================================================================
# DualFloat.days() normalization tests
# ===========================================================================

class TestDaysNormalization:
    def test_frac_in_range(self):
        """Frac must always be in [0, 1) after days()."""
        d = DualFloat.days(jnp.array([59000.0, 59001.0]),
                           jnp.array([1.3, -0.2]))
        assert jnp.all(d.frac >= 0.0)
        assert jnp.all(d.frac < 1.0)

    def test_preserves_total(self):
        """int + frac should equal the original total."""
        int_in = jnp.array([59000.0, 59001.0])
        frac_in = jnp.array([1.3, -0.2])
        d = DualFloat.days(int_in, frac_in)
        expected = int_in + frac_in
        assert jnp.allclose(d.total, expected, atol=ulp_tol(expected))

    def test_large_overflow(self):
        """Large fractional values are carried correctly."""
        d = DualFloat.days(jnp.array(59000.0), jnp.array(3.7))
        assert jnp.allclose(d.int, jnp.array(59003.0))
        assert jnp.allclose(d.frac, jnp.array(0.7), atol=1e-15)

    def test_negative_overflow(self):
        """Negative fractional values carry correctly."""
        d = DualFloat.days(jnp.array(59003.0), jnp.array(-2.3))
        assert jnp.allclose(d.int, jnp.array(59000.0))
        assert jnp.allclose(d.frac, jnp.array(0.7), atol=1e-15)

    @given(dual_floats_days())
    @settings(deadline=None)
    def test_hypothesis_frac_range(self, d):
        """Frac is always in [0, 1) for random inputs."""
        assert float(d.frac) >= 0.0
        assert float(d.frac) < 1.0

    @given(dual_floats_days())
    @settings(deadline=None)
    def test_hypothesis_idempotent(self, d):
        """days(d.int, d.frac) == d (idempotent)."""
        d2 = DualFloat.days(d.int, d.frac)
        assert jnp.array_equal(d.int, d2.int)
        assert jnp.array_equal(d.frac, d2.frac)


# ===========================================================================
# DualFloat.cycles() normalization tests (verify same as PhaseResult)
# ===========================================================================

class TestCyclesNormalization:
    def test_frac_in_range(self):
        """Frac must always be in [-0.5, 0.5) after cycles()."""
        d = DualFloat.cycles(jnp.array([0.0, 5.0, -3.0]),
                              jnp.array([0.7, -0.8, 1.3]))
        assert jnp.all(d.frac >= -0.5)
        assert jnp.all(d.frac < 0.5)

    def test_preserves_total(self):
        int_in = jnp.array([3.0, -2.0, 0.0])
        frac_in = jnp.array([0.7, -1.3, 0.49999])
        d = DualFloat.cycles(int_in, frac_in)
        expected = int_in + frac_in
        assert jnp.allclose(d.total, expected, atol=ulp_tol(expected))


# ===========================================================================
# Cross-construction tests
# ===========================================================================

class TestCrossConstruction:
    def test_cycles_then_days(self):
        """Converting between normalization conventions preserves total."""
        c = DualFloat.cycles(jnp.array(5.0), jnp.array(-0.3))
        d = DualFloat.days(c.int, c.frac)
        assert jnp.allclose(c.total, d.total, atol=ulp_tol(c.total))

    def test_days_then_cycles(self):
        """Converting between normalization conventions preserves total."""
        d = DualFloat.days(jnp.array(59000.0), jnp.array(0.7))
        c = DualFloat.cycles(d.int, d.frac)
        assert jnp.allclose(d.total, c.total, atol=ulp_tol(d.total))

    def test_subtraction_of_days_values(self):
        """Subtracting two days-normalized DualFloats gives correct total."""
        a = DualFloat.days(jnp.array(59001.0), jnp.array(0.3))
        b = DualFloat.days(jnp.array(59000.0), jnp.array(0.7))
        diff = a - b  # Uses cycles() normalization
        expected = (59001.3 - 59000.7)
        assert jnp.allclose(diff.total, jnp.array(expected), atol=ulp_tol(expected))


# ===========================================================================
# backward-compat alias
# ===========================================================================

class TestBackwardCompat:
    def test_create_alias(self):
        """DualFloat.create() is an alias for DualFloat.cycles()."""
        a = DualFloat.create(jnp.array(1.0), jnp.array(0.7))
        b = DualFloat.cycles(jnp.array(1.0), jnp.array(0.7))
        assert jnp.array_equal(a.int, b.int)
        assert jnp.array_equal(a.frac, b.frac)

    def test_phase_result_alias(self):
        """PhaseResult is DualFloat."""
        from jaxpint.phase_result import PhaseResult
        assert PhaseResult is DualFloat


# ===========================================================================
# JAX compatibility
# ===========================================================================

class TestJAXCompat:
    def test_jit_cycles(self):
        @jax.jit
        def f(x, y):
            return DualFloat.cycles(x, y).total
        assert jnp.allclose(f(jnp.array(1.0), jnp.array(0.7)),
                            jnp.array(2.0 - 0.3))

    def test_jit_days(self):
        @jax.jit
        def f(x, y):
            return DualFloat.days(x, y).total
        assert jnp.allclose(f(jnp.array(59000.0), jnp.array(1.3)),
                            jnp.array(59001.3))

    def test_grad_through_arithmetic(self):
        def loss(frac):
            a = DualFloat.cycles(jnp.array(1.0), frac)
            b = DualFloat.cycles(jnp.array(2.0), jnp.array(0.1))
            return (a + b).total

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.array(0.3))
        assert jnp.allclose(g, 1.0)

    def test_grad_through_days(self):
        def loss(frac):
            d = DualFloat.days(jnp.array(59000.0), frac)
            return d.total

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.array(0.5))
        assert jnp.allclose(g, 1.0)
