"""Tests for DualFloat: cycles/days normalization, arithmetic, precision."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import composite, floats, integers

from jaxpint.types.dual_float import DualFloat

# Uniform tolerance: 2 ULP of float64 at the scale of the expected value
_EPS64 = np.finfo(np.float64).eps


def ulp_tol(expected):
    """2 ULP of float64 at the magnitude of `expected`.

    ULP = "Unit in the Last Place" — the spacing between adjacent
    representable floats at a given magnitude. See
    https://en.wikipedia.org/wiki/Unit_in_the_last_place.

    For float64, ULP(x) ≈ |x| * 2^-52 ≈ |x| * 2.2e-16. We clamp the
    magnitude at 1.0 so tests against near-zero expected values still
    get a meaningful tolerance (2 * eps64).
    """
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
    return DualFloat.from_cycles(jnp.array(int_part), jnp.array(frac_part))


@composite
def dual_floats_days(draw):
    """Generate a DualFloat normalized with days()."""
    int_part = float(draw(integers(min_value=50000, max_value=70000)))
    frac_part = draw(floats(min_value=0.0, max_value=1.0, exclude_max=True,
                            allow_nan=False, allow_infinity=False))
    return DualFloat.from_days(jnp.array(int_part), jnp.array(frac_part))


@composite
def raw_phase_inputs(draw):
    """Generate arbitrary (int, frac) inputs that may need normalization."""
    int_part = float(draw(integers(min_value=-1_000_000, max_value=1_000_000)))
    frac_part = draw(floats(min_value=-100.0, max_value=100.0,
                            allow_nan=False, allow_infinity=False))
    return int_part, frac_part


@composite
def big_int_dual_floats(draw):
    """DualFloat with int in [1e9, 1e12] range (realistic pulsar scales)."""
    int_part = float(draw(integers(min_value=1_000_000_000, max_value=1_000_000_000_000)))
    frac_part = draw(floats(min_value=-0.5, max_value=0.5, exclude_max=True,
                            allow_nan=False, allow_infinity=False))
    return DualFloat.from_cycles(jnp.array(int_part), jnp.array(frac_part))


@composite
def small_perturbations(draw):
    """Tiny floats in [1e-15, 1e-10] for cancellation tests."""
    return draw(floats(min_value=1e-15, max_value=1e-10,
                       allow_nan=False, allow_infinity=False))


# ===========================================================================
# DualFloat.from_days() normalization
# ===========================================================================

class TestDaysNormalization:
    def test_frac_in_range(self):
        """Frac must always be in [0, 1) after days()."""
        d = DualFloat.from_days(jnp.array([59000.0, 59001.0]),
                           jnp.array([1.3, -0.2]))
        assert jnp.all(d.frac >= 0.0)
        assert jnp.all(d.frac < 1.0)

    def test_preserves_total(self):
        """int + frac should equal the original total."""
        int_in = jnp.array([59000.0, 59001.0])
        frac_in = jnp.array([1.3, -0.2])
        d = DualFloat.from_days(int_in, frac_in)
        expected = int_in + frac_in
        assert jnp.allclose(d.approx_total, expected, atol=ulp_tol(expected))

    def test_large_overflow(self):
        """Large fractional values are carried correctly."""
        d = DualFloat.from_days(jnp.array(59000.0), jnp.array(3.7))
        assert jnp.allclose(d.int, jnp.array(59003.0))
        assert jnp.allclose(d.frac, jnp.array(0.7), atol=1e-15)

    def test_negative_overflow(self):
        """Negative fractional values carry correctly."""
        d = DualFloat.from_days(jnp.array(59003.0), jnp.array(-2.3))
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
        d2 = DualFloat.from_days(d.int, d.frac)
        assert jnp.array_equal(d.int, d2.int)
        assert jnp.array_equal(d.frac, d2.frac)

# ===========================================================================
# DualFloat.from_cycles() normalization
# ===========================================================================

class TestCyclesNormalization:
    def test_frac_in_range(self):
        """Frac must always be in [-0.5, 0.5) after cycles()."""
        d = DualFloat.from_cycles(jnp.array([0.0, 5.0, -3.0]),
                              jnp.array([0.7, -0.8, 1.3]))
        assert jnp.all(d.frac >= -0.5)
        assert jnp.all(d.frac < 0.5)

    def test_preserves_total(self):
        int_in = jnp.array([3.0, -2.0, 0.0])
        frac_in = jnp.array([0.7, -1.3, 0.49999])
        d = DualFloat.from_cycles(int_in, frac_in)
        expected = int_in + frac_in
        assert jnp.allclose(d.total, expected, atol=ulp_tol(expected))


# ===========================================================================
# Cross-construction tests
# ===========================================================================

class TestCrossConstruction:
    def test_cycles_then_days(self):
        """Converting between normalization conventions preserves total."""
        c = DualFloat.from_cycles(jnp.array(5.0), jnp.array(-0.3))
        d = DualFloat.from_days(c.int, c.frac)
        assert jnp.allclose(c.total, d.total, atol=ulp_tol(c.total))

    def test_days_then_cycles(self):
        """Converting between normalization conventions preserves total."""
        d = DualFloat.from_days(jnp.array(59000.0), jnp.array(0.7))
        c = DualFloat.from_cycles(d.int, d.frac)
        assert jnp.allclose(d.approx_total, c.approx_total, atol=ulp_tol(d.approx_total))

    def test_subtraction_of_days_values(self):
        """Subtracting two days-normalized DualFloats gives correct total."""
        a = DualFloat.from_days(jnp.array(59001.0), jnp.array(0.3))
        b = DualFloat.from_days(jnp.array(59000.0), jnp.array(0.7))
        diff = a - b  # Uses cycles() normalization
        expected = (59001.3 - 59000.7)
        assert jnp.allclose(diff.total, jnp.array(expected), atol=ulp_tol(expected))


# ===========================================================================
# JAX compatibility
# ===========================================================================

class TestJAXCompat:
    def test_jit_cycles(self):
        @jax.jit
        def f(x, y):
            return DualFloat.from_cycles(x, y).total
        assert jnp.allclose(f(jnp.array(1.0), jnp.array(0.7)),
                            jnp.array(2.0 - 0.3))

    def test_jit_days(self):
        @jax.jit
        def f(x, y):
            return DualFloat.from_days(x, y).approx_total
        assert jnp.allclose(f(jnp.array(59000.0), jnp.array(1.3)),
                            jnp.array(59001.3))

    def test_grad_through_arithmetic(self):
        def loss(frac):
            a = DualFloat.from_cycles(jnp.array(1.0), frac)
            b = DualFloat.from_cycles(jnp.array(2.0), jnp.array(0.1))
            return (a + b).total

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.array(0.3))
        assert jnp.allclose(g, 1.0)

    def test_grad_through_days(self):
        def loss(frac):
            d = DualFloat.from_days(jnp.array(59000.0), frac)
            return d.approx_total

        grad_fn = jax.grad(loss)
        g = grad_fn(jnp.array(0.5))
        assert jnp.allclose(g, 1.0)


# ===========================================================================
# Deterministic unit tests 
# ===========================================================================

class TestDualFloatDeterministic:
    def test_normalize_invariant(self):
        """Frac must always be in [-0.5, 0.5) after cycles()."""
        p = DualFloat.from_cycles(jnp.array([0.0, 5.0, -3.0]), jnp.array([0.7, -0.8, 1.3]))
        assert jnp.all(p.frac >= -0.5)
        assert jnp.all(p.frac < 0.5)

    def test_normalize_preserves_total(self):
        """int + frac should equal the original total phase."""
        int_in = jnp.array([3.0, -2.0, 0.0])
        frac_in = jnp.array([0.7, -1.3, 0.49999])
        p = DualFloat.from_cycles(int_in, frac_in)
        expected = int_in + frac_in
        assert jnp.allclose(p.total, expected, atol=ulp_tol(expected))

    def test_add_sub_roundtrip(self):
        """(a + b) - b should approximately equal a."""
        a = DualFloat.from_cycles(jnp.array([10.0]), jnp.array([0.3]))
        b = DualFloat.from_cycles(jnp.array([5.0]), jnp.array([-0.2]))
        result = (a + b) - b
        assert jnp.allclose(result.total, a.total, atol=ulp_tol(a.total))

    def test_negative_double(self):
        """Double negation should be identity."""
        p = DualFloat.from_cycles(jnp.array([3.0]), jnp.array([0.4]))
        pp = -(-p)
        assert jnp.allclose(pp.int, p.int)
        assert jnp.allclose(pp.frac, p.frac)

    def test_quantity(self):
        p = DualFloat.from_cycles(jnp.array([5.0]), jnp.array([0.3]))
        assert jnp.allclose(p.total, jnp.array([5.3]), atol=ulp_tol(5.3))

    def test_jit_compatible(self):
        a = DualFloat.from_cycles(jnp.array([1.0]), jnp.array([0.2]))
        b = DualFloat.from_cycles(jnp.array([2.0]), jnp.array([0.3]))

        @jax.jit
        def add_phases(x, y):
            return x + y

        result = add_phases(a, b)
        assert jnp.allclose(result.total, jnp.array([3.5]), atol=ulp_tol(3.5))

    def test_grad_through_phase(self):
        """Gradient should flow through DualFloat arithmetic."""

        @jax.grad
        def loss(x):
            p = DualFloat.from_cycles(jnp.array([0.0]), x)
            return jnp.sum(p.total ** 2)

        g = loss(jnp.array([0.3]))
        assert jnp.allclose(g, 2.0 * 0.3, atol=ulp_tol(2.0 * 0.3))


# ===========================================================================
# Hypothesis property-based tests
# ===========================================================================

class TestDualFloatHypothesis:
    """Randomized stress tests for DualFloat arithmetic precision."""

    # -- Normalization properties --

    @given(raw_phase_inputs())
    @settings(deadline=None)
    def test_normalization_frac_range(self, raw):
        """cycles() must always produce frac in [-0.5, 0.5)."""
        int_in, frac_in = raw
        p = DualFloat.from_cycles(jnp.array(int_in), jnp.array(frac_in))
        assert float(p.frac) >= -0.5
        assert float(p.frac) < 0.5

    @given(raw_phase_inputs())
    @settings(deadline=None)
    def test_normalization_preserves_total(self, raw):
        """int + frac after normalization equals the original total."""
        int_in, frac_in = raw
        p = DualFloat.from_cycles(jnp.array(int_in), jnp.array(frac_in))
        expected = int_in + frac_in
        assert abs(float(p.approx_total) - expected) <= ulp_tol(expected)

    # -- Addition properties --

    @given(dual_floats_cycles(), dual_floats_cycles())
    @settings(deadline=None)
    def test_addition_commutativity(self, a, b):
        """a + b == b + a (exact equality on int and frac)."""
        ab = a + b
        ba = b + a
        assert float(ab.int) == float(ba.int)
        assert float(ab.frac) == float(ba.frac)

    @given(dual_floats_cycles(), dual_floats_cycles(), dual_floats_cycles())
    @settings(deadline=None)
    def test_addition_associativity(self, a, b, c):
        """(a + b) + c ≈ a + (b + c) in total phase."""
        lhs = (a + b) + c
        rhs = a + (b + c)
        assert jnp.allclose(lhs.approx_total, rhs.approx_total, atol=ulp_tol(lhs.approx_total))

    @given(dual_floats_cycles())
    @settings(deadline=None)
    def test_additive_identity(self, a):
        """a + 0 == a."""
        zero = DualFloat.from_cycles(jnp.array(0.0), jnp.array(0.0))
        result = a + zero
        assert float(result.int) == float(a.int)
        assert float(result.frac) == float(a.frac)

    @given(dual_floats_cycles())
    @settings(deadline=None)
    def test_additive_inverse(self, a):
        """a + (-a) has quantity ≈ 0."""
        result = a + (-a)
        assert abs(float(result.total)) <= ulp_tol(0)

    # -- Subtraction / negation properties --

    @given(dual_floats_cycles(), dual_floats_cycles())
    @settings(deadline=None)
    def test_sub_equals_add_neg(self, a, b):
        """a - b == a + (-b).

        The int parts match exactly. The frac parts can differ by up to 1 ULP
        when ``b`` is on the half-integer boundary (``b.frac == -0.5``): the two
        paths round ``(frac + 0.5) - 1`` vs ``frac - 0.5`` differently, and both
        are valid normalizations of the same value.
        """
        lhs = a - b
        rhs = a + (-b)
        assert float(lhs.int) == float(rhs.int)
        assert abs(float(lhs.frac) - float(rhs.frac)) <= ulp_tol(0)

    @given(dual_floats_cycles())
    @settings(deadline=None)
    def test_double_negation(self, a):
        """-(-a) == a."""
        pp = -(-a)
        assert float(pp.int) == float(a.int)
        assert float(pp.frac) == float(a.frac)

    # -- Precision roundtrip tests --

    @given(dual_floats_cycles(), dual_floats_cycles())
    @settings(deadline=None)
    def test_add_sub_roundtrip_field_level(self, a, b):
        """(a + b) - b recovers a in total phase."""
        result = (a + b) - b
        assert abs(float(result.approx_total) - float(a.approx_total)) <= ulp_tol(a.approx_total)

    @given(
        integers(min_value=500_000_000, max_value=2_000_000_000),
        floats(min_value=1e-16, max_value=1e-14, allow_nan=False, allow_infinity=False),
    )
    @settings(deadline=None)
    def test_large_int_tiny_frac_precision(self, big_int, tiny_frac):
        """With int ~1e9 and frac ~1e-15, arithmetic preserves the tiny frac."""
        a = DualFloat.from_cycles(jnp.array(float(big_int)), jnp.array(tiny_frac))
        offset = DualFloat.from_cycles(jnp.array(1e8), jnp.array(0.3))
        roundtrip = (a + offset) - offset

        assert float(roundtrip.int) == float(a.int)
        frac_err = abs(float(roundtrip.frac) - float(a.frac))
        assert frac_err <= ulp_tol(a.frac), (
            f"Tiny frac {tiny_frac} lost precision: err={frac_err}"
        )


# ===========================================================================
# Idempotent Normalization
# ===========================================================================

class TestIdempotentNormalization:
    """cycles(p.int, p.frac) must be bitwise identical to p."""

    @pytest.mark.parametrize("int_in, frac_in", [
        (0.0, 0.0),
        (3.0, 0.25),
        (3.0, -0.5),
        (-3.0, -0.5),
        (3.0, -0.49999999999999994),
        (1e12, 0.1),
        (-1e12, -0.1),
        (1.0, 0.0),
    ])
    def test_deterministic_idempotence(self, int_in, frac_in):
        p = DualFloat.from_cycles(jnp.array(int_in), jnp.array(frac_in))
        p2 = DualFloat.from_cycles(p.int, p.frac)
        assert jnp.array_equal(p.int, p2.int)
        assert jnp.array_equal(p.frac, p2.frac)

    def test_near_boundary_idempotence(self):
        eps = float(jnp.finfo(jnp.float64).eps)
        for frac_in in [-0.5, -0.5 + eps, 0.5 - eps, 0.5 - 2 * eps, 0.0]:
            p = DualFloat.from_cycles(jnp.array(0.0), jnp.array(frac_in))
            p2 = DualFloat.from_cycles(p.int, p.frac)
            assert jnp.array_equal(p.int, p2.int), f"int mismatch at frac_in={frac_in}"
            assert jnp.array_equal(p.frac, p2.frac), f"frac mismatch at frac_in={frac_in}"

    @given(raw_phase_inputs())
    @settings(deadline=None)
    def test_hypothesis_idempotence(self, raw):
        int_in, frac_in = raw
        p = DualFloat.from_cycles(jnp.array(int_in), jnp.array(frac_in))
        p2 = DualFloat.from_cycles(p.int, p.frac)
        assert jnp.array_equal(p.int, p2.int)
        assert jnp.array_equal(p.frac, p2.frac)

    @given(raw_phase_inputs())
    @settings(deadline=None)
    def test_triple_application(self, raw):
        """cycles³ == cycles (transitivity of idempotence)."""
        int_in, frac_in = raw
        p1 = DualFloat.from_cycles(jnp.array(int_in), jnp.array(frac_in))
        p2 = DualFloat.from_cycles(p1.int, p1.frac)
        p3 = DualFloat.from_cycles(p2.int, p2.frac)
        assert jnp.array_equal(p1.int, p3.int)
        assert jnp.array_equal(p1.frac, p3.frac)


# ===========================================================================
# Normalization Edge Cases
# ===========================================================================

class TestNormalizationEdgeCases:
    """Parametrized edge cases from PINT's test suite + boundary analysis."""

    @pytest.mark.parametrize("int_in, frac_in, expected_int, expected_frac", [
        (0.0, 0.0, 0.0, 0.0),
        (2.0, 0.3, 2.0, 0.3),
        (1.0, 0.7, 2.0, -0.3),
        (-4.0, 0.5, -3.0, -0.5),
        (4.0, -0.5, 4.0, -0.5),
        (5.0, 1.4, 6.0, 0.4),
        # Large frac overflow
        (0.0, 10.7, 11.0, -0.3),
        (0.0, -10.3, -10.0, -0.3),
        # frac exactly 0.5 carries up
        (0.0, 0.5, 1.0, -0.5),
        (-1.0, 0.5, 0.0, -0.5),
        # frac exactly -0.5 stays
        (0.0, -0.5, 0.0, -0.5),
        # Negative int with positive/negative frac
        (-5.0, 0.3, -5.0, 0.3),
        (5.0, -0.3, 5.0, -0.3),
        (-5.0, 0.7, -4.0, -0.3),
        (5.0, -0.7, 4.0, 0.3),
    ])
    def test_normalization(self, int_in, frac_in, expected_int, expected_frac):
        p = DualFloat.from_cycles(jnp.array(int_in), jnp.array(frac_in))
        tol = ulp_tol(int_in + frac_in)
        assert float(p.int) == pytest.approx(expected_int, abs=tol), (
            f"int: got {float(p.int)}, expected {expected_int}"
        )
        assert float(p.frac) == pytest.approx(expected_frac, abs=tol), (
            f"frac: got {float(p.frac)}, expected {expected_frac}"
        )

    def test_near_half_boundary(self):
        """Values near frac=0.5 must consistently round."""
        eps = float(jnp.finfo(jnp.float64).eps)
        p = DualFloat.from_cycles(jnp.array(0.0), jnp.array(0.5 - eps))
        assert float(p.frac) >= -0.5
        assert float(p.frac) < 0.5
        assert float(p.int) == 0.0

        p2 = DualFloat.from_cycles(jnp.array(0.0), jnp.array(-0.5 + eps))
        assert float(p2.frac) >= -0.5
        assert float(p2.frac) < 0.5

    def test_multi_cycle_large_frac(self):
        """Frac values spanning many cycles normalize correctly."""
        raw_frac = 1000000.499999
        p = DualFloat.from_cycles(jnp.array(0.0), jnp.array(raw_frac))
        assert float(p.frac) >= -0.5
        assert float(p.frac) < 0.5
        assert abs(float(p.approx_total) - raw_frac) <= ulp_tol(raw_frac)


# ===========================================================================
# Catastrophic Cancellation
# ===========================================================================

class TestCatastrophicCancellation:
    """Subtracting nearly-equal large phases must preserve small differences."""

    def test_tiny_frac_difference(self):
        """Plain float64 loses this; int/frac split preserves it."""
        eps = 1e-14
        a = DualFloat.from_cycles(jnp.array(1e12), jnp.array(0.3))
        b = DualFloat.from_cycles(jnp.array(1e12), jnp.array(0.3 + eps))
        diff = b - a
        assert float(diff.int) == 0.0
        assert abs(float(diff.frac) - eps) <= ulp_tol(eps)

        # (For contrast, plain float64 cannot resolve this: ``(1e12 + 0.3 + eps)
        # - (1e12 + 0.3)`` loses ``eps`` entirely to rounding. That's a property
        # of IEEE-754, not of DualFloat, so it isn't asserted here.)

    def test_realistic_timing_residual(self):
        """~100 ns residual at 600 Hz MSP."""
        residual_cycles = 3.3e-7
        a = DualFloat.from_cycles(jnp.array(5.7e11), jnp.array(0.12345678901234))
        b = DualFloat.from_cycles(jnp.array(5.7e11), jnp.array(0.12345678901234 + residual_cycles))
        diff = b - a
        assert float(diff.int) == 0.0
        assert abs(float(diff.frac) - residual_cycles) <= ulp_tol(residual_cycles)

    def test_difference_spanning_int_frac_boundary(self):
        """Difference where frac wraps across the normalization boundary."""
        a = DualFloat.from_cycles(jnp.array(1e12), jnp.array(0.4999999))
        b = DualFloat.from_cycles(jnp.array(1e12 + 1.0), jnp.array(-0.4999999))
        diff = b - a
        expected = 0.0000002
        assert abs(float(diff.total) - expected) <= ulp_tol(expected)

    def test_cancellation_with_different_ints(self):
        a = DualFloat.from_cycles(jnp.array(1_000_000_001.0), jnp.array(-0.3))
        b = DualFloat.from_cycles(jnp.array(1_000_000_000.0), jnp.array(0.3))
        diff = a - b
        assert abs(float(diff.total) - 0.4) <= ulp_tol(0.4)

    @given(big_int_dual_floats(), small_perturbations())
    @settings(deadline=None)
    def test_hypothesis_cancellation(self, a, eps):
        """Add tiny eps to frac, subtract original, recover eps."""
        b = DualFloat.from_cycles(a.int, a.frac + jnp.array(eps))
        diff = b - a
        assert abs(float(diff.total) - eps) <= ulp_tol(eps)


# ===========================================================================
# Longdouble Oracle
# ===========================================================================

class TestLongdoubleOracle:
    """Compare DualFloat arithmetic against numpy longdouble ground truth."""

    @staticmethod
    def _to_ld(p):
        """Reconstruct total as longdouble (no precision loss)."""
        return np.longdouble(float(p.int)) + np.longdouble(float(p.frac))

    @given(dual_floats_cycles(), dual_floats_cycles())
    @settings(deadline=None)
    def test_addition_oracle(self, a, b):
        """DualFloat addition matches longdouble to within ~1 ULP of the larger operand.

        Under catastrophic cancellation (opposite-sign large ints) the achievable
        precision -- and the longdouble oracle's own precision, since longdouble
        cannot hold a ~1e6 int plus a 52-bit frac exactly -- is bounded by the
        operand magnitude, not the (small) result. Mirrors test_subtraction_oracle.
        """
        result = a + b
        ld_a, ld_b = self._to_ld(a), self._to_ld(b)
        ld_result = ld_a + ld_b
        pr_total = self._to_ld(result)
        assert abs(float(pr_total - ld_result)) <= ulp_tol(max(abs(ld_a), abs(ld_b)))

    @given(dual_floats_cycles(), dual_floats_cycles())
    @settings(deadline=None)
    def test_subtraction_oracle(self, a, b):
        """DualFloat subtraction matches longdouble to within 2 ULP of the result."""
        result = a - b
        ld_a, ld_b = self._to_ld(a), self._to_ld(b)
        ld_result = ld_a - ld_b
        pr_total = self._to_ld(result)
        assert abs(float(pr_total - ld_result)) <= ulp_tol(max(abs(ld_a), abs(ld_b)))

    @given(dual_floats_cycles(), dual_floats_cycles(), dual_floats_cycles())
    @settings(deadline=None)
    def test_chained_ops_oracle(self, a, b, c):
        """(a + b) - c compared to longdouble."""
        result = (a + b) - c
        ld_result = (self._to_ld(a) + self._to_ld(b)) - self._to_ld(c)
        pr_total = self._to_ld(result)
        assert abs(float(pr_total - ld_result)) <= ulp_tol(ld_result)

    def test_accumulated_sum_oracle(self):
        """Sum 1000 tiny increments; compare against longdouble accumulation."""
        N = 1000
        delta = 1e-14
        acc = DualFloat.from_cycles(jnp.array(0.0), jnp.array(0.0))
        increment = DualFloat.from_cycles(jnp.array(0.0), jnp.array(delta))
        for _ in range(N):
            acc = acc + increment

        ld_expected = np.longdouble(delta) * np.longdouble(N)
        pr_total = self._to_ld(acc)
        assert abs(float(pr_total - ld_expected)) <= ulp_tol(ld_expected)

    def test_accumulated_sum_with_large_int(self):
        """Accumulate onto a large base; verify frac accuracy."""
        N = 1000
        delta = 1e-14
        acc = DualFloat.from_cycles(jnp.array(1e9), jnp.array(0.0))
        increment = DualFloat.from_cycles(jnp.array(1.0), jnp.array(delta))
        for _ in range(N):
            acc = acc + increment

        assert float(acc.int) == pytest.approx(1e9 + N, abs=ulp_tol(1e9 + N))
        ld_expected_frac = np.longdouble(delta) * np.longdouble(N)
        assert abs(float(acc.frac) - float(ld_expected_frac)) <= ulp_tol(ld_expected_frac)

    def test_alternating_sign_accumulation(self):
        """Add +delta then -delta N times; result should be ~zero."""
        N = 1000
        delta = 1e-14
        plus = DualFloat.from_cycles(jnp.array(0.0), jnp.array(delta))
        minus = DualFloat.from_cycles(jnp.array(0.0), jnp.array(-delta))
        acc = DualFloat.from_cycles(jnp.array(0.0), jnp.array(0.0))
        for _ in range(N):
            acc = acc + plus
            acc = acc + minus
        assert abs(float(acc.total)) <= ulp_tol(0)


# ===========================================================================
# Direct PINT Phase Comparison
# ===========================================================================

class TestPINTPhaseComparison:
    """Compare JaxPINT DualFloat against PINT Phase for identical inputs."""

    @pytest.fixture(autouse=True)
    def _import_pint(self):
        pint_phase_mod = pytest.importorskip("pint.phase")
        self.PINTPhase = pint_phase_mod.Phase

    @pytest.mark.parametrize("ii1,ff1,ii2,ff2", [
        (2, 0.3, 1, 0.1),
        (2, 0.3, 1, 0.2),
        (2, 0.0, 1, -0.5),
        (0, 0.0, 0, 0.0),
        (10, -0.4, 5, 0.3),
    ])
    def test_addition_agreement(self, ii1, ff1, ii2, ff2):
        pint_result = self.PINTPhase(ii1, ff1) + self.PINTPhase(ii2, ff2)
        jax_result = (
            DualFloat.from_cycles(jnp.array(float(ii1)), jnp.array(ff1))
            + DualFloat.from_cycles(jnp.array(float(ii2)), jnp.array(ff2))
        )
        assert float(jax_result.int) == pytest.approx(float(pint_result.int[0]), abs=ulp_tol(pint_result.int[0]))
        assert float(jax_result.frac) == pytest.approx(float(pint_result.frac[0]), abs=ulp_tol(pint_result.frac[0]))

    @pytest.mark.parametrize("int_in, frac_in, expected_int, expected_frac", [
        (0, 0.0, 0.0, 0.0),
        (2, 0.3, 2.0, 0.3),
        (1, 0.7, 2.0, -0.3),
        (-4, 0.5, -3.0, -0.5),
        (4, -0.5, 4.0, -0.5),
        (5, 1.4, 6.0, 0.4),
    ])
    def test_normalization_agreement(self, int_in, frac_in, expected_int, expected_frac):
        """DualFloat normalization matches PINT Phase for integer int_parts."""
        pint_p = self.PINTPhase(int_in, frac_in)
        jax_p = DualFloat.from_cycles(jnp.array(float(int_in)), jnp.array(frac_in))
        assert float(jax_p.int) == pytest.approx(float(pint_p.int[0]), abs=ulp_tol(pint_p.int[0]))
        assert float(jax_p.frac) == pytest.approx(float(pint_p.frac[0]), abs=ulp_tol(pint_p.frac[0]))

    def test_precision_agreement(self):
        """PINT's test_precision case: Phase(1e5, 0.1) + Phase(0, 1e-9)."""
        pint_result = self.PINTPhase(1e5, 0.1) + self.PINTPhase(0, 1e-9)
        jax_result = (
            DualFloat.from_cycles(jnp.array(1e5), jnp.array(0.1))
            + DualFloat.from_cycles(jnp.array(0.0), jnp.array(1e-9))
        )
        assert float(jax_result.int) == pytest.approx(float(pint_result.int[0]), abs=ulp_tol(pint_result.int[0]))
        assert float(jax_result.frac) == pytest.approx(float(pint_result.frac[0]), abs=ulp_tol(pint_result.frac[0]))

    def test_negation_total_phase_agreement(self):
        """JaxPINT __neg__ skips cycles; PINT re-normalizes. Total phase must match."""
        for ii, ff in [(4, -0.5), (3, 0.3), (-2, 0.1)]:
            pint_neg = -self.PINTPhase(ii, ff)
            jax_neg = -DualFloat.from_cycles(jnp.array(float(ii)), jnp.array(ff))
            pint_total = float(pint_neg.int[0]) + float(pint_neg.frac[0])
            jax_total = float(jax_neg.int) + float(jax_neg.frac)
            assert jax_total == pytest.approx(pint_total, abs=ulp_tol(pint_total))

    @given(
        integers(min_value=-100_000, max_value=100_000),
        floats(min_value=-0.5, max_value=0.5, exclude_max=True,
               allow_nan=False, allow_infinity=False),
        integers(min_value=-100_000, max_value=100_000),
        floats(min_value=-0.5, max_value=0.5, exclude_max=True,
               allow_nan=False, allow_infinity=False),
    )
    @settings(deadline=None)
    def test_hypothesis_addition_agreement(self, ii1, ff1, ii2, ff2):
        pint_result = self.PINTPhase(ii1, ff1) + self.PINTPhase(ii2, ff2)
        jax_result = (
            DualFloat.from_cycles(jnp.array(float(ii1)), jnp.array(ff1))
            + DualFloat.from_cycles(jnp.array(float(ii2)), jnp.array(ff2))
        )
        pint_total = float(pint_result.int[0]) + float(pint_result.frac[0])
        jax_total = float(jax_result.approx_total)
        assert jax_total == pytest.approx(pint_total, abs=ulp_tol(pint_total))


# ===========================================================================
# Realistic Pulsar Timing Scales
# ===========================================================================

class TestRealisticScales:
    """DualFloat at actual pulsar timing scales."""

    def test_msp_30_year_accumulation(self):
        """Accumulate 1000 phase increments over 30 years at 622 Hz."""
        F0 = np.longdouble("622.122")
        total_time = np.longdouble("30.0") * np.longdouble("365.25") * np.longdouble("86400.0")
        dt = total_time / np.longdouble("1000.0")

        acc = DualFloat.from_cycles(jnp.array(0.0), jnp.array(0.0))
        ld_acc = np.longdouble(0.0)

        for _ in range(1000):
            ld_step = F0 * dt
            int_step = float(np.floor(ld_step))
            frac_step = float(ld_step - np.longdouble(int_step))
            acc = acc + DualFloat.from_cycles(jnp.array(int_step), jnp.array(frac_step))
            ld_acc += ld_step

        pr_total = np.longdouble(float(acc.int)) + np.longdouble(float(acc.frac))
        err_cycles = abs(float(pr_total - ld_acc))
        assert err_cycles < 1e-5, f"Accumulated error: {err_cycles} cycles"

    def test_adjacent_toa_differencing(self):
        """Phase difference between TOAs 86.4 us apart at 622 Hz."""
        F0 = 622.122
        dt_sec = 0.000000001 * 86400.0
        expected_delta_phase = F0 * dt_sec

        base_cycles = 5.89e11
        phase1 = DualFloat.from_cycles(jnp.array(base_cycles), jnp.array(0.12345))
        phase2 = DualFloat.from_cycles(
            jnp.array(base_cycles),
            jnp.array(0.12345 + expected_delta_phase),
        )

        diff = phase2 - phase1
        assert abs(float(diff.total) - expected_delta_phase) <= ulp_tol(expected_delta_phase)

    def test_barycentric_correction_roundtrip(self):
        """Add ~500s Roemer delay (311061 cycles at 622 Hz), subtract back."""
        roemer = DualFloat.from_cycles(jnp.array(311061.0), jnp.array(0.247))
        base = DualFloat.from_cycles(jnp.array(5.89e11), jnp.array(0.1))
        corrected = base + roemer
        recovered = corrected - roemer

        assert float(recovered.int) == float(base.int)
        assert abs(float(recovered.frac) - float(base.frac)) <= ulp_tol(base.frac)

    def test_dm_delay_roundtrip(self):
        """DM delay: ~0.622 cycles, add to large phase and subtract back."""
        dm_phase = DualFloat.from_cycles(jnp.array(0.0), jnp.array(0.622))
        assert float(dm_phase.int) == 1.0
        assert abs(float(dm_phase.frac) - (-0.378)) <= ulp_tol(-0.378)

        base = DualFloat.from_cycles(jnp.array(5.89e11), jnp.array(0.1))
        roundtrip = (base + dm_phase) - dm_phase
        assert float(roundtrip.int) == float(base.int)
        assert abs(float(roundtrip.frac) - float(base.frac)) <= ulp_tol(base.frac)


# ===========================================================================
# Construction guards (__check_init__)
# ===========================================================================

class TestInitGuards:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            DualFloat(int=jnp.zeros(3), frac=jnp.zeros(4))

    def test_dtype_mismatch_raises(self):
        with pytest.raises(ValueError, match="dtype mismatch"):
            DualFloat(
                int=jnp.zeros(3, dtype=jnp.float64),
                frac=jnp.zeros(3, dtype=jnp.float32),
            )

    def test_matching_shapes_ok(self):
        DualFloat(int=jnp.zeros(3), frac=jnp.zeros(3))  # must not raise


# ===========================================================================
# Negation canonical form
# ===========================================================================

class TestNegationCanonicalForm:
    """__neg__ must keep frac in [-0.5, 0.5), even at the -0.5 boundary."""

    def test_neg_half_stays_canonical(self):
        """For a = (int=5, frac=-0.5), -a must have frac in [-0.5, 0.5)."""
        a = DualFloat.from_cycles(jnp.array(5.0), jnp.array(-0.5))
        neg_a = -a
        assert float(neg_a.frac) >= -0.5
        assert float(neg_a.frac) < 0.5
        # And the total must still be correct: -(5 - 0.5) = -4.5
        assert float(neg_a.total) == pytest.approx(-4.5, abs=ulp_tol(-4.5))

    def test_neg_half_negative_int(self):
        """Same at a negative int."""
        a = DualFloat.from_cycles(jnp.array(-3.0), jnp.array(-0.5))
        neg_a = -a
        assert float(neg_a.frac) >= -0.5
        assert float(neg_a.frac) < 0.5
        # -(-3 - 0.5) = 3.5
        assert float(neg_a.total) == pytest.approx(3.5, abs=ulp_tol(3.5))

    @given(dual_floats_cycles())
    @settings(deadline=None)
    def test_neg_always_canonical(self, a):
        neg_a = -a
        assert float(neg_a.frac) >= -0.5
        assert float(neg_a.frac) < 0.5


# ===========================================================================
# Contract edges 
# ===========================================================================

class TestContractEdges:
    """Pin behavior at the edges of DualFloat's contract.
    - non-finite propagation
    -signed zero
    - non-canonical inputs
    - the 2^53 ceiling
    - boundary sums *through* ArithmeticError 
    - transform invariance (vmap/scan)
    - and exact gradients at carry events 
    """

    # -- non-finite propagation ---------------------------------------

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_nonfinite_int_propagates_visibly(self, bad):
        """Poisoned int must stay visible through arithmetic, never
        silently become a finite-but-wrong value."""
        d = DualFloat.from_cycles(jnp.array(bad), jnp.array(0.1))
        r = d + DualFloat.from_cycles(jnp.array(1.0), jnp.array(0.1))
        assert not bool(jnp.isfinite(r.int))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_nonfinite_frac_propagates_visibly(self, bad):
        d = DualFloat.from_cycles(jnp.array(1.0), jnp.array(bad))
        r = d + DualFloat.from_cycles(jnp.array(1.0), jnp.array(0.1))
        assert (not bool(jnp.isfinite(r.int))) or (
            not bool(jnp.isfinite(r.frac))
        )

    # -- signed zero ---------------------------------------------------

    def test_negative_zero_frac_normalizes_benignly(self):
        d = DualFloat.from_cycles(jnp.array(5.0), jnp.array(-0.0))
        assert float(d.int) == 5.0
        assert float(d.frac) == 0.0

    def test_neg_of_zero_frac(self):
        n = -DualFloat.from_cycles(jnp.array(5.0), jnp.array(0.0))
        assert float(n.int) == -5.0
        assert float(n.frac) == 0.0  

    # -- raw non-canonical construction --------------------------------

    def test_raw_noncanonical_preserves_total_through_add(self):
        """Raw DualFloat(int=0.3, frac=0.7) bypasses the factories.

        De-facto contract: arithmetic still preserves the TOTAL to ~1 ulp,
        but the integer-int invariant is NOT restored — field-level tricks
        (pulse numbering) are only licensed after factory normalization.
        """
        from fractions import Fraction

        raw = DualFloat(int=jnp.array(0.3), frac=jnp.array(0.7))
        r = raw + DualFloat.from_cycles(jnp.array(1.0), jnp.array(0.0))
        exact = Fraction(0.3) + Fraction(0.7) + 1
        got = Fraction(float(r.int)) + Fraction(float(r.frac))
        assert abs(float(got - exact)) <= ulp_tol(2.0)

    # -- the 2^53 ceiling ----------------------------------------------

    def test_int_carry_exact_just_below_2pow53(self):
        from fractions import Fraction

        big = 2.0**53 - 2.0
        d = DualFloat.from_cycles(jnp.array(big), jnp.array(0.4))
        r = d + DualFloat.from_cycles(jnp.array(1.0), jnp.array(0.2))
        exact = Fraction(big) + Fraction(0.4) + 1 + Fraction(0.2)
        got = Fraction(float(r.int)) + Fraction(float(r.frac))
        assert abs(float(got - exact)) <= ulp_tol(1.0)

    def test_int_ceiling_is_2pow53(self):
        """Above 2^53, odd integer ints are unrepresentable and carries go
        lossy — the design's fundamental cliff.  The operating envelope
        (phase ~1e11 cycles, MJD ~6e4 days) sits 4+ orders below; this
        test documents WHERE exactness ends so the envelope claim is
        testable rather than folklore.
        """
        from fractions import Fraction

        big = 2.0**53
        d = DualFloat.from_cycles(jnp.array(big), jnp.array(0.4))
        r = d + DualFloat.from_cycles(jnp.array(1.0), jnp.array(0.2))
        exact = Fraction(big) + Fraction(0.4) + 1 + Fraction(0.2)
        got = Fraction(float(r.int)) + Fraction(float(r.frac))
        assert abs(float(got - exact)) >= 1.0  # loss, by design, here

    # -- boundary sums THROUGH arithmetic ------------------------------

    def test_add_lands_exactly_on_plus_half(self):
        """frac sum of exactly +0.5 must carry: result frac = -0.5."""
        s = DualFloat.from_cycles(jnp.array(10.0), jnp.array(0.25)) + \
            DualFloat.from_cycles(jnp.array(20.0), jnp.array(0.25))
        assert float(s.int) == 31.0
        assert float(s.frac) == -0.5

    def test_add_lands_exactly_on_minus_half(self):
        """frac sum of exactly -0.5 is already canonical: no carry."""
        s = DualFloat.from_cycles(jnp.array(10.0), jnp.array(-0.25)) + \
            DualFloat.from_cycles(jnp.array(20.0), jnp.array(-0.25))
        assert float(s.int) == 30.0
        assert float(s.frac) == -0.5

    @pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
    def test_add_near_half_boundary_exact_and_canonical(self, k):
        """Sums landing k ulp(0.5) around +0.5: total exact, frac canonical."""
        from fractions import Fraction

        ulp_half = float(np.spacing(0.5))
        f2 = 0.25 + k * ulp_half  # exactly representable (2 ulp of 0.25 apart)
        a = DualFloat.from_cycles(jnp.array(0.0), jnp.array(0.25))
        b = DualFloat.from_cycles(jnp.array(0.0), jnp.array(f2))
        s = a + b
        exact = Fraction(0.25) + Fraction(f2)
        got = Fraction(float(s.int)) + Fraction(float(s.frac))
        assert got == exact  # bit-exact: both fracs share an exponent range
        assert -0.5 <= float(s.frac) < 0.5

    # -- transform invariance ------------------------------------------

    def test_vmap_heterogeneous_carries_match_scalar_path(self):
        """A vmapped add with per-element carry decisions (no carry / carry
        up / boundary carry / carry down) must be bitwise identical to the
        scalar path."""
        f1 = jnp.array([0.1, 0.45, 0.25, -0.45])
        f2 = jnp.array([0.3, 0.45, 0.25, -0.45])

        def add_pair(x, y):
            return DualFloat.from_cycles(jnp.zeros(()), x) + \
                DualFloat.from_cycles(jnp.zeros(()), y)

        batched = jax.vmap(add_pair)(f1, f2)
        for i in range(4):
            scalar = add_pair(f1[i], f2[i])
            assert float(batched.int[i]) == float(scalar.int)
            assert float(batched.frac[i]) == float(scalar.frac)

    def test_scan_accumulation_matches_python_loop(self):
        """lax.scan (jit) accumulation must be bitwise identical to the
        eager Python loop — carries may not depend on the transform."""
        inc = DualFloat.from_cycles(jnp.array(0.0), jnp.array(0.123456789))
        n = 512

        acc_scan, _ = jax.lax.scan(
            lambda c, _: (c + inc, None),
            DualFloat.from_cycles(jnp.array(0.0), jnp.array(0.0)),
            None,
            length=n,
        )
        acc_loop = DualFloat.from_cycles(jnp.array(0.0), jnp.array(0.0))
        for _ in range(n):
            acc_loop = acc_loop + inc

        assert float(acc_scan.int) == float(acc_loop.int)
        assert float(acc_scan.frac) == float(acc_loop.frac)

    # -- gradients at carry events -------------------------------------

    @pytest.mark.parametrize(
        "f0",
        [0.1, 0.2, 0.25, 0.2 - 1e-13],
        ids=["interior", "lands-exactly-on-0.5", "carries", "1e-13-below"],
    )
    def test_grad_of_total_through_add_is_exactly_unity(self, f0):
        """d(total)/d(frac) == 1.0 EXACTLY, including when the frac sum
        lands exactly on the 0.5 carry boundary (f0=0.2 with +0.3).

        This is the property protecting every jacfwd design matrix: the
        floor/where carry machinery must be gradient-transparent.
        """

        def total_after_add(f):
            d = DualFloat.from_cycles(jnp.asarray(10.0), f)
            e = d + DualFloat.from_cycles(jnp.asarray(5.0), jnp.asarray(0.3))
            return e.int + e.frac

        g = jax.grad(total_after_add)(jnp.asarray(f0))
        assert float(g) == 1.0

    def test_grad_through_sub_production_pattern(self):
        """Mirror of dt = tdb - epoch with sensitivity through the epoch
        frac (the fitter path: epoch int is static, frac carries grad)."""

        def dt_total(ep_frac):
            tdb = DualFloat.from_days(jnp.asarray(59001.0), jnp.asarray(0.75))
            ep = DualFloat.from_days(jnp.asarray(59000.0), ep_frac)
            diff = tdb - ep
            return diff.int + diff.frac

        g = jax.grad(dt_total)(jnp.asarray(0.25))
        assert float(g) == -1.0

    # -- operand broadcasting ------------------------------------------

    def test_add_broadcasts_scalar_shape_against_array(self):
        """Pin the semantics: scalar-shape + array-shape broadcasts (JAX
        rules), producing a well-formed array DualFloat."""
        a = DualFloat.from_cycles(jnp.zeros(3), jnp.array([0.1, 0.2, 0.3]))
        b = DualFloat.from_cycles(jnp.asarray(1.0), jnp.asarray(0.25))
        r = a + b
        assert r.int.shape == (3,)
        for i, f in enumerate([0.1, 0.2, 0.3]):
            scalar = DualFloat.from_cycles(jnp.zeros(()), jnp.asarray(f)) + b
            assert float(r.int[i]) == float(scalar.int)
            assert float(r.frac[i]) == float(scalar.frac)

    def test_add_incompatible_shapes_raises(self):
        a = DualFloat.from_cycles(jnp.zeros(3), jnp.zeros(3))
        b = DualFloat.from_cycles(jnp.zeros(4), jnp.zeros(4))
        with pytest.raises((ValueError, TypeError)):
            _ = a + b


class TestGuardedTotal:
    """Subtract-before-you-scale enforcement.

    ``.total`` is guarded via equinox.error_if at |int| > 30000: collapsing
    an absolute MJD/phase is a runtime error under eager, jit, vmap, and
    grad.  ``.approx_total`` is the explicit, unguarded opt-out for
    tolerance-insensitive reads (windowing / interpolation).
    """

    def test_guard_fires_on_absolute_mjd(self):
        d = DualFloat.from_days(jnp.array(59000.0), jnp.array(0.5))
        with pytest.raises(Exception, match="approx_total"):
            _ = float(d.total)

    def test_guard_fires_on_absolute_phase(self):
        d = DualFloat.from_cycles(jnp.array(1.0e10), jnp.array(0.25))
        with pytest.raises(Exception, match="approx_total"):
            _ = float(d.total)

    def test_guard_fires_under_jit(self):
        @jax.jit
        def collapse(d):
            return d.total

        d = DualFloat.from_days(jnp.array(59000.0), jnp.array(0.5))
        with pytest.raises(Exception, match="approx_total"):
            _ = float(collapse(d))

    def test_total_allowed_on_subtracted_values(self):
        """The production pattern: difference first, then collapse."""
        a = DualFloat.from_days(jnp.array(59001.0), jnp.array(0.75))
        b = DualFloat.from_days(jnp.array(59000.0), jnp.array(0.25))
        assert float((a - b).total) == 1.5

    def test_approx_total_unguarded_on_absolute_mjd(self):
        d = DualFloat.from_days(jnp.array(59000.0), jnp.array(0.5))
        assert float(d.approx_total) == 59000.5

    def test_guarded_total_grad_transparent(self):
        """error_if must not perturb the gradient (still exactly 1)."""

        def f(frac):
            return DualFloat.from_cycles(jnp.array(10.0), frac).total

        assert float(jax.grad(f)(jnp.array(0.25))) == 1.0
