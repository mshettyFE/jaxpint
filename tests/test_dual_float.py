"""Tests for DualFloat: cycles/days normalization, arithmetic, precision."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, assume, settings
from hypothesis.strategies import composite, floats, integers

from jaxpint.dual_float import DualFloat

# Uniform tolerance: 2 ULP of float64 at the scale of the expected value
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


@composite
def raw_phase_inputs(draw):
    """Generate arbitrary (int, frac) inputs that may need normalization."""
    int_part = float(draw(integers(min_value=-1_000_000, max_value=1_000_000)))
    frac_part = draw(floats(min_value=-100.0, max_value=100.0,
                            allow_nan=False, allow_infinity=False))
    return int_part, frac_part


def scalars():
    """Non-zero, finite scalars for multiplication tests."""
    return floats(min_value=-1000.0, max_value=1000.0,
                  allow_nan=False, allow_infinity=False).filter(lambda x: abs(x) > 1e-10)


@composite
def big_int_dual_floats(draw):
    """DualFloat with int in [1e9, 1e12] range (realistic pulsar scales)."""
    int_part = float(draw(integers(min_value=1_000_000_000, max_value=1_000_000_000_000)))
    frac_part = draw(floats(min_value=-0.5, max_value=0.5, exclude_max=True,
                            allow_nan=False, allow_infinity=False))
    return DualFloat.cycles(jnp.array(int_part), jnp.array(frac_part))


@composite
def small_perturbations(draw):
    """Tiny floats in [1e-15, 1e-10] for cancellation tests."""
    return draw(floats(min_value=1e-15, max_value=1e-10,
                       allow_nan=False, allow_infinity=False))


# ===========================================================================
# DualFloat.days() normalization
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
# DualFloat.cycles() normalization
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


# ===========================================================================
# Deterministic unit tests (from original PhaseResult tests)
# ===========================================================================

class TestDualFloatDeterministic:
    def test_normalize_invariant(self):
        """Frac must always be in [-0.5, 0.5) after cycles()."""
        p = DualFloat.cycles(jnp.array([0.0, 5.0, -3.0]), jnp.array([0.7, -0.8, 1.3]))
        assert jnp.all(p.frac >= -0.5)
        assert jnp.all(p.frac < 0.5)

    def test_normalize_preserves_total(self):
        """int + frac should equal the original total phase."""
        int_in = jnp.array([3.0, -2.0, 0.0])
        frac_in = jnp.array([0.7, -1.3, 0.49999])
        p = DualFloat.cycles(int_in, frac_in)
        expected = int_in + frac_in
        assert jnp.allclose(p.total, expected, atol=ulp_tol(expected))

    def test_add_sub_roundtrip(self):
        """(a + b) - b should approximately equal a."""
        a = DualFloat.cycles(jnp.array([10.0]), jnp.array([0.3]))
        b = DualFloat.cycles(jnp.array([5.0]), jnp.array([-0.2]))
        result = (a + b) - b
        assert jnp.allclose(result.total, a.total, atol=ulp_tol(a.total))

    def test_mul_scalar(self):
        p = DualFloat.cycles(jnp.array([1.0]), jnp.array([0.25]))
        doubled = p * 2.0
        assert jnp.allclose(doubled.total, jnp.array([2.5]), atol=ulp_tol(2.5))

    def test_rmul(self):
        p = DualFloat.cycles(jnp.array([1.0]), jnp.array([0.25]))
        doubled = 2.0 * p
        assert jnp.allclose(doubled.total, jnp.array([2.5]), atol=ulp_tol(2.5))

    def test_negative_double(self):
        """Double negation should be identity."""
        p = DualFloat.cycles(jnp.array([3.0]), jnp.array([0.4]))
        pp = -(-p)
        assert jnp.allclose(pp.int, p.int)
        assert jnp.allclose(pp.frac, p.frac)

    def test_quantity(self):
        p = DualFloat.cycles(jnp.array([5.0]), jnp.array([0.3]))
        assert jnp.allclose(p.total, jnp.array([5.3]), atol=ulp_tol(5.3))

    def test_jit_compatible(self):
        a = DualFloat.cycles(jnp.array([1.0]), jnp.array([0.2]))
        b = DualFloat.cycles(jnp.array([2.0]), jnp.array([0.3]))

        @jax.jit
        def add_phases(x, y):
            return x + y

        result = add_phases(a, b)
        assert jnp.allclose(result.total, jnp.array([3.5]), atol=ulp_tol(3.5))

    def test_grad_through_phase(self):
        """Gradient should flow through DualFloat arithmetic."""

        @jax.grad
        def loss(x):
            p = DualFloat.cycles(jnp.array([0.0]), x)
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
        p = DualFloat.cycles(jnp.array(int_in), jnp.array(frac_in))
        assert float(p.frac) >= -0.5
        assert float(p.frac) < 0.5

    @given(raw_phase_inputs())
    @settings(deadline=None)
    def test_normalization_preserves_total(self, raw):
        """int + frac after normalization equals the original total."""
        int_in, frac_in = raw
        p = DualFloat.cycles(jnp.array(int_in), jnp.array(frac_in))
        expected = int_in + frac_in
        assert abs(float(p.total) - expected) <= ulp_tol(expected)

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
        assert jnp.allclose(lhs.total, rhs.total, atol=ulp_tol(lhs.total))

    @given(dual_floats_cycles())
    @settings(deadline=None)
    def test_additive_identity(self, a):
        """a + 0 == a."""
        zero = DualFloat.cycles(jnp.array(0.0), jnp.array(0.0))
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
        """a - b == a + (-b)."""
        lhs = a - b
        rhs = a + (-b)
        assert float(lhs.int) == float(rhs.int)
        assert float(lhs.frac) == float(rhs.frac)

    @given(dual_floats_cycles())
    @settings(deadline=None)
    def test_double_negation(self, a):
        """-(-a) == a."""
        pp = -(-a)
        assert float(pp.int) == float(a.int)
        assert float(pp.frac) == float(a.frac)

    # -- Scalar multiplication properties --

    @given(dual_floats_cycles(), scalars())
    @settings(deadline=None)
    def test_scalar_mul_distributes(self, a, s):
        """s * (a + b) ≈ s*a + s*b for a random b."""
        b = DualFloat.cycles(jnp.array(42.0), jnp.array(0.123))
        lhs = s * (a + b)
        rhs = (s * a) + (s * b)
        assert jnp.allclose(lhs.total, rhs.total, atol=ulp_tol(lhs.total))

    @given(dual_floats_cycles())
    @settings(deadline=None)
    def test_mul_by_one_is_identity(self, a):
        """1 * a == a."""
        result = 1.0 * a
        assert float(result.int) == float(a.int)
        assert float(result.frac) == float(a.frac)

    # -- Precision roundtrip tests --

    @given(dual_floats_cycles(), dual_floats_cycles())
    @settings(deadline=None)
    def test_add_sub_roundtrip_field_level(self, a, b):
        """(a + b) - b recovers a in total phase."""
        result = (a + b) - b
        assert abs(float(result.total) - float(a.total)) <= ulp_tol(a.total)

    @given(
        integers(min_value=500_000_000, max_value=2_000_000_000),
        floats(min_value=1e-16, max_value=1e-14, allow_nan=False, allow_infinity=False),
    )
    @settings(deadline=None)
    def test_large_int_tiny_frac_precision(self, big_int, tiny_frac):
        """With int ~1e9 and frac ~1e-15, arithmetic preserves the tiny frac."""
        a = DualFloat.cycles(jnp.array(float(big_int)), jnp.array(tiny_frac))
        offset = DualFloat.cycles(jnp.array(1e8), jnp.array(0.3))
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
        p = DualFloat.cycles(jnp.array(int_in), jnp.array(frac_in))
        p2 = DualFloat.cycles(p.int, p.frac)
        assert jnp.array_equal(p.int, p2.int)
        assert jnp.array_equal(p.frac, p2.frac)

    def test_near_boundary_idempotence(self):
        eps = float(jnp.finfo(jnp.float64).eps)
        for frac_in in [-0.5, -0.5 + eps, 0.5 - eps, 0.5 - 2 * eps, 0.0]:
            p = DualFloat.cycles(jnp.array(0.0), jnp.array(frac_in))
            p2 = DualFloat.cycles(p.int, p.frac)
            assert jnp.array_equal(p.int, p2.int), f"int mismatch at frac_in={frac_in}"
            assert jnp.array_equal(p.frac, p2.frac), f"frac mismatch at frac_in={frac_in}"

    @given(raw_phase_inputs())
    @settings(deadline=None)
    def test_hypothesis_idempotence(self, raw):
        int_in, frac_in = raw
        p = DualFloat.cycles(jnp.array(int_in), jnp.array(frac_in))
        p2 = DualFloat.cycles(p.int, p.frac)
        assert jnp.array_equal(p.int, p2.int)
        assert jnp.array_equal(p.frac, p2.frac)

    @given(raw_phase_inputs())
    @settings(deadline=None)
    def test_triple_application(self, raw):
        """cycles³ == cycles (transitivity of idempotence)."""
        int_in, frac_in = raw
        p1 = DualFloat.cycles(jnp.array(int_in), jnp.array(frac_in))
        p2 = DualFloat.cycles(p1.int, p1.frac)
        p3 = DualFloat.cycles(p2.int, p2.frac)
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
        p = DualFloat.cycles(jnp.array(int_in), jnp.array(frac_in))
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
        p = DualFloat.cycles(jnp.array(0.0), jnp.array(0.5 - eps))
        assert float(p.frac) >= -0.5
        assert float(p.frac) < 0.5
        assert float(p.int) == 0.0

        p2 = DualFloat.cycles(jnp.array(0.0), jnp.array(-0.5 + eps))
        assert float(p2.frac) >= -0.5
        assert float(p2.frac) < 0.5

    def test_multi_cycle_large_frac(self):
        """Frac values spanning many cycles normalize correctly."""
        raw_frac = 1000000.499999
        p = DualFloat.cycles(jnp.array(0.0), jnp.array(raw_frac))
        assert float(p.frac) >= -0.5
        assert float(p.frac) < 0.5
        assert abs(float(p.total) - raw_frac) <= ulp_tol(raw_frac)


# ===========================================================================
# Catastrophic Cancellation
# ===========================================================================

class TestCatastrophicCancellation:
    """Subtracting nearly-equal large phases must preserve small differences."""

    def test_tiny_frac_difference(self):
        """Plain float64 loses this; int/frac split preserves it."""
        eps = 1e-14
        a = DualFloat.cycles(jnp.array(1e12), jnp.array(0.3))
        b = DualFloat.cycles(jnp.array(1e12), jnp.array(0.3 + eps))
        diff = b - a
        assert float(diff.int) == 0.0
        assert abs(float(diff.frac) - eps) <= ulp_tol(eps)

        # Demonstrate that plain float64 cannot do this
        a_f64 = 1e12 + 0.3
        b_f64 = 1e12 + 0.3 + eps
        diff_f64 = b_f64 - a_f64
        assert abs(diff_f64 - eps) > eps * 0.01, (
            "float64 should lose precision here"
        )

    def test_realistic_timing_residual(self):
        """~100 ns residual at 600 Hz MSP."""
        residual_cycles = 3.3e-7
        a = DualFloat.cycles(jnp.array(5.7e11), jnp.array(0.12345678901234))
        b = DualFloat.cycles(jnp.array(5.7e11), jnp.array(0.12345678901234 + residual_cycles))
        diff = b - a
        assert float(diff.int) == 0.0
        assert abs(float(diff.frac) - residual_cycles) <= ulp_tol(residual_cycles)

    def test_difference_spanning_int_frac_boundary(self):
        """Difference where frac wraps across the normalization boundary."""
        a = DualFloat.cycles(jnp.array(1e12), jnp.array(0.4999999))
        b = DualFloat.cycles(jnp.array(1e12 + 1.0), jnp.array(-0.4999999))
        diff = b - a
        expected = 0.0000002
        assert abs(float(diff.total) - expected) <= ulp_tol(expected)

    def test_cancellation_with_different_ints(self):
        a = DualFloat.cycles(jnp.array(1_000_000_001.0), jnp.array(-0.3))
        b = DualFloat.cycles(jnp.array(1_000_000_000.0), jnp.array(0.3))
        diff = a - b
        assert abs(float(diff.total) - 0.4) <= ulp_tol(0.4)

    @given(big_int_dual_floats(), small_perturbations())
    @settings(deadline=None)
    def test_hypothesis_cancellation(self, a, eps):
        """Add tiny eps to frac, subtract original, recover eps."""
        b = DualFloat.cycles(a.int, a.frac + jnp.array(eps))
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
        """DualFloat addition matches longdouble to within 1 ULP of the result."""
        result = a + b
        ld_result = self._to_ld(a) + self._to_ld(b)
        pr_total = self._to_ld(result)
        assert abs(float(pr_total - ld_result)) <= ulp_tol(ld_result)

    @given(dual_floats_cycles(), dual_floats_cycles())
    @settings(deadline=None)
    def test_subtraction_oracle(self, a, b):
        """DualFloat subtraction matches longdouble to within 2 ULP of the result."""
        result = a - b
        ld_a, ld_b = self._to_ld(a), self._to_ld(b)
        ld_result = ld_a - ld_b
        pr_total = self._to_ld(result)
        assert abs(float(pr_total - ld_result)) <= ulp_tol(max(abs(ld_a), abs(ld_b)))

    @given(dual_floats_cycles(), scalars())
    @settings(deadline=None)
    def test_scalar_mul_oracle(self, a, s):
        result = a * s
        ld_result = self._to_ld(a) * np.longdouble(s)
        pr_total = self._to_ld(result)
        assert abs(float(pr_total - ld_result)) <= ulp_tol(ld_result)

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
        acc = DualFloat.cycles(jnp.array(0.0), jnp.array(0.0))
        increment = DualFloat.cycles(jnp.array(0.0), jnp.array(delta))
        for _ in range(N):
            acc = acc + increment

        ld_expected = np.longdouble(delta) * np.longdouble(N)
        pr_total = self._to_ld(acc)
        assert abs(float(pr_total - ld_expected)) <= ulp_tol(ld_expected)

    def test_accumulated_sum_with_large_int(self):
        """Accumulate onto a large base; verify frac accuracy."""
        N = 1000
        delta = 1e-14
        acc = DualFloat.cycles(jnp.array(1e9), jnp.array(0.0))
        increment = DualFloat.cycles(jnp.array(1.0), jnp.array(delta))
        for _ in range(N):
            acc = acc + increment

        assert float(acc.int) == pytest.approx(1e9 + N, abs=ulp_tol(1e9 + N))
        ld_expected_frac = np.longdouble(delta) * np.longdouble(N)
        assert abs(float(acc.frac) - float(ld_expected_frac)) <= ulp_tol(ld_expected_frac)

    def test_alternating_sign_accumulation(self):
        """Add +delta then -delta N times; result should be ~zero."""
        N = 1000
        delta = 1e-14
        plus = DualFloat.cycles(jnp.array(0.0), jnp.array(delta))
        minus = DualFloat.cycles(jnp.array(0.0), jnp.array(-delta))
        acc = DualFloat.cycles(jnp.array(0.0), jnp.array(0.0))
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
            DualFloat.cycles(jnp.array(float(ii1)), jnp.array(ff1))
            + DualFloat.cycles(jnp.array(float(ii2)), jnp.array(ff2))
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
        jax_p = DualFloat.cycles(jnp.array(float(int_in)), jnp.array(frac_in))
        assert float(jax_p.int) == pytest.approx(float(pint_p.int[0]), abs=ulp_tol(pint_p.int[0]))
        assert float(jax_p.frac) == pytest.approx(float(pint_p.frac[0]), abs=ulp_tol(pint_p.frac[0]))

    def test_precision_agreement(self):
        """PINT's test_precision case: Phase(1e5, 0.1) + Phase(0, 1e-9)."""
        pint_result = self.PINTPhase(1e5, 0.1) + self.PINTPhase(0, 1e-9)
        jax_result = (
            DualFloat.cycles(jnp.array(1e5), jnp.array(0.1))
            + DualFloat.cycles(jnp.array(0.0), jnp.array(1e-9))
        )
        assert float(jax_result.int) == pytest.approx(float(pint_result.int[0]), abs=ulp_tol(pint_result.int[0]))
        assert float(jax_result.frac) == pytest.approx(float(pint_result.frac[0]), abs=ulp_tol(pint_result.frac[0]))

    def test_negation_total_phase_agreement(self):
        """JaxPINT __neg__ skips cycles; PINT re-normalizes. Total phase must match."""
        for ii, ff in [(4, -0.5), (3, 0.3), (-2, 0.1)]:
            pint_neg = -self.PINTPhase(ii, ff)
            jax_neg = -DualFloat.cycles(jnp.array(float(ii)), jnp.array(ff))
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
            DualFloat.cycles(jnp.array(float(ii1)), jnp.array(ff1))
            + DualFloat.cycles(jnp.array(float(ii2)), jnp.array(ff2))
        )
        pint_total = float(pint_result.int[0]) + float(pint_result.frac[0])
        jax_total = float(jax_result.total)
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

        acc = DualFloat.cycles(jnp.array(0.0), jnp.array(0.0))
        ld_acc = np.longdouble(0.0)

        for _ in range(1000):
            ld_step = F0 * dt
            int_step = float(np.floor(ld_step))
            frac_step = float(ld_step - np.longdouble(int_step))
            acc = acc + DualFloat.cycles(jnp.array(int_step), jnp.array(frac_step))
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
        phase1 = DualFloat.cycles(jnp.array(base_cycles), jnp.array(0.12345))
        phase2 = DualFloat.cycles(
            jnp.array(base_cycles),
            jnp.array(0.12345 + expected_delta_phase),
        )

        diff = phase2 - phase1
        assert abs(float(diff.total) - expected_delta_phase) <= ulp_tol(expected_delta_phase)

    def test_barycentric_correction_roundtrip(self):
        """Add ~500s Roemer delay (311061 cycles at 622 Hz), subtract back."""
        roemer = DualFloat.cycles(jnp.array(311061.0), jnp.array(0.247))
        base = DualFloat.cycles(jnp.array(5.89e11), jnp.array(0.1))
        corrected = base + roemer
        recovered = corrected - roemer

        assert float(recovered.int) == float(base.int)
        assert abs(float(recovered.frac) - float(base.frac)) <= ulp_tol(base.frac)

    def test_dm_delay_roundtrip(self):
        """DM delay: ~0.622 cycles, add to large phase and subtract back."""
        dm_phase = DualFloat.cycles(jnp.array(0.0), jnp.array(0.622))
        assert float(dm_phase.int) == 1.0
        assert abs(float(dm_phase.frac) - (-0.378)) <= ulp_tol(-0.378)

        base = DualFloat.cycles(jnp.array(5.89e11), jnp.array(0.1))
        roundtrip = (base + dm_phase) - dm_phase
        assert float(roundtrip.int) == float(base.int)
        assert abs(float(roundtrip.frac) - float(base.frac)) <= ulp_tol(base.frac)
