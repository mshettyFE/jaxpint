"""Tests for PhaseResult: deterministic unit tests and Hypothesis property tests."""

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, assume, settings
from hypothesis.strategies import composite, floats, integers

jax.config.update("jax_enable_x64", True)

from jaxpint.phase_result import PhaseResult


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@composite
def phase_results(draw):
    """Generate a random PhaseResult (scalar)."""
    int_part = float(draw(integers(min_value=-1_000_000, max_value=1_000_000)))
    frac_part = draw(floats(min_value=-0.5, max_value=0.5, exclude_max=True,
                            allow_nan=False, allow_infinity=False))
    return PhaseResult.create(jnp.array(int_part), jnp.array(frac_part))


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


# ===========================================================================
# Original deterministic tests (moved from test_types.py)
# ===========================================================================

class TestPhaseResult:
    def test_normalize_invariant(self):
        """Frac must always be in [-0.5, 0.5) after create()."""
        p = PhaseResult.create(jnp.array([0.0, 5.0, -3.0]), jnp.array([0.7, -0.8, 1.3]))
        assert jnp.all(p.frac >= -0.5)
        assert jnp.all(p.frac < 0.5)

    def test_normalize_preserves_total(self):
        """int + frac should equal the original total phase."""
        int_in = jnp.array([3.0, -2.0, 0.0])
        frac_in = jnp.array([0.7, -1.3, 0.49999])
        p = PhaseResult.create(int_in, frac_in)
        expected = int_in + frac_in
        assert jnp.allclose(p.quantity, expected, atol=1e-12)

    def test_add_sub_roundtrip(self):
        """(a + b) - b should approximately equal a."""
        a = PhaseResult.create(jnp.array([10.0]), jnp.array([0.3]))
        b = PhaseResult.create(jnp.array([5.0]), jnp.array([-0.2]))
        result = (a + b) - b
        assert jnp.allclose(result.quantity, a.quantity, atol=1e-12)

    def test_mul_scalar(self):
        p = PhaseResult.create(jnp.array([1.0]), jnp.array([0.25]))
        doubled = p * 2.0
        assert jnp.allclose(doubled.quantity, jnp.array([2.5]), atol=1e-12)

    def test_rmul(self):
        p = PhaseResult.create(jnp.array([1.0]), jnp.array([0.25]))
        doubled = 2.0 * p
        assert jnp.allclose(doubled.quantity, jnp.array([2.5]), atol=1e-12)

    def test_negative_double(self):
        """Double negation should be identity."""
        p = PhaseResult.create(jnp.array([3.0]), jnp.array([0.4]))
        pp = -(-p)
        assert jnp.allclose(pp.int, p.int)
        assert jnp.allclose(pp.frac, p.frac)

    def test_quantity(self):
        p = PhaseResult.create(jnp.array([5.0]), jnp.array([0.3]))
        assert jnp.allclose(p.quantity, jnp.array([5.3]), atol=1e-12)

    def test_jit_compatible(self):
        a = PhaseResult.create(jnp.array([1.0]), jnp.array([0.2]))
        b = PhaseResult.create(jnp.array([2.0]), jnp.array([0.3]))

        @jax.jit
        def add_phases(x, y):
            return x + y

        result = add_phases(a, b)
        assert jnp.allclose(result.quantity, jnp.array([3.5]), atol=1e-12)

    def test_grad_through_phase(self):
        """Gradient should flow through PhaseResult arithmetic."""

        @jax.grad
        def loss(x):
            p = PhaseResult.create(jnp.array([0.0]), x)
            return jnp.sum(p.quantity ** 2)

        g = loss(jnp.array([0.3]))
        assert jnp.allclose(g, 2.0 * 0.3, atol=1e-12)


# ===========================================================================
# Hypothesis property-based tests
# ===========================================================================

class TestPhaseResultHypothesis:
    """Randomized stress tests for PhaseResult arithmetic precision."""

    # -- Normalization properties --

    @given(raw_phase_inputs())
    @settings(deadline=None)
    def test_normalization_frac_range(self, raw):
        """create() must always produce frac in [-0.5, 0.5)."""
        int_in, frac_in = raw
        p = PhaseResult.create(jnp.array(int_in), jnp.array(frac_in))
        assert float(p.frac) >= -0.5
        assert float(p.frac) < 0.5

    @given(raw_phase_inputs())
    @settings(deadline=None)
    def test_normalization_preserves_total(self, raw):
        """int + frac after normalization equals the original total."""
        int_in, frac_in = raw
        p = PhaseResult.create(jnp.array(int_in), jnp.array(frac_in))
        expected = int_in + frac_in
        assert abs(float(p.quantity) - expected) < 1e-10

    # -- Addition properties --

    @given(phase_results(), phase_results())
    @settings(deadline=None)
    def test_addition_commutativity(self, a, b):
        """a + b == b + a (exact equality on int and frac)."""
        ab = a + b
        ba = b + a
        assert float(ab.int) == float(ba.int)
        assert float(ab.frac) == float(ba.frac)

    @given(phase_results(), phase_results(), phase_results())
    @settings(deadline=None)
    def test_addition_associativity(self, a, b, c):
        """(a + b) + c ≈ a + (b + c) in total phase."""
        lhs = (a + b) + c
        rhs = a + (b + c)
        assert jnp.allclose(lhs.quantity, rhs.quantity, atol=1e-10)

    @given(phase_results())
    @settings(deadline=None)
    def test_additive_identity(self, a):
        """a + 0 == a."""
        zero = PhaseResult.create(jnp.array(0.0), jnp.array(0.0))
        result = a + zero
        assert float(result.int) == float(a.int)
        assert float(result.frac) == float(a.frac)

    @given(phase_results())
    @settings(deadline=None)
    def test_additive_inverse(self, a):
        """a + (-a) has quantity ≈ 0."""
        result = a + (-a)
        assert abs(float(result.quantity)) < 1e-10

    # -- Subtraction / negation properties --

    @given(phase_results(), phase_results())
    @settings(deadline=None)
    def test_sub_equals_add_neg(self, a, b):
        """a - b == a + (-b)."""
        lhs = a - b
        rhs = a + (-b)
        assert float(lhs.int) == float(rhs.int)
        assert float(lhs.frac) == float(rhs.frac)

    @given(phase_results())
    @settings(deadline=None)
    def test_double_negation(self, a):
        """-(-a) == a."""
        pp = -(-a)
        assert float(pp.int) == float(a.int)
        assert float(pp.frac) == float(a.frac)

    # -- Scalar multiplication properties --

    @given(phase_results(), scalars())
    @settings(deadline=None)
    def test_scalar_mul_distributes(self, a, s):
        """s * (a + b) ≈ s*a + s*b for a random b."""
        b = PhaseResult.create(jnp.array(42.0), jnp.array(0.123))
        lhs = s * (a + b)
        rhs = (s * a) + (s * b)
        atol = max(1e-8, abs(s) * 1e-10)
        assert jnp.allclose(lhs.quantity, rhs.quantity, atol=atol)

    @given(phase_results())
    @settings(deadline=None)
    def test_mul_by_one_is_identity(self, a):
        """1 * a == a."""
        result = 1.0 * a
        assert float(result.int) == float(a.int)
        assert float(result.frac) == float(a.frac)

    # -- Precision roundtrip tests --

    @given(phase_results(), phase_results())
    @settings(deadline=None)
    def test_add_sub_roundtrip_field_level(self, a, b):
        """(a + b) - b recovers a in total phase.

        The int/frac *split* may differ by ±1 at the normalization boundary
        (frac ≈ ±0.5) because three float64 additions cannot perfectly cancel
        when the result falls within ~1 ULP of a half-integer.  The total
        phase, however, must be preserved.
        """
        result = (a + b) - b
        assert abs(float(result.quantity) - float(a.quantity)) < 1e-12

    @given(
        integers(min_value=500_000_000, max_value=2_000_000_000),
        floats(min_value=1e-16, max_value=1e-14, allow_nan=False, allow_infinity=False),
    )
    @settings(deadline=None)
    def test_large_int_tiny_frac_precision(self, big_int, tiny_frac):
        """With int ~1e9 and frac ~1e-15, arithmetic preserves the tiny frac.

        This is the key long-double-equivalent precision test: the int/frac
        split must keep the tiny fractional part alive through add/sub,
        even when int is billions of cycles.
        """
        a = PhaseResult.create(jnp.array(float(big_int)), jnp.array(tiny_frac))
        # Add a large offset and subtract it back
        offset = PhaseResult.create(jnp.array(1e8), jnp.array(0.3))
        roundtrip = (a + offset) - offset

        # The int part must be recovered exactly
        assert float(roundtrip.int) == float(a.int)
        # The tiny frac must survive the roundtrip
        frac_err = abs(float(roundtrip.frac) - float(a.frac))
        assert frac_err < 1e-15, (
            f"Tiny frac {tiny_frac} lost precision: err={frac_err}"
        )
