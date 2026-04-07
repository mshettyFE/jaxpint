"""DualFloat: Faux long double via integer + fractional part split.

Represents a value as the sum of an integer part and a fractional part,
each stored in float64. This gives ~30 digits of precision for values
that would otherwise lose low-order bits in a single float64.

Two normalization conventions are supported via factory methods:
- ``DualFloat.cycles()``: frac in [-0.5, 0.5), for phase (cycles)
- ``DualFloat.days()``:   frac in [0, 1),      for MJD / time (days)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class DualFloat(eqx.Module):
    """A value split into integer and fractional parts for extended precision.

    Both fields are float64 JAX arrays of the same shape. The fractional
    part's range depends on which factory method was used to create the
    instance, but arithmetic always normalizes to [-0.5, 0.5) (cycles
    convention).
    """

    int: Float[Array, "..."]
    frac: Float[Array, "..."]

    # -- Factory methods (normalization) --

    @staticmethod
    def cycles(int_part: Float[Array, "..."], frac_part: Float[Array, "..."]) -> DualFloat:
        """Create a DualFloat, normalizing frac to [-0.5, 0.5).

        Suitable for phase (cycles). Assumes ``int_part`` holds integer
        values. If ``frac_part`` is outside [-0.5, 0.5), the overflow is
        carried into the integer part.
        """
        int_part = jnp.asarray(int_part, dtype=jnp.float64)
        frac_part = jnp.asarray(frac_part, dtype=jnp.float64)

        # Carry overflow from frac into int.
        # Avoid ``floor(frac + 0.5)`` — the intermediate addition loses
        # precision near half-integers, making normalization non-idempotent.
        # Instead compute the fractional remainder of floor directly.
        fl = jnp.floor(frac_part)
        remainder = frac_part - fl          # in [0, 1), precise
        carry = jnp.where(remainder >= 0.5, fl + 1.0, fl)
        ff = frac_part - carry
        ii = int_part + carry

        return DualFloat(int=ii, frac=ff)

    @staticmethod
    def days(int_part: Float[Array, "..."], frac_part: Float[Array, "..."]) -> DualFloat:
        """Create a DualFloat, normalizing frac to [0, 1).

        Suitable for MJD / time values (days). Overflow from the
        fractional part is carried into the integer part.
        """
        int_part = jnp.asarray(int_part, dtype=jnp.float64)
        frac_part = jnp.asarray(frac_part, dtype=jnp.float64)

        carry = jnp.floor(frac_part)
        ff = frac_part - carry
        ii = int_part + carry

        return DualFloat(int=ii, frac=ff)

    # -- Properties --

    @property
    def total(self) -> Float[Array, "..."]:
        """Collapse int + frac into a single float64 array.

        Safe when ``int`` is small (e.g. phase residuals, time differences
        of a few days). Unsafe when ``int`` is large (e.g. absolute MJD
        ~60000 or absolute phase ~10^10 cycles) — the addition discards
        the low-order bits of ``frac``.
        """
        return self.int + self.frac

    # -- Arithmetic --

    def __add__(self, other: DualFloat) -> DualFloat:
        return DualFloat.cycles(self.int + other.int, self.frac + other.frac)

    def __sub__(self, other: DualFloat) -> DualFloat:
        return DualFloat.cycles(self.int - other.int, self.frac - other.frac)

    def __neg__(self) -> DualFloat:
        # Bypass normalization so that negation is exact and a - b == a + (-b)
        # by construction (both paths feed identical values into cycles).
        # The only out-of-range value this can produce is frac = 0.5
        # (when the input had frac = -0.5); cycles normalizes it on the
        # next arithmetic operation.
        return DualFloat(int=-self.int, frac=-self.frac)

    def __mul__(self, scalar) -> DualFloat:
        scalar = jnp.asarray(scalar, dtype=jnp.float64)
        return DualFloat.cycles(self.int * scalar, self.frac * scalar)

    def __rmul__(self, scalar) -> DualFloat:
        return self.__mul__(scalar)

    # Backward-compatible alias: PhaseResult.create() still works
    create = cycles
