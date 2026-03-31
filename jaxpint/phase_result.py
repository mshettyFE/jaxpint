"""PhaseResult: Pulse phase as integer + fractional parts."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


class PhaseResult(eqx.Module):
    """Pulse phase as integer and fractional parts.

    Mirrors PINT's ``Phase`` class. The fractional part is always in [-0.5, 0.5),
    giving a unique representation. Both fields are float64 JAX arrays of the
    same shape.

    Units: dimensionless (cycles).

    NOTE: Open interval is meant to avoid cycle ambiguity; -0.5 to 0.5 is arbitrary though. Any interval of length 1 works

    """

    int: Float[Array, "..."]
    frac: Float[Array, "..."]

    @staticmethod
    def create(int_part: Float[Array, "..."], frac_part: Float[Array, "..."]) -> PhaseResult:
        """Create a PhaseResult, normalizing frac to [-0.5, 0.5).

        Assumes ``int_part`` holds integer values. If ``frac_part`` is outside
        [-0.5, 0.5), the overflow is carried into the integer part.
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

        return PhaseResult(int=ii, frac=ff)

    @property
    def quantity(self) -> Float[Array, "..."]:
        """Collapse int + frac into a single float64 array.

        Safe to call when ``int`` is small (e.g. phase residuals where the
        integer pulse count has been subtracted). In that regime the sum
        fits comfortably in float64 with no precision loss.

        Unsafe when ``int`` is large (e.g. absolute pulse phase accumulated
        over decades -- ~10^10 cycles). The addition discards the low-order
        bits of ``frac``, defeating the purpose of the int/frac split.
        """
        return self.int + self.frac

    def __add__(self, other: PhaseResult) -> PhaseResult:
        return PhaseResult.create(self.int + other.int, self.frac + other.frac)

    def __sub__(self, other: PhaseResult) -> PhaseResult:
        return PhaseResult.create(self.int - other.int, self.frac - other.frac)

    def __neg__(self) -> PhaseResult:
        # Bypass create so that negation is exact and a - b == a + (-b)
        # by construction (both paths feed identical values into create).
        # The only out-of-range value this can produce is frac = 0.5
        # (when the input had frac = -0.5); create normalizes it on the
        # next arithmetic operation.
        return PhaseResult(int=-self.int, frac=-self.frac)

    def __mul__(self, scalar) -> PhaseResult:
        scalar = jnp.asarray(scalar, dtype=jnp.float64)
        return PhaseResult.create(self.int * scalar, self.frac * scalar)

    def __rmul__(self, scalar) -> PhaseResult:
        return self.__mul__(scalar)
