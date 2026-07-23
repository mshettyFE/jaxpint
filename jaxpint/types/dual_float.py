"""DualFloat: Faux long double via integer + fractional part split.

Represents a value as the sum of an integer part and a fractional part,
each stored in float64. This gives ~30 digits of precision for values
that would otherwise lose low-order bits in a single float64.

Two normalization conventions are supported via factory methods:
- ``DualFloat.from_cycles()``: frac in [-0.5, 0.5), for phase (cycles)
- ``DualFloat.from_days()``:   frac in [0, 1),      for MJD / time (days)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

# Boundary between "safe to collapse" (subtracted values: day/cycle
# differences, |int| <= ~2e4 even for 55-yr baselines) and "must
# difference first" (absolute MJD >= ~4e4; absolute phase >= 1e9).
# Try to think about weather you care about precision loss when
# choosing between the .total and .approx_total
_TOTAL_GUARD_THRESHOLD = 30_000.0


class DualFloat(eqx.Module):
    """A value split into integer and fractional parts for extended precision.

    Both fields are float64 JAX arrays of the same shape. The fractional
    part's range depends on which factory method was used to create the
    instance, but arithmetic always normalizes to [-0.5, 0.5) (cycles
    convention).
    """

    int: Float[Array, "..."]
    frac: Float[Array, "..."]

    def __check_init__(self):
        if self.int.shape != self.frac.shape:
            raise ValueError(
                f"DualFloat int/frac shape mismatch: "
                f"{self.int.shape} vs {self.frac.shape}"
            )
        if self.int.dtype != self.frac.dtype:
            raise ValueError(
                f"DualFloat int/frac dtype mismatch: "
                f"{self.int.dtype} vs {self.frac.dtype}"
            )

    @staticmethod
    def from_cycles(
        int_part: Float[Array, "..."], frac_part: Float[Array, "..."]
    ) -> DualFloat:
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
        remainder = frac_part - fl  # in [0, 1), precise
        carry = jnp.where(remainder >= 0.5, fl + 1.0, fl)
        ff = frac_part - carry
        ii = int_part + carry

        return DualFloat(int=ii, frac=ff)

    @staticmethod
    def from_days(
        int_part: Float[Array, "..."], frac_part: Float[Array, "..."]
    ) -> DualFloat:
        """Create a DualFloat, normalizing frac to [0, 1).

        Suitable for MJD / time values (days). Overflow from the
        fractional part is carried into the integer part.

        .. note::
            Arithmetic operators (``+``, ``-``, ``*``) always renormalize
            to the cycles convention (``frac in [-0.5, 0.5)``). If you
            need days-form output after arithmetic, call
            ``DualFloat.from_days(result.int, result.frac)`` explicitly.
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
        """Collapse int + frac into a single float64 array — GUARDED.

        Raises (via :func:`equinox.error_if`; jit/vmap/grad-safe) when
        ``|int| > 30000``, where the collapse discards low-order bits of
        ``frac`` (error ~ulp(int)).  For tolerance-insensitive reads of absolute values
        (window membership, interpolation grids) use
        :attr:`approx_total`, which is unguarded by explicit contract.
        """
        guarded_int = eqx.error_if(
            self.int,
            jnp.any(jnp.abs(self.int) > _TOTAL_GUARD_THRESHOLD),
            "DualFloat.total on |int| > 3e4: collapsing an absolute "
            "MJD/phase discards the precision the int/frac split exists "
            "to preserve.  Difference first ((a - b).total), or use "
            ".approx_total for tolerance-insensitive uses "
            "(windowing / interpolation).",
        )
        return guarded_int + self.frac

    @property
    def approx_total(self) -> Float[Array, "..."]:
        """``int + frac`` WITHOUT the large-int guard.

        Precision is ~ulp(int): ~6e-12 days (~0.5 us) at MJD 59000.
        For tolerance-insensitive uses only — window-membership tests,
        interpolation lookups, year fractions.  Anything feeding a
        residual or phase must difference first and use :attr:`total`.
        """
        return self.int + self.frac

    # -- Arithmetic --

    def __add__(self, other: DualFloat) -> DualFloat:
        return DualFloat.from_cycles(self.int + other.int, self.frac + other.frac)

    def __sub__(self, other: DualFloat) -> DualFloat:
        return DualFloat.from_cycles(self.int - other.int, self.frac - other.frac)

    def __neg__(self) -> DualFloat:
        # Plain negation gives frac in (-0.5, 0.5]; the value 0.5 is out
        # of canonical range. Patch the boundary by carrying when frac
        # == -0.5: total -> -total = -int + 0.5 -> (-int + 1) + (-0.5).
        on_boundary = self.frac == -0.5
        new_int = jnp.where(on_boundary, -self.int + 1.0, -self.int)
        new_frac = jnp.where(
            on_boundary, jnp.asarray(-0.5, dtype=self.frac.dtype), -self.frac
        )
        return DualFloat(int=new_int, frac=new_frac)

    # NOTE: scalar multiplication (__mul__/__rmul__) was removed
    # No production code ever needed it: the pipeline's discipline is
    # "subtract before you scale" (dt = tdb - epoch, then dt.total * s on
    # the small-int result), and the one precision-critical scaling
    # (phase accumulation) lives in taylor_horner_phase's compensated
    # Horner, which operates on the int/frac fields directly.
