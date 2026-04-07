"""Piecewise spindown phase component.

Ports PINT's ``PiecewiseSpindown`` class as a pure Equinox module.  The
phase is modelled as a Taylor expansion within user-defined time bins:

    phase(t) = Σ_n [ PWPH_n + PWF0_n*dt + PWF1_n*dt^2/2! + PWF2_n*dt^3/3! ]
               for t in [PWSTART_n, PWSTOP_n)  where dt = t - PWEP_n

All derivatives are handled by ``jax.jacobian`` through ``__call__``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import PhaseComponent
from jaxpint.constants import SECS_PER_DAY
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import taylor_horner


class PiecewiseSpindown(PhaseComponent):
    """Piecewise Taylor-expansion spindown model.

    Parameters
    ----------
    n_pieces : int
        Number of piecewise segments.
    pwstart_names : tuple[str, ...]
        Names of segment start epoch parameters (MJD).
    pwstop_names : tuple[str, ...]
        Names of segment stop epoch parameters (MJD).
    pwep_names : tuple[str, ...]
        Names of segment reference epoch parameters (MJD).
    pwph_names : tuple[str, ...]
        Names of segment phase offset parameters (dimensionless cycles).
    pwf0_names : tuple[str, ...]
        Names of segment frequency parameters (Hz).
    pwf1_names : tuple[str, ...]
        Names of segment frequency derivative parameters (Hz/s).
    pwf2_names : tuple[str, ...]
        Names of segment second derivative parameters (Hz/s^2).
    """

    n_pieces: int = eqx.field(static=True)
    pwstart_names: tuple[str, ...] = eqx.field(static=True)
    pwstop_names: tuple[str, ...] = eqx.field(static=True)
    pwep_names: tuple[str, ...] = eqx.field(static=True)
    pwph_names: tuple[str, ...] = eqx.field(static=True)
    pwf0_names: tuple[str, ...] = eqx.field(static=True)
    pwf1_names: tuple[str, ...] = eqx.field(static=True)
    pwf2_names: tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self):
        if self.n_pieces < 1:
            raise ValueError("PiecewiseSpindown requires at least one piece")
        for attr in (
            "pwstart_names", "pwstop_names", "pwep_names",
            "pwph_names", "pwf0_names", "pwf1_names", "pwf2_names",
        ):
            if len(getattr(self, attr)) != self.n_pieces:
                raise ValueError(
                    f"Length of {attr} ({len(getattr(self, attr))}) "
                    f"does not match n_pieces ({self.n_pieces})"
                )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> DualFloat:
        """Compute piecewise spindown phase contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data.
        params : ParameterVector
            Timing-model parameters.
        delay : array, shape (n_toas,)
            Accumulated signal delay from prior components in seconds.

        Returns
        -------
        DualFloat
            Phase contribution in cycles (int + frac split).
        """
        toa_tdb = toa_data.tdb.total
        phase = jnp.zeros(toa_data.n_toas)

        for i in range(self.n_pieces):
            # Segment boundaries
            start = params.epoch_dual(self.pwstart_names[i]).total
            stop = params.epoch_dual(self.pwstop_names[i]).total
            affected = (toa_tdb >= start) & (toa_tdb < stop)

            # Time since segment epoch (DualFloat precision)
            ep = params.epoch_dual(self.pwep_names[i])
            dt = (toa_data.tdb - ep).total * SECS_PER_DAY - delay

            # Taylor coefficients: [PWPH, PWF0, PWF1, PWF2]
            pwph = params.param_value(self.pwph_names[i])
            pwf0 = params.param_value(self.pwf0_names[i])
            pwf1 = params.param_value(self.pwf1_names[i])
            pwf2 = params.param_value(self.pwf2_names[i])
            coeffs = jnp.array([pwph, pwf0, pwf1, pwf2])

            piece_phase = taylor_horner(dt, coeffs)
            phase = phase + jnp.where(affected, piece_phase, 0.0)

        return DualFloat.cycles(jnp.zeros(toa_data.n_toas), phase)
