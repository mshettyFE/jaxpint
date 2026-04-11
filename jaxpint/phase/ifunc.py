"""Interpolation function phase component (IFunc).

Ports PINT's ``IFunc`` class as a pure Equinox module.  The phase is
modelled by interpolating tabulated (MJD, delay) pairs and converting
to phase via F0:

    phase(t) = interp(t) * F0

Supports piecewise-constant (SIFUNC=0) and linear (SIFUNC=2) interpolation.

The control points are pre-extracted at bridge time and stored as fixed
arrays (not fittable parameters).

All derivatives w.r.t. F0 are handled by ``jax.jacobian`` through ``__call__``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import PhaseComponent
from jaxpint.constants import SECS_PER_DAY
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector


class IFunc(PhaseComponent):
    """Interpolation function model.

    Parameters
    ----------
    interp_type : int
        Interpolation type: 0 = piecewise constant, 2 = linear.
    control_mjds : array, shape (n_points,)
        MJD control-point times (sorted, ascending).
    control_delays : array, shape (n_points,)
        Delay values at control points (seconds).
    f0_name : str
        Name of the spin frequency parameter (default ``"F0"``).

    Raises
    ------
    ValueError
        If ``interp_type`` is not 0 or 2.
    ValueError
        If fewer than one control point is provided.
    """

    interp_type: int = eqx.field(static=True)
    control_mjds: tuple[float, ...] = eqx.field(static=True)
    control_delays: tuple[float, ...] = eqx.field(static=True)
    f0_name: str = eqx.field(static=True, default="F0")

    def __check_init__(self):
        if self.interp_type not in (0, 2):
            raise ValueError(f"IFunc interp_type must be 0 or 2, got {self.interp_type}")
        if len(self.control_mjds) < 1:
            raise ValueError("IFunc requires at least one control point")

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> DualFloat:
        """Compute IFunc phase contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data.
        params : ParameterVector
            Timing-model parameters containing F0.
        delay : array, shape (n_toas,)
            Accumulated signal delay from prior components in seconds.

        Returns
        -------
        DualFloat
            Phase contribution in cycles (int + frac split).
        """
        f0 = params.param_value(self.f0_name)
        t = toa_data.tdb.total - delay / SECS_PER_DAY

        mjds = jnp.array(self.control_mjds)
        delays = jnp.array(self.control_delays)

        if self.interp_type == 0:
            # Piecewise constant: use nearest preceding control point
            idx = jnp.searchsorted(mjds, t, side="right") - 1
            idx = jnp.clip(idx, 0, len(self.control_mjds) - 1)
            interp_delay = delays[idx]
        else:
            # Linear interpolation
            interp_delay = jnp.interp(t, mjds, delays)

        phase = interp_delay * f0

        return DualFloat.cycles(jnp.zeros(toa_data.n_toas), phase)
