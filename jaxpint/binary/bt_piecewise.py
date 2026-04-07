"""Piecewise Blandford & Teukolsky (1976) binary delay model.

Extends BT with per-piece T0 and A1 values over non-overlapping
time intervals.  Each piece defines [XR1, XR2) with its own T0X/A1X.

Reference
---------
Blandford & Teukolsky (1976), ApJ, 205, 580-591.
PINT ``stand_alone_psr_binaries/BT_piecewise.py``.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector
from jaxpint.constants import SECS_PER_DAY
from jaxpint.binary.common import (
    compute_tt0,
    compute_orbital_phase,
    compute_eccentric_anomaly,
    compute_ecc,
    compute_a1,
    compute_omega_bt,
)


class BinaryBTPiecewise(DelayComponent):
    """Piecewise BT binary delay model.

    Allows different T0 and A1 values for non-overlapping time intervals.
    The number of pieces and their parameter names are static (set at
    construction time from the parfile).  TOAs outside all defined pieces
    use the global T0 and A1.

    Parameters
    ----------
    n_pieces : int
        Number of piecewise intervals.
    t0x_names : tuple of str
        Parameter names for piecewise T0 values (epoch parameters).
    a1x_names : tuple of str
        Parameter names for piecewise A1 values.
    xr1_names, xr2_names : tuple of str
        Parameter names for piece lower/upper boundaries (MJD, stored
        as regular parameters in ParameterVector).
    """

    pb_name: str = eqx.field(static=True, default="PB")
    t0_name: str = eqx.field(static=True, default="T0")
    a1_name: str = eqx.field(static=True, default="A1")
    ecc_name: str = eqx.field(static=True, default="ECC")
    om_name: str = eqx.field(static=True, default="OM")

    pbdot_name: Optional[str] = eqx.field(static=True, default=None)
    omdot_name: Optional[str] = eqx.field(static=True, default=None)
    edot_name: Optional[str] = eqx.field(static=True, default=None)
    a1dot_name: Optional[str] = eqx.field(static=True, default=None)
    gamma_name: Optional[str] = eqx.field(static=True, default=None)
    xpbdot_name: Optional[str] = eqx.field(static=True, default=None)

    # Piecewise configuration
    n_pieces: int = eqx.field(static=True, default=0)
    t0x_names: tuple[str, ...] = eqx.field(static=True, default=())
    a1x_names: tuple[str, ...] = eqx.field(static=True, default=())
    xr1_names: tuple[str, ...] = eqx.field(static=True, default=())
    xr2_names: tuple[str, ...] = eqx.field(static=True, default=())

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        # --- Extract global parameters ---
        pb_d = params.param_value(self.pb_name)
        t0 = params.epoch_dual(self.t0_name)
        a1_ls = params.param_value(self.a1_name)
        ecc0 = params.param_value(self.ecc_name)
        om_rad = params.param_value(self.om_name)

        pbdot = params.param_value_or(self.pbdot_name)
        omdot = params.param_value_or(self.omdot_name)
        edot = params.param_value_or(self.edot_name)
        a1dot = params.param_value_or(self.a1dot_name)
        gamma = params.param_value_or(self.gamma_name)
        xpbdot = params.param_value_or(self.xpbdot_name)

        n_toas = toa_data.n_toas
        toa_mjd = toa_data.tdb.total

        # --- Build per-TOA T0 and A1 from piecewise intervals ---
        # Start with global values
        t0_int_per_toa = jnp.full(n_toas, t0.int)
        t0_frac_per_toa = jnp.full(n_toas, t0.frac)
        a1_per_toa = jnp.full(n_toas, a1_ls)

        for i in range(self.n_pieces):
            xr1 = params.param_value(self.xr1_names[i])
            xr2 = params.param_value(self.xr2_names[i])
            in_piece = (toa_mjd >= xr1) & (toa_mjd < xr2)

            if self.t0x_names and i < len(self.t0x_names):
                t0x = params.epoch_dual(self.t0x_names[i])
                t0_int_per_toa = jnp.where(in_piece, t0x.int, t0_int_per_toa)
                t0_frac_per_toa = jnp.where(in_piece, t0x.frac, t0_frac_per_toa)

            if self.a1x_names and i < len(self.a1x_names):
                a1x = params.param_value(self.a1x_names[i])
                a1_per_toa = jnp.where(in_piece, a1x, a1_per_toa)

        # --- Compute time since (piecewise) T0 (corrected for accumulated delay) ---
        dt = toa_data.tdb - DualFloat(int=t0_int_per_toa, frac=t0_frac_per_toa)
        tt0_s = dt.total * SECS_PER_DAY - delay

        # --- Time-dependent orbital elements ---
        ecc = compute_ecc(ecc0, edot, tt0_s)
        a1 = compute_a1(a1_per_toa, a1dot, tt0_s)
        omega = compute_omega_bt(om_rad, omdot, tt0_s)

        # --- Solve Kepler's equation ---
        # Use the piecewise T0 for orbital phase computation
        t0_per_toa = DualFloat(int=t0_int_per_toa, frac=t0_frac_per_toa)
        M = _compute_orbital_phase_piecewise(
            toa_data.tdb, t0_per_toa,
            pb_d, pbdot, xpbdot, delay=delay,
        )
        E = compute_eccentric_anomaly(ecc, M)

        sinE = jnp.sin(E)
        cosE = jnp.cos(E)
        sin_omega = jnp.sin(omega)
        cos_omega = jnp.cos(omega)
        sqrt_1me2 = jnp.sqrt(1.0 - ecc ** 2)

        # --- BT delay formula ---
        L1 = a1 * sin_omega * (cosE - ecc)
        L2 = (a1 * cos_omega * sqrt_1me2 + gamma) * sinE

        pb_s = (pb_d + pbdot * tt0_s / SECS_PER_DAY) * SECS_PER_DAY
        num = a1 * cos_omega * sqrt_1me2 * cosE - a1 * sin_omega * sinE
        den = 1.0 - ecc * cosE
        R = 1.0 - 2.0 * jnp.pi * num / (den * pb_s)

        return (L1 + L2) * R


def _compute_orbital_phase_piecewise(
    tdb, epoch, pb_d, pbdot, xpbdot,
    delay=None,
):
    """Orbital phase with per-TOA epoch (vectorized version of compute_orbital_phase)."""
    dt = tdb - epoch
    dt_int_days = dt.int
    dt_frac_days = dt.frac
    if delay is not None:
        dt_frac_days = dt_frac_days - delay / SECS_PER_DAY

    n_orbits = jnp.floor(dt_int_days / pb_d)
    rem_int_days = dt_int_days - n_orbits * pb_d

    rem_days = rem_int_days + dt_frac_days
    extra = jnp.floor(rem_days / pb_d)
    rem_days = rem_days - extra * pb_d

    frac_orbit = rem_days / pb_d

    tt0_s = (dt_int_days + dt_frac_days) * SECS_PER_DAY
    pb_s = pb_d * SECS_PER_DAY
    ratio = tt0_s / pb_s
    pbdot_corr = -0.5 * (pbdot + xpbdot) * ratio ** 2

    frac_total = frac_orbit + pbdot_corr
    frac_total = frac_total - jnp.floor(frac_total)

    return 2.0 * jnp.pi * frac_total
