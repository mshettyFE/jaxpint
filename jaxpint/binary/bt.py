"""Blandford & Teukolsky (1976) binary delay model.

The simplest binary model: Roemer delay with a relativistic correction
factor.  No Shapiro delay.

Reference
---------
Blandford & Teukolsky (1976), ApJ, 205, 580-591, eq. 2.33.
PINT ``stand_alone_psr_binaries/BT_model.py``.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent
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


class BinaryBT(DelayComponent):
    """Blandford-Teukolsky binary delay model.

    All hand-coded derivatives are omitted; ``jax.jacobian`` through
    ``__call__`` replaces PINT's ``d_BTdelay_d_*`` functions.

    Parameters
    ----------
    pb_name : str
        Name of binary period parameter (days).
    t0_name : str
        Name of periastron epoch parameter (MJD, int/frac split).
    a1_name : str
        Name of projected semi-major axis parameter (light-seconds).
    ecc_name : str
        Name of eccentricity parameter (dimensionless).
    om_name : str
        Name of longitude of periastron parameter (radians in ParameterVector).
    pbdot_name, omdot_name, edot_name, a1dot_name, gamma_name, xpbdot_name :
        Optional parameter names for secular derivatives.  ``None`` disables.
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

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute BT binary delay.

        Returns
        -------
        array, shape (n_toas,)
            Binary delay in seconds.
        """
        # --- Extract parameters ---
        pb_d = params.param_value(self.pb_name)       # days
        t0 = params.epoch_dual(self.t0_name)
        a1_ls = params.param_value(self.a1_name)      # light-seconds
        ecc0 = params.param_value(self.ecc_name)
        om_rad = params.param_value(self.om_name)     # radians

        pbdot = params.param_value_or(self.pbdot_name)
        omdot = params.param_value_or(self.omdot_name)  # deg/yr
        edot = params.param_value_or(self.edot_name)
        a1dot = params.param_value_or(self.a1dot_name)
        gamma = params.param_value_or(self.gamma_name)
        xpbdot = params.param_value_or(self.xpbdot_name)

        # --- Compute time since periastron (corrected for accumulated delay) ---
        tt0_s = compute_tt0(toa_data.tdb, t0, delay=delay)

        # --- Time-dependent orbital elements ---
        ecc = compute_ecc(ecc0, edot, tt0_s)
        a1 = compute_a1(a1_ls, a1dot, tt0_s)  # light-seconds = seconds
        omega = compute_omega_bt(om_rad, omdot, tt0_s)

        # --- Solve Kepler's equation ---
        M = compute_orbital_phase(
            toa_data.tdb, t0,
            pb_d, pbdot, xpbdot, delay=delay,
        )
        E = compute_eccentric_anomaly(ecc, M)

        sinE = jnp.sin(E)
        cosE = jnp.cos(E)
        sin_omega = jnp.sin(omega)
        cos_omega = jnp.cos(omega)
        sqrt_1me2 = jnp.sqrt(1.0 - ecc ** 2)

        # --- BT delay formula ---
        # L1 = a1 * sin(omega) * (cos(E) - ecc)
        L1 = a1 * sin_omega * (cosE - ecc)

        # L2 = (a1 * cos(omega) * sqrt(1-ecc^2) + GAMMA) * sin(E)
        L2 = (a1 * cos_omega * sqrt_1me2 + gamma) * sinE

        # Relativistic correction factor R
        # R = 1 - 2*pi * (a1*cos(omega)*sqrt(1-ecc^2)*cos(E) - a1*sin(omega)*sin(E))
        #         / ((1 - ecc*cos(E)) * PB_s)
        # BT uses instantaneous period pb() = PB + PBDOT*tt0
        pb_s = (pb_d + pbdot * tt0_s / SECS_PER_DAY) * SECS_PER_DAY
        num = a1 * cos_omega * sqrt_1me2 * cosE - a1 * sin_omega * sinE
        den = 1.0 - ecc * cosE
        R = 1.0 - 2.0 * jnp.pi * num / (den * pb_s)

        return (L1 + L2) * R
