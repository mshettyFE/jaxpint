"""Damour & Deruelle (1986) binary delay model and variants.

Implements the DD, DDS, and DDH binary models as pure Equinox modules.
The DD model computes three delay components:
  1. Inverse timing delay (Roemer + Einstein, corrected to 2nd order)
  2. Shapiro delay (gravitational time dilation from companion)
  3. Aberration delay (A0/B0 terms)

Reference
---------
T. Damour and N. Deruelle (1986), Ann. Inst. H. Poincaré, 44, 263.
PINT ``stand_alone_psr_binaries/DD_model.py``.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent
from jaxpint.types import TOAData, ParameterVector
from jaxpint.constants import SECS_PER_DAY, TSUN
from jaxpint.binary.common import (
    compute_tt0,
    compute_orbital_phase,
    compute_orbits_pb,
    compute_eccentric_anomaly,
    compute_ecc,
    compute_a1,
    compute_true_anomaly,
    compute_omega_dd,
    dd_inverse_timing,
    dd_shapiro_delay,
    dd_aberration_delay,
)


class BinaryDD(DelayComponent):
    """Damour-Deruelle binary delay model.

    All hand-coded derivatives are omitted; ``jax.jacobian`` through
    ``__call__`` replaces PINT's ``d_DDdelay_d_*`` functions.

    Supports three Shapiro delay parameterizations via ``shapiro_mode``:

    - ``"standard"`` (DD): Uses ``SINI`` and ``M2`` directly.
    - ``"shapmax"`` (DDS): Uses ``SHAPMAX = -ln(1 - sin(i))``.
    - ``"h3stigma"`` (DDH): Uses ``H3`` and ``STIGMA``.
    """

    pb_name: str = eqx.field(static=True, default="PB")
    t0_name: str = eqx.field(static=True, default="T0")
    a1_name: str = eqx.field(static=True, default="A1")
    ecc_name: str = eqx.field(static=True, default="ECC")
    om_name: str = eqx.field(static=True, default="OM")

    # Optional secular derivatives
    pbdot_name: Optional[str] = eqx.field(static=True, default=None)
    omdot_name: Optional[str] = eqx.field(static=True, default=None)
    edot_name: Optional[str] = eqx.field(static=True, default=None)
    a1dot_name: Optional[str] = eqx.field(static=True, default=None)
    xpbdot_name: Optional[str] = eqx.field(static=True, default=None)

    # DD-specific parameters
    gamma_name: Optional[str] = eqx.field(static=True, default=None)
    dr_name: Optional[str] = eqx.field(static=True, default=None)
    dth_name: Optional[str] = eqx.field(static=True, default=None)
    a0_name: Optional[str] = eqx.field(static=True, default=None)
    b0_name: Optional[str] = eqx.field(static=True, default=None)

    # Shapiro delay parameters (mode-dependent)
    shapiro_mode: str = eqx.field(static=True, default="standard")
    m2_name: Optional[str] = eqx.field(static=True, default=None)
    sini_name: Optional[str] = eqx.field(static=True, default=None)
    shapmax_name: Optional[str] = eqx.field(static=True, default=None)
    h3_name: Optional[str] = eqx.field(static=True, default=None)
    stigma_name: Optional[str] = eqx.field(static=True, default=None)

    def _get_sini_m2(self, params: ParameterVector):
        """Compute sin(i) and M2 based on Shapiro parameterization mode."""
        if self.shapiro_mode == "standard":
            sini = params.param_value(self.sini_name) if self.sini_name else 0.0
            m2 = params.param_value(self.m2_name) if self.m2_name else 0.0
        elif self.shapiro_mode == "shapmax":
            shapmax = params.param_value(self.shapmax_name)
            sini = 1.0 - jnp.exp(-shapmax)
            m2 = params.param_value(self.m2_name) if self.m2_name else 0.0
        elif self.shapiro_mode == "h3stigma":
            h3 = params.param_value(self.h3_name)
            stigma = params.param_value(self.stigma_name)
            sini = 2.0 * stigma / (1.0 + stigma ** 2)
            m2 = h3 / (stigma ** 3 * TSUN)
        else:
            sini = 0.0
            m2 = 0.0
        return sini, m2

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute DD binary delay.

        Returns
        -------
        array, shape (n_toas,)
            Binary delay in seconds.
        """
        # --- Extract parameters ---
        pb_d = params.param_value(self.pb_name)
        t0_int, t0_frac = params.epoch_value(self.t0_name)
        a1_ls = params.param_value(self.a1_name)
        ecc0 = params.param_value(self.ecc_name)
        om_rad = params.param_value(self.om_name)

        pbdot = params.param_value(self.pbdot_name) if self.pbdot_name else 0.0
        omdot = params.param_value(self.omdot_name) if self.omdot_name else 0.0
        edot = params.param_value(self.edot_name) if self.edot_name else 0.0
        a1dot = params.param_value(self.a1dot_name) if self.a1dot_name else 0.0
        xpbdot = params.param_value(self.xpbdot_name) if self.xpbdot_name else 0.0

        gamma = params.param_value(self.gamma_name) if self.gamma_name else 0.0
        dr = params.param_value(self.dr_name) if self.dr_name else 0.0
        dth = params.param_value(self.dth_name) if self.dth_name else 0.0
        A0 = params.param_value(self.a0_name) if self.a0_name else 0.0
        B0 = params.param_value(self.b0_name) if self.b0_name else 0.0

        sini, m2 = self._get_sini_m2(params)

        # --- Compute time since periastron ---
        tt0_s = compute_tt0(toa_data.tdb_int, toa_data.tdb_frac, t0_int, t0_frac)

        # --- Time-dependent orbital elements ---
        ecc = compute_ecc(ecc0, edot, tt0_s)
        a1 = compute_a1(a1_ls, a1dot, tt0_s)  # light-seconds = seconds

        # --- Solve Kepler's equation ---
        # Use int/frac split for precision-preserving mean anomaly.
        M = compute_orbital_phase(
            toa_data.tdb_int, toa_data.tdb_frac, t0_int, t0_frac,
            pb_d, pbdot, xpbdot,
        )
        E = compute_eccentric_anomaly(ecc, M)
        # Full orbit count still needed for cumulative true anomaly.
        orbits = compute_orbits_pb(tt0_s, pb_d, pbdot, xpbdot)
        nu = compute_true_anomaly(E, ecc, orbits, M)

        # --- DD omega: OM + nu*k where k = OMDOT/n, n uses instantaneous period ---
        omega = compute_omega_dd(om_rad, omdot, nu, pb_d, pbdot, tt0_s)

        sinE = jnp.sin(E)
        cosE = jnp.cos(E)
        sin_omega = jnp.sin(omega)
        cos_omega = jnp.cos(omega)

        # --- DD-specific eccentricities ---
        er = ecc * (1.0 + dr)
        eTheta = ecc * (1.0 + dth)

        # --- DD intermediate quantities (eqs. [46]-[47]) ---
        # alpha = a1 * sin(omega)   (a1 in light-seconds = seconds)
        # beta = a1 * sqrt(1-eTheta^2) * cos(omega)
        alpha = a1 * sin_omega
        beta = a1 * jnp.sqrt(1.0 - eTheta ** 2) * cos_omega

        # --- Roemer + Einstein delay (Dre, eq. [48]) ---
        # Dre = alpha*(cos(E)-er) + (beta+gamma)*sin(E)
        Dre = alpha * (cosE - er) + (beta + gamma) * sinE

        # --- Dre derivatives w.r.t. u (eqs. [49]-[50]) ---
        Drep = -alpha * sinE + (beta + gamma) * cosE
        Drepp = -alpha * cosE - (beta + gamma) * sinE

        # --- nhat (eq. [51]) --- uses instantaneous period
        pb_prime_s = pb_d * SECS_PER_DAY + pbdot * tt0_s
        nhat = 2.0 * jnp.pi / pb_prime_s / (1.0 - ecc * cosE)

        # --- 1. Inverse timing delay (eq. [52]) ---
        delay_inverse = dd_inverse_timing(Dre, Drep, Drepp, nhat, ecc, sinE, cosE)

        # --- 2. Shapiro delay (eq. [26]) ---
        delay_shapiro = dd_shapiro_delay(ecc, cosE, sinE, sin_omega, cos_omega, sini, m2)

        # --- 3. Aberration delay (eq. [27]) ---
        delay_aberration = dd_aberration_delay(A0, B0, sin_omega, cos_omega, nu, omega, ecc)

        return delay_inverse + delay_shapiro + delay_aberration


class BinaryDDS(BinaryDD):
    """DD model with SHAPMAX parameterization (DDS).

    Uses ``SHAPMAX = -ln(1 - sin(i))`` instead of ``SINI``.
    """

    shapiro_mode: str = eqx.field(static=True, default="shapmax")


class BinaryDDH(BinaryDD):
    """DD model with H3/STIGMA Shapiro parameterization (DDH).

    Uses ``H3`` and ``STIGMA`` instead of ``M2`` and ``SINI``.
    """

    shapiro_mode: str = eqx.field(static=True, default="h3stigma")
