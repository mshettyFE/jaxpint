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
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent
from jaxpint.types import TOAData, ParameterVector
from jaxpint.binary.common import (
    compute_tt0,
    compute_orbital_phase,
    compute_orbits_pb,
    compute_eccentric_anomaly,
    compute_ecc,
    compute_a1,
    compute_true_anomaly,
    compute_omega_dd,
    dd_core_delay,
    get_sini_m2,
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

        pbdot = params.param_value_or(self.pbdot_name)
        omdot = params.param_value_or(self.omdot_name)
        edot = params.param_value_or(self.edot_name)
        a1dot = params.param_value_or(self.a1dot_name)
        xpbdot = params.param_value_or(self.xpbdot_name)

        gamma = params.param_value_or(self.gamma_name)
        dr = params.param_value_or(self.dr_name)
        dth = params.param_value_or(self.dth_name)
        A0 = params.param_value_or(self.a0_name)
        B0 = params.param_value_or(self.b0_name)

        sini, m2 = get_sini_m2(
            params, self.shapiro_mode, self.sini_name, self.m2_name,
            self.shapmax_name, self.h3_name, self.stigma_name,
        )

        # --- Compute time since periastron (corrected for accumulated delay) ---
        tt0_s = compute_tt0(toa_data.tdb_int, toa_data.tdb_frac, t0_int, t0_frac, delay=delay)

        # --- Time-dependent orbital elements ---
        ecc = compute_ecc(ecc0, edot, tt0_s)
        a1 = compute_a1(a1_ls, a1dot, tt0_s)  # light-seconds = seconds

        # --- Solve Kepler's equation ---
        # Use int/frac split for precision-preserving mean anomaly.
        M = compute_orbital_phase(
            toa_data.tdb_int, toa_data.tdb_frac, t0_int, t0_frac,
            pb_d, pbdot, xpbdot, delay=delay,
        )
        E = compute_eccentric_anomaly(ecc, M)
        # Full orbit count still needed for cumulative true anomaly.
        orbits = compute_orbits_pb(tt0_s, pb_d, pbdot, xpbdot)
        nu = compute_true_anomaly(E, ecc, orbits, M)

        # --- DD omega: OM + nu*k where k = OMDOT/n, n uses instantaneous period ---
        omega = compute_omega_dd(om_rad, omdot, nu, pb_d, pbdot, tt0_s)

        return dd_core_delay(
            E, ecc, omega, nu, a1, tt0_s, pb_d, pbdot,
            gamma, dr, dth, A0, B0, sini, m2,
        )
