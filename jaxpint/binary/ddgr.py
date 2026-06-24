"""Damour & Deruelle model with GR-derived post-Keplerian parameters (DDGR).

Derives all PK parameters (SINI, GAMMA, PBDOT, OMDOT, DR, DTH) from
the total system mass MTOT and companion mass M2 via General Relativity,
then computes the standard DD binary delay.

Reference
---------
Taylor & Weisberg (1989), ApJ, 345, 434.
PINT ``stand_alone_psr_binaries/DDGR_model.py``.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.binary._param_decls import BINARY_CORE
from jaxpint.types import TOAData, ParameterVector
from jaxpint.constants import SECS_PER_DAY, TSUN, C_M_PER_S
from jaxpint.binary.common import (
    compute_tt0,
    compute_orbital_phase,
    compute_orbits_pb,
    compute_eccentric_anomaly,
    compute_ecc,
    compute_a1,
    compute_true_anomaly,
    dd_core_delay,
)


# G * Msun in SI (m^3 / s^2).  Derived from TSUN = G*Msun/c^3.
_GM_SUN_SI = TSUN * C_M_PER_S**3

# Number of fixed iterations for relativistic Kepler equation.
_REL_KEPLER_NITER = 20


def _solve_relativistic_kepler(mtot, m1, m2, n):
    """Solve relativistic Kepler's third law by fixed-point iteration.

    Taylor & Weisberg (1989), Eq. 15.

    Parameters
    ----------
    mtot, m1, m2 : float
        Masses in solar masses.
    n : float
        Orbital angular frequency in rad/s.

    Returns
    -------
    arr0 : float
        Non-relativistic semi-major axis in metres.
    arr : float
        Relativistic semi-major axis in metres.
    """
    gm_tot = mtot * _GM_SUN_SI
    arr0 = (gm_tot / n**2) ** (1.0 / 3.0)
    c2 = C_M_PER_S**2

    def body(_, arr_prev):
        return arr0 * (
            1.0 + (m1 * m2 / mtot**2 - 9.0) * (gm_tot / (2.0 * arr_prev * c2))
        ) ** (2.0 / 3.0)

    arr = jax.lax.fori_loop(0, _REL_KEPLER_NITER, body, arr0)
    return arr0, arr


class BinaryDDGR(DelayComponent):
    """DD model with GR-derived post-Keplerian parameters.

    Input masses MTOT and M2 determine all PK parameters via GR.
    XOMDOT and XPBDOT allow for excess beyond GR predictions.
    """

    PARAMS = (
        *BINARY_CORE,
        ParamDecl("T0", kind="mjd"),
        ParamDecl("GAMMA"),
        ParamDecl("A0"),
        ParamDecl("B0"),
        ParamDecl("DR"),
        ParamDecl("DTH", aliases=("DTHETA",)),
        ParamDecl("M2"),
        ParamDecl("SINI"),
        ParamDecl("MTOT"),
        ParamDecl("XOMDOT", unit="deg / yr"),
        ParamDecl("XPBDOT", scale=1e-12, scale_threshold=1e-7),
    )

    pb_name: str = eqx.field(static=True, default="PB")
    t0_name: str = eqx.field(static=True, default="T0")
    a1_name: str = eqx.field(static=True, default="A1")
    ecc_name: str = eqx.field(static=True, default="ECC")
    om_name: str = eqx.field(static=True, default="OM")

    mtot_name: str = eqx.field(static=True, default="MTOT")
    m2_name: str = eqx.field(static=True, default="M2")

    edot_name: Optional[str] = eqx.field(static=True, default=None)
    a1dot_name: Optional[str] = eqx.field(static=True, default=None)
    xomdot_name: Optional[str] = eqx.field(static=True, default=None)
    xpbdot_name: Optional[str] = eqx.field(static=True, default=None)
    a0_name: Optional[str] = eqx.field(static=True, default=None)
    b0_name: Optional[str] = eqx.field(static=True, default=None)

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute DDGR binary delay with GR-derived post-Keplerian parameters.

        Derives SINI, GAMMA, PBDOT, OMDOT, DR, and DTH from the total
        system mass (MTOT) and companion mass (M2) via General Relativity,
        then evaluates the standard DD delay formula.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times, etc.).
        params : ParameterVector
            Timing-model parameters containing orbital elements (PB, T0,
            A1, ECC, OM) and masses (MTOT, M2).
        delay : array, shape (n_toas,)
            Accumulated signal delay in seconds, used to correct
            the time of arrival to emission time.

        Returns
        -------
        array, shape (n_toas,)
            Binary delay in seconds.
        """
        # --- Extract orbital parameters ---
        pb_d = params.param_value(self.pb_name)
        t0 = params.epoch_dual(self.t0_name)
        a1_ls = params.param_value(self.a1_name)
        ecc0 = params.param_value(self.ecc_name)
        om_rad = params.param_value(self.om_name)

        mtot = params.param_value(self.mtot_name)
        m2 = params.param_value(self.m2_name)
        m1 = mtot - m2

        edot = params.param_value_or(self.edot_name)
        a1dot = params.param_value_or(self.a1dot_name)
        xomdot = params.param_value_or(self.xomdot_name)
        xpbdot = params.param_value_or(self.xpbdot_name)
        A0 = params.param_value_or(self.a0_name)
        B0 = params.param_value_or(self.b0_name)

        # --- Derive PK parameters from GR (Taylor & Weisberg 1989) ---
        pb_s = pb_d * SECS_PER_DAY
        n = 2.0 * jnp.pi / pb_s  # orbital angular frequency (rad/s)
        gm_tot = mtot * _GM_SUN_SI  # G * Mtot (m^3 s^{-2})
        c = C_M_PER_S
        c2 = c**2
        c5 = c**5

        arr0, arr = _solve_relativistic_kepler(mtot, m1, m2, n)

        # Pulsar component of semi-major axis (metres)
        ar = arr * m2 / mtot

        # SINI (Eq. 20): sin(i) = a1 * c / ar
        sini = a1_ls * c / ar

        # GAMMA (Eq. 17, uses arr0 per tempo convention)
        gamma = ecc0 * gm_tot * m2 * (m1 + 2.0 * m2) / (n * c2 * arr0 * mtot**2)

        # PBDOT (Eq. 18): GW orbital decay
        # Re-expressed as: (-192*pi/5) * (G*Mtot*n)^{5/3} * M1*M2/Mtot^2 * fe / c^5
        fe = (1.0 + (73.0 / 24.0) * ecc0**2 + (37.0 / 96.0) * ecc0**4) * (
            1.0 - ecc0**2
        ) ** (-7.0 / 2.0)
        pbdot = (
            (-192.0 * jnp.pi / 5.0)
            * (gm_tot * n) ** (5.0 / 3.0)
            * (m1 * m2 / mtot**2)
            * fe
            / c5
        ) + xpbdot

        # k: periastron advance rate (Eq. 16, uses arr0 per tempo convention)
        k = 3.0 * gm_tot / (c2 * arr0 * (1.0 - ecc0**2))

        # DR (Eq. 24) and DTH (Eq. 25): relativistic eccentricity corrections
        gr_factor = _GM_SUN_SI / (c2 * mtot * arr)
        dr = gr_factor * (3.0 * m1**2 + 6.0 * m1 * m2 + 2.0 * m2**2)
        dth = gr_factor * (3.5 * m1**2 + 6.0 * m1 * m2 + 2.0 * m2**2)

        # --- Compute time since periastron (corrected for accumulated delay) ---
        tt0_s = compute_tt0(toa_data.tdb, t0, delay=delay)

        # --- Time-dependent orbital elements ---
        ecc = compute_ecc(ecc0, edot, tt0_s)
        a1 = compute_a1(a1_ls, a1dot, tt0_s)

        # --- Solve Kepler's equation ---
        M = compute_orbital_phase(
            toa_data.tdb,
            t0,
            pb_d,
            pbdot,
            xpbdot,
            delay=delay,
        )
        E = compute_eccentric_anomaly(ecc, M)
        orbits = compute_orbits_pb(tt0_s, pb_d, pbdot, xpbdot)
        nu = compute_true_anomaly(E, ecc, orbits, M)

        # --- omega: OM + nu * k_eff where k_eff includes XOMDOT contribution ---
        k_eff = k + xomdot / n
        omega = om_rad + nu * k_eff

        return dd_core_delay(
            E,
            ecc,
            omega,
            nu,
            a1,
            tt0_s,
            pb_d,
            pbdot,
            gamma,
            dr,
            dth,
            A0,
            B0,
            sini,
            m2,
        )
