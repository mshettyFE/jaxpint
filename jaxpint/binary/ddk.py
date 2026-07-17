"""Kopeikin-corrected DD model (DDK).

Extends DD with corrections to a1 and omega for:
  - Annual-orbital parallax (Kopeikin 1995, Eq. 18-19)
  - Proper motion (Kopeikin 1996, Eq. 8-10) when K96=True

Uses KIN (inclination angle) instead of SINI.

Reference
---------
Kopeikin (1995), ApJ, 439, L5.
Kopeikin (1996), ApJ, 467, L93.
PINT ``stand_alone_psr_binaries/DDK_model.py``.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import BinaryDelayComponent, ParamDecl
from jaxpint.binary._param_decls import BINARY_CORE
from jaxpint.types import TOAData, ParameterVector
from jaxpint.constants import SECS_PER_DAY, RAD_PER_MAS, KPC_TO_KM
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
)


class BinaryDDK(BinaryDelayComponent):
    """DDK binary delay model.

    Parameters
    ----------
    kin_name : str
        Inclination angle KIN (radians in ParameterVector).
    kom_name : str
        Longitude of ascending node KOM (radians in ParameterVector).
    k96 : bool
        If True, apply proper motion corrections (Kopeikin 1996).
    raj_name, decj_name : str
        Pulsar position parameter names (radians).
    pmra_name, pmdec_name : str, optional
        Proper motion parameter names (mas/yr or equivalent).
    posepoch_name : str, optional
        Proper motion reference epoch.
    px_name : str
        Parallax parameter name (mas).
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
        ParamDecl("KIN", unit="deg"),
        ParamDecl("KOM", unit="deg"),
        ParamDecl("K96", kind="bool"),
    )

    pb_name: str = eqx.field(static=True, default="PB")
    t0_name: str = eqx.field(static=True, default="T0")
    a1_name: str = eqx.field(static=True, default="A1")
    ecc_name: str = eqx.field(static=True, default="ECC")
    om_name: str = eqx.field(static=True, default="OM")

    pbdot_name: Optional[str] = eqx.field(static=True, default=None)
    omdot_name: Optional[str] = eqx.field(static=True, default=None)
    edot_name: Optional[str] = eqx.field(static=True, default=None)
    a1dot_name: Optional[str] = eqx.field(static=True, default=None)
    xpbdot_name: Optional[str] = eqx.field(static=True, default=None)

    gamma_name: Optional[str] = eqx.field(static=True, default=None)
    dr_name: Optional[str] = eqx.field(static=True, default=None)
    dth_name: Optional[str] = eqx.field(static=True, default=None)
    a0_name: Optional[str] = eqx.field(static=True, default=None)
    b0_name: Optional[str] = eqx.field(static=True, default=None)

    # Shapiro delay (DDK uses KIN → SINI = sin(KIN), M2 still needed)
    m2_name: Optional[str] = eqx.field(static=True, default=None)

    # DDK-specific parameters
    kin_name: str = eqx.field(static=True, default="KIN")
    kom_name: str = eqx.field(static=True, default="KOM")
    k96: bool = eqx.field(static=True, default=True)

    # Astrometry parameters (needed for Kopeikin corrections)
    raj_name: str = eqx.field(static=True, default="RAJ")
    decj_name: str = eqx.field(static=True, default="DECJ")
    pmra_name: Optional[str] = eqx.field(static=True, default=None)
    pmdec_name: Optional[str] = eqx.field(static=True, default=None)
    posepoch_name: Optional[str] = eqx.field(static=True, default=None)
    px_name: str = eqx.field(static=True, default="PX")

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute DDK binary delay with Kopeikin parallax and proper-motion corrections.

        Extends the DD model by correcting the projected semi-major axis
        (A1) and longitude of periastron (OM) for annual-orbital parallax
        (Kopeikin 1995) and, optionally, proper motion (Kopeikin 1996).

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times, SSB-observatory positions, etc.).
        params : ParameterVector
            Timing-model parameters containing DD orbital elements,
            Kopeikin parameters (KIN, KOM, PX), and astrometry (RAJ, DECJ).
        delay : array, shape (n_toas,)
            Accumulated signal delay in seconds, used to correct
            the time of arrival to emission time.

        Returns
        -------
        array, shape (n_toas,)
            Binary delay in seconds.
        """
        # --- Extract DD parameters ---
        pb_d = params.param_value(self.pb_name)
        t0 = params.epoch_dual(self.t0_name)
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

        m2 = params.param_value_or(self.m2_name)

        # --- DDK-specific parameters ---
        kin = params.param_value(self.kin_name)  # radians
        kom = params.param_value(self.kom_name)  # radians
        px_mas = params.param_value(self.px_name)  # mas

        # --- Compute pulsar direction (unit vector) ---
        ra = params.param_value(self.raj_name)  # radians
        dec = params.param_value(self.decj_name)  # radians

        # Apply proper motion correction to position if available
        if self.pmra_name and self.pmdec_name and self.posepoch_name:
            pmra = params.param_value(self.pmra_name)  # mas/yr
            pmdec = params.param_value(self.pmdec_name)  # mas/yr

            posepoch = params.epoch_dual(self.posepoch_name)
            dt_pos_days = (toa_data.tdb - posepoch).total
            dt_pos_yr = dt_pos_days / 365.25

            # Update RA/DEC with PM (per TOA)
            ra_toa = ra + (pmra * RAD_PER_MAS / jnp.cos(dec)) * dt_pos_yr
            dec_toa = dec + (pmdec * RAD_PER_MAS) * dt_pos_yr
        else:
            pmra = 0.0
            pmdec = 0.0
            ra_toa = ra
            dec_toa = dec

        # Trig functions of pulsar position (Kopeikin 1995, Eq 10)
        sin_long = jnp.sin(ra_toa)
        cos_long = jnp.cos(ra_toa)
        sin_lat = jnp.sin(dec_toa)
        cos_lat = jnp.cos(dec_toa)

        sin_KOM = jnp.sin(kom)
        cos_KOM = jnp.cos(kom)

        # --- Compute time since T0 (corrected for accumulated delay) ---
        tt0_s = compute_tt0(toa_data.tdb, t0, delay=delay)

        # --- Base a1 and omega (before Kopeikin corrections) ---
        a1_base = compute_a1(a1_ls, a1dot, tt0_s)

        # --- Kopeikin proper motion corrections (K96) ---
        # Proper motion in mas/yr → rad/s
        pm_long_rad_per_s = pmra * RAD_PER_MAS / (365.25 * SECS_PER_DAY)
        pm_lat_rad_per_s = pmdec * RAD_PER_MAS / (365.25 * SECS_PER_DAY)

        if self.k96:
            # delta_kin (Kopeikin 1996, Eq 10)
            delta_kin = (
                -pm_long_rad_per_s * sin_KOM + pm_lat_rad_per_s * cos_KOM
            ) * tt0_s
            kin_eff = kin + delta_kin

            # delta_a1 from proper motion (Eq 8)
            delta_a1_pm = a1_base * delta_kin / jnp.tan(kin_eff)

            # delta_omega from proper motion (Eq 9)
            delta_omega_pm = (
                (pm_long_rad_per_s * cos_KOM + pm_lat_rad_per_s * sin_KOM)
                / jnp.sin(kin_eff)
                * tt0_s
            )
        else:
            kin_eff = kin
            delta_a1_pm = 0.0
            delta_omega_pm = 0.0

        # --- Kopeikin parallax corrections (Kopeikin 1995, Eq 15-19) ---
        # obs_pos is SSB → observatory in km (ICRS or ECL depending on model)
        obs_pos = toa_data.ssb_obs_pos  # (n_toas, 3) in km

        # delta_I0 (Eq 15) and delta_J0 (Eq 16)
        # These are projections of the observatory position onto the sky plane
        delta_I0 = -obs_pos[:, 0] * sin_long + obs_pos[:, 1] * cos_long
        delta_J0 = (
            -obs_pos[:, 0] * sin_lat * cos_long
            - obs_pos[:, 1] * sin_lat * sin_long
            + obs_pos[:, 2] * cos_lat
        )

        # Distance from parallax: PX in mas → d in kpc → d_km
        d_kpc = 1.0 / px_mas  # kpc
        d_km = d_kpc * KPC_TO_KM

        # delta_a1 from parallax (Eq 18)
        a1_for_parallax = a1_base + delta_a1_pm if self.k96 else a1_base
        delta_a1_par = (
            a1_for_parallax
            / jnp.tan(kin_eff)
            / d_km
            * (delta_I0 * sin_KOM - delta_J0 * cos_KOM)
        )

        # delta_omega from parallax (Eq 19)
        delta_omega_par = (
            -1.0 / jnp.sin(kin_eff) / d_km * (delta_I0 * cos_KOM + delta_J0 * sin_KOM)
        )

        # --- Apply corrections ---
        a1 = a1_base + delta_a1_pm + delta_a1_par
        sini = jnp.sin(kin_eff)

        # --- Time-dependent eccentricity ---
        ecc = compute_ecc(ecc0, edot, tt0_s)

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

        # --- DD omega with Kopeikin corrections ---
        omega_dd = compute_omega_dd(om_rad, omdot, nu, pb_d, pbdot, tt0_s)
        omega = omega_dd + delta_omega_pm + delta_omega_par

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
