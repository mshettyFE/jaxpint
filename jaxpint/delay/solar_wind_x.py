"""Piecewise solar wind dispersion delay component (SWX).

Each segment has its own maximum DM at conjunction (``SWXDM_XXXX``), radial power-law index
(``SWXP_XXXX``), and time range (``SWXR1_XXXX`` / ``SWXR2_XXXX``).

Unlike the standard ``SolarWindDispersion``, this model represents *excess* DM:
it goes to 0 at opposition and scales to ``SWXDM`` at conjunction.  The scaling
for each TOA is:

    dm = SWXDM * (G(toa) - G_opp) / (G_conj - G_opp)

where ``G`` is the Hazboun et al. (2022) geometry factor, ``G_conj`` and
``G_opp`` are the geometry at conjunction and opposition respectively (computed
at 1 AU with the pulsar's ecliptic latitude as the elongation angle).

References
----------
- Edwards et al. 2006, MNRAS, 372, 1549
- Madison et al. 2019, ApJ, 872, 150
- Hazboun et al. 2022, ApJ, 929, 39
- You et al. 2012, MNRAS, 422, 1160
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.constants import AU_KM, DMCONST
from jaxpint.delay.solar_wind import _solar_wind_geometry_swm1, _sun_angle_and_distance
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import compute_pulsar_direction, ecl_to_icrs_rotation


class SolarWindDispersionX(DelayComponent):
    """Piecewise solar wind dispersion delay (SWX model).

    Parameters
    ----------
    n_bins : int
        Number of SWX segments.
    swxdm_names : tuple[str, ...]
        Names of SWXDM parameters, e.g. ``("SWXDM_0001", "SWXDM_0002")``.
    swxp_names : tuple[str, ...]
        Names of SWXP parameters, e.g. ``("SWXP_0001", "SWXP_0002")``.
    swxr1_names : tuple[str, ...]
        Names of bin-start MJD epoch parameters.
    swxr2_names : tuple[str, ...]
        Names of bin-end MJD epoch parameters.
    theta0 : float
        Elongation at conjunction in radians (precomputed by the bridge).
    raj_name, decj_name : str
        Astrometry coordinate parameter names.
    pmra_name, pmdec_name : str or None
        Proper-motion parameter names.
    posepoch_name : str or None
        Position epoch parameter name.
    obliquity_arcsec : float or None
        When set, coordinates are ecliptic; rotate to ICRS.

    Raises
    ------
    ValueError
        If ``n_bins`` is less than 1.
    ValueError
        If the length of ``swxdm_names``, ``swxp_names``, ``swxr1_names``,
        or ``swxr2_names`` does not match ``n_bins``.
    """

    PARAMS = (
        ParamDecl("SWXDM_0001", prefix="SWXDM_"),
        ParamDecl("SWXP_0001", prefix="SWXP_"),
        ParamDecl("SWXR1_0001", kind="mjd", prefix="SWXR1_"),
        ParamDecl("SWXR2_0001", kind="mjd", prefix="SWXR2_"),
    )

    n_bins: int = eqx.field(static=True)
    swxdm_names: tuple[str, ...] = eqx.field(static=True)
    swxp_names: tuple[str, ...] = eqx.field(static=True)
    swxr1_names: tuple[str, ...] = eqx.field(static=True)
    swxr2_names: tuple[str, ...] = eqx.field(static=True)
    theta0: float = eqx.field(static=True)

    raj_name: str = eqx.field(static=True, default="RAJ")
    decj_name: str = eqx.field(static=True, default="DECJ")
    pmra_name: Optional[str] = eqx.field(static=True, default=None)
    pmdec_name: Optional[str] = eqx.field(static=True, default=None)
    posepoch_name: Optional[str] = eqx.field(static=True, default=None)
    obliquity_arcsec: Optional[float] = eqx.field(static=True, default=None)

    def __check_init__(self):
        self.check_name_tuples(
            "n_bins",
            "swxdm_names",
            "swxp_names",
            "swxr1_names",
            "swxr2_names",
            label="segment",
        )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute piecewise solar wind dispersion delay.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times, frequencies, positions).
        params : ParameterVector
            Timing-model parameters containing SWXDM, SWXP, SWXR1, SWXR2.
        delay : array, shape (n_toas,)
            Accumulated delay from prior components (not used).

        Returns
        -------
        array, shape (n_toas,)
            Solar wind dispersion delay in seconds.
        """
        # 1. Pulsar direction (unit vector, ICRS).
        psr_dir = compute_pulsar_direction(
            toa_data,
            params,
            raj_name=self.raj_name,
            decj_name=self.decj_name,
            pmra_name=self.pmra_name,
            pmdec_name=self.pmdec_name,
            posepoch_name=self.posepoch_name,
        )
        if self.obliquity_arcsec is not None:
            psr_dir = psr_dir @ ecl_to_icrs_rotation(self.obliquity_arcsec)

        # 2. Per-TOA sun angle and distance.
        theta, r_km = _sun_angle_and_distance(toa_data, psr_dir)

        # 3. TOA MJD for bin assignment (UTC, matching PINT's mjd_float).
        toa_mjd = toa_data.mjd.total

        # 4. Fiducial angles for conjunction/opposition (1-element arrays for
        #    compatibility with _solar_wind_geometry_swm1).
        theta0_arr = jnp.array([self.theta0])
        theta0_opp_arr = jnp.array([jnp.pi - self.theta0])
        r0_arr = jnp.array([AU_KM])

        # 5. Loop over segments, accumulate DM.
        dm = jnp.zeros(toa_data.n_toas)

        for i in range(self.n_bins):
            r1 = params.epoch_dual(self.swxr1_names[i]).total
            r2 = params.epoch_dual(self.swxr2_names[i]).total
            in_bin = (toa_mjd >= r1) & (toa_mjd <= r2)

            swxdm = params.param_value(self.swxdm_names[i])
            p = params.param_value(self.swxp_names[i])

            toa_geom = _solar_wind_geometry_swm1(theta, r_km, p)

            # Conjunction and opposition geometry (scalar, via 1-element array)
            conj_geom = _solar_wind_geometry_swm1(theta0_arr, r0_arr, p)[0]
            opp_geom = _solar_wind_geometry_swm1(theta0_opp_arr, r0_arr, p)[0]

            # Scaling: (G(toa) - G_opp) / (G_conj - G_opp)
            denom = conj_geom - opp_geom
            safe_denom = jnp.where(denom == 0.0, 1.0, denom)
            scaling = jnp.where(denom == 0.0, 0.0, (toa_geom - opp_geom) / safe_denom)

            dm = dm + jnp.where(in_bin, swxdm * scaling, 0.0)

        return dm * DMCONST / toa_data.freq**2
