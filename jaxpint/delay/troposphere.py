"""Troposphere delay component.

Ports PINT's ``TroposphereDelay`` as a pure Equinox module.  Computes the
atmospheric delay for topocentric TOAs using:

- Davis zenith hydrostatic delay (Davis et al. 1985, Appendix A)
- Niell mapping functions (Niell 1996, Eq 4)
- US Standard Atmosphere pressure–altitude relation

The target elevation angle, observatory latitude, and observatory height
are pre-computed in the bridge layer (requires astropy AltAz transforms).
This module performs only the numerical delay calculation in pure JAX.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.dual_float import DualFloat
from jaxpint.constants import (
    C_M_PER_S,
    NIELL_A_AMP,
    NIELL_A_AVG,
    NIELL_A_HT,
    NIELL_AW,
    NIELL_B_AMP,
    NIELL_B_AVG,
    NIELL_B_HT,
    NIELL_BW,
    NIELL_C_AMP,
    NIELL_C_AVG,
    NIELL_C_HT,
    NIELL_CW,
    NIELL_DOY_OFFSET,
    NIELL_LAT_BREAKS,
)
from jaxpint.types import TOAData, ParameterVector


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def _herring_map(
    sin_alt: Float[Array, " n"],
    a: Float[Array, " n"],
    b: Float[Array, " n"],
    c: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Niell/Herring continued-fraction mapping function (Eq 4).

    Maps zenith delay to the delay at elevation angle ``alt``.
    At zenith (sin_alt=1) this returns 1.0.

    Parameters
    ----------
    sin_alt : (n,)
        Sine of the source elevation angle.
    a, b, c : (n,)
        Continued-fraction coefficients, interpolated from latitude
        and seasonal tables (Niell 1996, Table 1).
    """
    numer = 1.0 + a / (1.0 + b / (1.0 + c))
    denom = sin_alt + a / (sin_alt + b / (sin_alt + c))
    return numer / denom


def _herring_map_scalar(
    sin_alt: Float[Array, " n"],
    a: float,
    b: float,
    c: float,
) -> Float[Array, " n"]:
    """Herring continued-fraction mapping with scalar coefficients.

    Same formula as :func:`_herring_map` but takes scalar ``a``, ``b``,
    ``c`` (used for the height-correction mapping where the coefficients
    are fixed constants, not latitude-dependent).

    Parameters
    ----------
    sin_alt : (n,)
        Sine of the source elevation angle.
    a, b, c : float
        Fixed continued-fraction coefficients.
    """
    numer = 1.0 + a / (1.0 + b / (1.0 + c))
    denom = sin_alt + a / (sin_alt + b / (sin_alt + c))
    return numer / denom


def _interp_lat(
    abs_lat_rad: Float[Array, " n"],
    coeff: Float[Array, " 7"],
) -> Float[Array, " n"]:
    """Linearly interpolate a Niell latitude coefficient table.

    The Niell (1996) mapping function tables have 7 entries at
    latitudes [15, 30, 45, 60, 75, 90] degrees (stored in
    ``NIELL_LAT_BREAKS`` as radians, with a 0-degree sentinel).
    This function interpolates between entries for arbitrary latitudes.

    Parameters
    ----------
    abs_lat_rad : (n,)
        Absolute geodetic latitude in radians.
    coeff : (7,)
        Coefficient table (one value per latitude break).
    """
    idx = jnp.searchsorted(NIELL_LAT_BREAKS, abs_lat_rad, side="right") - 1
    idx = jnp.clip(idx, 0, 5)
    x0 = NIELL_LAT_BREAKS[idx]
    x1 = NIELL_LAT_BREAKS[idx + 1]
    y0 = coeff[idx]
    y1 = coeff[idx + 1]
    t = (abs_lat_rad - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _year_fraction(
    tdb_mjd: Float[Array, " n"],
    lat_rad: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Fractional year from TDB MJD, with southern hemisphere offset."""
    season_offset = jnp.where(lat_rad < 0.0, 0.5, 0.0)
    # NIELL_DOY_OFFSET due to seasonal variation of troposphere. 
    # Shifting "start of year" so that troposphere model aligns with the peak in this variation
    days_since_J2000  =  (tdb_mjd - 51544.5 + NIELL_DOY_OFFSET)
    # Convert to years, then add back in year 2000, plus have a year offset if in southern hemisphere
    return jnp.mod(
        2000.0 + days_since_J2000 / 365.25 + season_offset,
        1.0,
    )


def _pressure_from_height_km(H_km: Float[Array, " n"]) -> Float[Array, " n"]:
    """Atmospheric pressure in kPa from height in km (US Standard Atmosphere).

    Valid for heights below ~11 km.
    """
    H_m = H_km * 1000.0
    T = 288.15 - 0.0065 * H_m  # temperature lapse (uses geometric height, matching PINT)
    return 101.325 * (288.15 / T) ** (-5.25575)


def _zenith_hydrostatic_delay(
    lat_rad: Float[Array, " n"],
    H_km: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Davis zenith hydrostatic delay in seconds."""
    p_kpa = _pressure_from_height_km(H_km)
    return (p_kpa / 43.921) / (C_M_PER_S * (1.0 - 0.00266 * jnp.cos(2.0 * lat_rad) - 0.00028 * H_km))


def _hydrostatic_mapping(
    sin_alt: Float[Array, " n"],
    abs_lat_rad: Float[Array, " n"],
    H_km: Float[Array, " n"],
    year_frac: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Niell hydrostatic mapping function with height correction."""
    # Interpolate coefficients with annual variation
    a_avg = _interp_lat(abs_lat_rad, NIELL_A_AVG)
    a_amp = _interp_lat(abs_lat_rad, NIELL_A_AMP)
    b_avg = _interp_lat(abs_lat_rad, NIELL_B_AVG)
    b_amp = _interp_lat(abs_lat_rad, NIELL_B_AMP)
    c_avg = _interp_lat(abs_lat_rad, NIELL_C_AVG)
    c_amp = _interp_lat(abs_lat_rad, NIELL_C_AMP)

    cos_yf = jnp.cos(2.0 * jnp.pi * year_frac)
    a = a_avg + a_amp * cos_yf
    b = b_avg + b_amp * cos_yf
    c = c_avg + c_amp * cos_yf

    base_map = _herring_map(sin_alt, a, b, c)

    # Height correction
    f_correction = _herring_map_scalar(sin_alt, NIELL_A_HT, NIELL_B_HT, NIELL_C_HT)
    height_correction = (1.0 / sin_alt - f_correction) * H_km

    return base_map + height_correction


def _wet_mapping(
    sin_alt: Float[Array, " n"],
    abs_lat_rad: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Niell wet mapping function (no annual variation, no height correction)."""
    a = _interp_lat(abs_lat_rad, NIELL_AW)
    b = _interp_lat(abs_lat_rad, NIELL_BW)
    c = _interp_lat(abs_lat_rad, NIELL_CW)
    return _herring_map(sin_alt, a, b, c)


# ---------------------------------------------------------------------------
# TroposphereDelay component
# ---------------------------------------------------------------------------

class TroposphereDelay(DelayComponent):
    """Troposphere delay for topocentric TOAs.

    All input data (elevation angle, observatory latitude/height) is
    pre-computed in the bridge layer and stored on ``TOAData``.  This
    component performs the pure-JAX numerical calculation.

    Has no fittable parameters — ``CORRECT_TROPOSPHERE`` is handled at
    the bridge level (if False, this component is not added to the model).
    """

    PARAMS = (
        ParamDecl("CORRECT_TROPOSPHERE", kind="bool"),
    )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute troposphere delay using Niell mapping functions.

        Applies the Davis zenith hydrostatic delay mapped to the source
        elevation angle via the Niell (1996) continued-fraction mapping.
        Returns zero for TOAs without valid elevation data.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data including elevation angles, observatory
            latitude, and height (pre-computed by the bridge layer).
        params : ParameterVector
            Timing-model parameters (unused; no fittable parameters).
        delay : array, shape (n_toas,)
            Accumulated signal delay in seconds (unused).

        Returns
        -------
        array, shape (n_toas,)
            Troposphere delay in seconds.
        """
        if toa_data.tropo_alt is None:
            return jnp.zeros(toa_data.n_toas)
        # These geodetic fields are populated together with tropo_alt.
        assert toa_data.obs_geodetic_lat is not None
        assert toa_data.obs_height_km is not None
        assert toa_data.tropo_alt_valid is not None

        tdb_mjd = toa_data.tdb.total
        sin_alt = jnp.sin(toa_data.tropo_alt)
        abs_lat = jnp.abs(toa_data.obs_geodetic_lat)
        year_frac = _year_fraction(tdb_mjd, toa_data.obs_geodetic_lat)

        # Hydrostatic: zenith delay * mapping function
        zenith = _zenith_hydrostatic_delay(toa_data.obs_geodetic_lat, toa_data.obs_height_km)
        hydro_map = _hydrostatic_mapping(sin_alt, abs_lat, toa_data.obs_height_km, year_frac)

        # Wet: currently zero (matching PINT default)
        wet_zenith = 0.0
        wet_map = _wet_mapping(sin_alt, abs_lat)

        total = zenith * hydro_map + wet_zenith * wet_map

        # Zero out invalid altitudes
        total = jnp.where(toa_data.tropo_alt_valid, total, 0.0)

        return total
