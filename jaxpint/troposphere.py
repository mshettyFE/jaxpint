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

from jaxpint.components import DelayComponent
from jaxpint.types import TOAData, ParameterVector

# ---------------------------------------------------------------------------
# Niell mapping function coefficients (with pole/equator padding pre-applied)
# Latitude breakpoints: [0, 15, 30, 45, 60, 75, 90] degrees
# Index 0 and 6 are copies of index 1 and 5 respectively, providing
# constant extrapolation within 15 degrees of the equator and poles.
# ---------------------------------------------------------------------------

_LAT_BREAKS = jnp.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]) * (jnp.pi / 180.0)

# Hydrostatic average coefficients
_A_AVG = jnp.array([1.2769934, 1.2769934, 1.2683230, 1.2465397, 1.2196049, 1.2045996, 1.2045996]) * 1e-3
_B_AVG = jnp.array([2.9153695, 2.9153695, 2.9152299, 2.9288445, 2.9022565, 2.9024912, 2.9024912]) * 1e-3
_C_AVG = jnp.array([62.610505, 62.610505, 62.837393, 63.721774, 63.824265, 64.258455, 64.258455]) * 1e-3

# Hydrostatic amplitude coefficients
_A_AMP = jnp.array([0.0, 0.0, 1.2709626, 2.6523662, 3.4000452, 4.1202191, 4.1202191]) * 1e-5
_B_AMP = jnp.array([0.0, 0.0, 2.1414979, 3.0160779, 7.2562722, 11.723375, 11.723375]) * 1e-5
_C_AMP = jnp.array([0.0, 0.0, 9.0128400, 4.3497037, 84.795348, 170.37206, 170.37206]) * 1e-5

# Height correction coefficients (scalar)
_A_HT: float = 2.53e-5
_B_HT: float = 5.49e-3
_C_HT: float = 1.14e-3

# Wet mapping coefficients (no annual variation)
_AW = jnp.array([5.8021897, 5.8021897, 5.6794847, 5.8118019, 5.9727542, 6.1641693, 6.1641693]) * 1e-4
_BW = jnp.array([1.4275268, 1.4275268, 1.5138625, 1.4572752, 1.5007428, 1.7599082, 1.7599082]) * 1e-3
_CW = jnp.array([4.3472961, 4.3472961, 4.6729510, 4.3908931, 4.4626982, 5.4736038, 5.4736038]) * 1e-2

# Other constants
_DOY_OFFSET: int = -28
_EARTH_R_KM: float = 6356.766  # Earth radius at 45 deg latitude in km
_C_LIGHT: float = 299792458.0  # speed of light in m/s


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
    """Herring map with scalar coefficients (for height correction)."""
    numer = 1.0 + a / (1.0 + b / (1.0 + c))
    denom = sin_alt + a / (sin_alt + b / (sin_alt + c))
    return numer / denom


def _interp_lat(
    abs_lat_rad: Float[Array, " n"],
    coeff: Float[Array, " 7"],
) -> Float[Array, " n"]:
    """Linear interpolation into a 7-element latitude coefficient array.

    Uses ``jnp.searchsorted`` for vectorized bin lookup — no Python loops.
    """
    idx = jnp.searchsorted(_LAT_BREAKS, abs_lat_rad, side="right") - 1
    idx = jnp.clip(idx, 0, 5)
    x0 = _LAT_BREAKS[idx]
    x1 = _LAT_BREAKS[idx + 1]
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
    return jnp.mod(
        2000.0 + (tdb_mjd - 51544.5 + _DOY_OFFSET) / 365.25 + season_offset,
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
    return (p_kpa / 43.921) / (_C_LIGHT * (1.0 - 0.00266 * jnp.cos(2.0 * lat_rad) - 0.00028 * H_km))


def _hydrostatic_mapping(
    sin_alt: Float[Array, " n"],
    abs_lat_rad: Float[Array, " n"],
    H_km: Float[Array, " n"],
    year_frac: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Niell hydrostatic mapping function with height correction."""
    # Interpolate coefficients with annual variation
    a_avg = _interp_lat(abs_lat_rad, _A_AVG)
    a_amp = _interp_lat(abs_lat_rad, _A_AMP)
    b_avg = _interp_lat(abs_lat_rad, _B_AVG)
    b_amp = _interp_lat(abs_lat_rad, _B_AMP)
    c_avg = _interp_lat(abs_lat_rad, _C_AVG)
    c_amp = _interp_lat(abs_lat_rad, _C_AMP)

    cos_yf = jnp.cos(2.0 * jnp.pi * year_frac)
    a = a_avg + a_amp * cos_yf
    b = b_avg + b_amp * cos_yf
    c = c_avg + c_amp * cos_yf

    base_map = _herring_map(sin_alt, a, b, c)

    # Height correction
    f_correction = _herring_map_scalar(sin_alt, _A_HT, _B_HT, _C_HT)
    height_correction = (1.0 / sin_alt - f_correction) * H_km

    return base_map + height_correction


def _wet_mapping(
    sin_alt: Float[Array, " n"],
    abs_lat_rad: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Niell wet mapping function (no annual variation, no height correction)."""
    a = _interp_lat(abs_lat_rad, _AW)
    b = _interp_lat(abs_lat_rad, _BW)
    c = _interp_lat(abs_lat_rad, _CW)
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

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        if toa_data.tropo_alt is None:
            return jnp.zeros(toa_data.n_toas)

        tdb_mjd = toa_data.tdb_int + toa_data.tdb_frac
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
