"""Physical, astronomical, and model-specific constants for JaxPINT."""

from __future__ import annotations

import jax.numpy as jnp

# ── Physical constants ──────────────────────────────────────────────

#: Speed of light (km/s)
C_KM_PER_S: float = 299_792.458
#: Speed of light (m/s)
C_M_PER_S: float = 299_792_458.0

# ── Astronomical constants ──────────────────────────────────────────

#: Astronomical unit (km), IAU 2012
AU_KM: float = 149_597_870.7
#: 1 kiloparsec (km)
KPC_TO_KM: float = 3.0856775814913673e16
#: 1 parsec (km)
PC_TO_KM: float = 3.0856775814913673e13
#: GM_sun / c^3 (s)
TSUN: float = 4.92549094830932e-6
#: Earth radius at 45 deg latitude (km)
EARTH_R_KM: float = 6356.766

#: Planet mass parameters; ``T_planet = T_sun / mass_ratio`` in seconds.
PLANET_MASSES: dict[str, float] = {
    "jupiter": TSUN / 1047.3486,
    "saturn":  TSUN / 3497.898,
    "venus":   TSUN / 408523.71,
    "uranus":  TSUN / 22902.98,
    "neptune": TSUN / 19412.24,
}
#: Names of planets used in Shapiro delay calculation
PLANET_NAMES: tuple[str, ...] = ("jupiter", "saturn", "venus", "uranus", "neptune")

#: Ecliptic obliquity (arcseconds), from PINT's ecliptic.dat
OBLIQUITY_ARCSEC: dict[str, float] = {
    "IAU1976":  84381.448,
    "IERS1992": 84381.412,
    "DE403":    84381.412,
    "IERS2003": 84381.4059,
    "IERS2010": 84381.406,
    "IAU2005":  84381.406,
    "DEFAULT":  84381.406,
}

# ── Time conversions ───────────────────────────────────────────────

#: Seconds per day
SECS_PER_DAY: float = 86_400.0
#: Days per Julian year
DAYS_PER_JULIAN_YEAR: float = 365.25
#: Seconds per Julian year
SECS_PER_JULIAN_YEAR: float = 365.25 * 86_400.0
#: 1/year in Hz
FYR: float = 1.0 / SECS_PER_JULIAN_YEAR
#: JD to MJD offset
JD_MJD_OFFSET: float = 2_400_000.5

# ── Unit conversions ──────────────────────────────────────────────

#: Radians per milliarcsecond
RAD_PER_MAS: float = jnp.pi / (180.0 * 3600.0 * 1000.0)
#: Arcseconds to radians
ARCSEC_TO_RAD: float = jnp.pi / (180.0 * 3600.0)

# ── Dispersion ────────────────────────────────────────────────────

#: Dispersion constant (MHz^2 * s * cm^3 / pc).
#: ``delay = DM * DMCONST / freq_MHz^2``.
#: See Lorimer & Kramer, Handbook of Pulsar Astronomy, 2nd ed., p. 86 note 1.
DMCONST: float = 1.0 / 2.41e-4

# ── Troposphere (Niell 1996 mapping function) ─────────────────────

#: Latitude breakpoints in radians; corresponds to 0, 15, 30, 45, 60, 75, 90 degrees.
#: Indices 0 and 6 are copies of 1 and 5 for constant extrapolation.
NIELL_LAT_BREAKS = jnp.array(
    [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
) * (jnp.pi / 180.0)

#: Hydrostatic average coefficient a (Niell 1996, Table 1)
NIELL_A_AVG = jnp.array([1.2769934, 1.2769934, 1.2683230, 1.2465397, 1.2196049, 1.2045996, 1.2045996]) * 1e-3
#: Hydrostatic average coefficient b (Niell 1996, Table 1)
NIELL_B_AVG = jnp.array([2.9153695, 2.9153695, 2.9152299, 2.9288445, 2.9022565, 2.9024912, 2.9024912]) * 1e-3
#: Hydrostatic average coefficient c (Niell 1996, Table 1)
NIELL_C_AVG = jnp.array([62.610505, 62.610505, 62.837393, 63.721774, 63.824265, 64.258455, 64.258455]) * 1e-3

#: Hydrostatic amplitude coefficient a (Niell 1996, Table 3)
NIELL_A_AMP = jnp.array([0.0, 0.0, 1.2709626, 2.6523662, 3.4000452, 4.1202191, 4.1202191]) * 1e-5
#: Hydrostatic amplitude coefficient b (Niell 1996, Table 3)
NIELL_B_AMP = jnp.array([0.0, 0.0, 2.1414979, 3.0160779, 7.2562722, 11.723375, 11.723375]) * 1e-5
#: Hydrostatic amplitude coefficient c (Niell 1996, Table 3)
NIELL_C_AMP = jnp.array([0.0, 0.0, 9.0128400, 4.3497037, 84.795348, 170.37206, 170.37206]) * 1e-5

#: Height correction coefficient a (Niell 1996, Eq 6)
NIELL_A_HT: float = 2.53e-5
#: Height correction coefficient b (Niell 1996, Eq 6)
NIELL_B_HT: float = 5.49e-3
#: Height correction coefficient c (Niell 1996, Eq 6)
NIELL_C_HT: float = 1.14e-3

#: Wet mapping coefficient a (Niell 1996, Table 2)
NIELL_AW = jnp.array([5.8021897, 5.8021897, 5.6794847, 5.8118019, 5.9727542, 6.1641693, 6.1641693]) * 1e-4
#: Wet mapping coefficient b (Niell 1996, Table 2)
NIELL_BW = jnp.array([1.4275268, 1.4275268, 1.5138625, 1.4572752, 1.5007428, 1.7599082, 1.7599082]) * 1e-3
#: Wet mapping coefficient c (Niell 1996, Table 2)
NIELL_CW = jnp.array([4.3472961, 4.3472961, 4.6729510, 4.3908931, 4.4626982, 5.4736038, 5.4736038]) * 1e-2

#: Day-of-year offset for seasonal variation (Niell 1996, Eq 2)
NIELL_DOY_OFFSET: int = -28

# ── Kepler solver ─────────────────────────────────────────────────

#: Halley's method iterations for Kepler equation (cubically convergent;
#: 5 iterations reaches machine epsilon even at e=0.95 with Danby initial guess)
KEPLER_N_ITER: int = 5

# ── Bridge layer ──────────────────────────────────────────────────

#: PINT parameter types that map to numeric values
NUMERIC_PARAM_TYPES = frozenset({"floatParameter", "MJDParameter", "AngleParameter"})
#: Planets used for Shapiro delay and position lookups
PLANETS = ("jupiter", "saturn", "venus", "uranus", "neptune", "earth")
