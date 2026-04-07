"""Component ordering mirroring PINT's DEFAULT_ORDER.

Defines the canonical execution order for timing model components.
Used by :func:`jaxpint.parfile._model_builder.build_model` to process
components via a priority queue so that delays are chained in the
correct physical order.
"""

from __future__ import annotations

from jaxpint.parfile._registry import Component

# Mirrors PINT's DEFAULT_ORDER (timing_model.py:119-135), extended with
# categories PINT doesn't list (they sort to end in PINT) and noise
# components at the bottom.
DEFAULT_ORDER: tuple[Component, ...] = (
    # --- Delay components (PINT ordering) ---
    Component.ASTROMETRY_EQUATORIAL,
    Component.ASTROMETRY_ECLIPTIC,
    Component.TROPOSPHERE_DELAY,
    Component.SOLAR_SYSTEM_SHAPIRO,
    Component.SOLAR_WIND_DISPERSION,
    Component.SOLAR_WIND_DISPERSION_X,
    Component.DISPERSION_DM,
    Component.DISPERSION_DMX,
    Component.DISPERSION_JUMP,
    Component.BINARY,
    Component.BINARY_BT_PIECEWISE,
    Component.FREQUENCY_DEPENDENT,
    Component.FD_JUMP,
    Component.CHROMATIC_CM,
    Component.CHROMATIC_CMX,
    Component.EXPONENTIAL_DIP,
    Component.WAVE_X,
    Component.DM_WAVE_X,
    Component.CM_WAVE_X,
    # --- Phase components ---
    Component.SPINDOWN,
    Component.GLITCH,
    Component.PIECEWISE_SPINDOWN,
    Component.PHASE_JUMP,
    Component.WAVE,
    Component.IFUNC,
    # --- Noise components ---
    Component.SCALE_TOA_ERROR,
    Component.SCALE_DM_ERROR,
    Component.ECORR_NOISE,
    Component.PL_RED_NOISE,
    Component.PL_DM_NOISE,
    Component.PL_CHROM_NOISE,
    Component.PL_SW_NOISE,
)

# Priority lookup: Component → position in DEFAULT_ORDER.
# Components not in DEFAULT_ORDER get len(DEFAULT_ORDER) (sort to end).
PRIORITY: dict[Component, int] = {comp: i for i, comp in enumerate(DEFAULT_ORDER)}
