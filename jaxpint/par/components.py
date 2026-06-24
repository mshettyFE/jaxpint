"""Component auto-detection for the native ``.par`` parser.

Detection is driven by the components' own ``PARAMS`` declarations (aggregated
in :mod:`jaxpint.par.spec`): a parameter marked ``triggers=True`` activates its
component when present.  ``SolarSystemShapiro`` is auto-added with astrometry,
and the binary model comes from the ``BINARY`` line.  This produces the same
``(component_set, binary_model)`` the PINT bridge reads from ``model.components``.

Also owns ``PINT_COMPONENT_MAP`` (PINT component class name -> JaxPINT
``Component`` enum), used only by the PINT bridge.  PINT-free.
"""

from __future__ import annotations

import logging
from typing import Optional

from jaxpint.par import spec as S
from jaxpint.par.registry import BinaryModel, Component

log = logging.getLogger(__name__)


# PINT component class name -> JaxPINT Component enum.  Used only by the PINT
# bridge (which reads PINT's own model.components); the native detector uses the
# components' trigger declarations instead.
PINT_COMPONENT_MAP: dict[str, Component] = {
    "Spindown": Component.SPINDOWN,
    "PhaseOffset": Component.PHASE_OFFSET,
    "AstrometryEquatorial": Component.ASTROMETRY_EQUATORIAL,
    "AstrometryEcliptic": Component.ASTROMETRY_ECLIPTIC,
    "DispersionDM": Component.DISPERSION_DM,
    "DispersionDMX": Component.DISPERSION_DMX,
    "DispersionJump": Component.DISPERSION_JUMP,
    "SolarSystemShapiro": Component.SOLAR_SYSTEM_SHAPIRO,
    "SolarWindDispersion": Component.SOLAR_WIND_DISPERSION,
    "SolarWindDispersionX": Component.SOLAR_WIND_DISPERSION_X,
    "TroposphereDelay": Component.TROPOSPHERE_DELAY,
    "PhaseJump": Component.PHASE_JUMP,
    "ScaleToaError": Component.SCALE_TOA_ERROR,
    "ScaleDmError": Component.SCALE_DM_ERROR,
    "EcorrNoise": Component.ECORR_NOISE,
    "PLRedNoise": Component.PL_RED_NOISE,
    "PLDMNoise": Component.PL_DM_NOISE,
    "PLChromNoise": Component.PL_CHROM_NOISE,
    "PLSWNoise": Component.PL_SW_NOISE,
    "WaveX": Component.WAVE_X,
    "DMWaveX": Component.DM_WAVE_X,
    "CMWaveX": Component.CM_WAVE_X,
    "ChromaticCM": Component.CHROMATIC_CM,
    "ChromaticCMX": Component.CHROMATIC_CMX,
    "FD": Component.FREQUENCY_DEPENDENT,
    "FDJump": Component.FD_JUMP,
    "SimpleExponentialDip": Component.EXPONENTIAL_DIP,
    "PiecewiseSpindown": Component.PIECEWISE_SPINDOWN,
    "Wave": Component.WAVE,
    "IFunc": Component.IFUNC,
    "Glitch": Component.GLITCH,
}

_ASTROMETRY = {Component.ASTROMETRY_EQUATORIAL, Component.ASTROMETRY_ECLIPTIC}


def _detect_binary(
    templates: set[str], binary_value: Optional[str]
) -> tuple[Optional[BinaryModel], Optional[Component]]:
    """Resolve the binary model from the BINARY line (guessed fallback if absent)."""
    name = binary_value
    if name is None:
        if not (templates & S.BINARY_PARAMS):
            return None, None
        name = S.BINARY_PRIORITY[0]  # priority-ordered fallback (rare; no BINARY line)
        log.warning("No BINARY line but binary params present; guessing %r", name)

    try:
        binary_model = BinaryModel(name)
    except ValueError:
        log.warning("Unknown binary model %r", name)
        return None, None

    comp = Component.BINARY_BT_PIECEWISE if name == "BT_piecewise" else Component.BINARY
    return binary_model, comp


def detect_components(
    templates: set[str], binary_value: Optional[str] = None
) -> tuple[set[Component], Optional[BinaryModel]]:
    """Return ``(component_set, binary_model)`` for the present parameters.

    A present (template) parameter activates its component via
    ``spec.TRIGGER_MAP``; ``SolarSystemShapiro`` is auto-added with astrometry;
    the binary model comes from the ``BINARY`` line.
    """
    trigger_map = S.TRIGGER_MAP
    component_set: set[Component] = {
        trigger_map[t] for t in templates if t in trigger_map
    }

    if component_set & _ASTROMETRY:
        component_set.add(Component.SOLAR_SYSTEM_SHAPIRO)

    binary_model, binary_comp = _detect_binary(templates, binary_value)
    if binary_comp is not None:
        component_set.add(binary_comp)

    return component_set, binary_model
