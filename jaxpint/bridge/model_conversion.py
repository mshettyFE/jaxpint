"""Model conversion: PINT TimingModel ↔ JaxPINT ParameterVector.

Handles extracting numeric parameters from a PINT model into a flat
JAX array (with unit conversion for angles and epochs), and writing
fitted values back.
"""

from __future__ import annotations

import logging
from typing import Optional

import astropy.units as u
import jax.numpy as jnp
from pint.models.parameter import (
    AngleParameter,
    MJDParameter,
    boolParameter,
    intParameter,
    maskParameter,
    pairParameter,
    prefixParameter,
    strParameter,
)

from pint.models.timing_model import TimingModel as PINTTimingModel
from pint.models.pulsar_binary import PulsarBinary as PINTPulsarBinary

from jaxpint.bridge.toa_conversion import _split_mjd_time
from jaxpint.bridge._aliases import (
    synthesize_pb_from_fb,
    synthesize_tnredamp_from_rnamp,
)
from jaxpint.bridge._param_builder import ParResult, MaskInfo
from jaxpint.bridge._registry import BinaryModel, Component
from jaxpint.types import ParameterVector

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PINT component name → JaxPINT Component enum mapping
# ---------------------------------------------------------------------------

_PINT_COMPONENT_MAP: dict[str, Component] = {
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _convert_deg_to_rad(quantity):
    """Convert a quantity with degree-based units to radian-based units.

    Uses astropy's ``dimensionless_angles()`` equivalency to replace
    degrees with radians in any compound unit (e.g. deg → rad, deg/yr → rad/s).

    Returns (value, unit_string) if conversion was applied, or None if the
    quantity does not contain degrees.
    """
    try:
        unit = quantity.unit
        if u.deg not in unit.bases:
            return None
    except (AttributeError, TypeError):
        return None

    # Build the target unit by replacing deg with rad in the decomposition
    rad_unit = unit
    for base, power in zip(unit.bases, unit.powers):
        if base == u.deg:
            rad_unit = rad_unit * (u.rad / u.deg) ** power

    converted = quantity.to(rad_unit, equivalencies=u.dimensionless_angles())
    return float(converted.value), str(converted.unit)


def _split_epoch_jd(quantity) -> tuple[float, float]:
    """Split a single astropy Time (from an MJDParameter) into MJD int + frac."""
    mjd_int, mjd_frac = _split_mjd_time(quantity)
    return float(mjd_int), float(mjd_frac)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pint_model_to_params(model: PINTTimingModel) -> ParResult:
    """Convert a PINT TimingModel to a JaxPINT :class:`ParResult`.

    Iterates all parameters, skipping non-numeric types (str, bool, int,
    func) for the ParameterVector but collecting them into metadata,
    bool_params, and int_params dicts.  MJD epochs are split into a
    static integer day and a dynamic fractional day.  Angles are
    converted to radians.

    Parameters
    ----------
    model : pint.models.TimingModel
        The timing model to extract parameters from.

    Returns
    -------
    ParResult
        A container holding the ``ParameterVector``, the set of detected
        timing-model components, optional binary model identifier,
        string metadata, mask info, integer parameters, and boolean
        parameters extracted from the PINT model.
    """
    names: list[str] = []
    values: list[float] = []
    units: list[str] = []
    frozen_mask: list[bool] = []
    epoch_int_values: dict[str, float] = {}

    metadata: dict[str, str] = {}
    int_params: dict[str, int] = {}
    bool_params: dict[str, bool] = {}
    mask_info: dict[str, MaskInfo] = {}

    for pname in model.params:
        param = getattr(model, pname)

        # Collect non-numeric parameter types into side dicts
        if isinstance(param, strParameter):
            if param.value is not None:
                metadata[pname] = str(param.value)
            continue
        if isinstance(param, boolParameter):
            if param.value is not None:
                bool_params[pname] = bool(param.value)
            continue
        if isinstance(param, intParameter):
            if param.value is not None:
                int_params[pname] = int(param.value)
            continue

        # funcParameter has no .value — skip anything without it
        if not hasattr(param, "value") or not hasattr(param, "quantity"):
            continue
        # Skip unset parameters
        if param.quantity is None:
            continue

        # Collect mask_info for maskParameters
        if isinstance(param, maskParameter):
            if hasattr(param, 'key') and param.key is not None:
                key_value2 = None
                if hasattr(param, 'key_value') and isinstance(param.key_value, (list, tuple)) and len(param.key_value) == 2:
                    key_value2 = str(param.key_value[1])
                    kv = str(param.key_value[0])
                else:
                    kv = str(param.key_value) if param.key_value is not None else ""
                mask_info[pname] = MaskInfo(
                    name=pname,
                    key=str(param.key),
                    key_value=kv,
                    key_value2=key_value2,
                )

        # Pair parameters (e.g. WAVE, IFUNC) store [value_a, value_b] lists.
        # Detected via pairParameter type OR prefixParameter with parameter_type="pair".
        # Split into two ParameterVector entries with _A / _B suffixes.
        is_pair = isinstance(param, pairParameter) or (
            hasattr(param, "parameter_type") and param.parameter_type == "pair"
        )
        if is_pair:
            pair = param.quantity
            if pair is None:
                continue
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            val_a = float(pair[0].value) if hasattr(pair[0], "value") else float(pair[0])
            val_b = float(pair[1].value) if hasattr(pair[1], "value") else float(pair[1])
            unit_str = str(param.units) if param.units is not None else ""
            for suffix, val in [("_A", val_a), ("_B", val_b)]:
                names.append(pname + suffix)
                values.append(val)
                units.append(unit_str)
                frozen_mask.append(param.frozen)
            continue

        if isinstance(param, MJDParameter):
            mjd_int, mjd_frac = _split_epoch_jd(param.quantity)
            epoch_int_values[pname] = mjd_int
            values.append(mjd_frac)
            units.append("day")

        elif isinstance(param, AngleParameter):
            val_rad = float(param.quantity.to(u.rad).value)
            values.append(val_rad)
            units.append("rad")

        elif isinstance(param, maskParameter) and (
            pname.startswith("EQUAD") or pname.startswith("ECORR")
        ):
            # EQUAD/ECORR are stored in microseconds in PINT; convert to
            # seconds to match TOAData.error convention.
            values.append(float(param.quantity.to(u.s).value))
            units.append("s")

        elif (
            isinstance(param, prefixParameter)
            and getattr(param, "parameter_type", None) == "MJD"
        ):
            # MJD-type prefix parameters (e.g. GLEP_N, DMXR1_N, PWEP_N):
            # split the raw MJD float into integer day + fractional day.
            mjd_val = float(param.value)
            mjd_int = float(int(mjd_val))
            mjd_frac = mjd_val - mjd_int
            epoch_int_values[pname] = mjd_int
            values.append(mjd_frac)
            units.append("day")

        else:
            # floatParameter, prefixParameter, maskParameter, etc.
            # Convert degree-based units to radian-based (e.g. OM deg → rad,
            # OMDOT deg/yr → rad/s) so binary models get radians throughout.
            deg_result = _convert_deg_to_rad(param.quantity)
            if deg_result is not None:
                val, unit_str = deg_result
                values.append(val)
                units.append(unit_str)
            else:
                values.append(float(param.value))
                units.append(str(param.units))

        names.append(pname)
        frozen_mask.append(param.frozen)

    # Some PINT floatParameters are semantically integers and the standalone
    # parser puts them in int_params.  Mirror that here so _model_builder
    # picks them up via par.int_params.get(...).
    _INT_VALUED_FLOATS = {"TNREDC", "TNDMC", "TNCHROMC", "TNSWC", "SWM", "SIFUNC"}
    for ivname in _INT_VALUED_FLOATS:
        if hasattr(model, ivname):
            p = getattr(model, ivname)
            if p.value is not None:
                int_params[ivname] = int(p.value)

    # Synthesize canonical parameter names from alternate par-file conventions.
    # Each alias mutates the in-progress lists in place; see jaxpint.bridge._aliases.
    synthesize_tnredamp_from_rnamp(model, names, values, units, frozen_mask)
    synthesize_pb_from_fb(model, names, values, units, frozen_mask)

    # Build component_set from PINT model's component registry
    component_set: set[Component] = set()
    binary_model: Optional[BinaryModel] = None

    for comp_name, comp in model.components.items():
        if comp_name in ("AbsPhase",):
            continue
        if comp_name in _PINT_COMPONENT_MAP:
            component_set.add(_PINT_COMPONENT_MAP[comp_name])
            #  TODO: HACKY way of storing theta0. Need to find a workaround 
            # Precompute theta0 for SolarWindDispersionX from PINT
            if comp_name == "SolarWindDispersionX":
                try:
                    theta0_rad = float(comp.theta0.to(u.rad).value)
                    metadata["_SWX_THETA0_RAD"] = str(theta0_rad)
                except Exception:
                    log.warning("Could not compute SWX theta0 from PINT")
        elif isinstance(comp, PINTPulsarBinary):
            bname = comp.binary_model_name
            try:
                binary_model = BinaryModel(bname)
            except ValueError:
                log.warning("Unknown binary model %r", bname)
            if bname == "BT_piecewise":
                component_set.add(Component.BINARY_BT_PIECEWISE)
            else:
                component_set.add(Component.BINARY)
        else:
            log.warning(
                "Unknown PINT component %r (%s) — skipping for component_set",
                comp_name, type(comp).__name__,
            )

    param_vector = ParameterVector(
        values=jnp.asarray(values, dtype=jnp.float64),
        frozen_mask=tuple(frozen_mask),
        names=tuple(names),
        units=tuple(units),
        epoch_int_values=epoch_int_values,
    )

    return ParResult(
        params=param_vector,
        component_set=component_set,
        binary_model=binary_model,
        metadata=metadata,
        mask_info=mask_info,
        int_params=int_params,
        bool_params=bool_params,
    )


def params_to_pint_model(
    params: ParameterVector,
    model: PINTTimingModel,
) -> PINTTimingModel:
    """Write JaxPINT parameter values back into a PINT TimingModel.

    Modifies *model* in-place and returns it.  The caller should copy
    the model first (``copy.deepcopy(model)``) if the original must be
    preserved.

    Parameters
    ----------
    params : ParameterVector
        The (possibly fitted) parameter values.
    model : pint.models.TimingModel
        The PINT model to update.

    Returns
    -------
    pint.models.TimingModel
        The same *model* instance, modified in-place with the updated
        parameter values (angles converted back to degrees, epochs
        reconstructed from integer + fractional day, etc.).
    """
    # Collect pair parameter halves (_A / _B) for recombination
    pair_halves: dict[str, dict[str, float]] = {}  # base_name → {"_A": val, "_B": val}

    for i, pname in enumerate(params.names):
        # Handle pair parameter suffixes
        if pname.endswith("_A") or pname.endswith("_B"):
            base = pname[:-2]
            suffix = pname[-2:]
            if hasattr(model, base):
                p = getattr(model, base)
                is_pair = isinstance(p, pairParameter) or (
                    hasattr(p, "parameter_type") and p.parameter_type == "pair"
                )
                if is_pair:
                    pair_halves.setdefault(base, {})[suffix] = float(params.values[i])
                    continue

        param = getattr(model, pname)
        val = float(params.values[i])

        if isinstance(param, MJDParameter):
            # Reconstruct full MJD from integer + fractional day
            full_mjd = params.epoch_int_values[pname] + val
            param.value = full_mjd

        elif (
            isinstance(param, prefixParameter)
            and pname in params.epoch_int_values
        ):
            # MJD-type prefix parameters: reconstruct full MJD
            param.value = params.epoch_int_values[pname] + val

        elif isinstance(param, AngleParameter):
            # Convert radians back to the parameter's native angle unit
            native_value = float((val * u.rad).to(param.units).value)
            param.value = native_value

        elif isinstance(param, maskParameter) and (
            pname.startswith("EQUAD") or pname.startswith("ECORR")
        ):
            # Convert seconds back to the parameter's native unit (microseconds)
            native_value = float((val * u.s).to(param.units).value)
            param.value = native_value

        else:
            # If we converted deg→rad on the way in, convert back using
            # the stored unit string to reconstruct the radian-based unit.
            stored_unit_str = params.units[i]
            native_unit = param.units
            if native_unit is not None and stored_unit_str != str(native_unit):
                stored_unit = u.Unit(stored_unit_str)
                native_value = float(
                    (val * stored_unit).to(
                        native_unit, equivalencies=u.dimensionless_angles()
                    ).value
                )
                param.value = native_value
            else:
                param.value = val

    # Recombine pair parameter halves
    for base, halves in pair_halves.items():
        param = getattr(model, base)
        val_a = halves.get("_A", 0.0)
        val_b = halves.get("_B", 0.0)
        param.value = (val_a, val_b)

    return model
