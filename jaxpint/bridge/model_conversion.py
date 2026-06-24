"""Model conversion: PINT TimingModel <-> JaxPINT ParameterVector.

The forward path (:func:`pint_model_to_params`) is a thin PINT adapter:
:func:`_pint_to_raw_params` reads each PINT parameter object into an
adapter-neutral :class:`~jaxpint.par.raw_params.RawParam`, and
:func:`_pint_detect_components` reads PINT's authoritative component registry.
All unit coercion, alias synthesis, classification, and ``ParameterVector``
assembly then happen in the shared, PINT-free
:func:`jaxpint.par.core.raw_params_to_result` -- the same core the future native
``.par`` parser will use.

The reverse path (:func:`params_to_pint_model`) writes fitted values back into a
PINT model; it is PINT-specific and not shared.
"""

from __future__ import annotations

import logging
from typing import Optional, cast

import astropy.units as u
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
from jaxpint.par.components import PINT_COMPONENT_MAP as _PINT_COMPONENT_MAP
from jaxpint.par.core import raw_params_to_result
from jaxpint.par.raw_params import ParamKind, RawParam
from jaxpint.par.registry import BinaryModel, Component
from jaxpint.par.result import ParResult
from jaxpint.types import ParameterVector

log = logging.getLogger(__name__)


# PINT float parameters that are metadata, not fitted quantities, and whose
# values may legitimately be non-finite (e.g. `TZRFRQ=inf` for an
# asymptotic-frequency reference TOA). They're stashed in `metadata` and kept
# out of the JAX-backed values vector, where `inf` would trip `JAX_DEBUG_INFS`
# at array construction.
_METADATA_ONLY_FLOAT_PARAMS: frozenset[str] = frozenset({"TZRFRQ"})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _split_epoch_jd(quantity) -> tuple[float, float]:
    """Split a single astropy Time (from an MJDParameter) into MJD int + frac."""
    mjd_int, mjd_frac = _split_mjd_time(quantity)
    return float(mjd_int), float(mjd_frac)


# ---------------------------------------------------------------------------
# PINT adapter: PINT TimingModel -> list[RawParam] + detected components
# ---------------------------------------------------------------------------


def _bridge_uncertainty(param) -> Optional[float]:
    """PINT 1-sigma in the parameter's native unit, or None.

    Maps PINT's ``0.0`` placeholder (which it stores for a fitted/frozen param
    that has a fit flag but no explicit sigma) to None, so the bridge matches the
    native parser, where such a line yields "no uncertainty" (NaN). A genuine
    0.0 sigma is meaningless, so this is safe."""
    unc = getattr(param, "uncertainty_value", None)
    try:
        unc = float(unc) if unc is not None else None
    except (TypeError, ValueError):
        return None
    return None if (unc is None or unc == 0.0) else unc


def _bridge_uncertainty_rad(param) -> Optional[float]:
    """Angle 1-sigma converted to radians (None if unset / 0.0 placeholder).

    The native ANGLE path stores the uncertainty in radians, so the bridge must
    too -- ``uncertainty_value`` would be in hourangle/deg. Uses the Quantity."""
    q = getattr(param, "uncertainty", None)
    if q is None:
        return None
    try:
        val = float(q.to(u.rad).value)
    except (AttributeError, TypeError, ValueError):
        return None
    return None if val == 0.0 else val


def _pint_to_raw_params(model: PINTTimingModel) -> list[RawParam]:
    """Read each set PINT parameter object into an adapter-neutral RawParam.

    Does only the source-specific extraction (precision-preserving MJD split via
    astropy ``jd1/jd2``, angle -> radians, mask key/value extraction, pair
    splitting); all unit-algebra and assembly is deferred to
    :func:`jaxpint.par.core.raw_params_to_result`.  Parameters in
    ``model.params`` order; unset parameters (``quantity is None``) are skipped.
    """
    raw: list[RawParam] = []

    for pname in model.params:
        param = getattr(model, pname)

        # Non-numeric parameter types -> side dicts (handled by the core).
        if isinstance(param, strParameter):
            if param.value is not None:
                raw.append(RawParam(pname, ParamKind.STR, str_value=str(param.value)))
            continue
        if isinstance(param, boolParameter):
            if param.value is not None:
                raw.append(RawParam(pname, ParamKind.BOOL, bool_value=bool(param.value)))
            continue
        if isinstance(param, intParameter):
            if param.value is not None:
                raw.append(RawParam(pname, ParamKind.INT, int_value=int(param.value)))
            continue

        # Metadata-only floats (e.g. TZRFRQ=inf): stash as a string so the value
        # stays out of the JAX array.
        if pname in _METADATA_ONLY_FLOAT_PARAMS:
            if getattr(param, "value", None) is not None:
                raw.append(RawParam(pname, ParamKind.STR, str_value=str(param.value)))
            continue

        # funcParameter has no .value -- skip anything without it; skip unset.
        if not hasattr(param, "value") or not hasattr(param, "quantity"):
            continue
        if param.quantity is None:
            continue

        # Pair parameters (WAVE, IFUNC): [value_a, value_b].  Detected via
        # pairParameter type OR prefixParameter with parameter_type="pair".
        is_pair = isinstance(param, pairParameter) or (
            hasattr(param, "parameter_type") and param.parameter_type == "pair"
        )
        if is_pair:
            pair = param.quantity
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            val_a = float(pair[0].value) if hasattr(pair[0], "value") else float(pair[0])
            val_b = float(pair[1].value) if hasattr(pair[1], "value") else float(pair[1])
            unit_str = str(param.units) if param.units is not None else ""
            raw.append(RawParam(
                pname, ParamKind.PAIR, value_pair=(val_a, val_b),
                unit=unit_str, frozen=param.frozen,
            ))
            continue

        # Mask parameters (JUMP/EFAC/EQUAD/ECORR/...): carry both the scalar
        # value and the key info; the core builds MaskInfo and converts
        # EQUAD/ECORR us->s.
        if isinstance(param, maskParameter):
            mask_key = None
            mkv = None
            mkv2 = None
            if hasattr(param, "key") and param.key is not None:
                kv = param.key_value
                if isinstance(kv, (list, tuple)) and len(kv) == 2:
                    mkv = str(kv[0])
                    mkv2 = str(kv[1])
                elif isinstance(kv, (list, tuple)) and len(kv) == 1:
                    # single-value key (e.g. -fe / -f / tel): store the bare
                    # value, not the list repr, so it matches the native parser
                    # and is usable for TOA flag matching.
                    mkv = str(kv[0])
                elif kv is not None:
                    mkv = str(kv)
                else:
                    mkv = ""
                mask_key = str(param.key)
            raw.append(RawParam(
                pname, ParamKind.MASK, value=float(cast(float, param.value)),
                uncertainty=_bridge_uncertainty(param),   # native unit; core converts
                unit=str(param.units), frozen=param.frozen,
                mask_key=mask_key, mask_key_value=mkv, mask_key_value2=mkv2,
            ))
            continue

        if isinstance(param, MJDParameter):
            mjd_int, mjd_frac = _split_epoch_jd(param.quantity)
            raw.append(RawParam(
                pname, ParamKind.MJD, mjd_split=(mjd_int, mjd_frac),
                uncertainty=_bridge_uncertainty(param),   # days
                frozen=param.frozen,
            ))
            continue

        if isinstance(param, AngleParameter):
            val_rad = float(param.quantity.to(u.rad).value)
            raw.append(RawParam(
                pname, ParamKind.ANGLE, value=val_rad,
                uncertainty=_bridge_uncertainty_rad(param),
                frozen=param.frozen,
            ))
            continue

        if (
            isinstance(param, prefixParameter)
            and getattr(param, "parameter_type", None) == "MJD"
        ):
            # MJD-type prefix parameters (GLEP_N, DMXR1_N, PWEP_N): split the raw
            # MJD float into integer day + fractional day.
            mjd_val = float(param.value)
            mjd_int = float(int(mjd_val))
            mjd_frac = mjd_val - mjd_int
            raw.append(RawParam(
                pname, ParamKind.MJD, mjd_split=(mjd_int, mjd_frac),
                uncertainty=_bridge_uncertainty(param),   # days
                frozen=param.frozen,
            ))
            continue

        # floatParameter, non-MJD prefixParameter, etc.  Pass the native unit
        # string through; the core converts deg-based units to radian-based.
        # _bridge_uncertainty mirrors PINT's 1-sigma (native unit) so the bridge
        # path matches the native parser (text_adapter).
        raw.append(RawParam(
            pname, ParamKind.FLOAT, value=float(cast(float, param.value)),
            uncertainty=_bridge_uncertainty(param),
            unit=str(param.units), frozen=param.frozen,
        ))

    return raw


def _pint_detect_components(
    model: PINTTimingModel,
) -> tuple[set[Component], Optional[BinaryModel], dict[str, str]]:
    """Read PINT's authoritative component registry into JaxPINT's vocabulary.

    Returns ``(component_set, binary_model, metadata_extra)`` where
    ``metadata_extra`` carries derived metadata that isn't a parameter line
    (currently the SolarWindDispersionX ``theta0``).
    """
    component_set: set[Component] = set()
    binary_model: Optional[BinaryModel] = None
    metadata_extra: dict[str, str] = {}

    for comp_name, comp in model.components.items():
        if comp_name in ("AbsPhase",):
            continue
        if comp_name in _PINT_COMPONENT_MAP:
            component_set.add(_PINT_COMPONENT_MAP[comp_name])
            #  TODO: HACKY way of storing theta0. Need to find a workaround.
            # Precompute theta0 for SolarWindDispersionX from PINT.
            if comp_name == "SolarWindDispersionX":
                try:
                    theta0_rad = float(comp.theta0.to(u.rad).value)
                    metadata_extra["_SWX_THETA0_RAD"] = str(theta0_rad)
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

    return component_set, binary_model, metadata_extra


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pint_model_to_params(model: PINTTimingModel) -> ParResult:
    """Convert a PINT TimingModel to a JaxPINT :class:`ParResult`.

    Thin composition: read the PINT model into ``RawParam``s and detect its
    components, then hand both to the shared
    :func:`jaxpint.par.core.raw_params_to_result`.  MJD epochs are split into a
    static integer day and a dynamic fractional day; angles are converted to
    radians; degree-based rates to radian-based; EQUAD/ECORR us->s; pair
    parameters split into ``_A``/``_B``.

    Parameters
    ----------
    model : pint.models.TimingModel
        The timing model to extract parameters from.

    Returns
    -------
    ParResult
        Container with the ``ParameterVector``, detected components, optional
        binary model, string metadata, mask info, integer and boolean params.
    """
    raw = _pint_to_raw_params(model)
    component_set, binary_model, metadata_extra = _pint_detect_components(model)
    return raw_params_to_result(raw, component_set, binary_model, metadata_extra)


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
