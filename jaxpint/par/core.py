"""Shared parameter-conversion core.

:func:`raw_params_to_result` turns an adapter-neutral ``list[RawParam]`` (plus
the separately-detected component set and binary model) into a
:class:`~jaxpint.par.result.ParResult`.  It owns all the unit-algebra and
structural work that is identical regardless of where the parameters came from
(the PINT bridge today, the native ``.par`` parser later):

- deg->rad / (deg/yr)->(rad/s) for ``FLOAT`` params carrying a deg-based unit,
- us->s for ``EQUAD``/``ECORR`` mask params,
- pair (``WAVEn``/``IFUNC``) -> ``_A``/``_B`` value entries,
- MJD epoch routing (integer day -> ``epoch_int_values``, fractional day -> values),
- ``INT``/``BOOL``/``STR`` -> the side dicts,
- ``MaskInfo`` construction,
- alias synthesis (``RNAMP``->``TNRED*``, ``FB``->``PB``),
- the non-finite guard, and ``ParameterVector`` assembly.

This module is PINT-free.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import astropy.units as u
import jax.numpy as jnp

from jaxpint.par.aliases import apply_aliases
from jaxpint.par.raw_params import ParamKind, RawParam
from jaxpint.par.registry import BinaryModel, Component
from jaxpint.par.result import MaskInfo, ParResult
from jaxpint.types import ParameterVector

log = logging.getLogger(__name__)


# Some adapters classify these semantically-integer quantities as floats (they
# also live in the ParameterVector); mirror PINT by additionally exposing them
# via ParResult.int_params so the model builder can read them with `.get(...)`.
_INT_VALUED_FLOATS: frozenset[str] = frozenset(
    {"TNREDC", "TNDMC", "TNCHROMC", "TNSWC", "SWM", "SIFUNC"}
)


def _convert_deg_to_rad(value: float, unit_str: str):
    """Convert a (value, unit-string) carrying degrees into radian-based units.

    Replaces ``deg`` with ``rad`` in any compound unit (deg -> rad, deg/yr ->
    rad/s) using astropy's ``dimensionless_angles`` equivalency.  Returns
    ``(value, unit_str)`` if a conversion was applied, or ``None`` if the unit
    does not contain degrees.  Mirrors the former
    ``bridge.model_conversion._convert_deg_to_rad`` but operates on a
    value/unit-string pair instead of an astropy ``Quantity``.
    """
    if not unit_str:
        return None
    try:
        unit = u.Unit(unit_str)
    except (ValueError, TypeError):
        return None
    if u.deg not in getattr(unit, "bases", ()):
        return None

    rad_unit = unit
    for base, power in zip(unit.bases, unit.powers):
        if base == u.deg:
            rad_unit = rad_unit * (u.rad / u.deg) ** power

    converted = (value * unit).to(rad_unit, equivalencies=u.dimensionless_angles())
    return float(converted.value), str(converted.unit)


def _coerce_float(value: float, unit_str: str) -> tuple[float, str]:
    """deg->rad if the unit has degrees, else pass through unchanged."""
    res = _convert_deg_to_rad(value, unit_str)
    if res is not None:
        return res
    return float(value), unit_str


# Masked-parameter families PINT validates for duplicate selectors.  Plain
# ``startswith`` is unambiguous: ``DMEFAC1`` does not start with ``EFAC``.
_MASK_DUP_FAMILIES = ("EFAC", "EQUAD", "ECORR", "DMEFAC", "DMEQUAD")

# Timescales JaxPINT's timing model actually implements.  A par file with no
# UNITS line is TDB by convention (TEMPO1 predates the distinction), so absence
# is accepted.
_SUPPORTED_UNITS = frozenset({"TDB"})

# Recognized-but-unimplemented timescales, mapped to the remedy we can offer.
_KNOWN_UNSUPPORTED_UNITS = {
    "TCB": (
        "TCB differs from TDB by a rate factor of ~1.55e-8 (~0.5 s/yr in epoch, "
        "with a proportional rescaling of F0/F1 and the Keplerian parameters), "
        "so reading a TCB par as TDB silently produces wrong timing. "
        "Convert it first -- PINT ships a `tcb2tdb` command-line tool -- or use "
        "a TDB par if the release provides one (IPTA DR1/DR2 ship `.TDB.par` "
        "siblings; EPTA DR2 does not). Note the conversion is not exact and a "
        "refit is required for reliable results."
    ),
}


def _mask_selector(info: MaskInfo) -> tuple[str, str, Optional[str]]:
    """Normalized selector identity used for duplicate detection.

    PINT
    compares ``(key, key_value)`` verbatim and so misses ``EFAC -f A`` /
    ``EFAC f A`` as a duplicate even though both select identical TOAs;
    normalizing makes this check a strict superset of PINT's.
    """
    return (info.key.lstrip("-").lower(), info.key_value, info.key_value2)


def validate_mask_duplicates(mask_info: dict[str, MaskInfo]) -> None:
    """Raise if two masked parameters of one family share a selector.

    Mirrors PINT's ``ScaleToaError.validate`` / ``EcorrNoise.validate`` /
    ``ScaleDmError.validate``.  Without it, two ``EFAC -f 430_ASP`` lines
    silently become ``EFAC1`` and ``EFAC2`` over *identical* masks and the
    per-parameter fold in :func:`jaxpint.noise._white_common.apply_efac_equad`
    applies both -- a wrong sigma rather than an error.

    The leading sentence of the message is PINT's verbatim, so parity tests can
    share a single ``match=`` pattern across both stacks.
    """
    for family in _MASK_DUP_FAMILIES:
        seen: dict[tuple[str, str, Optional[str]], str] = {}
        for name in sorted(mask_info):
            if not name.startswith(family):
                continue
            info = mask_info[name]
            if not info.key:
                continue  # unset selector cannot collide
            sel = _mask_selector(info)
            if sel in seen:
                extra = f" value2={info.key_value2!r}" if info.key_value2 else ""
                raise ValueError(
                    f"'{family}s' have duplicated keys and key values. "
                    f"{seen[sel]}, {name} both select key={info.key!r} "
                    f"value={info.key_value!r}{extra}"
                )
            seen[sel] = name


def validate_units(metadata: dict[str, str]) -> None:
    """Raise unless the par file's ``UNITS`` is a timescale JaxPINT implements.

    ``UNITS`` is stored as metadata but nothing downstream acts on it, so
    without this guard a ``UNITS TCB`` par loads clean and is then treated as
    TDB -- a wrong answer rather than an error.  That is not a corner case:
    every ``UNITS`` line in IPTA DR1 (145 par files) and EPTA DR2 (76) is TCB.

    This **matches PINT's default behaviour**, it does not diverge from it.
    ``TimingModel.validate`` (``timing_model.py:412-427``) raises on ``UNITS
    TCB`` unless the caller passes ``allow_tcb=True``, and ``get_model``
    defaults it to ``False``; the message points at PINT's ``tcb2tdb`` tool.
    PINT converts only on explicit opt-in, warning that the conversion is
    approximate and the model must be re-fit.  PINT likewise raises on a
    ``UNITS`` value outside ``{None, TDB, TCB}``, which is why an unrecognized
    value raises here too.

    Implementing the conversion (see ``pint.models.tcb_conversion``) would let
    this accept TCB behind a similar opt-in flag.

    An absent ``UNITS`` is accepted as TDB (TEMPO1 predates the distinction).
    Unrecognized values raise rather than being ignored -- a typo'd timescale
    is a corrupt par file, not a default.
    """
    units = metadata.get("UNITS")
    if units is None:
        return
    key = units.strip().upper()
    if key in _SUPPORTED_UNITS:
        return
    if key in _KNOWN_UNSUPPORTED_UNITS:
        raise NotImplementedError(
            f"par file declares UNITS {units}, which JaxPINT does not support "
            f"(only {'/'.join(sorted(_SUPPORTED_UNITS))}). "
            f"{_KNOWN_UNSUPPORTED_UNITS[key]}"
        )
    raise ValueError(
        f"par file declares an unrecognized UNITS value {units!r}; "
        f"expected one of {sorted(_SUPPORTED_UNITS | set(_KNOWN_UNSUPPORTED_UNITS))} "
        "or no UNITS line at all."
    )


def raw_params_to_result(
    raw: list[RawParam],
    component_set: set[Component],
    binary_model: Optional[BinaryModel] = None,
    metadata_extra: Optional[dict[str, str]] = None,
) -> ParResult:
    """Assemble a :class:`~jaxpint.par.result.ParResult` from adapter-neutral parsed parameters.

    Parameters
    ----------
    raw
        Parsed parameters in the order they should appear in the
        ``ParameterVector`` (alias-synthesized entries are appended here).
    component_set
        Detected timing-model components (computed by the adapter; not derived
        here, so the eventual native parser's detector can be validated against
        PINT's authoritative answer).
    binary_model
        Detected binary orbital model, or ``None``.
    metadata_extra
        Adapter-derived metadata not present as a parameter line (e.g. PINT's
        precomputed ``_SWX_THETA0_RAD``).
    """
    raw = list(raw)
    apply_aliases(raw)

    names: list[str] = []
    values: list[float] = []
    units: list[str] = []
    frozen_mask: list[bool] = []
    # 1-sigma fit uncertainty per ParameterVector entry, aligned with ``values``
    # (NaN where the source reported none). Only FLOAT params carry one today
    # (e.g. PX); other kinds append NaN to stay index-aligned.
    uncertainties: list[float] = []
    epoch_int_values: dict[str, float] = {}

    metadata: dict[str, str] = {}
    int_params: dict[str, int] = {}
    bool_params: dict[str, bool] = {}
    mask_info: dict[str, MaskInfo] = {}

    for rp in raw:
        match rp.kind:
            case ParamKind.STR:
                if rp.str_value is not None:
                    metadata[rp.name] = rp.str_value

            case ParamKind.BOOL:
                if rp.bool_value is not None:
                    bool_params[rp.name] = bool(rp.bool_value)

            case ParamKind.INT:
                if rp.int_value is not None:
                    int_params[rp.name] = int(rp.int_value)

            case ParamKind.MASK:
                if rp.mask_key is not None:
                    mask_info[rp.name] = MaskInfo(
                        name=rp.name,
                        key=rp.mask_key,
                        key_value=rp.mask_key_value
                        if rp.mask_key_value is not None
                        else "",
                        key_value2=rp.mask_key_value2,
                    )
                # EQUAD/ECORR are stored in microseconds; convert to seconds to
                # match the TOAData.error convention.  All other mask params
                # (JUMP, EFAC, ...) take the float path (deg->rad is a no-op).
                # The uncertainty rides the same conversion as the value.
                assert rp.value is not None
                if rp.name.startswith("EQUAD") or rp.name.startswith("ECORR"):
                    val = float((rp.value * u.Unit(rp.unit)).to(u.s).value)
                    unit_str = "s"
                    unc = (
                        math.nan
                        if rp.uncertainty is None
                        else float((rp.uncertainty * u.Unit(rp.unit)).to(u.s).value)
                    )
                else:
                    val, unit_str = _coerce_float(rp.value, rp.unit)
                    unc = (
                        math.nan
                        if rp.uncertainty is None
                        else _coerce_float(rp.uncertainty, rp.unit)[0]
                    )
                names.append(rp.name)
                values.append(val)
                units.append(unit_str)
                frozen_mask.append(rp.frozen)
                uncertainties.append(unc)

            case ParamKind.PAIR:
                assert rp.value_pair is not None
                val_a, val_b = rp.value_pair
                for suffix, val in (("_A", val_a), ("_B", val_b)):
                    names.append(rp.name + suffix)
                    values.append(float(val))
                    units.append(rp.unit)
                    frozen_mask.append(rp.frozen)
                    uncertainties.append(math.nan)

            case ParamKind.MJD:
                assert rp.mjd_split is not None
                mjd_int, mjd_frac = rp.mjd_split
                epoch_int_values[rp.name] = float(mjd_int)
                values.append(float(mjd_frac))
                units.append("day")
                names.append(rp.name)
                frozen_mask.append(rp.frozen)
                # Epoch uncertainty is in days (adapter passes it through as-is).
                uncertainties.append(
                    math.nan if rp.uncertainty is None else float(rp.uncertainty)
                )

            case ParamKind.ANGLE:
                assert rp.value is not None
                values.append(float(rp.value))
                units.append("rad")
                names.append(rp.name)
                frozen_mask.append(rp.frozen)
                # Adapter already converted the angle uncertainty to radians.
                uncertainties.append(
                    math.nan if rp.uncertainty is None else float(rp.uncertainty)
                )

            case ParamKind.FLOAT:
                assert rp.value is not None
                val, unit_str = _coerce_float(rp.value, rp.unit)
                values.append(val)
                units.append(unit_str)
                names.append(rp.name)
                frozen_mask.append(rp.frozen)
                # Uncertainty rides the same deg->rad coercion as the value.
                uncertainties.append(
                    math.nan
                    if rp.uncertainty is None
                    else _coerce_float(rp.uncertainty, rp.unit)[0]
                )

    # Semantically-integer floats are exposed via int_params in addition to the
    # ParameterVector (mirrors PINT bridge behaviour).
    for rp in raw:
        if rp.name in _INT_VALUED_FLOATS:
            v = rp.value if rp.value is not None else rp.int_value
            if v is not None:
                int_params[rp.name] = int(v)

    if metadata_extra:
        metadata.update(metadata_extra)

    non_finite = [
        (n, v, u_)
        for n, v, u_ in zip(names, values, units)
        if not math.isfinite(float(v))
    ]
    if non_finite:
        items = ", ".join(f"{n}={v} [{u_}]" for n, v, u_ in non_finite[:10])
        more = f" (+{len(non_finite) - 10} more)" if len(non_finite) > 10 else ""
        raise ValueError(
            f"Non-finite parameter value(s): {items}{more}. "
            "Check the par file for missing or unset parameters that "
            "synthesize aliases (e.g. PB from FB0=0, TNREDAMP from RNAMP=0)."
        )

    # Runs after the MASK loop (so TNEQ->EQUAD synthesis in apply_aliases is
    # already reflected), mirroring PINT's validate()-after-setup() ordering.
    validate_mask_duplicates(mask_info)

    # After metadata_extra is merged, so an adapter-supplied UNITS is seen too.
    validate_units(metadata)

    param_vector = ParameterVector(
        values=jnp.asarray(values, dtype=jnp.float64),
        frozen_mask=tuple(frozen_mask),
        names=tuple(names),
        units=tuple(units),
        uncertainties=tuple(uncertainties),
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
