"""Native ``.par`` text adapter: ``ParLine`` list -> ``list[RawParam]``.

The native analogue of ``jaxpint.bridge.model_conversion._pint_to_raw_params``:
it resolves each par name to its canonical name + type via the spec aggregated
from each component's ``PARAMS`` (:mod:`jaxpint.par.spec`), does the
source-specific extraction
(sexagesimal angles via astropy, direct-string MJD split, mask key parsing,
pair splitting, repeatable-family indexing), and emits ``RawParam``s.  All
unit-algebra / classification / assembly is then done by
:func:`jaxpint.par.core.raw_params_to_result`.

"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

from jaxpint.par import spec as S
from jaxpint.par.parfile import ParLine
from jaxpint.par.raw_params import ParamKind, RawParam

log = logging.getLogger(__name__)

_TRAIL_INT = re.compile(r"^(.*?)(\d+)$")
_BOOL_TRUE = {"Y", "YES", "T", "TRUE", "1"}
_BOOL_FALSE = {"N", "NO", "F", "FALSE", "0"}


@dataclass
class ParsedPar:
    """Result of the text adapter."""
    raw_params: list[RawParam] = field(default_factory=list)
    templates: set[str] = field(default_factory=set)   # canonical first-init names present
    binary_value: Optional[str] = None                  # the BINARY line's value, if any


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def _fortran(s: str) -> str:
    """Normalize Fortran-style exponent markers (1.2D-3 -> 1.2e-3)."""
    return s.replace("D", "e").replace("d", "e")


def _fortran_float(s: str) -> float:
    return float(_fortran(s))


def _parse_bool(s: str) -> bool:
    t = s.strip().upper()
    if t in _BOOL_TRUE:
        return True
    if t in _BOOL_FALSE:
        return False
    return bool(float(s))


def _split_mjd_string(tok: str) -> tuple[float, float]:
    """Split an MJD string into (int_day, frac_day) at longdouble precision.

    PINT-free; matches PINT's pulsar_mjd split to ~1 ULP for ordinary days
    (leap-second-day exactness is deferred to the .tim/clock phase)."""
    ld = np.longdouble(_fortran(tok))
    mjd_int = np.floor(ld)
    return float(mjd_int), float(ld - mjd_int)


def _frozen(trailing: tuple[str, ...], default: bool = True) -> bool:
    """Determine frozen from the tokens after the value: a leading 0/1 is the
    fit flag (1 -> free, 0 -> frozen); anything else is an uncertainty.  With no
    fit flag the param keeps its PINT default frozen state (*default*)."""
    if trailing:
        t0 = trailing[0]
        if t0 == "1":
            return False
        if t0 == "0":
            return True
    return default


def _num_key_values(key: str) -> int:
    """How many key-values a mask key takes (mjd/freq are inclusive ranges)."""
    return 2 if key in ("mjd", "freq") else 1


# ---------------------------------------------------------------------------
# Name resolution (alias + prefix), mirroring PINT alias_to_pint_param
# ---------------------------------------------------------------------------


def _split_trailing_int(name: str):
    m = _TRAIL_INT.match(name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(2)


def _next_family_index(template: str, counters: dict[str, int]) -> str:
    """Next sequential canonical name for a repeatable family written by its bare
    base (e.g. repeated ``EQUAD`` -> EQUAD1, EQUAD2; ``JUMP`` -> JUMP1, ...)."""
    prefix = S.CANONICAL_PREFIX[template]
    base = _split_trailing_int(template)[1]
    n = counters.get(template, 0)
    counters[template] = n + 1
    return f"{prefix}{base + n}"


def _resolve(name: str, counters: dict[str, int]):
    """Resolve a raw par name to (canonical_name, template_name, spec) or None.

    A plain float resolves to a default spec (its unit is documentation the
    runtime ignores), so it need not have an explicit ``PARAM_SPEC`` entry.
    """
    name = name.upper()

    # Exact known param (RAJ, DM, PEPOCH, F0, JUMP1, DMX_0001, BINARY, ...).
    s = S.spec_for(name)
    if s is not None:
        return name, name, s

    # Non-indexed alias (RA->RAJ, T2EQUAD->EQUAD1, EQUAD->EQUAD1, JUMP->JUMP1).
    if name in S.ALIAS_MAP:
        template = S.ALIAS_MAP[name]
        tspec = S.spec_for(template)
        if tspec is None:
            return None
        if tspec.get("is_prefix"):
            # bare repeatable family base -> next sequential index
            return _next_family_index(template, counters), template, tspec
        return template, template, tspec

    # Prefix + index (F2, DMX_0023, EQUAD3, T2EQUAD2, ...).
    split = _split_trailing_int(name)
    if split:
        pfx, _idx, digits = split
        if pfx in S.PREFIX_MAP:
            template = S.PREFIX_MAP[pfx]
            tspec = S.spec_for(template)
            canon_pfx = S.CANONICAL_PREFIX.get(template, pfx)
            return canon_pfx + digits, template, tspec

    return None


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _emit(canonical: str, spec: dict, tokens: tuple[str, ...]) -> Optional[RawParam]:
    kind = spec["kind"]
    unit = spec.get("unit", "")
    fd = spec.get("frozen_default", True)

    if not tokens:
        return None

    match kind:
        case "mask":
            key = tokens[0]
            nkv = _num_key_values(key)
            if len(tokens) < 1 + nkv + 1:
                return None
            kvs = tokens[1:1 + nkv]
            trailing = tokens[2 + nkv:]
            # mjd/freq ranges are numeric in PINT (str(float(...))); flags stay raw.
            numeric = key in ("mjd", "freq")
            kv1 = str(float(kvs[0])) if numeric else kvs[0]
            kv2 = (str(float(kvs[1])) if numeric else kvs[1]) if nkv == 2 else None
            return RawParam(
                canonical, ParamKind.MASK, value=_fortran_float(tokens[1 + nkv]),
                unit=unit, frozen=_frozen(trailing, fd),
                mask_key=key, mask_key_value=kv1, mask_key_value2=kv2,
            )

        case "str":
            return RawParam(canonical, ParamKind.STR, str_value=tokens[0])

        case "bool":
            return RawParam(canonical, ParamKind.BOOL, bool_value=_parse_bool(tokens[0]))

        case "int":
            return RawParam(canonical, ParamKind.INT, int_value=int(float(tokens[0])))

        case "angle":
            rad = float(Angle(tokens[0], unit=u.Unit(unit)).to(u.rad).value)
            return RawParam(canonical, ParamKind.ANGLE, value=rad, frozen=_frozen(tokens[1:], fd))

        case "mjd":
            return RawParam(
                canonical, ParamKind.MJD, mjd_split=_split_mjd_string(tokens[0]),
                frozen=_frozen(tokens[1:], fd),
            )

        case "pair":
            if len(tokens) < 2:
                return None
            return RawParam(
                canonical, ParamKind.PAIR,
                value_pair=(_fortran_float(tokens[0]), _fortran_float(tokens[1])),
                unit=unit, frozen=_frozen(tokens[2:], fd),
            )

        case _:  # "float" (incl. semantically-int floats like TNREDC; the core
                 # dual-exposes them to int_params)
            val = _fortran_float(tokens[0])
            scale = spec.get("scale")
            if scale is not None and abs(val) > spec.get("scale_threshold", 0.0):
                # PINT auto-scaling: "PBDOT 1.59" -> 1.59e-12 (applied only above
                # the threshold, so an already-small literal passes through).
                val *= scale
            return RawParam(
                canonical, ParamKind.FLOAT, value=val,
                unit=unit, frozen=_frozen(tokens[1:], fd),
            )


def to_raw_params(parlines: list[ParLine]) -> ParsedPar:
    """Convert tokenized ``.par`` lines into the adapter-neutral parse result."""
    out = ParsedPar()
    counters: dict[str, int] = {}

    for pl in parlines:
        resolved = _resolve(pl.name, counters)
        if resolved is None:
            log.debug("Skipping unrecognized .par parameter %r", pl.name)
            continue
        canonical, template, spec = resolved

        if canonical == "BINARY" and pl.tokens:
            out.binary_value = pl.tokens[0]

        rp = _emit(canonical, spec, pl.tokens)
        if rp is None:
            log.debug("Skipping %r: could not parse tokens %r", pl.name, pl.tokens)
            continue
        out.raw_params.append(rp)
        out.templates.add(template)

    return out
