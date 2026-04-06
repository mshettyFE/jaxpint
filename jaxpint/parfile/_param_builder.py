"""Parameter builder: tokenized .par lines → ParameterVector.

This is the core of the standalone parser.  It classifies each line,
parses the value according to its type, applies unit conversions, and
assembles the result into a :class:`ParameterVector` and a
:class:`ParResult` that carries all the metadata the model builder needs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax.numpy as jnp

from jaxpint.parfile._converters import (
    CONVERTERS,
    parse_dms_to_rad,
    parse_float,
    parse_hms_to_rad,
    split_mjd,
    tcb_scale_parameter,
    tcb_transform_mjd,
)

from jaxpint.parfile._registry import BinaryModel, Component, ParamMeta, ParamType, lookup, split_prefixed_name
from jaxpint.parfile._tokenizer import RawLine, tokenize
from jaxpint.types import ParameterVector

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intermediate data
# ---------------------------------------------------------------------------

# NOTE: Don't construct this manually
@dataclass
class ParsedParam:
    """A single parsed parameter ready for ParameterVector."""
    name: str
    value: float
    unit: str
    frozen: bool
    component: Component

# NOTE: Don't construct this manually
@dataclass
class MaskInfo:
    """Metadata for a mask parameter (JUMP, EFAC, etc.) needed for TOA matching."""
    name: str          # e.g. "JUMP1"
    key: str           # e.g. "-fe" or "-sys"
    key_value: str     # e.g. "Rcvr_800" or "430"
    key_value2: Optional[str] = None  # Second value for range-type keys (mjd, freq)


# NOTE: Don't construct this manually
@dataclass
class ParResult:
    """Complete result of parsing a .par file."""
    params: ParameterVector # JIT-able values that jax can trace
    component_set: set[str] = field(default_factory=set) # What components need to be built
    binary_model: Optional[BinaryModel] = None # What binary model you are assuming 
    metadata: dict[str, str] = field(default_factory=dict) # Non-numeric .par parameters 
    # Idea is that some of the .par parameters denote mask operators. This means that this particular value 
    # only applies to a subset of TOAs which match some criteria as specified by key and key_value in MaskInfo
    # You can have multiple mask parameters of the same values, hence the dictionary
    mask_info: dict[str, MaskInfo] = field(default_factory=dict)
    int_params: dict[str, int] = field(default_factory=dict) # Non-jittable integer parameters 
    bool_params: dict[str, bool] = field(default_factory=dict) # Non-jittable boolean parameters


# ---------------------------------------------------------------------------
# Fit-flag / uncertainty detection
# ---------------------------------------------------------------------------


def _parse_fit_flag_and_uncertainty(
    tokens: list[str], start: int
) -> tuple[bool, Optional[float]]:
    """Parse optional fit_flag and uncertainty from tokens[start:].

    .par format after the value:
        [fit_flag] [uncertainty]
    where fit_flag is 0 (frozen) or 1 (free).  Sometimes the uncertainty
    appears without a fit_flag.
   """
    frozen = True  # default: frozen
    uncertainty = None

    remaining = tokens[start:]
    if len(remaining) == 0:
        return frozen, uncertainty
    elif len(remaining) == 1:
        tok = remaining[0]
        if tok in ("0", "1"):
            frozen = tok == "0"
        else:
            # Assume it's an uncertainty (no fit flag)
            try:
                uncertainty = parse_float(tok)
            except ValueError:
                pass
    elif len(remaining) >= 2:
        if remaining[0] in ("0", "1"):
            frozen = remaining[0] == "0"
            try:
                uncertainty = parse_float(remaining[1])
            except ValueError:
                pass
        else:
            # Two values but no fit flag
            try:
                uncertainty = parse_float(remaining[0])
            except ValueError:
                pass

    return frozen, uncertainty


# ---------------------------------------------------------------------------
# Mask parameter parsing
# ---------------------------------------------------------------------------

# Mask key types that take two key values (range selection)
_RANGE_KEYS = {"mjd", "freq", "MJD", "FREQ"}


def _parse_mask_line(
    name: str, tokens: list[str], meta: ParamMeta, counter: dict[str, int]
) -> tuple[list[ParsedParam], MaskInfo | None]:
    """Parse a mask parameter line (JUMP, EFAC, EQUAD, ECORR, etc.).

    Format: NAME key key_value [key_value2] param_value [fit_flag] [uncertainty]
    """
    if len(tokens) < 2:
        log.warning("Mask parameter %s has too few tokens: %s", name, tokens)
        return [], None

    key = tokens[0]

    # Determine how many key-value tokens there are
    if key.lstrip("-") in _RANGE_KEYS:
        # Range key: two key values
        if len(tokens) < 4:
            log.warning("Mask parameter %s (range key) has too few tokens", name)
            return [], None
        key_value = tokens[1]
        key_value2 = tokens[2]
        value_start = 3
    else:
        key_value = tokens[1]
        key_value2 = None
        value_start = 2

    if value_start >= len(tokens):
        log.warning("Mask parameter %s missing value token", name)
        return [], None

    try:
        raw_value = parse_float(tokens[value_start])
    except ValueError:
        log.warning("Cannot parse mask value for %s: %s", name, tokens[value_start])
        return [], None

    frozen, _ = _parse_fit_flag_and_uncertainty(tokens, value_start + 1)

    # Apply conversion
    if meta.convert and meta.convert in CONVERTERS:
        raw_value = CONVERTERS[meta.convert](raw_value)

    # Assign sequential index
    base = name
    count = counter.get(base, 0) + 1
    counter[base] = count
    indexed_name = f"{base}{count}"

    parsed = ParsedParam(
        name=indexed_name,
        value=raw_value,
        unit=meta.default_unit,
        frozen=frozen,
        component=meta.component,
    )
    mask = MaskInfo(
        name=indexed_name, key=key,
        key_value=key_value, key_value2=key_value2,
    )
    return [parsed], mask


# ---------------------------------------------------------------------------
# Pair parameter parsing
# ---------------------------------------------------------------------------


def _parse_pair_line(
    name: str, tokens: list[str], meta: ParamMeta
) -> list[ParsedParam]:
    """Parse a pair parameter line (WAVE, IFUNC).

    Format: NAME value_a value_b
    """
    if len(tokens) < 2:
        log.warning("Pair parameter %s has too few tokens", name)
        return []

    try:
        val_a = parse_float(tokens[0])
        val_b = parse_float(tokens[1])
    except ValueError:
        log.warning("Cannot parse pair values for %s", name)
        return []

    frozen, _ = _parse_fit_flag_and_uncertainty(tokens, 2)

    return [
        ParsedParam(f"{name}_A", val_a, meta.default_unit, frozen, meta.component),
        ParsedParam(f"{name}_B", val_b, meta.default_unit, frozen, meta.component),
    ]


# ---------------------------------------------------------------------------
# FDnJUMPm pattern detection
# ---------------------------------------------------------------------------

_FDJUMP_RE = re.compile(r"^FD(\d+)JUMP(\d+)$")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_par(source: str | Path) -> ParResult:
    """Parse a .par file into a :class:`ParResult`.
    Handles syntax parsing, not semantic parsing (this happens in build_model())

    Parameters
    ----------
    source : str or Path
        File path or raw text content of a .par file.
    """
    raw_lines = tokenize(source)

    # Accumulators
    parsed_params: list[ParsedParam] = []
    metadata: dict[str, str] = {}
    int_params: dict[str, int] = {}
    bool_params: dict[str, bool] = {}
    mask_info: dict[str, MaskInfo] = {}
    epoch_int_values: dict[str, float] = {}
    mask_counters: dict[str, int] = {}
    binary_model: Optional[BinaryModel] = None

    for line in raw_lines:
        name = line.name
        tokens = line.tokens

        # Special case: BINARY
        if name == "BINARY":
            raw_binary = tokens[0] if tokens else None
            if raw_binary is not None:
                try:
                    binary_model = BinaryModel(raw_binary)
                except ValueError:
                    raise ValueError(
                        f"Unknown binary model {raw_binary!r}. "
                        f"Supported: {[m.value for m in BinaryModel]}"
                    )
            metadata["BINARY"] = raw_binary or ""
            continue

        # Special case: FDnJUMPm pattern
        m = _FDJUMP_RE.match(name)
        if m:
            if not tokens:
                continue
            try:
                raw_value = parse_float(tokens[0])
            except ValueError:
                log.warning("Cannot parse FDJump value for %s", name)
                continue
            frozen, _ = _parse_fit_flag_and_uncertainty(tokens, 1)
            parsed_params.append(ParsedParam(
                name=name, value=raw_value, unit="s",
                frozen=frozen, component=Component.FD_JUMP,
            ))
            continue

        # Look up in registry
        result = lookup(name)
        if result is None:
            log.warning("Unknown parameter %r — skipping", name)
            continue

        meta, canon_name = result

        match meta.param_type:
            # -- Non-numeric types (not added to ParameterVector) --
            case ParamType.STR:
                metadata[canon_name] = " ".join(tokens) if tokens else ""
                continue
            case ParamType.INT:
                try:
                    int_params[canon_name] = int(tokens[0]) if tokens else 0
                except ValueError:
                    log.warning("Cannot parse int for %s: %s", canon_name, tokens)
                continue
            case ParamType.BOOL:
                val = tokens[0] if tokens else "0"
                bool_params[canon_name] = val.upper() in ("1", "Y", "YES", "TRUE")
                continue

            # -- Repeatable / compound types --
            case ParamType.MASK:
                params, mi = _parse_mask_line(canon_name, tokens, meta, mask_counters)
                parsed_params.extend(params)
                if mi is not None:
                    mask_info[mi.name] = mi
                continue
            case ParamType.PAIR:
                parsed_params.extend(_parse_pair_line(name, tokens, meta))
                continue

            # -- Numeric types (added to ParameterVector) --
            case ParamType.ANGLE_HMS:
                value = parse_hms_to_rad(tokens[0])
                unit = "rad"
            case ParamType.ANGLE_DMS:
                value = parse_dms_to_rad(tokens[0])
                unit = "rad"
            case ParamType.MJD:
                raw_mjd = parse_float(tokens[0])
                mjd_int, mjd_frac = split_mjd(raw_mjd)
                epoch_int_values[name] = mjd_int
                value = mjd_frac
                unit = "day"
            case ParamType.FLOAT:
                try:
                    value = parse_float(tokens[0])
                except ValueError:
                    log.warning("Cannot parse float for %s: %s", name, tokens[0])
                    continue
                unit = meta.default_unit
            case _:
                log.warning("Unexpected param_type %r for %s — skipping", meta.param_type, name)
                continue

        # Post-processing for numeric types (ANGLE_HMS, ANGLE_DMS, MJD, FLOAT)

        # This is a convenience thing; if the user accientally forgets to include "e-12", 
        # then the parser tacitly assumes they forgot to include it 
        # e.g. "PBDOT 7.2" is interpreted as "PBDOT 7.2E-12", since the scale_factor is 1E-12 and the scale_threshold is 1E-7
        if meta.scale_factor is not None and abs(value) > meta.scale_threshold:
            value *= meta.scale_factor

        # Apply conversion to standard units in jaxpint (e.g. deg_to_rad, us_to_s). 
        if meta.convert and meta.convert in CONVERTERS:
            value = CONVERTERS[meta.convert](value)
            # Update unit for degree conversions
            if meta.convert == "deg_to_rad":
                unit = "rad"
            elif meta.convert == "deg_per_yr_to_rad_per_s":
                unit = "rad/s"
            elif meta.convert == "us_to_s":
                unit = "s"

        frozen, _ = _parse_fit_flag_and_uncertainty(tokens, 1)

        parsed_params.append(ParsedParam(
            name=name, value=value, unit=unit,
            frozen=frozen, component=meta.component,
        ))

    # -- TCB → TDB conversion --
    # Performed in post processing
    units_system = metadata.get("UNITS", "TDB").upper()
    if units_system == "TCB":
        _apply_tcb_to_tdb(parsed_params, epoch_int_values)
        metadata["UNITS"] = "TDB"
        log.warning(
            "Converted timing model from TCB to TDB. "
            "The conversion is approximate — the model should be re-fit."
        )

    # -- Build ParameterVector --
    # Filter to only params with a real component (skip metadata-only params)
    numeric_params = [p for p in parsed_params if p.component is not Component.NONE]
    numeric_names = {p.name for p in numeric_params}

    # Remove epoch_int_values for params that were filtered out
    epoch_int_values = {
        k: v for k, v in epoch_int_values.items() if k in numeric_names
    }

    names = tuple(p.name for p in numeric_params)
    values = jnp.asarray([p.value for p in numeric_params], dtype=jnp.float64)
    frozen_mask = tuple(p.frozen for p in numeric_params)
    units = tuple(p.unit for p in numeric_params)
    component_set = {p.component for p in numeric_params}
    param_vector = ParameterVector(
        values=values,
        frozen_mask=frozen_mask,
        names=names,
        units=units,
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


# ---------------------------------------------------------------------------
# TCB → TDB post-processing
# ---------------------------------------------------------------------------


def _apply_tcb_to_tdb(
    params: list[ParsedParam],
    epoch_int_values: dict[str, float],
) -> None:
    """Apply TCB → TDB conversion in-place to parsed parameters.
        Very old pulsar timing codes use TCB. JAXPint assumes TDB convention. 
       See Irwin, A. W. & Fukushima, T. (1999) — "A numerical time ephemeris of the Earth." Astronomy & Astrophysics, 348, 642–652.
    """
    for p in params:
        result = lookup(p.name)
        if result is None:
            continue
        meta, _ = result
        n = meta.tcb2tdb_n
        if n is None:
            continue

        if meta.is_epoch or meta.param_type is ParamType.MJD:
            # Reconstruct full MJD, transform, re-split
            full_mjd = epoch_int_values.get(p.name, 0.0) + p.value
            converted = tcb_transform_mjd(full_mjd)
            new_int = float(int(converted))
            epoch_int_values[p.name] = new_int
            p.value = converted - new_int
        elif n != 0:
            p.value = tcb_scale_parameter(p.value, n)
