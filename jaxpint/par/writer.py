"""``.par`` writing: :class:`ParResult` (+ optionally fitted values) -> par text.

The inverse of the native parser, inverting each conversion the core applies:

* rad -> deg (and rad/s -> deg/yr etc.) for angle-bearing FLOAT params,
* RAJ/DECJ back to sexagesimal (h:m:s / d:m:s),
* EQUAD/ECORR seconds -> microseconds,
* epoch int/frac recombination at full split precision (digit concatenation,
  never through a single float64 -- that costs ~1 us at MJD 55000, which is
  fatal for TZRMJD),
* ``_A``/``_B`` pair entries -> one ``WAVEn a b`` line,
* ``frozen_mask`` -> fit flags, ``uncertainties`` -> the 4th column,
* ``MaskInfo`` -> ``EFAC -f <val> <value>`` selector lines.

``as_parfile(par_result)`` mirrors PINT's ``TimingModel.as_parfile``; pass a
fitted :class:`~jaxpint.types.ParameterVector` as *params* to persist a fit.
The output targets round-trip fidelity through :func:`jaxpint.par.get_model`
and readability by PINT -- both are pinned by tests, not assumed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Union

import astropy.units as u
import numpy as np

from .result import ParResult

__all__ = ["as_parfile", "write_parfile"]


# Internal metadata the adapters stash for the model builder; never par lines.
_INTERNAL_METADATA_PREFIX = "_"

# Written specially, not as generic float lines.
_SEXAGESIMAL = {"RAJ": u.hourangle, "DECJ": u.deg}


def _fmt(x: float) -> str:
    """Round-trip-exact float formatting (17 significant digits, trimmed)."""
    return f"{float(x):.17g}"


def _epoch_str(int_day: float, frac_day: float) -> str:
    """MJD string from the int/frac split without a float64 recombination."""
    if not 0.0 <= frac_day < 1.0:
        # Fitted epoch fractions can walk out of [0, 1); renormalize.
        carry = math.floor(frac_day)
        int_day += carry
        frac_day -= carry
    return f"{int(int_day)}.{f'{frac_day:.16f}'[2:]}"


def _rad_to_display(value: float, unit_str: str) -> tuple[float, str]:
    """Invert the core's deg->rad conversion: rad-based -> deg-based units."""
    if not unit_str or "rad" not in unit_str:
        return value, unit_str
    try:
        unit = u.Unit(unit_str)
    except (ValueError, TypeError):
        return value, unit_str
    if u.rad not in getattr(unit, "bases", ()):
        return value, unit_str
    deg_unit = unit
    for base, power in zip(unit.bases, unit.powers):
        if base == u.rad:
            deg_unit = deg_unit * (u.deg / u.rad) ** power
    converted = (value * unit).to(deg_unit, equivalencies=u.dimensionless_angles())
    return float(converted.value), str(converted.unit)


def _sexagesimal(name: str, value_rad: float) -> str:
    from astropy.coordinates import Angle

    angle = Angle(value_rad, unit=u.rad)
    if name == "RAJ":
        return angle.to_string(unit=u.hourangle, sep=":", precision=10, pad=True)
    return angle.to_string(unit=u.deg, sep=":", precision=9, pad=True, alwayssign=True)


def _family(mask_name: str) -> str:
    """``EFAC1`` -> ``EFAC``: strip the parser's per-instance index."""
    return re.sub(r"\d+$", "", mask_name)


def as_parfile(par_result: ParResult, params=None) -> str:
    """Serialize to par text. *params* overrides the values (e.g. a fitted set)."""
    p = params if params is not None else par_result.params
    names = list(p.names)
    values = np.asarray(p.values, dtype=np.float64)
    frozen = p.frozen_mask
    units = p.units
    uncs = p.uncertainties if p.uncertainties else (math.nan,) * len(names)

    lines: list[str] = []
    md = dict(par_result.metadata)

    # --- header: PSR first, then the other string params -------------------
    if "PSR" in md:
        lines.append(f"PSR {md.pop('PSR')}")
    if par_result.binary_model is not None and "BINARY" not in md:
        md["BINARY"] = getattr(
            par_result.binary_model, "value", str(par_result.binary_model)
        )

    # --- float parameters, in vector order ---------------------------------
    mask_names = set(par_result.mask_info)
    pair_bases_done: set[str] = set()
    for i, name in enumerate(names):
        if name in mask_names:
            continue  # written from mask_info below, with their selectors
        if name.endswith(("_A", "_B")):
            base = name[:-2]
            if base in pair_bases_done:
                continue
            pair_bases_done.add(base)
            try:
                a = values[names.index(base + "_A")]
                b = values[names.index(base + "_B")]
            except ValueError:
                a, b = values[i], 0.0
            lines.append(f"{base} {_fmt(a)} {_fmt(b)}")
            continue

        fit = " 1" if not frozen[i] else ""
        unc = float(uncs[i]) if i < len(uncs) else math.nan

        if name in p.epoch_int_values:
            val_str = _epoch_str(p.epoch_int_values[name], float(values[i]))
        elif name in _SEXAGESIMAL:
            val_str = _sexagesimal(name, float(values[i]))
            if math.isfinite(unc):
                # Uncertainty column for sexagesimal params is in seconds (of
                # time for RAJ, of arc for DECJ) in PINT's output; the native
                # parser tolerates its absence, which is the safe choice here.
                unc = math.nan
        else:
            val, unit_str = _rad_to_display(float(values[i]), units[i])
            if unit_str == "s" and (
                name.startswith("EQUAD") or name.startswith("ECORR")
            ):
                val *= 1e6  # stored seconds -> written microseconds
            val_str = _fmt(val)

        line = f"{name} {val_str}{fit}"
        if math.isfinite(unc):
            v_disp, _ = _rad_to_display(unc, units[i])
            line += f"{' 0' if frozen[i] else ''} {_fmt(v_disp)}"
        lines.append(line)

    # --- masked parameters (EFAC/EQUAD/ECORR/JUMP ...) ----------------------
    for name, info in par_result.mask_info.items():
        try:
            i = names.index(name)
        except ValueError:
            continue
        val = float(values[i])
        if name.startswith(("EQUAD", "ECORR")):
            val *= 1e6
        key = info.key if info.key.startswith("-") else f"-{info.key}"
        sel = f"{key} {info.key_value}"
        if info.key_value2:
            sel += f" {info.key_value2}"
        fit = " 1" if not frozen[i] else ""
        lines.append(f"{_family(name)} {sel} {_fmt(val)}{fit}")

    # --- int / bool / remaining string params -------------------------------
    for name, iv in par_result.int_params.items():
        if name in names:
            continue  # int-valued floats already written from the vector
        lines.append(f"{name} {iv}")
    for name, bv in par_result.bool_params.items():
        lines.append(f"{name} {'Y' if bv else 'N'}")
    for name, sv in md.items():
        if name.startswith(_INTERNAL_METADATA_PREFIX):
            continue
        lines.append(f"{name} {sv}")

    return "\n".join(lines) + "\n"


def write_parfile(par_result: ParResult, path: Union[str, Path], params=None) -> None:
    """Write :func:`as_parfile` output to *path* (overwritten if present)."""
    Path(path).write_text(as_parfile(par_result, params))
