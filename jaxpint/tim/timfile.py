"""``.tim`` (TOA) text reader

Produces a raw, pre-clock-correction TOA table (:class:`ParsedTim`) from a
``.tim`` file.  This is *purely* a text->records stage: it applies EFAC/EQUAD,
EMIN/EMAX/FMIN/FMAX filtering, and the command flags (``TIME``/``PHASE``/
``JUMP``/``INFO``), but performs **no** clock corrections, TT/TDB conversion, or
ephemeris math -- exactly the surface of PINT's :func:`pint.toa.read_toa_file`
(``toa.py:701``), which it is bit-for-bit diffable against. 

Only the **Tempo2** line format (``name freq MJD err obs -flag val ...``) is
supported; the fixed-column legacy formats (Princeton/Parkes/ITOA) raise
``NotImplementedError`` rather than be silently mis-parsed.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

from .raw_toa import (
    KnownFlag,
    ParsedTim,
    RawTOA,
    _validate_flag_value,
    normalize_flag_key,
)

log = logging.getLogger(__name__)

# Recognised command keywords (mirrors PINT's ``toa_commands``, ``toa.py:68``).
TOA_COMMANDS = (
    "DITHER",
    "EFAC",
    "EMAX",
    "EMAP",
    "EMIN",
    "EQUAD",
    "FMAX",
    "FMIN",
    "INCLUDE",
    "INFO",
    "JUMP",
    "MODE",
    "NOSKIP",
    "PHA1",
    "PHA2",
    "PHASE",
    "SEARCH",
    "SIGMA",
    "SIM",
    "SKIP",
    "TIME",
    "TRACK",
    "ZAWGT",
    "FORMAT",
    "END",
)

# Flag keys that would collide with TOA parameters; rejected, as in PINT
# (``toa.py:527``).  Compared against the lowercased key
_RESERVED_FLAG_KEYS = frozenset(
    {"error", "freq", "scale", "mjd", "flags", "obs", "name"}
)


class LineFormat(StrEnum):
    """Classification of a ``.tim`` line, mirroring PINT's ``_toa_format`` labels.

    Doubles as the running FORMAT state in ``cdict["FORMAT"]``, where only
    ``UNKNOWN`` (initial) and ``TEMPO2`` (after a ``FORMAT 1`` command) occur --
    the legacy formats are detected by line shape, not selected by a command.
    """

    PRINCETON = "Princeton"
    COMMENT = "Comment"
    COMMAND = "Command"
    BLANK = "Blank"
    PARKES = "Parkes"
    TEMPO2 = "Tempo2"
    ITOA = "ITOA"
    UNKNOWN = "Unknown"


def _new_cdict() -> dict:
    """Fresh command/state dict, matching PINT's initialisation (``toa.py:747``).

    Units are implicit: EMIN/EMAX/EQUAD in microseconds, FMIN/FMAX in MHz,
    TIME in seconds, EFAC dimensionless.
    """
    return {
        "EFAC": 1.0,  # error multiplier (set/replace; dimensionless)
        "EQUAD": 0.0,  # error added in quadrature (set/replace; microseconds)
        "EMIN": 0.0,  # drop TOAs with raw error below this (set/replace; us)
        "EMAX": math.inf,  # drop TOAs with raw error above this (set/replace; us)
        "FMIN": 0.0,  # drop TOAs with freq below this (set/replace; MHz)
        "FMAX": math.inf,  # drop TOAs with freq above this (set/replace; MHz)
        "INFO": None,  # current -info string tag, or None (set/replace)
        "SKIP": False,  # while True, parsed TOAs are discarded (toggle)
        "TIME": 0.0,  # cumulative clock/time offset (accumulates; seconds)
        "PHASE": 0,  # cumulative integer pulse-turn offset (accumulates)
        "JUMP": [False, 0],  # [block currently open?, blocks-closed counter] (toggle)
        "FORMAT": LineFormat.UNKNOWN,  # running line-format state (TEMPO2 after FORMAT 1)
        "END": False,  # set True by END; stops parsing the file
    }


def _classify_line(line: str, fmt: LineFormat = LineFormat.UNKNOWN) -> LineFormat:
    """Classify a ``.tim`` line, mirroring PINT's ``_toa_format`` (``toa.py:441``).

    ``fmt`` is the running FORMAT state (``LineFormat.TEMPO2`` once a ``FORMAT 1``
    command has been seen).
    """
    if re.match(r"[0-9a-z@] ", line):
        return LineFormat.PRINCETON
    elif line.startswith(("C ", "c ", "#", "CC ")):
        return LineFormat.COMMENT
    elif line.upper().lstrip().startswith(TOA_COMMANDS):
        return LineFormat.COMMAND
    elif re.match(r"^\s*$", line):
        return LineFormat.BLANK
    elif re.match(r"^ ", line) and len(line) > 41 and line[41] == ".":
        return LineFormat.PARKES
    elif len(line) > 80 or fmt == LineFormat.TEMPO2:
        return LineFormat.TEMPO2
    elif re.match(r"\S\S", line) and len(line) > 14 and line[14] == ".":
        return LineFormat.ITOA
    else:
        return LineFormat.UNKNOWN


def _parse_tempo2_line(line: str) -> tuple:
    """Parse one Tempo2-format TOA line, mirroring PINT (``toa.py:504``).

    Returns ``(mjd_int, mjd_frac, freq_mhz, error_us, obs, line_flags)``.  The
    MJD is split on the literal ``.`` exactly as PINT does, so the integer and
    fractional day match what PINT feeds into its ``Time`` object.
    """
    fields = line.split()
    if len(fields) < 5:
        raise ValueError(f"malformed Tempo2 TOA line (need >=5 fields): {line!r}")
    # fields[0] is the name -- a label only; not retained (PINT leaks it into
    # flags via **kwargs, we deliberately do not).
    freq_mhz = float(fields[1])
    if "." in fields[2]:
        ii, ff = fields[2].split(".")
        mjd_int, mjd_frac = float(int(ii)), float(f"0.{ff}")
    else:
        mjd_int, mjd_frac = float(int(fields[2])), 0.0
    error_us = float(fields[3])
    obs = fields[4]  # raw token; canonical resolution is a downstream concern

    rest = fields[5:]
    if len(rest) % 2 != 0:
        raise ValueError(f"flags must come in -key value pairs: {' '.join(rest)!r}")
    line_flags: dict[str, str] = {}
    for i in range(0, len(rest), 2):
        k = normalize_flag_key(rest[i])
        if k in _RESERVED_FLAG_KEYS:
            raise ValueError(f"flag {k!r} would overwrite a TOA parameter")
        line_flags[k] = _validate_flag_value(rest[i + 1])
    return mjd_int, mjd_frac, freq_mhz, error_us, obs, line_flags


# Per-format TOA line parsers.  Each returns the shared 6-tuple
# ``(mjd_int, mjd_frac, freq_mhz, error_us, obs, line_flags)``.  A classified
# format absent from this table is unsupported (-> NotImplementedError); add an
# entry (plus its parser and a LineFormat member) to support a new format.
_PARSERS: dict[LineFormat, Callable[[str], tuple]] = {
    LineFormat.TEMPO2: _parse_tempo2_line,
}


def read_tim(path, *, process_includes: bool = True) -> ParsedTim:
    """Read a ``.tim`` file into a :class:`ParsedTim` (raw TOA table).

    Parameters
    ----------
    path :
        Path to the ``.tim`` file.
    process_includes :
        Whether to follow ``INCLUDE`` directives (relative to the including
        file's directory).
    """
    path = Path(path)
    return _read_tim(path, _new_cdict(), path.parent, process_includes)

def _read_tim(
    path: Path,
    cdict: dict,
    base_dir: Path,
    process_includes: bool,
) -> ParsedTim:
    """Recursive worker for :func:`read_tim`.

    ``cdict`` is the running command state, shared by reference so that command
    effects (``EFAC``, an open ``JUMP``, accumulated ``TIME``/``PHASE`` ...) flow
    into and back out of ``INCLUDE``d files.  ``base_dir`` resolves relative
    ``INCLUDE`` paths against the including file's directory.
    """
    toas: list[RawTOA] = []
    commands: list[tuple[list[str], int]] = []
    ntoas = 0

    for line in path.read_text().splitlines():
        fmt_line = _classify_line(line, cdict["FORMAT"])

        if fmt_line == LineFormat.COMMAND:
            tokens = line.split()
            cmd = tokens[0].upper()
            commands.append((tokens, ntoas))
            match cmd:
                case "SKIP":
                    cdict["SKIP"] = True
                case "NOSKIP":
                    cdict["SKIP"] = False
                case "END":
                    cdict["END"] = True
                    break
                case "TIME" | "PHASE":
                    cdict[cmd] += float(tokens[1])
                case "EMIN" | "EMAX" | "EQUAD":
                    cdict[cmd] = float(tokens[1])  # microseconds
                case "FMIN" | "FMAX":
                    cdict[cmd] = float(tokens[1])  # MHz
                case "EFAC" | "PHA1" | "PHA2":
                    cdict[cmd] = float(tokens[1])
                case "INFO":
                    cdict["INFO"] = tokens[1]
                case "FORMAT":
                    if tokens[1] == "1":
                        cdict["FORMAT"] = LineFormat.TEMPO2
                case "JUMP":
                    if cdict["JUMP"][0]:
                        cdict["JUMP"][0] = False
                        cdict["JUMP"][1] += 1
                    else:
                        cdict["JUMP"][0] = True
                case "INCLUDE" if process_includes:
                    inc = base_dir / tokens[1]
                    saved_fmt = cdict["FORMAT"]
                    cdict["FORMAT"] = LineFormat.UNKNOWN
                    sub = _read_tim(inc, cdict, inc.parent, process_includes)
                    cdict["FORMAT"] = saved_fmt
                    toas.extend(sub.toas)
                    commands.extend(sub.commands)
                    ntoas += len(sub.toas)
                case "MODE":
                    log.warning("MODE command is not supported; ignoring: %r", line)
                case _:
                    log.warning("Unknown/unsupported .tim command: %r", line)
            continue

        if cdict["SKIP"] or fmt_line in (
            LineFormat.COMMENT,
            LineFormat.BLANK,
            LineFormat.UNKNOWN,
        ):
            continue

        parser = _PARSERS.get(fmt_line)
        if parser is None:
            raise NotImplementedError(
                f"{fmt_line}-format TOA lines are not supported by the native "
                f".tim parser (Tempo2 only): {line!r}"
            )
        mjd_int, mjd_frac, freq_mhz, error_us, obs, line_flags = parser(line)
        if freq_mhz == 0.0:
            freq_mhz = math.inf  # PINT's 0 -> inf convention

        # Filter on the RAW error/freq, before EFAC/EQUAD (matches PINT order).
        if (
            cdict["EMIN"] > error_us
            or cdict["EMAX"] < error_us
            or cdict["FMIN"] > freq_mhz
            or cdict["FMAX"] < freq_mhz
        ):
            continue

        err_us = math.hypot(error_us * cdict["EFAC"], cdict["EQUAD"])
        error_s = err_us * 1e-6

        flags = dict(line_flags)
        if cdict["INFO"]:
            flags[KnownFlag.INFO] = cdict["INFO"]
        if cdict["JUMP"][0]:
            flags[KnownFlag.JUMP] = str(cdict["JUMP"][1] + 1)
            flags[KnownFlag.TIM_JUMP] = str(cdict["JUMP"][1] + 1)
        if cdict["PHASE"] != 0:
            flags[KnownFlag.PHASE] = str(cdict["PHASE"])
        if cdict["TIME"] != 0.0:
            flags[KnownFlag.TO] = str(cdict["TIME"])

        toas.append(
            RawTOA(
                mjd_int=mjd_int,
                mjd_frac=mjd_frac,
                error_s=error_s,
                freq_mhz=freq_mhz,
                obs=obs,
                flags=flags,
            )
        )
        ntoas += 1

    return ParsedTim(toas=toas, commands=commands)
