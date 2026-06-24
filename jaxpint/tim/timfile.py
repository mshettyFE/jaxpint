"""Native ``.tim`` (TOA) text reader

Produces a raw, pre-clock-correction TOA table (:class:`ParsedTim`) from a
``.tim`` file.  This is *purely* a text->records stage: it applies EFAC/EQUAD,
EMIN/EMAX/FMIN/FMAX filtering, and the command flags (``TIME``/``PHASE``/
``JUMP``/``INFO``), but performs **no** clock corrections, TT/TDB conversion, or
ephemeris math -- exactly the surface of PINT's :func:`pint.toa.read_toa_file`
(``toa.py:701``), which it is bit-for-bit diffable against.  PINT-free.

Only the **Tempo2** line format (``name freq MJD err obs -flag val ...``) is
supported; the fixed-column legacy formats (Princeton/Parkes/ITOA) raise
``NotImplementedError`` rather than be silently mis-parsed.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

from .raw_toa import (
    ParsedTim,
    RawTOA,
    _validate_flag_value,
    normalize_flag_key,
)

log = logging.getLogger(__name__)

# Recognised command keywords (mirrors PINT's ``toa_commands``, ``toa.py:68``).
TOA_COMMANDS = (
    "DITHER", "EFAC", "EMAX", "EMAP", "EMIN", "EQUAD", "FMAX", "FMIN",
    "INCLUDE", "INFO", "JUMP", "MODE", "NOSKIP", "PHA1", "PHA2", "PHASE",
    "SEARCH", "SIGMA", "SIM", "SKIP", "TIME", "TRACK", "ZAWGT", "FORMAT", "END",
)

# Flag keys that would collide with TOA parameters; rejected, as in PINT
# (``toa.py:527``).  Compared against the lowercased key (slightly stricter than
# PINT, which compares the raw-case key -- harmless for real corpora).
_RESERVED_FLAG_KEYS = frozenset(
    {"error", "freq", "scale", "mjd", "flags", "obs", "name"}
)


def _new_cdict() -> dict:
    """Fresh command/state dict, matching PINT's initialisation (``toa.py:747``).

    Units are implicit: EMIN/EMAX/EQUAD in microseconds, FMIN/FMAX in MHz,
    TIME in seconds, EFAC dimensionless.
    """
    return {
        "EFAC": 1.0,
        "EQUAD": 0.0,
        "EMIN": 0.0,
        "EMAX": math.inf,
        "FMIN": 0.0,
        "FMAX": math.inf,
        "INFO": None,
        "SKIP": False,
        "TIME": 0.0,
        "PHASE": 0,
        "JUMP": [False, 0],
        "FORMAT": "Unknown",
        "END": False,
    }


def _classify_line(line: str, fmt: str = "Unknown") -> str:
    """Classify a ``.tim`` line, mirroring PINT's ``_toa_format`` (``toa.py:441``).

    Returns one of: Princeton, Comment, Command, Blank, Parkes, Tempo2, ITOA,
    Unknown.  ``fmt`` is the running FORMAT state (``"Tempo2"`` once a
    ``FORMAT 1`` command has been seen).
    """
    if re.match(r"[0-9a-z@] ", line):
        return "Princeton"
    elif line.startswith(("C ", "c ", "#", "CC ")):
        return "Comment"
    elif line.upper().lstrip().startswith(TOA_COMMANDS):
        return "Command"
    elif re.match(r"^\s*$", line):
        return "Blank"
    elif re.match(r"^ ", line) and len(line) > 41 and line[41] == ".":
        return "Parkes"
    elif len(line) > 80 or fmt == "Tempo2":
        return "Tempo2"
    elif re.match(r"\S\S", line) and len(line) > 14 and line[14] == ".":
        return "ITOA"
    else:
        return "Unknown"


def _parse_tempo2_line(line: str):
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
        raise ValueError(
            f"flags must come in -key value pairs: {' '.join(rest)!r}"
        )
    line_flags: dict[str, str] = {}
    for i in range(0, len(rest), 2):
        k = normalize_flag_key(rest[i])
        if k in _RESERVED_FLAG_KEYS:
            raise ValueError(f"flag {k!r} would overwrite a TOA parameter")
        line_flags[k] = _validate_flag_value(rest[i + 1])
    return mjd_int, mjd_frac, freq_mhz, error_us, obs, line_flags


def read_tim(
    path,
    *,
    process_includes: bool = True,
    _cdict: dict | None = None,
    _dir: Path | None = None,
) -> ParsedTim:
    """Read a ``.tim`` file into a :class:`ParsedTim` (raw TOA table).

    Parameters
    ----------
    path :
        Path to the ``.tim`` file.
    process_includes :
        Whether to follow ``INCLUDE`` directives (relative to the including
        file's directory).
    _cdict, _dir :
        Internal -- used to share command state and the base directory across
        ``INCLUDE`` recursion.  Do not set these directly.
    """
    path = Path(path)
    if _dir is None:
        _dir = path.parent
    cdict = _cdict if _cdict is not None else _new_cdict()

    toas: list[RawTOA] = []
    commands: list[tuple] = []
    ntoas = 0

    for line in path.read_text().splitlines():
        fmt_line = _classify_line(line, cdict["FORMAT"])

        if fmt_line == "Command":
            tokens = line.split()
            cmd = tokens[0].upper()
            commands.append((tokens, ntoas))
            if cmd == "SKIP":
                cdict["SKIP"] = True
            elif cmd == "NOSKIP":
                cdict["SKIP"] = False
            elif cmd == "END":
                cdict["END"] = True
                break
            elif cmd in ("TIME", "PHASE"):
                cdict[cmd] += float(tokens[1])
            elif cmd in ("EMIN", "EMAX", "EQUAD"):
                cdict[cmd] = float(tokens[1])          # microseconds
            elif cmd in ("FMIN", "FMAX"):
                cdict[cmd] = float(tokens[1])          # MHz
            elif cmd in ("EFAC", "PHA1", "PHA2"):
                cdict[cmd] = float(tokens[1])
            elif cmd == "INFO":
                cdict["INFO"] = tokens[1]
            elif cmd == "FORMAT":
                if tokens[1] == "1":
                    cdict["FORMAT"] = "Tempo2"
            elif cmd == "JUMP":
                if cdict["JUMP"][0]:
                    cdict["JUMP"][0] = False
                    cdict["JUMP"][1] += 1
                else:
                    cdict["JUMP"][0] = True
            elif cmd == "INCLUDE" and process_includes:
                inc = _dir / tokens[1]
                saved_fmt = cdict["FORMAT"]
                cdict["FORMAT"] = "Unknown"
                sub = read_tim(
                    inc,
                    process_includes=process_includes,
                    _cdict=cdict,
                    _dir=inc.parent,
                )
                cdict["FORMAT"] = saved_fmt
                toas.extend(sub.toas)
                commands.extend(sub.commands)
                ntoas += len(sub.toas)
            elif cmd == "MODE":
                log.warning("MODE command is not supported; ignoring: %r", line)
            else:
                log.warning("Unknown/unsupported .tim command: %r", line)
            continue

        # Non-command line.
        if cdict["SKIP"] or fmt_line in ("Comment", "Blank", "Unknown"):
            continue
        if fmt_line in ("Princeton", "Parkes", "ITOA"):
            raise NotImplementedError(
                f"{fmt_line}-format TOA lines are not supported by the native "
                f".tim parser (Tempo2 only): {line!r}"
            )

        # Tempo2 TOA line.
        mjd_int, mjd_frac, freq_mhz, error_us, obs, line_flags = _parse_tempo2_line(line)
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

        # Scale: error = hypot(EFAC * error, EQUAD), in microseconds -> seconds.
        err_us = math.hypot(error_us * cdict["EFAC"], cdict["EQUAD"])
        error_s = err_us * 1e-6

        flags = dict(line_flags)
        if cdict["INFO"]:
            flags["info"] = cdict["INFO"]
        if cdict["JUMP"][0]:
            flags["jump"] = str(cdict["JUMP"][1] + 1)
            flags["tim_jump"] = str(cdict["JUMP"][1] + 1)
        if cdict["PHASE"] != 0:
            flags["phase"] = str(cdict["PHASE"])
        if cdict["TIME"] != 0.0:
            flags["to"] = str(cdict["TIME"])

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
