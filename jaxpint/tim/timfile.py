"""``.tim`` (TOA) text reader

Produces a raw, pre-clock-correction TOA table (:class:`ParsedTim`) from a
``.tim`` file.  This is *purely* a text->records stage: it applies EFAC/EQUAD,
EMIN/EMAX/FMIN/FMAX filtering, and the command flags (``TIME``/``PHASE``/
``JUMP``/``INFO``), but performs **no** clock corrections, TT/TDB conversion, or
ephemeris math -- exactly the surface of PINT's :func:`pint.toa.read_toa_file`
(``toa.py:701``), which it is bit-for-bit diffable against.

The **Tempo2** line format (``name freq MJD err obs -flag val ...``) and the
fixed-column **Princeton** and **Parkes** formats are supported. **ITOA** still
raises ``NotImplementedError`` rather than be silently mis-parsed -- PINT has no
implementation to port, and one file in ~4,400 surveyed uses it.
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
    """Classify a ``.tim`` line.

    ``fmt`` is the running FORMAT state (``LineFormat.TEMPO2`` once a ``FORMAT 1``
    command has been seen).

    **Deliberate divergence from PINT** (``toa.py:441``), which tests its
    fixed-column heuristics *before* the Tempo2 branch, so the ``fmt`` state set
    by ``FORMAT 1`` cannot protect a line.  Any Tempo2 TOA line that starts with
    a space and happens to carry a ``.`` in column 42 is claimed by the Parkes
    branch, which then slices bytes out of the middle of the filename.  That is
    not hypothetical: it makes 64 EPTA DR2 files and 1 IPTA DR1 file (5,857 TOA
    lines) unreadable, e.g. a TOA whose first field is
    ``20120209/75624/J0613-0200-20120209-75624.cal`` -- the ``.cal`` extension
    lands its dot on column 42.  Pure filename-length luck.

    Here ``FORMAT 1`` is authoritative: once declared, the only special lines
    are comments, commands and blanks, and everything else is a Tempo2 TOA.
    Fixed-column dialects cannot appear in a file that declared itself Tempo2.
    The legacy branch below (no ``FORMAT 1`` seen) keeps PINT's original order
    byte-for-byte, so non-Tempo2 files classify exactly as they did.

    """
    if fmt == LineFormat.TEMPO2:
        if line.startswith(("C ", "c ", "#", "CC ")):
            # Also fixes PINT's lowercase-'c' hazard: there, ``c `` matches the
            # Princeton regex before the comment check (PINT carries a FIXME on
            # that line), so a lowercase comment parses as a TOA.
            return LineFormat.COMMENT
        if line.upper().lstrip().startswith(TOA_COMMANDS):
            return LineFormat.COMMAND
        if re.match(r"^\s*$", line):
            return LineFormat.BLANK
        return LineFormat.TEMPO2

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
    elif len(line) > 80:
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


def _parse_princeton_line(line: str) -> tuple:
    """Parse one Princeton (TEMPO fixed-column) TOA line.

    Mirrors PINT's ``toa.py:481-503``. Columns (1-indexed, per the TEMPO and
    tempo2 manuals, which agree byte-for-byte)::

        1      observatory code, one character ('@' = barycentre)
        2      must be blank
        16-24  observing frequency (MHz)
        25-44  TOA (decimal point in column 30 or 31)
        45-53  TOA uncertainty (microseconds)
        69-78  DM correction (pc cm^-3), optional

    Fixed-column, so fields are sliced rather than split -- a Princeton line's
    name/label field may contain spaces, and the DM column is often absent
    entirely.

    The MJD is split on the literal ``.`` exactly as the Tempo2 parser does, so
    the int/frac decomposition that carries this codebase's sub-ns precision is
    identical across formats.
    """
    if len(line) < 53:
        raise ValueError(f"malformed Princeton TOA line (too short): {line!r}")
    obs = line[0]
    freq_mhz = float(line[15:24])
    error_us = float(line[44:53])

    mjd_field = line[24:44]
    if "." not in mjd_field:
        raise ValueError(f"Princeton TOA has no decimal point: {mjd_field!r}")
    ii, ff = mjd_field.split(".")
    mjd_int = int(ii)
    # Very old TOAs were recorded on a different day count; TEMPO applies this
    # offset and PINT mirrors it (toa.py:497). It silently rewrites dates, so it
    # only fires well below any plausible modern MJD.
    # https://tempo.sourceforge.net/ref_man_sections/toa.txt
    if mjd_int < 40000:
        mjd_int += 39126

    # Optional DM correction. PINT stores it as a flag and defaults to 0.0 when
    # the column is absent or unparseable; do the same rather than failing, since
    # most real Princeton files stop at column 53.
    line_flags: dict[str, str] = {}
    try:
        line_flags["ddm"] = str(float(line[68:78]))
    except (ValueError, IndexError):
        line_flags["ddm"] = str(0.0)

    return float(mjd_int), float(f"0.{ff}"), freq_mhz, error_us, obs, line_flags


def _parse_parkes_line(line: str) -> tuple:
    """Parse one Parkes (fixed-column) TOA line.

    Mirrors PINT's ``toa.py:535-556``. Columns (1-indexed, per the TEMPO and
    tempo2 manuals)::

        1      must be blank
        2-25   name / label
        26-34  observing frequency (MHz)
        35-55  TOA (decimal point in column 42)
        56-63  phase offset, fraction of P0, ADDED to the TOA
        64-71  TOA uncertainty (microseconds)
        80     observatory code, one character

    The MJD is split *around* the fixed decimal column rather than on the
    literal ``.`` -- ``line[34:41]`` and ``line[42:55]`` -- because the column
    position is what the format guarantees. Same int/frac decomposition the
    other parsers produce, so sub-ns precision carries across formats.

    A nonzero phase offset raises, matching PINT: the column asks for a
    fractional-turn shift of the TOA that neither implementation applies, and
    silently ignoring it would move the TOA by a fraction of a pulse period.
    The phase field is read across its full width, which PINT does not do; see
    the comment at that slice.
    """
    if len(line) < 71:
        raise ValueError(f"malformed Parkes TOA line (too short): {line!r}")
    freq_mhz = float(line[25:34])

    ii, ff = line[34:41], line[42:55]
    if line[41] != ".":
        raise ValueError(
            f"Parkes TOA decimal point must be in column 42, found {line[41]!r}"
        )
    mjd_int, mjd_frac = float(int(ii)), float(f"0.{ff}")

    # NOTE: Bug in PINT. Doesnt slice up to 63
    phase_offset = float(line[55:63])
    if phase_offset != 0:
        raise NotImplementedError(
            f"Parkes phase offset {phase_offset} is not applied (column 56-63 "
            "shifts the TOA by a fraction of P0). PINT raises here too; "
            "ignoring it would silently move the TOA."
        )

    error_us = float(line[63:71])
    # Column 80 is the observatory; some writers stop short of it.
    obs = line[79] if len(line) > 79 else "@"
    return mjd_int, mjd_frac, freq_mhz, error_us, obs, {}


# Per-format TOA line parsers.  Each returns the shared 6-tuple
# ``(mjd_int, mjd_frac, freq_mhz, error_us, obs, line_flags)``.  A classified
# format absent from this table is unsupported (-> NotImplementedError); add an
# entry (plus its parser and a LineFormat member) to support a new format.
_PARSERS: dict[LineFormat, Callable[[str], tuple]] = {
    LineFormat.TEMPO2: _parse_tempo2_line,
    LineFormat.PRINCETON: _parse_princeton_line,
    LineFormat.PARKES: _parse_parkes_line,
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
                    # MODE selects the fit's weighting: 1 = weight by TOA
                    # uncertainty, 0 = unweighted (ordinary least squares).
                    # JaxPINT always weights, so MODE 1 is a no-op and stays
                    # silent -- it is near-universal (632 of 635 MODE lines in
                    # the local corpus) and warning on it was pure noise.
                    #
                    # MODE 0 asks for the *opposite* of what we do. Ignoring it
                    # silently honours the inverse of the request, so it raises.
                    # PINT only warns here; this is a deliberate divergence, and
                    # a cheap one -- no file in the local corpus uses MODE 0.
                    mode = tokens[1] if len(tokens) > 1 else None
                    if mode == "0":
                        raise NotImplementedError(
                            "MODE 0 (unweighted / ordinary least squares) is not "
                            "supported: JaxPINT always weights by TOA "
                            "uncertainty. Remove the MODE 0 line to accept "
                            "weighted fitting, which is what would happen "
                            "anyway."
                        )
                    if mode != "1":
                        log.warning(
                            "Unrecognized MODE value %r; ignoring: %r", mode, line
                        )
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
                # PINT's delta_pulse_number: the accumulated PHASE-command turns
                # (integer, cumulative) plus this TOA's -padd flag (possibly
                # fractional, per-TOA); same sign as each source.
                delta_pulse_number=float(cdict["PHASE"])
                + float(flags.get("padd", 0.0)),
            )
        )
        ntoas += 1

    return ParsedTim(toas=toas, commands=commands)
