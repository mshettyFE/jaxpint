"""Clock-file readers + interpolation (PINT-free, numpy-only).

Reimplements PINT's TEMPO2 / TEMPO clock-file parsers and ``ClockFile.evaluate``
closely enough to reproduce its corrections bit-for-bit.  Everything is stored
internally in **microseconds**

This is the data-reading half of the clock-correction chain; the chain that sums
site/GPS/BIPM/TIME terms lives in :mod:`jaxpint.clock.correction`.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ClockCorrectionOutOfRange(UserWarning):
    """A TOA falls outside a clock file's covered MJD range.

    Distinct from :class:`jaxpint.clock.paths.StaleClockWarning` (which is
    about snapshot *age*); this is about a specific TOA being past the *data*.
    ``np.interp`` clamps to the endpoint value regardless -- this only governs
    whether we warn (``limits="warn"``) or raise (``limits="error"``).
    """


# Verbatim from PINT (observatory/clock_file.py): the header line names the two
# timescales; data rows are two scanf-style floats with an optional trailing
# comment.  Anything that doesn't match a data row is treated as a comment.
_HDRLINE_RE = re.compile(r"#\s*(\S+)\s+(\S+)\s+(\d+)?(.*)")
_CLKCORR_RE = re.compile(
    r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?)"
    r"\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?)"
    r" ?(.*)"
)

# TEMPO fixed-column layout (Fortran FORMAT, verbatim from PINT): the fields
# live at fixed character offsets, and a *blank* column means "absent" (-> 0.0).
# Whitespace splitting cannot represent a blank column -- it would shift later
# fields left and silently misread the corrections -- so these slices are
# load-bearing, not stylistic.
_COL_MJD = slice(0, 9)
_COL_CLKCORR1 = slice(9, 21)
_COL_CLKCORR2 = slice(21, 33)


def _col_float(line: str, col: slice):
    """Parse a fixed-width numeric column; ``None`` if blank or unparseable."""
    try:
        return float(line[col])
    except (ValueError, IndexError):
        return None


@dataclass(frozen=True)
class ClockFile:
    """A parsed clock file: a piecewise-linear correction (µs) vs MJD."""

    mjd: np.ndarray  # float64, ascending; the file's UTC/pulsar MJD axis
    clock_us: np.ndarray  # float64, microseconds
    valid_beyond_ends: bool = False
    friendly_name: str = ""

    def evaluate(self, t_mjd, *, limits: str = "warn") -> np.ndarray:
        """Linear-interpolate the correction (µs) at the given MJD(s).

        ``np.interp`` clamps to the endpoint values outside the range; if ``valid_beyond_ends``
        is False and any query is out of range we ``warn`` or ``raise`` per
        ``limits``.
        """
        t = np.asarray(t_mjd, dtype=np.float64)

        if len(self.mjd) == 0:
            self._out_of_range(
                limits,
                "the clock file is empty, so no correction can be interpolated",
                stale=True,
            )
            return np.zeros_like(t)

        if not self.valid_beyond_ends:
            lo, hi = self.mjd[0], self.mjd[-1]
            t_min, t_max = float(np.min(t)), float(np.max(t))
            if t_max > hi:
                # The common case: data ends before the observation -> stale snapshot.
                gap = t_max - hi
                self._out_of_range(
                    limits,
                    f"TOA at MJD {t_max:.4f} is {gap:.1f} day(s) past the last "
                    f"entry (MJD {hi:.4f}); the clock file ends before this "
                    f"observation",
                    stale=True,
                )
            elif t_min < lo:
                self._out_of_range(
                    limits,
                    f"TOA at MJD {t_min:.4f} is {lo - t_min:.1f} day(s) before "
                    f"the first entry (MJD {lo:.4f})",
                    stale=False,
                )

        return np.interp(t, self.mjd, self.clock_us)

    def _out_of_range(self, limits: str, detail: str, *, stale: bool) -> None:
        if limits == "ignore":
            return
        msg = (
            f"Clock correction out of range in {self.friendly_name!r}: {detail}. "
            f"The correction is clamped to the nearest endpoint value, which may "
            f"be inaccurate."
        )
        if stale:
            # Past-the-end almost always means the vendored snapshot is older than
            # the data; point the user at a refresh.
            msg += (
                " Your clock-correction snapshot is likely out of date — update it "
                "with `jaxpint.clock.update_clocks()`, or pin a specific IPTA "
                "release via $JAXPINT_CLOCK_REF."
            )
        if limits == "error":
            raise ClockCorrectionOutOfRange(msg)
        warnings.warn(msg, ClockCorrectionOutOfRange, stacklevel=3)


def _trim(mjd: list, clk: list, *, bogus_last_correction: bool):
    """Drop clock-file rows that are not real measurements.

    Two PINT-compatible cleanups on the parsed ``(mjd, clk)`` sample lists:

    1. If *bogus_last_correction* is set, drop the final row -- a sentinel
       placeholder some clock files append to pad the table (see
       :func:`read_tempo2_clock_file` for what the flag means).
    2. Strip any leading ``mjd == 0`` rows -- blank placeholder rows at the top
       of the file, whose zero MJD would otherwise corrupt the interpolation
       domain (a spurious sample at MJD 0 wrecks ``np.interp``'s range).

    Returns the trimmed ``(mjd, clk)`` lists.
    """
    if bogus_last_correction and mjd:
        mjd, clk = mjd[:-1], clk[:-1]
    while mjd and mjd[0] == 0:
        mjd, clk = mjd[1:], clk[1:]
    return mjd, clk


def read_tempo2_clock_file(
    path,
    *,
    bogus_last_correction: bool = False,
    valid_beyond_ends: bool = False,
    friendly_name: str | None = None,
) -> ClockFile:
    """Read a TEMPO2-format ``.clk`` file (values in seconds -> stored µs).

    The first line is the header (two timescale names); each data row is two
    floats (MJD, correction in seconds); anything else is a comment.

    Parameters
    ----------
    path : str or path-like
        The ``.clk`` file to read.
    bogus_last_correction : bool, default False
        Drop the file's final data row before interpolating.  Some TEMPO/TEMPO2
        clock files terminate with a sentinel last row -- a placeholder (not a
        real measurement) that pads the table -- which would otherwise extend
        the apparent valid range and skew the endpoint used when clamping
        out-of-range queries.  PINT carries the same per-file flag; set it only
        for files whose final row is known to be bogus.
    valid_beyond_ends : bool, default False
        Whether MJDs outside the file's covered span are treated as valid.  With
        the default (False), :meth:`ClockFile.evaluate` warns -- or raises, per
        its ``limits`` argument -- when a query falls before the first or past
        the last entry (``np.interp`` clamps to the endpoint value regardless).
        With True, out-of-range queries are silently clamped, no warning.
    friendly_name : str or None, default None
        Human-readable label for this file, used only in out-of-range warning
        messages.  Defaults to the file's base name.

    Returns
    -------
    ClockFile
        The parsed piecewise-linear correction (microseconds vs MJD).
    """
    path = Path(path)
    mjd: list[float] = []
    clk: list[float] = []  # seconds, as written in the file
    with open(path, "r") as f:
        hdr_seen = False
        for line in f:
            if not hdr_seen:
                # PINT: the FIRST line is unconditionally the header and must
                # name two timescales, else the file is malformed.
                if not _HDRLINE_RE.match(line):
                    raise ValueError(
                        f"Header line must start with # and contain two time "
                        f"scales: {line!r}"
                    )
                hdr_seen = True
                continue
            if line.startswith("#"):
                continue
            m = _CLKCORR_RE.match(line)
            if m is None:
                continue  # non-matching lines are comments (T2 sscanf behaviour)
            mjd.append(float(m.group(1)))
            clk.append(float(m.group(2)))

    mjd, clk = _trim(mjd, clk, bogus_last_correction=bogus_last_correction)
    return ClockFile(
        mjd=np.asarray(mjd, dtype=np.float64),
        clock_us=np.asarray(clk, dtype=np.float64) * 1e6,  # seconds -> µs
        valid_beyond_ends=valid_beyond_ends,
        friendly_name=friendly_name or path.name,
    )


def read_tempo_clock_file(
    path,
    *,
    bogus_last_correction: bool = False,
    valid_beyond_ends: bool = False,
    friendly_name: str | None = None,
) -> ClockFile:
    """Read a TEMPO-format ``.dat`` file (fixed columns; stored value is µs).

    Replicates PINT's quirks: fixed column slices, the ``clkcorr1 > 800 ->
    -= 818.8`` adjustment, the MJD sanity skip, header-line skipping, and storing
    ``clkcorr2 - clkcorr1`` in microseconds.  (INCLUDE directives are not present
    in the files we vendor and are not handled.)

    Parameters
    ----------
    path : str or path-like
        The ``.dat`` file to read.
    bogus_last_correction : bool, default False
        Drop the file's final data row before interpolating -- a sentinel
        placeholder row that pads the table.  See
        :func:`read_tempo2_clock_file` for the full rationale.
    valid_beyond_ends : bool, default False
        Whether MJDs outside the file's covered span are treated as valid; see
        :func:`read_tempo2_clock_file`.
    friendly_name : str or None, default None
        Human-readable label used only in out-of-range warning messages.
        Defaults to the file's base name.

    Returns
    -------
    ClockFile
        The parsed piecewise-linear correction (microseconds vs MJD).
    """
    path = Path(path)
    mjds: list[float] = []
    clkcorrs: list[float] = []  # microseconds
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            ls = line.split()
            if ls and (ls[0].upper().startswith("MJD") or ls[0].startswith("=====")):
                continue  # header lines

            mjd = _col_float(line, _COL_MJD)
            # Out-of-range MJD (before ~1965 or after ~2132) -- almost certainly
            # a fixed-column misparse of a non-data line, so skip it.  0 is
            # exempt: it is a blank-row placeholder that _trim removes.
            if mjd is not None and ((mjd < 39000 and mjd != 0) or mjd > 100000):
                mjd = None

            clkcorr1 = _col_float(line, _COL_CLKCORR1)
            clkcorr2 = _col_float(line, _COL_CLKCORR2)

            if mjd is None:
                continue
            if clkcorr1 is None and clkcorr2 is None:
                continue
            if clkcorr1 is None:
                clkcorr1 = 0.0
            if clkcorr2 is None:
                clkcorr2 = 0.0
            # Hard-coded in TEMPO (newsrc.f): clkcorr1 is the legacy LORAN-C
            # column (``xlor``); a reading above 800 carries the LORAN chain's
            # ~818.8 us coding delay, which TEMPO strips to recover the true
            # offset.  Undocumented constant, preserved verbatim for parity.
            if clkcorr1 > 800.0:
                clkcorr1 -= 818.8

            mjds.append(mjd)
            clkcorrs.append(clkcorr2 - clkcorr1)  # microseconds

    mjds, clkcorrs = _trim(mjds, clkcorrs, bogus_last_correction=bogus_last_correction)
    return ClockFile(
        mjd=np.asarray(mjds, dtype=np.float64),
        clock_us=np.asarray(clkcorrs, dtype=np.float64),  # already µs
        valid_beyond_ends=valid_beyond_ends,
        friendly_name=friendly_name or path.name,
    )


def load_clock_file(
    path,
    fmt: str,
    *,
    bogus_last_correction: bool = False,
    valid_beyond_ends: bool = False,
    friendly_name: str | None = None,
) -> ClockFile:
    """Dispatch to the TEMPO/TEMPO2 reader by format string.

    Parameters
    ----------
    path : str or path-like
        The clock file to read.
    fmt : {"tempo", "tempo2"}
        Which reader to use; raises ``ValueError`` for anything else.
    bogus_last_correction, valid_beyond_ends, friendly_name
        Forwarded verbatim to the selected reader; see
        :func:`read_tempo2_clock_file` for their meaning.

    Returns
    -------
    ClockFile
        The parsed piecewise-linear correction (microseconds vs MJD).
    """
    if fmt == "tempo2":
        return read_tempo2_clock_file(
            path,
            bogus_last_correction=bogus_last_correction,
            valid_beyond_ends=valid_beyond_ends,
            friendly_name=friendly_name,
        )
    if fmt == "tempo":
        return read_tempo_clock_file(
            path,
            bogus_last_correction=bogus_last_correction,
            valid_beyond_ends=valid_beyond_ends,
            friendly_name=friendly_name,
        )
    raise ValueError(f"unknown clock-file format {fmt!r} (expected tempo/tempo2)")
