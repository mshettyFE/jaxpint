"""Clock-file readers + interpolation (PINT-free, numpy-only).

Reimplements PINT's TEMPO2 / TEMPO clock-file parsers and ``ClockFile.evaluate``
closely enough to reproduce its corrections bit-for-bit.  Everything is stored
internally in **microseconds** (TEMPO2 files are read as seconds and converted;
TEMPO ``.dat`` files are already microseconds), so :meth:`ClockFile.evaluate`
returns microseconds with no further unit handling -- matching
``np.interp(t.mjd, file_mjd, clock.to(u.us).value)`` in PINT.

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

    Distinct from :class:`jaxpint.clock.staleness.StaleClockWarning` (which is
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


@dataclass(frozen=True)
class ClockFile:
    """A parsed clock file: a piecewise-linear correction (µs) vs MJD."""

    mjd: np.ndarray              # float64, ascending; the file's UTC/pulsar MJD axis
    clock_us: np.ndarray         # float64, microseconds
    valid_beyond_ends: bool = False
    friendly_name: str = ""

    def evaluate(self, t_mjd, *, limits: str = "warn") -> np.ndarray:
        """Linear-interpolate the correction (µs) at the given MJD(s).

        Mirrors PINT's ``np.interp(t.mjd, self.time.mjd, clock_us)``: ``np.interp``
        clamps to the endpoint values outside the range; if ``valid_beyond_ends``
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
    """Apply PINT's bogus-last-row drop, then strip leading ``mjd == 0`` rows."""
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
    """Read a TEMPO2-format ``.clk`` file (values in seconds -> stored µs)."""
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

            try:
                mjd = float(line[:9])
                if (mjd < 39000 and mjd != 0) or mjd > 100000:
                    mjd = None  # suspicious MJD -> treat line as comment
            except (ValueError, IndexError):
                mjd = None

            try:
                clkcorr1 = float(line[9:21])
            except (ValueError, IndexError):
                clkcorr1 = None
            try:
                clkcorr2 = float(line[21:33])
            except (ValueError, IndexError):
                clkcorr2 = None

            if mjd is None:
                continue
            if clkcorr1 is None and clkcorr2 is None:
                continue
            if clkcorr1 is None:
                clkcorr1 = 0.0
            if clkcorr2 is None:
                clkcorr2 = 0.0
            # Hard-coded in TEMPO:
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
    """Dispatch to the TEMPO/TEMPO2 reader by format string."""
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
