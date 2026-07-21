"""The per-TOA clock-correction chain (site clock -> TT(BIPM)).

Reproduces PINT's ``TOAs.apply_clock_corrections`` for a list of raw
:class:`~jaxpint.tim.raw_toa.RawTOA`: for each TOA the correction (seconds) is

    clkcorr = to_flag
            + Sum_i  site_clock_file_i.evaluate(mjd)        [us]
            + (apply_gps2utc)  gps2utc.evaluate(mjd)          [us]
            + (include_bipm and timescale != "tdb")
                  ( bipm.evaluate(mjd) - 32.184e6 )           [us]

with the microsecond terms converted to seconds and added to ``to_flag``.  This
matches the ``clkcorr`` flag PINT records (which *includes* ``-to``).  The
corrected MJD is still a **UTC-scale** ``pulsar_mjd`` value with the corrections
applied -- the corrections *target* the TT(BIPM) realization, but the actual
UTC -> TAI -> TT -> TDB scale conversion is done later by
``jaxpint.clock.timescale`` (and geometric barycentering is a later phase again).
PINT-free at runtime.

For the underlying time-transfer conventions (UTC -> TAI -> TT -> TT(BIPM)) see
the TEMPO2 timing-model paper, Edwards, Hobbs & Manchester (2006), MNRAS 372,
1549: https://arxiv.org/abs/astro-ph/0607664
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass

import numpy as np

from ..tim.raw_toa import RawTOA, get_time_offset
from .clockfile import ClockFile, load_clock_file
from .observatory import resolve_observatory
from .paths import clock_file_path, read_metadata

# TT(TAI) - TAI = 32.184 s, in microseconds.  The BIPM file holds TT(BIPM)-TAI,
# so subtracting this leaves only the (us-level) TT(BIPM)-TT(TAI) realization
# delta.  We strip the nominal 32.184 s here *because* the TT<->TAI conversion
# itself is applied later by timescale.to_tdb (astropy's UTC->TAI->TT step);
# leaving it in would double-count it.  So this leg contributes only the BIPM
# refinement, not a TT<->TAI conversion.  (Matches PINT.)
_TT_TAI_US = 32.184e6


@dataclass(frozen=True)
class UTCScaleTOAs:
    """Output of the clock-correction stage *only* -- not a finished time.

    The MJD is a UTC-scale ``pulsar_mjd`` with the clock corrections applied (the
    corrections target the TT(BIPM) realization).  The TT -> TDB conversion and
    geometric barycentering are still to come; see :func:`correct`.
    """

    mjd_int: np.ndarray  # float64, integer MJD day (UTC-scale, clock-corrected)
    mjd_frac: np.ndarray  # float64, fractional day in [0, 1)
    clkcorr_seconds: np.ndarray  # float64, total correction (incl. -to), seconds


@functools.cache
def _load_named(name: str) -> ClockFile:
    """Load+parse a clock file by basename once per process (cached)."""
    meta = read_metadata()["files"].get(name, {})
    fmt = meta.get("format")
    if fmt is None:
        ext = name.rsplit(".", 1)[-1]
        fmt = read_metadata()["format_by_extension"].get("." + ext, "tempo2")
    return load_clock_file(
        clock_file_path(name),
        fmt,
        bogus_last_correction=bool(meta.get("bogus_last_correction", False)),
        valid_beyond_ends=bool(meta.get("valid_beyond_ends", False)),
        friendly_name=name,
    )


def _gps_clock() -> ClockFile:
    return _load_named("gps2utc.clk")


def _bipm_clock(version: str) -> ClockFile:
    return _load_named(f"tai2tt_{version.lower()}.clk")


class UnsupportedClockRealization(UserWarning):
    """A par file requested a ``CLK`` realization JaxPINT cannot provide."""


def resolve_clock_config(
    clk: str | None,
    include_bipm: bool | None = None,
    bipm_version: str | None = None,
) -> tuple[bool, str | None]:
    """Turn a par file's ``CLK``/``CLOCK`` line into ``(include_bipm, bipm_version)``.

    Mirrors PINT's derivation in ``pint/toa.py:196-223``:

    ================== ============================================
    ``CLK`` value      result
    ================== ============================================
    ``TT(TAI)``        ``(False, None)`` -- no BIPM term at all
    ``UNCORR``         ``(False, None)`` -- uncorrected
    ``TT(BIPM)``       ``(True, default_bipm)``
    ``TT(BIPM2019)``   ``(True, "BIPM2019")``
    unrecognized       warn, fall back to ``(True, default_bipm)``
    absent             ``(True, default_bipm)``
    ================== ============================================

    Explicitly-passed arguments always win -- this only *derives* the values the
    caller left as ``None``, so ``get_TOAs(..., bipm_version="BIPM2019")`` still
    overrides the file.

    One **deliberate divergence from PINT**: naming a ``bipm_version`` while
    leaving ``include_bipm`` unset implies you want that realization *applied*,
    so it forces ``include_bipm=True`` even against a ``CLK TT(TAI)`` par.
    PINT's ordering would derive ``include_bipm=False`` from the file and
    silently ignore the version you asked for.  Pass ``include_bipm=False``
    explicitly if you really want the term off.

    Why this matters: the BIPM realization is not cosmetic.  ``TT(BIPM2015)``
    against ``BIPM2023`` differs by up to ~39 ns (concentrated after the earlier
    file's publication date, i.e. in recent data), and honouring ``TT(TAI)``
    matters far more -- the whole BIPM term is ~26.5-27.7 us, varying by ~1.2 us
    across a typical PTA span.  Clock errors are common-mode across pulsars, so
    they project onto the monopole and leak into correlated-signal searches.
    """
    if include_bipm is not None and bipm_version is not None:
        return include_bipm, bipm_version

    derived_include, derived_version = True, None
    if clk is not None:
        value = clk.strip()
        upper = value.upper()
        if upper in ("TT(TAI)", "UNCORR"):
            derived_include = False
        elif upper.startswith("TT(BIPM") and upper.endswith(")"):
            inner = value[3:-1]  # strip "TT(" and ")"
            derived_version = None if inner.upper() == "BIPM" else inner
        elif value:
            warnings.warn(
                f"CLOCK = {value} is not implemented; using "
                f"TT({read_metadata()['default_bipm']}) instead.",
                UnsupportedClockRealization,
                stacklevel=3,
            )

    if include_bipm is not None:
        include = include_bipm
    elif bipm_version is not None:
        include = True  # naming a realization means "apply it"
    else:
        include = derived_include
    version = derived_version if bipm_version is None else bipm_version
    return include, version


def clock_realization_label(include_bipm: bool, bipm_version: str | None) -> str:
    """Canonical ``CLK``-style label for the corrections actually applied.

    Resolves ``bipm_version=None`` to the packaged default, so the label names
    the realization used rather than repeating the request.  Stamped onto
    ``TOAData.clock_realization`` -- clock corrections are irreversible once
    applied, so this is the only surviving record of which one was used.
    """
    if not include_bipm:
        return "TT(TAI)"
    return f"TT({bipm_version or read_metadata()['default_bipm']})"


def correct(
    raw_toas: list[RawTOA],
    *,
    include_bipm: bool = True,
    bipm_version: str | None = None,
    limits: str = "warn",
) -> UTCScaleTOAs:
    """Apply the clock-correction chain (site clock -> TT(BIPM)) to raw TOAs.

    This is the clock-correction stage *only* (the analogue of PINT's
    ``apply_clock_corrections``): it returns a **UTC-scale** ``pulsar_mjd`` with
    the corrections applied.  The corrections *target* the TT(BIPM) realization,
    but the value is not yet on any TT/TDB scale and is *not* a usable
    barycentric time -- the UTC -> TT -> TDB conversion
    (``jaxpint.clock.timescale.to_tdb``) and the geometric barycentering
    (posvels + the model's Roemer delay) are separate subsequent steps, not
    performed here.

    Parameters
    ----------
    raw_toas:
        Parsed, pre-correction TOAs from ``jaxpint.tim.read_tim`` (MJD still
        in the raw site/UTC time scale).
    include_bipm:
        Apply the TT(TAI)->TT(BIPM) realization refinement (default True).
    bipm_version:
        BIPM realization, e.g. ``"BIPM2023"``; defaults to the metadata
        ``default_bipm``.
    limits:
        Out-of-range policy passed to each ``ClockFile.evaluate``
        (``"warn"`` | ``"error"`` | ``"ignore"``).

    Returns
    -------
    UTCScaleTOAs
        The corrected MJD (integer + fractional day, a **UTC-scale**
        ``pulsar_mjd`` with corrections applied) plus the total ``clkcorr``
        applied in seconds (including the ``-to`` flag offset).
    """
    n = len(raw_toas)
    mjd_int = np.array([t.mjd_int for t in raw_toas], dtype=np.float64)
    mjd_frac = np.array([t.mjd_frac for t in raw_toas], dtype=np.float64)
    to_s = np.array([get_time_offset(t.flags) for t in raw_toas], dtype=np.float64)
    raw_mjd = mjd_int + mjd_frac

    version = bipm_version or read_metadata()["default_bipm"]

    corr_us = np.zeros(n, dtype=np.float64)

    # Group indices by raw obs token (identical tokens resolve identically).
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(raw_toas):
        groups.setdefault(t.obs, []).append(i)

    for token, idx in groups.items():
        cfg = resolve_observatory(token)
        t = raw_mjd[idx]
        acc = np.zeros(len(idx), dtype=np.float64)
        if cfg.apply_gps2utc:
            acc += _gps_clock().evaluate(t, limits=limits)
        if include_bipm and cfg.timescale != "tdb":
            acc += _bipm_clock(version).evaluate(t, limits=limits) - _TT_TAI_US
        for name in cfg.clock_files:
            acc += _load_named(name).evaluate(t, limits=limits)
        corr_us[idx] = acc

    clkcorr_seconds = to_s + corr_us * 1e-6

    frac2 = mjd_frac + clkcorr_seconds / 86400.0
    carry = np.floor(frac2)
    return UTCScaleTOAs(
        mjd_int=mjd_int + carry,
        mjd_frac=frac2 - carry,
        clkcorr_seconds=clkcorr_seconds,
    )
