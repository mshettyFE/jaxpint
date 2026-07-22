"""``.tim`` writing: RawTOA records / TOAData -> Tempo2 ``FORMAT 1`` text.

:func:`write_tim` is the exact inverse of the parser: it takes
:class:`RawTOA` records (or a :class:`ParsedTim`) and emits ``FORMAT 1`` lines
carrying the same MJD digits, frequencies, errors, sites, and flags.
Round-trips bit-for-bit through :func:`~jaxpint.tim.read_tim` because nothing
is recomputed.

This module is pure text -- it never touches the clock subsystem. Writing a
*TOAData* back out (whose MJDs are clock-corrected) first requires un-applying
the corrections, which is loader-stage physics:
:func:`jaxpint.loaders.native.toa_data_to_raw` does that and feeds its records
here. (It lives with the loaders deliberately -- the
test_clock_subsystem_only_used_by_loaders invariant keeps parser-layer modules
clock-independent.)

Output is always Tempo2 ``FORMAT 1``. The fixed-column dialects (Princeton,
Parkes) are read-side compatibility only -- every modern consumer reads
``FORMAT 1``, and writing a legacy dialect would be a novelty, not a feature.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from .raw_toa import ParsedTim, RawTOA

__all__ = ["format_toa_line", "write_tim"]


def _shortest(x: float) -> str:
    """Shortest positional decimal that round-trips the float64 exactly.

    Fixed-width formats are traps here, and each trap was hit by a real file
    before this existed: ``%.6f`` truncated J1614's wideband frequencies, and
    16 fractional digits is one significant digit short of float64 for an MJD
    fraction in [0.1, 1) -- J1909 came back one ulp off. ``unique=True``
    positional formatting is exact by construction and never scientific
    notation (which the .tim grammar does not admit).
    """
    return np.format_float_positional(x, unique=True, trim="0")


def _exact_us(error_s: float) -> str:
    """Microseconds string whose parse (``* 1e-6``) reproduces *error_s* exactly.

    ``error_s * 1e6`` alone is wrong at the last ulp: the parser's ``us -> s``
    map rounds, so naively multiplying back lands next to -- not on -- the
    preimage (every error in the corpus round-tripped 1 ulp off before this).
    The true preimage, when one exists, is within one ulp of the product, so a
    three-candidate search finds it; a value with no preimage never came from
    microseconds, and the plain product is the honest best effort.
    """
    c = float(error_s) * 1e6
    for cand in (c, np.nextafter(c, np.inf), np.nextafter(c, -np.inf)):
        if float(cand) * 1e-6 == error_s:
            return _shortest(float(cand))
    return _shortest(c)


def format_toa_line(toa: RawTOA, name: str = "unk") -> str:
    """One Tempo2-format line: ``name freq mjd err_us site -flag value ...``.

    The MJD is printed from its int/frac split (never recombined through a
    single float64, which would cost ~1 us at MJD 55000): the fractional day's
    shortest-round-trip digits are appended to the integer day, preserving the
    split exactly. Frequency 0 <-> inf follows the parser's convention in
    reverse: an infinite frequency is written as 0.0.

    A nonzero ``delta_pulse_number`` is serialized as a ``-padd`` flag -- the
    per-TOA equivalent of the stateful ``PHASE`` commands the parser folded it
    from (any source ``padd`` flag is replaced: its contribution is already
    inside ``delta_pulse_number``). A stale ``phase`` flag, if present, is kept
    verbatim: the parser treats it as an inert record, not an instruction.
    """
    frac = float(toa.mjd_frac)
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"mjd_frac must be in [0, 1), got {frac!r}")
    mjd_str = f"{int(toa.mjd_int)}.{_shortest(frac)[2:]}"

    freq = float(toa.freq_mhz)
    freq_str = "0.0" if math.isinf(freq) else _shortest(freq)

    flags = dict(toa.flags)
    flags.pop("padd", None)
    dpn = float(toa.delta_pulse_number)
    if dpn != 0.0:
        flags["padd"] = _shortest(dpn)

    parts = [name, freq_str, mjd_str, _exact_us(float(toa.error_s)), toa.obs]
    for key, value in flags.items():
        parts.append(f"-{key}")
        parts.append(str(value))
    return " ".join(parts)


def write_tim(
    toas: Union[ParsedTim, Sequence[RawTOA]],
    path: Union[str, Path],
    *,
    name_fmt: str = "toa{i}",
) -> None:
    """Write TOAs as a Tempo2 ``FORMAT 1`` ``.tim`` file.

    Parameters
    ----------
    toas
        A :class:`ParsedTim` (as returned by :func:`~jaxpint.tim.read_tim`) or
        a plain sequence of :class:`RawTOA`.
    path
        Output path, overwritten if present.
    name_fmt
        Format string for the per-TOA name field (``{i}`` is the 0-based
        index). The name is a label only -- the parser on both sides discards
        it -- but the column must exist.

    Notes
    -----
    ``MODE 1`` is emitted after ``FORMAT 1``: PINT's reader warns without it,
    and it is what PINT's own ``write_TOA_file`` emits.

    Commands from a ``ParsedTim`` are **not** replayed: their effects
    (TIME/PHASE offsets, EFAC/EQUAD scaling, JUMP flags) were already applied
    to the records at read time, so replaying them would double-apply. The
    written file reproduces the *effective* TOAs, not the source text.
    """
    records = toas.toas if isinstance(toas, ParsedTim) else list(toas)
    lines = ["FORMAT 1", "MODE 1"]
    for i, t in enumerate(records):
        lines.append(format_toa_line(t, name=name_fmt.format(i=i)))
    Path(path).write_text("\n".join(lines) + "\n")
