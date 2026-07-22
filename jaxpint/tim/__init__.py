"""Native ``.tim`` (TOA) parser -- PINT-free raw TOA text reader.

The data-side analogue of :mod:`jaxpint.par`: turns ``.tim`` text into a raw,
pre-clock-correction TOA table (:class:`ParsedTim` of :class:`RawTOA`).  Clock
corrections, TT/TDB conversion, and ephemeris posvels are downstream.

:func:`~jaxpint.tim.masks.select_toa_mask` is the one consumer that lives
alongside the parser: it matches a masked-parameter selector against the parsed
``RawTOA`` flags/columns to produce the boolean ``TOAData.flag_masks``.
"""

from __future__ import annotations

from .raw_toa import (
    KnownFlag,
    ParsedTim,
    RawTOA,
    get_time_offset,
    normalize_flag_key,
)
from .masks import select_toa_mask
from .timfile import read_tim
from .writer import format_toa_line, write_tim

__all__ = [
    "read_tim",
    "write_tim",
    "format_toa_line",
    "select_toa_mask",
    "RawTOA",
    "ParsedTim",
    "KnownFlag",
    "normalize_flag_key",
    "get_time_offset",
]
