"""Native ``.tim`` (TOA) parser -- PINT-free raw TOA text reader.

The data-side analogue of :mod:`jaxpint.par`: turns ``.tim`` text into a raw,
pre-clock-correction TOA table (:class:`ParsedTim` of :class:`RawTOA`).  Clock
corrections, TT/TDB conversion, ephemeris posvels, and flag->mask binding are
downstream stages that consume this table; they are *not* done here.
"""

from __future__ import annotations

from .raw_toa import (
    KnownFlag,
    ParsedTim,
    RawTOA,
    get_info,
    get_jump,
    get_phase_offset,
    get_time_offset,
    normalize_flag_key,
)
from .timfile import read_tim

__all__ = [
    "read_tim",
    "RawTOA",
    "ParsedTim",
    "KnownFlag",
    "normalize_flag_key",
    "get_time_offset",
    "get_phase_offset",
    "get_jump",
    "get_info",
]
