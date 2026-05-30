"""Shared contract for the native ``.tim`` parser.

Defines the parser-side records produced by :mod:`jaxpint.tim.timfile` -- the
``.tim`` analogue of :class:`jaxpint.par.raw_params.RawParam` / ``ParsedPar``.
These are *raw*, pre-clock-correction TOA records: the MJD is still in the
site/UTC time scale, and **no** clock corrections, TT/TDB conversion, or
ephemeris posvels have been applied (those are downstream stages that consume
this table).  This module is PINT-free.

Flags handling -- permissive store, strict interpretation
---------------------------------------------------------
``.tim`` flag *keys* are an open, unbounded vocabulary (``-fe -be -f -sys
-group -pta -pp_dm`` ... plus dataset-specific ones), so :class:`RawTOA` stores
them as a plain ``dict[str, str]`` and the parser accepts *any* ``-key value``
pair -- restricting the vocabulary at parse time would silently drop valid
flags.  What we *do* validate is the **shape** (lowercase keys, no whitespace in
values), matching PINT's ``FlagDict`` so the dicts compare 1:1.

Strictness belongs at the *interpretation* boundary instead: the handful of
flags the rest of the code actually acts on are named in :class:`KnownFlag`, and
read through the typed accessors below.  Those guard consumption; they are never
used to filter what the parser ingests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawTOA:
    """One raw, pre-clock-correction time-of-arrival.

    Mirrors exactly what PINT's ``read_toa_file`` produces per ``TOA`` *before*
    ``apply_clock_corrections`` / ``compute_TDBs`` / ``compute_posvels``.
    """

    mjd_int: float            # integer MJD day (UTC/site scale, pre-correction)
    mjd_frac: float           # fractional day in [0, 1)
    error_s: float            # seconds, AFTER EFAC/EQUAD applied at read time
    freq_mhz: float           # MHz (0 -> inf convention, matching PINT)
    obs: str                  # raw observatory token as written (no canonicalisation)
    flags: dict[str, str] = field(default_factory=dict)
    delta_pulse_number: float = 0.0  # populated downstream from the 'phase' flag


@dataclass
class ParsedTim:
    """The whole-file result: the raw TOA list plus the command log.

    ``commands`` mirrors PINT's ``TOAs.commands`` -- ``(tokens, toa_count)``
    tuples for diagnostics/round-trip, not used to reconstruct the TOAs.
    """

    toas: list[RawTOA] = field(default_factory=list)
    commands: list[tuple] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Interpretation boundary: the flags the code acts on
# ---------------------------------------------------------------------------


class KnownFlag(str, Enum):
    """Flags synthesised by the reader and consumed downstream.

    These are the *acted-on* flags (phase/clock/jump logic).  The open-vocabulary
    line flags (``-fe``/``-be``/``-sys`` ...) used by the mask matcher are *not*
    enumerated here -- they stay opaque strings in :attr:`RawTOA.flags`.
    """

    TO = "to"               # cumulative TIME offset (seconds), -> clock chain
    PHASE = "phase"         # cumulative integer pulse turns, -> delta_pulse_number
    JUMP = "jump"           # phase-jump block index
    TIM_JUMP = "tim_jump"   # alias of JUMP that PINT also writes
    INFO = "info"           # INFO-command string tag


def normalize_flag_key(key: str) -> str:
    """Normalise a flag key to PINT-``FlagDict`` form: strip a leading ``-`` and
    lowercase.  Raises ``ValueError`` on an empty/whitespace key."""
    k = key.lstrip("-").lower()
    if not k or any(c.isspace() for c in k):
        raise ValueError(f"invalid flag key {key!r}")
    return k


def _validate_flag_value(value: str) -> str:
    if any(c.isspace() for c in value):
        raise ValueError(f"flag value may not contain whitespace: {value!r}")
    return value


# -- typed accessors (coerce str -> type only at point of use) ---------------


def get_time_offset(flags: dict[str, str]) -> float:
    """The ``-to`` (TIME command) offset in seconds; 0.0 if absent."""
    v = flags.get(KnownFlag.TO.value)
    return float(v) if v is not None else 0.0


def get_phase_offset(flags: dict[str, str]) -> int:
    """The cumulative integer ``-phase`` offset; 0 if absent."""
    v = flags.get(KnownFlag.PHASE.value)
    return int(v) if v is not None else 0


def get_jump(flags: dict[str, str]) -> Optional[int]:
    """The 1-based phase-jump block index, or ``None`` if this TOA is unjumped."""
    v = flags.get(KnownFlag.JUMP.value)
    return int(v) if v is not None else None


def get_info(flags: dict[str, str]) -> Optional[str]:
    """The ``-info`` string tag, or ``None``."""
    return flags.get(KnownFlag.INFO.value)
