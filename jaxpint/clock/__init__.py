"""Clock-correction subsystem.

A pinned IPTA ``pulsar-clock-corrections`` snapshot is downloaded into a cache
dir on first use and auto-refreshed on a TTL; pin via ``$JAXPINT_CLOCK_REF`` for
reproducible runs. See :func:`jaxpint.clock.config.describe` for the env vars.

On top of that data layer, this package provides the native (PINT-free)
correction chain: clock-file readers (:mod:`~jaxpint.clock.clockfile`),
observatory resolution (:mod:`~jaxpint.clock.observatory`), and the per-TOA
chain (:func:`correct`) that takes a raw site/UTC MJD to TT(BIPM). TT->TDB and
barycentering are later phases.
"""

from __future__ import annotations

from . import config
from ._pinned import SEED_CLOCK_DATE, SEED_CLOCK_REF
from .clockfile import (
    ClockCorrectionOutOfRange,
    ClockFile,
    load_clock_file,
    read_tempo2_clock_file,
    read_tempo_clock_file,
)
from .correction import CorrectedTOAs, correct
from .observatory import ObsClockConfig, UnknownObservatory, resolve_observatory
from .paths import (
    clock_dir,
    clock_file_path,
    ensure_fresh,
    read_index,
    read_metadata,
    snapshot_info,
)
from .staleness import StaleClockWarning, check_staleness
from .update import update_clocks

__all__ = [
    "config",
    "SEED_CLOCK_REF",
    "SEED_CLOCK_DATE",
    "clock_dir",
    "clock_file_path",
    "ensure_fresh",
    "read_index",
    "read_metadata",
    "snapshot_info",
    "StaleClockWarning",
    "check_staleness",
    "update_clocks",
    # correction chain
    "ClockFile",
    "ClockCorrectionOutOfRange",
    "load_clock_file",
    "read_tempo_clock_file",
    "read_tempo2_clock_file",
    "resolve_observatory",
    "ObsClockConfig",
    "UnknownObservatory",
    "correct",
    "CorrectedTOAs",
]
