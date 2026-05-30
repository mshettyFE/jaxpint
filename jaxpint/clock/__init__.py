"""Clock-correction data layer.

A pinned IPTA ``pulsar-clock-corrections`` snapshot is downloaded into a cache
dir on first use and auto-refreshed on a TTL; pin via ``$JAXPINT_CLOCK_REF`` for
reproducible runs. See :func:`jaxpint.clock.config.describe` for the env vars.

This is the *data* layer only — the clock-file reader, interpolation, and the
correction chain are a later phase.
"""

from __future__ import annotations

from . import config
from ._pinned import SEED_CLOCK_DATE, SEED_CLOCK_REF
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
]
