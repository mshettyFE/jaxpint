"""Clock-correction subsystem.

A pinned IPTA ``pulsar-clock-corrections`` snapshot is downloaded into a cache
dir on first use and auto-refreshed on a TTL; pin via ``$JAXPINT_CLOCK_REF`` for
reproducible runs. See :func:`jaxpint.clock.config.describe` for the env vars.

On top of that data layer, this package provides the native time
pipeline: clock-file readers (:mod:`~jaxpint.clock.clockfile`), observatory
resolution (:mod:`~jaxpint.clock.observatory`), the per-TOA correction chain
(:func:`correct`, site/UTC MJD -> TT(BIPM)), TT->TDB conversion
(:mod:`~jaxpint.clock.timescale`), and barycentric positions/velocities
(:mod:`~jaxpint.clock.posvels`).
"""

from __future__ import annotations

from . import config
from ._pinned import SEED_CLOCK_DATE, SEED_CLOCK_REF
from .correction import UTCScaleTOAs, correct
from .paths import StaleClockWarning, clock_dir, ensure_fresh, update_clocks

__all__ = [
    "config",
    "SEED_CLOCK_REF",
    "SEED_CLOCK_DATE",
    "clock_dir",
    "ensure_fresh",
    "StaleClockWarning",
    "update_clocks",
    "correct",
    "UTCScaleTOAs",
]
