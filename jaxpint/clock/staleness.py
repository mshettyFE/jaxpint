"""Staleness warning for the clock snapshot.

Standalone (no import from :mod:`jaxpint.clock.paths`, which imports this) so the
package has no import cycle.  With auto-update on, routine staleness is
self-healing; this fires only when we *couldn't* refresh (offline) and the cache
is genuinely old.  The out-of-range warning (a TOA past a clock file's last
sample) is the other safety net and lands with the reader in the next phase.
"""

from __future__ import annotations

import datetime
import warnings
from typing import Optional

HARD_STALE_DAYS = 180


class StaleClockWarning(UserWarning):
    """Emitted when clock data is old and could not be refreshed."""


def _to_date(value) -> datetime.date:
    if value is None:
        return datetime.date.today()
    if isinstance(value, str):
        return datetime.date.fromisoformat(value)
    return value


def check_staleness(
    snapshot: Optional[dict],
    *,
    today=None,
    hard_days: int = HARD_STALE_DAYS,
) -> Optional[int]:
    """Warn (``StaleClockWarning``) if the snapshot is older than ``hard_days``.

    Returns the age in days when a warning fired, else ``None``.
    """
    commit_date = (snapshot or {}).get("commit_date")
    if not commit_date:
        return None
    age = (_to_date(today) - datetime.date.fromisoformat(commit_date)).days
    if age > hard_days:
        warnings.warn(
            f"clock snapshot {(snapshot or {}).get('ref')!r} is {age} days old "
            f"and could not be refreshed (offline); corrections may be out of "
            f"date. Reconnect, or pin a known commit via $JAXPINT_CLOCK_REF.",
            StaleClockWarning,
            stacklevel=2,
        )
        return age
    return None
