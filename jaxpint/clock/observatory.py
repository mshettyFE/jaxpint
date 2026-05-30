"""Observatory resolution for the clock chain (PINT-free, metadata-driven).

A ``.tim`` ``obs`` token (``gbt``, ``ao``, ``1``, ``@`` ...) is mapped to the
clock-relevant config of an observatory, using the committed
``clock_metadata.json`` ``observatories`` slice.  This mirrors PINT's
``get_observatory`` resolution (case-insensitive over name + aliases, where the
generator has already folded ``tempo_code``/``itoa_code`` into the alias list).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .paths import read_metadata


class UnknownObservatory(KeyError):
    """The ``.tim`` obs token did not resolve to any known observatory."""


@dataclass(frozen=True)
class ObsClockConfig:
    """Everything the clock + time/geometry stages need about one observatory."""

    canonical: str
    clock_files: tuple[str, ...]   # site clock-file names (may be empty, e.g. chime)
    apply_gps2utc: bool
    timescale: str                 # "utc" | "tdb" (tdb gates BIPM off, e.g. barycenter)
    itrf_xyz: tuple[float, float, float] | None = None  # geocentric metres; None for barycenter


@functools.cache
def _resolution_map() -> dict[str, str]:
    """Build a case-folded ``token -> canonical name`` map (cached).

    Precedence: a canonical name always wins over an alias; among aliases the
    first observatory to claim a token keeps it (sorted iteration makes this
    deterministic).  The generator already lower-cases aliases/codes, but we
    fold again here defensively.
    """
    obs = read_metadata()["observatories"]
    mapping: dict[str, str] = {}
    # Canonical names first (highest precedence).
    for name in obs:
        mapping[name.lower()] = name
    # Then aliases / codes (do not clobber a canonical name).
    for name in sorted(obs):
        for alias in obs[name].get("aliases", []):
            key = str(alias).lower()
            mapping.setdefault(key, name)
    return mapping


def resolve_observatory(token: str) -> ObsClockConfig:
    """Resolve a ``.tim`` obs token to an :class:`ObsClockConfig`.

    Raises :class:`UnknownObservatory` if the token matches no observatory.
    """
    key = (token or "").strip().lower()
    if not key:
        raise UnknownObservatory("empty observatory token")
    canonical = _resolution_map().get(key)
    if canonical is None:
        raise UnknownObservatory(f"unknown observatory token {token!r}")
    entry = read_metadata()["observatories"][canonical]
    xyz = entry.get("itrf_xyz")
    return ObsClockConfig(
        canonical=canonical,
        clock_files=tuple(entry.get("clock_file", [])),
        apply_gps2utc=bool(entry.get("apply_gps2utc", True)),
        timescale=str(entry.get("timescale", "utc")).lower(),
        itrf_xyz=tuple(xyz) if xyz is not None else None,
    )
