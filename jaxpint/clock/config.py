"""Central registry for JaxPINT's environment variables.

Covers the ``JAXPINT_CLOCK_*`` clock-cache knobs plus ``JAXPINT_EPHEM_PATH``
(pre-staged ephemeris kernels). Every env read in the clock package goes
through :func:`get` so the lookup stays in one place.
"""

from __future__ import annotations

import os
from typing import Any, Callable, NamedTuple


class _Opt(NamedTuple):
    parser: Callable[[str], Any]
    default: Any
    help: str


def _as_str(v: str) -> str:
    return v


def _as_float(v: str) -> float:
    try:
        return float(v)
    except ValueError as exc:  # pragma: no cover - message exercised in tests
        raise ValueError(f"expected a number, got {v!r}") from exc


# The complete set of clock env vars.
OPTIONS: dict[str, _Opt] = {
    "JAXPINT_CLOCK_DIR": _Opt(
        _as_str,
        None,
        "Override the clock cache directory "
        "(default: the packaged jaxpint/data/clock).",
    ),
    "JAXPINT_CLOCK_REF": _Opt(
        _as_str,
        None,
        "Pin an exact IPTA pulsar-clock-corrections commit SHA. Makes runs "
        "reproducible and disables auto-update (pinning is how you freeze).",
    ),
    "JAXPINT_CLOCK_TTL_DAYS": _Opt(
        _as_float,
        7.0,
        "Auto-update cadence in days; the cache is refreshed from IPTA when "
        "older than this (0 = check every run).",
    ),
    "JAXPINT_EPHEM_PATH": _Opt(
        _as_str,
        None,
        "Path to a pre-staged ephemeris ``.bsp`` file or a directory of them; "
        "lets the posvel loader avoid the network on locked-down nodes.",
    ),
}


def get(name: str) -> Any:
    """Return the parsed value of a ``JAXPINT_CLOCK_*`` env var (or its default).

    Raises
    ------
    KeyError
        If ``name`` is not a registered option.
    ValueError
        If the env var is set but cannot be parsed.
    """
    opt = OPTIONS[name]
    raw = os.environ.get(name)
    if raw is None:
        return opt.default
    try:
        return opt.parser(raw)
    except ValueError as exc:
        raise ValueError(f"invalid {name}={raw!r}: {exc}") from exc


def describe() -> str:
    """Return a human-readable listing of every clock env var + default + help."""
    lines = ["JaxPINT clock environment variables:"]
    for name, opt in OPTIONS.items():
        cur = os.environ.get(name)
        set_note = f"  [currently set to {cur!r}]" if cur is not None else ""
        lines.append(f"  {name} (default {opt.default!r}){set_note}")
        lines.append(f"      {opt.help}")
    return "\n".join(lines)
