"""Forced clock refresh (the rare manual override; auto-update is the default)."""

from __future__ import annotations

from . import fetch, paths


def update_clocks(ref: str = "latest") -> dict:
    """Download a fresh clock snapshot into the active dir, return a diff.

    Parameters
    ----------
    ref:
        ``"latest"`` (the IPTA ``main`` HEAD) or an explicit commit SHA.

    Returns
    -------
    dict
        ``{ref_old, ref_new, added, removed}``.
    """
    dest = paths.clock_dir()
    if ref == "latest":
        sha, date = fetch.resolve_latest_sha()
    else:
        sha, date = ref, None
    diff = paths._download(sha, dest, commit_date=date)
    paths._ensured = True  # we just made it fresh
    return diff
