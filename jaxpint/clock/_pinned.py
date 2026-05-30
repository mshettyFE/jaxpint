"""The pinned IPTA ``pulsar-clock-corrections`` commit JaxPINT is tested against.

This module is the **single source of truth** for both *which* repository the
clock data comes from (``IPTA_REPO``, and the URLs derived from it, which
:mod:`jaxpint.clock.fetch` imports) and *which commit* is pinned
(``SEED_CLOCK_REF``).

``SEED_CLOCK_REF`` is the slow-moving anchor of the clock subsystem:

* the commit the test suite pins to (so CI is deterministic and never chases
  ``main``),
* the offline floor / first-download default,
* the value used when auto-update is frozen.

It is bumped **only** deliberately, by running ``tools/bump_clock_pin.py`` at
release / occasionally -- never automatically.  The fast-moving part is the
runtime auto-update to ``latest`` (see :mod:`jaxpint.clock.paths`).

To bump the seed by hand, browse the commits page (``IPTA_COMMITS_URL`` below,
or https://github.com/ipta/pulsar-clock-corrections/commits/main) and update
both constants (the date is the chosen commit's committer date).
"""

from __future__ import annotations

# --- repository identity (single source of truth for all clock URLs) -------
# "owner/repo" on GitHub. Everything else is derived from this.
IPTA_REPO = "ipta/pulsar-clock-corrections"

#: Human-facing repo + commits pages (for docs / bumping by hand).
IPTA_REPO_URL = f"https://github.com/{IPTA_REPO}"
IPTA_COMMITS_URL = f"{IPTA_REPO_URL}/commits/main"
#: Raw file content, suffixed with ``/<ref>/<path>`` by the fetcher.
IPTA_RAW_BASE = f"https://raw.githubusercontent.com/{IPTA_REPO}"
#: GitHub API endpoint for resolving the current ``main`` HEAD commit.
IPTA_API_COMMIT = f"https://api.github.com/repos/{IPTA_REPO}/commits/main"

# --- the pin ---------------------------------------------------------------
# IPTA pulsar-clock-corrections commit on `main` and its committer date.
# See the module docstring for how to bump these to a newer commit.
# Used for deterministic unit testing
SEED_CLOCK_REF = "c6731ec51d9e2e53e0b728ef494c100d0d620e07"
SEED_CLOCK_DATE = "2026-05-29"
