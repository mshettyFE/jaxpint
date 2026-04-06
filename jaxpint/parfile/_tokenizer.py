"""Line-level .par file tokenizer.

Reads raw text, strips comments and blank lines, resolves aliases, and
returns a list of RawLine records ready for classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from jaxpint.parfile._aliases import resolve_alias


@dataclass
class RawLine:
    """A single tokenized .par file line."""
    name: str          # Canonical uppercase parameter name
    tokens: list[str]  # Remaining whitespace-separated tokens (value, fit_flag, uncertainty)


def tokenize(source: str | Path) -> list[RawLine]:
    """Tokenize a .par file into a list of :class:`RawLine`.

    Parameters
    ----------
    source : str or Path
        Either a file path or the raw text content of a .par file.
        If *source* looks like an existing file path it is read;
        otherwise it is treated as raw text.
    """
    # Read file or use raw text
    path = Path(source) if not _is_multiline(source) else None
    if path is not None and path.is_file():
        text = path.read_text()
    else:
        text = str(source)

    lines: list[RawLine] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        # Skip comment lines
        if stripped.startswith("#"):
            continue
        if stripped.startswith("C ") or stripped == "C":
            continue

        parts = stripped.split()
        name = parts[0].upper()
        tokens = parts[1:]

        # Resolve alias
        name = resolve_alias(name)

        # Special rewrite: FDJUMP → FDnJUMPm pattern
        # (PINT stores these as FD1JUMP1, FD2JUMP1, etc.)
        # No rewrite needed here; the name comes from the file as-is.

        lines.append(RawLine(name=name, tokens=tokens))

    return lines


def _is_multiline(s: str | Path) -> bool:
    """Heuristic: if the string contains a newline, it's raw text, not a path."""
    return isinstance(s, str) and "\n" in s
