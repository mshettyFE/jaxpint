"""Tokenizer for ``.par`` files.

Splits a ``.par`` file into a list of :class:`ParLine` records -- one per
non-comment, non-blank line -- preserving order (so repeated prefix/mask lines
like ``DMX_0001`` or repeated ``JUMP`` keep their file order).  Interpretation
of the tokens (typing, unit coercion, fit-flag vs uncertainty) is the
:mod:`jaxpint.par.text_adapter`'s job.

Mirrors PINT's ``parse_parfile`` (``model_builder.py``) + the line-splitting in
``Parameter.from_parfile_line`` (``parameter.py``).  PINT-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParLine:
    """One parsed ``.par`` line: the (upper-cased) parameter name and the
    whitespace-separated tokens that follow it."""

    name: str  # upper-cased, as PINT does
    tokens: tuple[str, ...]  # everything after the name
    raw: str  # the original line (stripped), for diagnostics


_COMMENT_PREFIXES = ("#", "C ", "c ")


def _is_comment(line: str) -> bool:
    s = line.lstrip()
    if not s:
        return True
    if s in ("C", "c"):
        return True
    return s.startswith(_COMMENT_PREFIXES)


def tokenize_lines(lines: list[str]) -> list[ParLine]:
    """Tokenize raw ``.par`` text lines into :class:`ParLine` records."""
    out: list[ParLine] = []
    for raw in lines:
        line = raw.strip()
        if _is_comment(line):
            continue
        parts = line.split()
        if not parts:
            continue
        name = parts[0].upper()
        out.append(ParLine(name=name, tokens=tuple(parts[1:]), raw=line))
    return out


def tokenize(par_source) -> list[ParLine]:
    """Read and tokenize a ``.par`` from a path or an open file-like object.

    A file-like object (anything with ``.read()``, e.g. ``io.StringIO``) is
    consumed directly -- the same convention PINT's ``get_model`` uses, so par
    text held in memory (simulation setups, notebooks) needs no tempfile.
    """
    if hasattr(par_source, "read"):
        text = par_source.read()
    else:
        text = Path(par_source).read_text()
    return tokenize_lines(text.splitlines())
