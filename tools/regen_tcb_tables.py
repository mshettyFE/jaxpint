#!/usr/bin/env python
"""Dev tool (NOT shipped): regenerate ``jaxpint/par/_tcb_generated.py``.

The TCB->TDB conversion needs, per parameter, whether PINT converts it and its
"effective dimensionality" (the power of seconds in its SI dimension).  That is
physics we do not want to re-derive by hand, so the tables are extracted from a
PINT install and committed.  This script is the extractor.

Run it when PINT gains parameters, or when JaxPINT declares parameters it did
not before::

    python tools/regen_tcb_tables.py            # rewrite the committed file
    python tools/regen_tcb_tables.py --check    # verify it is up to date (CI)

Needs PINT installed; no network.  Output is restricted to parameters JaxPINT
declares (``spec.KNOWN_PARAMS``), so it stays in step with our own registry
rather than carrying PINT's whole vocabulary.

Two sources are swept, because PINT splits parameter ownership:

* every component in ``AllComponents()`` -- the bulk of the physics;
* ``TimingModel()`` itself, which owns START/FINISH/TRES/CHI2/CHI2R/NTOA/RM.
  Missing these was a real bug: the component-only sweep silently omitted them
  and the strict "unknown dimensionality" guard rejected real EPTA/IPTA pars.

The hand-written conversion logic lives in ``jaxpint/par/_tcb_tables.py`` and
imports from the generated module, so regenerating never clobbers code.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import warnings
from pathlib import Path

_OUT = Path(__file__).resolve().parents[1] / "jaxpint" / "par" / "_tcb_generated.py"


def _extract() -> tuple[dict[str, int], set[str], set[str]]:
    """Return ``(scale_dimensionality, mjd_params, not_convertible)`` from PINT."""
    warnings.simplefilter("ignore")
    from pint.models.model_builder import AllComponents
    from pint.models.parameter import (
        AngleParameter,
        MJDParameter,
        floatParameter,
        maskParameter,
        prefixParameter,
    )
    from pint.models.timing_model import TimingModel

    from jaxpint.par import spec as S

    known = set(S.KNOWN_PARAMS)
    scale: dict[str, int] = {}
    mjd: set[str] = set()
    noconv: set[str] = set()

    def visit(p) -> None:
        name = p.name
        if name in scale or name in mjd or name in noconv:
            return
        base = p.param_comp if isinstance(p, prefixParameter) else p
        if not getattr(p, "convert_tcb2tdb", False):
            noconv.add(name)
            return
        if isinstance(base, MJDParameter):
            mjd.add(name)
            return
        if not isinstance(base, (floatParameter, AngleParameter, maskParameter)):
            return
        # effective_dimensionality reads the parameter's own quantity, so an
        # unset parameter needs a placeholder value before it can be asked.
        if p.quantity is None:
            p.value = 1.0
        scale[name] = int(p.effective_dimensionality)

    ac = AllComponents()
    for comp in ac.components.values():
        for pname in comp.params:
            visit(getattr(comp, pname))

    tm = TimingModel()
    for pname in tm.params:
        visit(tm[pname])

    return (
        {k: v for k, v in sorted(scale.items()) if k in known},
        {k for k in mjd if k in known},
        {k for k in noconv if k in known},
    )


def _render(scale: dict[str, int], mjd: set[str], noconv: set[str]) -> str:
    def wrap(names: set[str]) -> str:
        body = " ".join(f"{n!r}," for n in sorted(names))
        return "\n".join(
            textwrap.wrap(body, 74, initial_indent="    ", subsequent_indent="    ")
        )

    entries = "\n".join(f"    {k!r}: {v}," for k, v in scale.items())
    return f'''"""TCB<->TDB parameter tables extracted from PINT.

DO NOT EDIT -- regenerate with ``python tools/regen_tcb_tables.py``.

Restricted to parameters JaxPINT declares.  The conversion logic that consumes
these lives in :mod:`jaxpint.par._tcb_tables`.
"""

from __future__ import annotations

# name -> effective dimensionality n; the scale factor is ``IFTE_K ** -n``
# (PINT: ``scale_parameter(model, par, -effective_dimensionality, ...)``).
SCALE_DIMENSIONALITY: dict[str, int] = {{
{entries}
}}

# Epochs: t_tdb = (t_tcb - IFTE_MJD0) / IFTE_K + IFTE_MJD0
MJD_PARAMS: frozenset[str] = frozenset({{
{wrap(mjd)}
}})

# PINT sets convert_tcb2tdb=False for these and leaves them untouched.
NOT_CONVERTIBLE: frozenset[str] = frozenset({{
{wrap(noconv)}
}})
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed file differs from a fresh extraction",
    )
    args = ap.parse_args(argv)

    scale, mjd, noconv = _extract()

    if args.check:
        # Compare the *tables*, not the file text: the committed module is
        # formatted by ruff (repr() emits single quotes, the formatter rewrites
        # them to double), so a byte comparison reports every formatter run as
        # staleness. Only a real change in the extracted physics should fail.
        try:
            from jaxpint.par import _tcb_generated as committed
        except ImportError:
            print(
                f"{_OUT.name} missing; run tools/regen_tcb_tables.py", file=sys.stderr
            )
            return 1
        drift = {
            "SCALE_DIMENSIONALITY": (
                set(scale.items()) ^ set(committed.SCALE_DIMENSIONALITY.items())
            ),
            "MJD_PARAMS": mjd ^ set(committed.MJD_PARAMS),
            "NOT_CONVERTIBLE": noconv ^ set(committed.NOT_CONVERTIBLE),
        }
        drift = {k: v for k, v in drift.items() if v}
        if drift:
            for table, diff in drift.items():
                print(f"{table}: {sorted(diff)[:8]}", file=sys.stderr)
            print(
                f"{_OUT.name} is stale; re-run tools/regen_tcb_tables.py",
                file=sys.stderr,
            )
            return 1
        print(f"{_OUT.name} is up to date")
        return 0

    text = _render(scale, mjd, noconv)

    _OUT.write_text(text)
    print(f"wrote {_OUT}")
    print(f"  scale={len(scale)} mjd={len(mjd)} not_convertible={len(noconv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
