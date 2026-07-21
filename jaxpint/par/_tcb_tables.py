"""TCB -> TDB conversion of a parsed ``.par``.

A timing model fitted in TCB cannot simply be relabelled TDB: TCB and TDB differ
by a rate factor, so every fitted quantity picks up a power of ``IFTE_K`` set by
its "effective dimensionality" (the power of seconds in its SI dimension), and
epochs transform affinely about ``IFTE_MJD0``.

The parameter tables live in :mod:`jaxpint.par._tcb_generated`, extracted from
PINT (each parameter's ``convert_tcb2tdb`` flag and ``effective_dimensionality``)
and restricted to parameters JaxPINT declares.  Regenerate them with::

    python tools/regen_tcb_tables.py            # rewrite
    python tools/regen_tcb_tables.py --check    # verify freshness

This module holds only the conversion logic, so regenerating never touches it.

**Strictness (deliberate divergence from PINT).**  PINT converts what it can and
silently leaves the rest.  Two of those omissions are refused here instead:

* ``TZRMJD``.  Epochs shift by ~15.9 s at MJD 55000, so converting
  PEPOCH/T0/TASC while leaving the absolute-phase anchor fixed corrupts absolute
  phase by ~2760 turns at F0 ~ 174 Hz.  Silently wrong, not approximately right.
* Any numeric parameter with no known dimensionality.  Scaling by the wrong
  power of ``IFTE_K`` is worse than refusing.

The result is still approximate in the way PINT's is: the model was *fitted* in
TCB, and rescaling each parameter is not the same as re-minimizing.  Re-fit
before trusting it.
"""

from __future__ import annotations

import numpy as np

# Re-exported so callers need only one import.
from jaxpint.par._tcb_generated import (
    MJD_PARAMS,
    NOT_CONVERTIBLE,
    SCALE_DIMENSIONALITY,
)

# Irwin & Fukushima (1999); identical to the constants tempo2 uses.
IFTE_MJD0 = np.longdouble("43144.0003725")
IFTE_KM1 = np.longdouble("1.55051979176e-8")
IFTE_K = 1 + IFTE_KM1

__all__ = [
    "IFTE_K",
    "IFTE_MJD0",
    "MJD_PARAMS",
    "NOT_CONVERTIBLE",
    "SCALE_DIMENSIONALITY",
    "convert_raw_params_tcb_to_tdb",
    "dimensionality_for",
]


def _scale(value, n):
    """x_tdb = x_tcb * IFTE_K ** -n  (PINT: scale_parameter with -eff_dim)."""
    return None if value is None else float(value * IFTE_K ** (-n))


def _transform_mjd(split):
    """t_tdb = (t_tcb - IFTE_MJD0) / IFTE_K + IFTE_MJD0, keeping the int/frac split.

    Done in longdouble on the *recombined* day and re-split, because the shift is
    ~0.5 s/yr of epoch -- far above the float64 resolution of a bare MJD (~1 us
    at MJD 55000)
    """
    if split is None:
        return None
    day = np.longdouble(split[0]) + np.longdouble(split[1])
    out = (day - IFTE_MJD0) / IFTE_K + IFTE_MJD0
    i = np.floor(out)
    return (float(i), float(out - i))


# Families whose trailing index is a *derivative order*, so the dimensionality
# changes with the index and the family cannot be collapsed onto one table key:
# F0 is s^-1, F1 is s^-2, F2 is s^-3 ...  ``base`` is n at index 0.
_DERIVATIVE_FAMILIES = {"F": -1, "DM": -1, "NE_SW": -2}


def _template_for(name: str) -> str:
    """Map an indexed name onto its canonical table key (DMX_0042 -> DMX_0001).

    The tables are keyed by each family's canonical first index, so repeatable
    *instance* families (DMX_, GLEP_, JUMP, WAVE, ...) must be resolved before
    lookup.  Uses the parser's own ``PREFIX_MAP`` rather than string-munging:
    several non-family names simply end in digits (``A1``, ``M2``, ``EPS1``).

    Derivative families are handled by :func:`dimensionality_for`, not here --
    collapsing ``F1`` onto ``F0`` would silently apply the wrong power.
    """
    from jaxpint.par import spec as S

    if name in SCALE_DIMENSIONALITY or name in MJD_PARAMS or name in NOT_CONVERTIBLE:
        return name
    best = ""
    for prefix in S.PREFIX_MAP:
        if len(prefix) > len(best) and name.startswith(prefix):
            if name[len(prefix) :].isdigit():
                best = prefix
    return S.PREFIX_MAP[best] if best else name


def dimensionality_for(name: str):
    """``n`` for a scalable parameter, ``"mjd"`` for an epoch, else ``None``.

    ``None`` means "not scalable and not an epoch" -- either explicitly
    non-convertible (PINT's ``convert_tcb2tdb=False``) or unknown to the table.
    Callers must treat unknown *numeric* parameters as a hard failure rather
    than passing them through unscaled.
    """
    for fam, base in _DERIVATIVE_FAMILIES.items():
        rest = name[len(fam) :]
        if name.startswith(fam) and (rest == "" or rest.isdigit()):
            return base - (int(rest) if rest else 0)
    key = _template_for(name)
    if key in MJD_PARAMS:
        return "mjd"
    return SCALE_DIMENSIONALITY.get(key)


def convert_raw_params_tcb_to_tdb(raw):
    """Convert a parsed TCB ``.par`` (list of ``RawParam``) to TDB.

    Scales each parameter by ``IFTE_K ** -n`` and transforms epochs about
    ``IFTE_MJD0``, mirroring PINT's ``convert_tcb_tdb``.

    **Refuses when ``TZRMJD`` is present.**  ``TZRMJD`` is the absolute-phase
    anchor and PINT marks it ``convert_tcb2tdb=False``, so a converted model has
    its spin epoch moved (~15.9 s at MJD 55000) while the anchor stays put.  At
    F0 ~ 174 Hz that is ~2760 turns of absolute phase -- silently wrong rather
    than approximately right.

    Parameters PINT also leaves alone (EFAC/EQUAD/ECORR, TempoNest noise, FD,
    WAVE, IFUNC) are left alone here too and do **not** trigger a refusal: they
    are either dimensionless (zero error by construction) or time-dimensioned at
    a relative 1.55e-8, i.e. ~8 femtoseconds on a 0.5 us EQUAD.  Refusing on
    those would reject essentially every real PTA par for nothing.

    The result is still approximate: the model was fitted in TCB, and rescaling
    is not the same as re-minimizing. Re-fit before trusting it.
    """
    import dataclasses

    blocked = sorted({rp.name for rp in raw if _template_for(rp.name) == "TZRMJD"})
    if blocked:
        raise NotImplementedError(
            "cannot convert this TCB par to TDB: it sets "
            f"{', '.join(blocked)}, which the conversion cannot transform "
            "(PINT marks TZRMJD convert_tcb2tdb=False). Epochs shift by ~15.9 s "
            "at MJD 55000, so converting PEPOCH/T0/TASC while leaving the "
            "absolute-phase anchor fixed corrupts absolute phase by thousands of "
            "turns. Remove TZRMJD and re-anchor, or refit the model in TDB."
        )

    # Numeric kinds carry physical dimension and must be scaled; STR/BOOL/INT
    # are metadata (PSR, EPHEM, NHARMS, PLANET_SHAPIRO...) and pass through.
    numeric = {"float", "angle", "mjd", "pair", "mask"}
    unknown = sorted(
        {
            rp.name
            for rp in raw
            if rp.kind.value in numeric
            and dimensionality_for(rp.name) is None
            and _template_for(rp.name) not in NOT_CONVERTIBLE
        }
    )
    if unknown:
        raise NotImplementedError(
            "cannot convert this TCB par to TDB: no TCB scaling is known for "
            f"{', '.join(unknown)}. Scaling an unknown parameter by the wrong "
            "power of IFTE_K is worse than refusing, so the conversion stops "
            "here. Add the parameter to SCALE_DIMENSIONALITY (from PINT's "
            "effective_dimensionality) if it should be converted."
        )

    out = []
    for rp in raw:
        n = dimensionality_for(rp.name)
        if n == "mjd" and rp.mjd_split is not None:
            out.append(dataclasses.replace(rp, mjd_split=_transform_mjd(rp.mjd_split)))
        elif isinstance(n, int):
            out.append(
                dataclasses.replace(
                    rp,
                    value=_scale(rp.value, n),
                    uncertainty=_scale(rp.uncertainty, n),
                    value_pair=(
                        None
                        if rp.value_pair is None
                        else (_scale(rp.value_pair[0], n), _scale(rp.value_pair[1], n))
                    ),
                )
            )
        else:
            out.append(rp)
    return out
