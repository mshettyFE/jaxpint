"""Parameter alias synthesis (adapter-neutral).

Some par-file conventions express the same physical quantity under different
parameter names (e.g. Tempo's ``RNAMP``/``RNIDX`` vs. TempoNest's
``TNREDAMP``/``TNREDGAM``; orbital-period ``PB`` vs. orbital-frequency Taylor
series ``FB0``/``FB1``/...).  Downstream JaxPINT code reads a single canonical
name (``TNREDAMP``, ``PB``, ...), so when only the alternate form is present the
shared core synthesizes the canonical form before assembling the
``ParameterVector``.

These functions operate on the adapter-neutral ``list[RawParam]`` (appending
synthesized entries), so they work identically for the PINT bridge and the ``.par`` parser.
Adding a new alias is one new function plus one call site in :func:`apply_aliases`.

"""

from __future__ import annotations

import logging
from typing import Optional

import jax.numpy as jnp

from jaxpint.par.raw_params import ParamKind, RawParam

log = logging.getLogger(__name__)


# Source: PINT's noise_model.py conversion between Tempo2 RNAMP and TempoNest
# TNREDAMP (https://github.com/nanograv/PINT/blob/master/src/pint/models/noise_model.py#L1132).
_JULIAN_YEAR_MICRO_SEC = 86400.0 * 365.24 * 1e6
_RNAMP_FAC = _JULIAN_YEAR_MICRO_SEC / (2.0 * jnp.pi * jnp.sqrt(3.0))


def _find(raw: list[RawParam], name: str) -> Optional[RawParam]:
    for rp in raw:
        if rp.name == name:
            return rp
    return None


def _has(raw: list[RawParam], name: str) -> bool:
    return _find(raw, name) is not None


def synthesize_tnredamp_from_rnamp(raw: list[RawParam]) -> None:
    """Synthesize TNREDAMP/TNREDGAM from RNAMP/RNIDX when only the latter are
    present.  NANOGrav 15-yr par files specify red noise via the Tempo2
    convention (``RNAMP``, ``RNIDX``) but ``PLRedNoise`` reads the TempoNest
    names.  Appends to *raw* in place.
    """
    rnamp = _find(raw, "RNAMP")
    rnidx = _find(raw, "RNIDX")
    if rnamp is None or rnidx is None:
        return
    if _has(raw, "TNREDAMP") or _has(raw, "TNREDGAM"):
        return

    assert rnamp.value is not None and rnidx.value is not None
    tnredamp = float(jnp.log10(float(rnamp.value) / _RNAMP_FAC))
    tnredgam = -float(rnidx.value)
    raw.append(
        RawParam(
            name="TNREDAMP",
            kind=ParamKind.FLOAT,
            value=tnredamp,
            unit="",
            frozen=bool(rnamp.frozen),
        )
    )
    raw.append(
        RawParam(
            name="TNREDGAM",
            kind=ParamKind.FLOAT,
            value=tnredgam,
            unit="",
            frozen=bool(rnidx.frozen),
        )
    )
    log.info(
        "Synthesized TNREDAMP=%.6f, TNREDGAM=%.6f from RNAMP=%g, RNIDX=%g",
        tnredamp,
        tnredgam,
        float(rnamp.value),
        float(rnidx.value),
    )


def synthesize_pb_from_fb(raw: list[RawParam]) -> None:
    """Synthesize ``PB`` (and ``PBDOT`` if available) from the orbital-frequency
    Taylor parameterization ``FB0``/``FB1``/... when only the latter are present.
    Four NANOGrav 15-yr binaries (J0023+0923, J0636+5128, J1705-1903,
    J2214+3000) use this form.  Appends to *raw* in place.

    Conversions::

        PB [days]    = 1 / (FB0 [Hz] * 86400)
        PBDOT [s/s]  = -FB1 / FB0**2

    Higher-order terms ``FB2..FBn`` are intentionally dropped -- the PB/PBDOT
    parameterization can't express them.  For the four affected pulsars these
    are tiny secular corrections; the proper fix for a future use case is native
    FB support in the binary components, not an extension of this synthesis.
    """
    fb0 = _find(raw, "FB0")
    if fb0 is None or fb0.value is None:
        return
    if _has(raw, "PB"):
        return

    fb0_hz = float(fb0.value)
    pb_days = 1.0 / (fb0_hz * 86400.0)
    raw.append(
        RawParam(
            name="PB",
            kind=ParamKind.FLOAT,
            value=pb_days,
            unit="d",
            frozen=bool(fb0.frozen),
        )
    )

    fb1 = _find(raw, "FB1")
    if fb1 is not None and fb1.value is not None and not _has(raw, "PBDOT"):
        fb1_val = float(fb1.value)
        pbdot = -fb1_val / (fb0_hz * fb0_hz)
        raw.append(
            RawParam(
                name="PBDOT",
                kind=ParamKind.FLOAT,
                value=pbdot,
                unit="s / s",
                frozen=bool(fb1.frozen),
            )
        )
        log.info(
            "Synthesized PB=%.9f d, PBDOT=%.6e from FB0=%g Hz, FB1=%g Hz/s",
            pb_days,
            pbdot,
            fb0_hz,
            fb1_val,
        )
    else:
        log.info("Synthesized PB=%.9f d from FB0=%g Hz", pb_days, fb0_hz)


def apply_aliases(raw: list[RawParam]) -> None:
    """Run all alias synthesizers over *raw* in place."""
    synthesize_tnredamp_from_rnamp(raw)
    synthesize_pb_from_fb(raw)
