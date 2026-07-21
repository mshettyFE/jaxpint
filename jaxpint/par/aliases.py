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
import math
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


def _mask_selector_key(rp: RawParam) -> tuple[str, Optional[str], Optional[str]]:
    """Dash-stripped, case-folded selector identity (see par.core._mask_selector)."""
    key = (rp.mask_key or "").lstrip("-").lower()
    return (key, rp.mask_key_value, rp.mask_key_value2)


def synthesize_equad_from_tneq(raw: list[RawParam]) -> None:
    """Convert TempoNest ``TNEQ`` parameters into the equivalent ``EQUAD``.

    ``TNEQ`` is EQUAD in log10(seconds); ``EQUAD`` is microseconds.  The TNEQ
    entries are removed once converted -- they are a source convention, not a
    model parameter, and the components discover their instances via
    ``names_with_prefix("EQUAD")``.

    Diverges from PINT in two places, both off the numerical path:

    * **index allocation** -- PINT reuses the TNEQ's index and silently
      overwrites a same-index EQUAD's value *and* key, destroying a
      user-specified parameter.  We allocate a fresh index, keep both, and warn.
      (Reusing it would also emit a duplicate name, which
      ``names_with_prefix`` then applies twice.)
    * **uncertainty** -- PINT drops it; we propagate by the delta method.
      ``x = 10**y`` gives ``dx/dy = x * ln(10)``, hence
      ``sigma_EQUAD[us] = EQUAD[us] * ln(10) * sigma_TNEQ[dex]``.  First-order
      only: the map is nonlinear, so a symmetric Gaussian in dex is asymmetric
      in linear units while ``uncertainties`` stores one symmetric 1-sigma.
    """
    tneqs = [
        rp
        for rp in raw
        if rp.kind is ParamKind.MASK
        and rp.name.startswith("TNEQ")
        and rp.mask_key is not None
        and rp.value is not None
    ]
    if not tneqs:
        return

    existing = {
        _mask_selector_key(rp): rp
        for rp in raw
        if rp.kind is ParamKind.MASK and rp.name.startswith("EQUAD")
    }
    # name -> human-readable selector, for the index-collision warning below
    existing_names = {
        rp.name: f"{rp.mask_key} {rp.mask_key_value}"
        for rp in raw
        if rp.kind is ParamKind.MASK and rp.name.startswith("EQUAD")
    }

    for tneq in tneqs:
        if tneq.value is None:
            # A TNEQ line with no value cannot be converted (the synthesis is
            # 10**value). Skip it rather than crashing in the exponentiation
            # below, matching the other `continue` branches in this loop; a
            # component that actually needs the EQUAD will fail loudly later in
            # validate_referenced_params.
            log.warning("%s has no value; cannot synthesize an EQUAD.", tneq.name)
            continue
        sel = _mask_selector_key(tneq)
        if sel in existing:
            # PINT's setup() prefers the explicit EQUAD when both describe the
            # same selector.  On the bridge path this is the normal case (PINT
            # already ran the conversion), so log at INFO to avoid a warning on
            # every load of a TempoNest par.
            log.info(
                "%s is already provided by %s for the same selector; "
                "using the EQUAD value.",
                tneq.name,
                existing[sel].name,
            )
            continue
        # Allocate a fresh, non-colliding index.  Reusing the TNEQ's own index
        # (as PINT does) would collide with an existing EQUAD carrying a
        # different selector -- see the docstring.
        taken = {
            rp.name
            for rp in raw
            if rp.kind is ParamKind.MASK and rp.name.startswith("EQUAD")
        }
        wanted = "".join(c for c in tneq.name if c.isdigit()) or "1"
        if f"EQUAD{wanted}" in taken:
            log.warning(
                "%s would map to EQUAD%s, which already exists with a different "
                "selector (%s). PINT silently overwrites that parameter's value "
                "and key here; JaxPINT keeps both and assigns a fresh index. The "
                "two stacks will disagree for this par file.",
                tneq.name,
                wanted,
                existing_names.get(f"EQUAD{wanted}", "unknown selector"),
            )
            n = 1
            while f"EQUAD{n}" in taken:
                n += 1
            wanted = str(n)
        equad_us = (10.0**tneq.value) * 1e6  # log10(s) -> us
        # Delta method: x = 10**y  =>  dx/dy = x * ln(10).  See the docstring for
        # why this is a first-order approximation and why propagating it is safe.
        unc_us = (
            None
            if tneq.uncertainty is None
            else equad_us * math.log(10.0) * float(tneq.uncertainty)
        )
        raw.append(
            RawParam(
                f"EQUAD{wanted}",
                ParamKind.MASK,
                value=equad_us,
                uncertainty=unc_us,
                unit="us",
                frozen=tneq.frozen,
                mask_key=tneq.mask_key,
                mask_key_value=tneq.mask_key_value,
                mask_key_value2=tneq.mask_key_value2,
            )
        )

    # TNEQ is a source convention, not a model parameter: drop it once converted.
    raw[:] = [rp for rp in raw if not rp.name.startswith("TNEQ")]


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

    Higher-order terms ``FB2..FBn`` are deliberately *not* folded in here --
    the PB/PBDOT parameterization cannot express them.  They are instead
    consumed natively by the ELL1 component.
    They are not negligible: dropping them
    cost 7.7e-06 s against tempo2 on J0023+0923 (FB0..FB3), 14x worse than
    PINT on the same file; consuming them brings it to 1.3e-08 s.  Nor are
    they rare -- 11 of the 318 NANOGrav 15-yr pars use FB2+, with indices up
    to FB5, mostly black widows and redbacks whose orbits vary.
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
    synthesize_equad_from_tneq(raw)
    synthesize_tnredamp_from_rnamp(raw)
    synthesize_pb_from_fb(raw)
