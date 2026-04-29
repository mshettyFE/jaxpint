"""Parameter alias synthesis for the PINT → JaxPINT bridge.

Some par-file conventions express the same physical quantity under
different parameter names (e.g. Tempo's ``RNAMP``/``RNIDX`` vs. TempoNest's
``TNREDAMP``/``TNREDGAM``; orbital-period ``PB`` vs. orbital-frequency
Taylor series ``FB0``/``FB1``/...). Downstream JaxPINT code reads a single
canonical name (``TNREDAMP``, ``PB``, ...), so when only the alternate
form is present the bridge synthesizes the canonical form before building
components.

Each function takes the in-progress lists from
``pint_model_to_params`` and mutates them in place when its conditions are
met. Adding a new alias is one new function plus one call site.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
from pint.models.timing_model import TimingModel as PINTTimingModel

log = logging.getLogger(__name__)


# Source: PINT's noise_model.py conversion between Tempo2 RNAMP and TempoNest
# TNREDAMP (https://github.com/nanograv/PINT/blob/master/src/pint/models/noise_model.py#L1132).
_JULIAN_YEAR_MICRO_SEC = 86400.0 * 365.24 * 1e6
_RNAMP_FAC = _JULIAN_YEAR_MICRO_SEC / (2.0 * jnp.pi * jnp.sqrt(3.0))


def synthesize_tnredamp_from_rnamp(
    model: PINTTimingModel,
    names: list[str],
    values: list,
    units: list[str],
    frozen_mask: list[bool],
) -> None:
    """Synthesize TNREDAMP/TNREDGAM from RNAMP/RNIDX when only the latter
    are populated. NANOGrav 15-yr par files specify red noise via the
    Tempo2 convention (``RNAMP``, ``RNIDX``) but ``PLRedNoise`` reads the
    TempoNest names.
    """
    rnamp = getattr(model, "RNAMP", None)
    rnidx = getattr(model, "RNIDX", None)
    tnredamp = getattr(model, "TNREDAMP", None)
    tnredgam = getattr(model, "TNREDGAM", None)
    if not (
        tnredamp is not None and tnredgam is not None
        and rnamp is not None and rnidx is not None
        and tnredamp.value is None and tnredgam.value is None
        and rnamp.value is not None and rnidx.value is not None
    ):
        return

    names.append("TNREDAMP")
    values.append(jnp.log10(float(rnamp.value) / _RNAMP_FAC))
    units.append("")
    frozen_mask.append(bool(rnamp.frozen))
    names.append("TNREDGAM")
    values.append(-float(rnidx.value))
    units.append("")
    frozen_mask.append(bool(rnidx.frozen))
    log.info(
        "Synthesized TNREDAMP=%.6f, TNREDGAM=%.6f from RNAMP=%g, RNIDX=%g",
        values[-2], values[-1], float(rnamp.value), float(rnidx.value),
    )


def synthesize_pb_from_fb(
    model: PINTTimingModel,
    names: list[str],
    values: list,
    units: list[str],
    frozen_mask: list[bool],
) -> None:
    """Synthesize ``PB`` (and ``PBDOT`` if available) from the orbital
    frequency Taylor parameterization ``FB0``/``FB1``/... when only the
    latter are populated. Four NANOGrav 15-yr binaries (J0023+0923,
    J0636+5128, J1705-1903, J2214+3000) use this form.

    Conversions:

        PB [days]    = 1 / (FB0 [Hz] * 86400)
        PBDOT [s/s]  = -FB1 / FB0**2

    Higher-order terms ``FB2..FBn`` are intentionally dropped — the
    PB/PBDOT parameterization can't express them. For the four affected
    pulsars these are tiny secular corrections; if a future use case
    needs them, the proper fix is native FB support in the binary
    components (re-derive orbital phase from the Taylor series in
    ``jaxpint/binary/common.py`` and add ``fb_names`` knobs to the
    binary classes), not an extension of this synthesis.
    """
    fb0 = getattr(model, "FB0", None)
    pb = getattr(model, "PB", None)
    if fb0 is None or fb0.value is None:
        return
    if pb is not None and pb.value is not None:
        return

    fb0_hz = float(fb0.value)
    pb_days = 1.0 / (fb0_hz * 86400.0)
    names.append("PB")
    values.append(pb_days)
    units.append("d")
    frozen_mask.append(bool(fb0.frozen))

    fb1 = getattr(model, "FB1", None)
    pbdot_p = getattr(model, "PBDOT", None)
    pbdot_synth = (
        fb1 is not None and fb1.value is not None
        and (pbdot_p is None or pbdot_p.value is None)
    )
    if pbdot_synth:
        fb1_val = float(fb1.value)
        pbdot = -fb1_val / (fb0_hz * fb0_hz)
        names.append("PBDOT")
        values.append(pbdot)
        units.append("s / s")
        frozen_mask.append(bool(fb1.frozen))
        log.info(
            "Synthesized PB=%.9f d, PBDOT=%.6e from FB0=%g Hz, FB1=%g Hz/s",
            pb_days, pbdot, fb0_hz, fb1_val,
        )
    else:
        log.info(
            "Synthesized PB=%.9f d from FB0=%g Hz", pb_days, fb0_hz,
        )
