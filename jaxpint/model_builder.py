"""Model builder: ParResult → TimingModel + NoiseModel.

Constructs JaxPINT timing and noise components from a
:class:`~jaxpint.par.result.ParResult`.  The PINT bridge delegates to
:func:`build_model` after converting a PINT model to ``ParResult`` via
:func:`~jaxpint.bridge.pint_model_to_params`.

Each component is constructed by a ``_build_<comp>(ctx)`` function registered in
:data:`_BUILDERS` (keyed by :class:`~jaxpint.par.registry.Component`).
:func:`build_model` resolves the active component set, then calls the builders in
PINT execution order and routes each result to the delay / phase / noise bucket
by its base class.  A builder returns ``None`` when its parameters are absent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Optional

from jaxpint.par.registry import BinaryModel, Component
from jaxpint._component_order import PRIORITY, DEFAULT_ORDER

import jax.numpy as jnp
import numpy as np

from jaxpint.par.result import ParResult
from jaxpint.types import TOAData
from jaxpint.utils import build_quantization_matrix as _build_quantization_matrix

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _param_is_set(par: ParResult, name: str) -> bool:
    """Check whether *name* exists in the ParameterVector and is non-zero."""
    idx = par.params._name_to_index.get(name)
    if idx is None:
        return False
    return float(par.params.values[idx]) != 0.0


def _opt_name(par: ParResult, name: str) -> Optional[str]:
    return name if _param_is_set(par, name) else None


def _has_param(par: ParResult, name: str) -> bool:
    return name in par.params._name_to_index


def _match_group1(pattern: str, s: str) -> str:
    """First capture group of ``pattern`` against ``s`` (caller guarantees a match)."""
    m = re.match(pattern, s)
    assert m is not None
    return m.group(1)


def _collect_prefix_indices(par: ParResult, prefix: str) -> list[int]:
    """Collect sorted integer indices for a prefix family (e.g. 'DMX_')."""
    indices = set()
    for pname in par.params.names:
        if pname.startswith(prefix):
            suffix = pname[len(prefix) :]
            try:
                indices.add(int(suffix))
            except ValueError:
                pass
    return sorted(indices)


def _resolve_astrometry(par: ParResult):
    """Resolve astrometry parameter names once, before the build loop.

    Returns ``(raj, decj, pmra, pmdec, posepoch, obliquity_arcsec)``.  The frame
    (equatorial vs ecliptic) is chosen from the detected component set.  These
    names are read by the astrometry, Shapiro, solar-wind and binary builders,
    so resolving them up front makes the result independent of build order
    (previously the astrometry arms mutated shared state later arms relied on).
    """
    cs = par.component_set
    raj, decj = "RAJ", "DECJ"
    pmra = pmdec = posepoch = None
    obliquity_arcsec = None

    if Component.ASTROMETRY_ECLIPTIC in cs:
        from jaxpint.constants import OBLIQUITY_ARCSEC

        raj, decj = "ELONG", "ELAT"
        if _param_is_set(par, "PMELONG"):
            pmra = "PMELONG"
        if _param_is_set(par, "PMELAT"):
            pmdec = "PMELAT"
        ecl_name = par.metadata.get("ECL", "IERS2010")
        obliquity_arcsec = OBLIQUITY_ARCSEC[ecl_name]
    elif Component.ASTROMETRY_EQUATORIAL in cs:
        if _param_is_set(par, "PMRA"):
            pmra = "PMRA"
        if _param_is_set(par, "PMDEC"):
            pmdec = "PMDEC"

    if pmra is not None or pmdec is not None:
        posepoch = "POSEPOCH" if _has_param(par, "POSEPOCH") else "PEPOCH"

    return raj, decj, pmra, pmdec, posepoch, obliquity_arcsec


def _tdb_seconds(toa_data) -> np.ndarray:
    """Per-TOA TDB time in seconds (``(tdb_int + tdb_frac) * 86400``)."""
    return (
        np.asarray(toa_data.tdb_int) * 86400.0 + np.asarray(toa_data.tdb_frac) * 86400.0
    )


def _span_seconds(par: ParResult, tdb_s, tspan_param: Optional[str] = None) -> float:
    """Observation span in seconds.

    The default is ``max - min`` of the TDB times; an explicit ``T...TSPAN``
    parameter (in days), when present, overrides it (matches PINT).
    """
    T = float(np.max(tdb_s) - np.min(tdb_s))
    if tspan_param is not None and _has_param(par, tspan_param):
        tspan_days = float(par.params.values[par.params._name_to_index[tspan_param]])
        T = tspan_days * 86400.0
    return T


@dataclass(frozen=True)
class BuildContext:
    """Shared inputs threaded to every ``_build_<comp>`` function.

    Bundles the parse result, optional TOA data, and the astrometry names
    resolved once up front (see :func:`_resolve_astrometry`).
    """

    par: ParResult
    toa_data: Optional[TOAData]
    raj: str
    decj: str
    pmra: Optional[str]
    pmdec: Optional[str]
    posepoch: Optional[str]
    obliquity_arcsec: Optional[float]


# ---------------------------------------------------------------------------
# Binary model construction
# ---------------------------------------------------------------------------


def _dd_common_kwargs(par: ParResult) -> dict:
    return dict(
        pb_name="PB",
        t0_name="T0",
        a1_name="A1",
        ecc_name="ECC",
        om_name="OM",
        pbdot_name=_opt_name(par, "PBDOT"),
        omdot_name=_opt_name(par, "OMDOT"),
        edot_name=_opt_name(par, "EDOT"),
        a1dot_name=_opt_name(par, "A1DOT"),
        xpbdot_name=_opt_name(par, "XPBDOT"),
        gamma_name=_opt_name(par, "GAMMA"),
        dr_name=_opt_name(par, "DR"),
        dth_name=_opt_name(par, "DTH"),
        a0_name=_opt_name(par, "A0"),
        b0_name=_opt_name(par, "B0"),
    )


def _ell1_common_kwargs(par: ParResult) -> dict:
    return dict(
        pb_name="PB",
        tasc_name="TASC",
        a1_name="A1",
        eps1_name="EPS1",
        eps2_name="EPS2",
        pbdot_name=_opt_name(par, "PBDOT"),
        a1dot_name=_opt_name(par, "A1DOT"),
        xpbdot_name=_opt_name(par, "XPBDOT"),
    )


def _build_binary(par: ParResult, astro_info: dict) -> object:
    """Construct the appropriate binary delay component."""
    from jaxpint.binary.bt import BinaryBT
    from jaxpint.binary.bt_piecewise import BinaryBTPiecewise
    from jaxpint.binary.dd import BinaryDD
    from jaxpint.binary.ddk import BinaryDDK
    from jaxpint.binary.ddgr import BinaryDDGR
    from jaxpint.binary.ell1 import BinaryELL1

    bname = par.binary_model
    if bname is None:
        raise ValueError("No BINARY model specified in .par file")

    match bname:
        case BinaryModel.BT:
            return BinaryBT(
                pb_name="PB",
                t0_name="T0",
                a1_name="A1",
                ecc_name="ECC",
                om_name="OM",
                pbdot_name=_opt_name(par, "PBDOT"),
                omdot_name=_opt_name(par, "OMDOT"),
                edot_name=_opt_name(par, "EDOT"),
                a1dot_name=_opt_name(par, "A1DOT"),
                gamma_name=_opt_name(par, "GAMMA"),
                xpbdot_name=_opt_name(par, "XPBDOT"),
            )

        case BinaryModel.DD:
            return BinaryDD(
                **_dd_common_kwargs(par),
                m2_name=_opt_name(par, "M2"),
                sini_name=_opt_name(par, "SINI"),
                shapiro_mode="standard",
            )

        case BinaryModel.DDS:
            return BinaryDD(
                **_dd_common_kwargs(par),
                m2_name=_opt_name(par, "M2"),
                shapmax_name="SHAPMAX",
                shapiro_mode="shapmax",
            )

        case BinaryModel.DDH:
            return BinaryDD(
                **_dd_common_kwargs(par),
                h3_name="H3",
                stigma_name="STIGMA",
                shapiro_mode="h3stigma",
            )

        case BinaryModel.ELL1:
            return BinaryELL1(
                **_ell1_common_kwargs(par),
                eps1dot_name=_opt_name(par, "EPS1DOT"),
                eps2dot_name=_opt_name(par, "EPS2DOT"),
                m2_name=_opt_name(par, "M2"),
                sini_name=_opt_name(par, "SINI"),
                shapiro_mode="standard" if _param_is_set(par, "M2") else "none",
            )

        case BinaryModel.ELL1H:
            if _param_is_set(par, "STIGMA"):
                shapiro_mode = "h3stigma"
            elif _param_is_set(par, "H4"):
                shapiro_mode = "h3h4"
            elif _param_is_set(par, "H3"):
                shapiro_mode = "h3nharms"
            else:
                shapiro_mode = "none"
            nharms = par.int_params.get("NHARMS", 7)
            return BinaryELL1(
                **_ell1_common_kwargs(par),
                eps1dot_name=_opt_name(par, "EPS1DOT"),
                eps2dot_name=_opt_name(par, "EPS2DOT"),
                h3_name=_opt_name(par, "H3"),
                stigma_name=_opt_name(par, "STIGMA"),
                h4_name=_opt_name(par, "H4"),
                shapiro_mode=shapiro_mode,
                nharms=nharms,
            )

        case BinaryModel.ELL1k:
            return BinaryELL1(
                **_ell1_common_kwargs(par),
                omdot_name=_opt_name(par, "OMDOT"),
                lnedot_name=_opt_name(par, "LNEDOT"),
                m2_name=_opt_name(par, "M2"),
                sini_name=_opt_name(par, "SINI"),
                shapiro_mode="standard" if _param_is_set(par, "M2") else "none",
            )

        case BinaryModel.DDK:
            k96 = par.bool_params.get("K96", False)
            return BinaryDDK(
                **_dd_common_kwargs(par),
                m2_name=_opt_name(par, "M2"),
                kin_name="KIN",
                kom_name="KOM",
                px_name="PX",
                raj_name=astro_info.get("raj_name", "RAJ"),
                decj_name=astro_info.get("decj_name", "DECJ"),
                pmra_name=astro_info.get("pmra_name"),
                pmdec_name=astro_info.get("pmdec_name"),
                posepoch_name=astro_info.get("posepoch_name"),
                k96=k96,
            )

        case BinaryModel.DDGR:
            return BinaryDDGR(
                pb_name="PB",
                t0_name="T0",
                a1_name="A1",
                ecc_name="ECC",
                om_name="OM",
                mtot_name="MTOT",
                m2_name="M2",
                edot_name=_opt_name(par, "EDOT"),
                a1dot_name=_opt_name(par, "A1DOT"),
                xomdot_name=_opt_name(par, "XOMDOT"),
                xpbdot_name=_opt_name(par, "XPBDOT"),
                a0_name=_opt_name(par, "A0"),
                b0_name=_opt_name(par, "B0"),
            )

        case BinaryModel.BT_PIECEWISE:
            t0x_names = sorted(n for n in par.params.names if n.startswith("T0X_"))
            a1x_names = sorted(n for n in par.params.names if n.startswith("A1X_"))
            xr1_names = sorted(n for n in par.params.names if n.startswith("XR1_"))
            xr2_names = sorted(n for n in par.params.names if n.startswith("XR2_"))
            n_pieces = len(xr1_names)

            return BinaryBTPiecewise(
                pb_name="PB",
                t0_name="T0",
                a1_name="A1",
                ecc_name="ECC",
                om_name="OM",
                pbdot_name=_opt_name(par, "PBDOT"),
                omdot_name=_opt_name(par, "OMDOT"),
                edot_name=_opt_name(par, "EDOT"),
                a1dot_name=_opt_name(par, "A1DOT"),
                gamma_name=_opt_name(par, "GAMMA"),
                xpbdot_name=_opt_name(par, "XPBDOT"),
                n_pieces=n_pieces,
                t0x_names=tuple(t0x_names),
                a1x_names=tuple(a1x_names),
                xr1_names=tuple(xr1_names),
                xr2_names=tuple(xr2_names),
            )

        case _:
            raise NotImplementedError(
                f"Binary model {bname!r} is not yet ported to JaxPINT"
            )


# ---------------------------------------------------------------------------
# Per-component builders.  Each takes a BuildContext and returns the constructed
# component, or None when its parameters are absent.  Imports are function-local
# to keep this module import-light and forcing only the active components.
# ---------------------------------------------------------------------------


# ---- Delay components ----


def _build_astrometry_equatorial(ctx: BuildContext):
    from jaxpint.delay.astrometry import AstrometryEquatorial

    return AstrometryEquatorial(
        raj_name=ctx.raj,
        decj_name=ctx.decj,
        pmra_name=ctx.pmra,
        pmdec_name=ctx.pmdec,
        px_name=_opt_name(ctx.par, "PX"),
        posepoch_name=ctx.posepoch,
    )


def _build_astrometry_ecliptic(ctx: BuildContext):
    from jaxpint.delay.astrometry import AstrometryEcliptic

    return AstrometryEcliptic(
        elong_name=ctx.raj,
        elat_name=ctx.decj,
        pmelong_name=ctx.pmra,
        pmelat_name=ctx.pmdec,
        px_name=_opt_name(ctx.par, "PX"),
        posepoch_name=ctx.posepoch,
        obliquity_arcsec=ctx.obliquity_arcsec,
    )


def _build_troposphere(ctx: BuildContext):
    from jaxpint.delay.troposphere import TroposphereDelay

    return TroposphereDelay()


def _build_shapiro(ctx: BuildContext):
    from jaxpint.delay.shapiro import SolarSystemShapiroDelay

    planet_shapiro = ctx.par.bool_params.get("PLANET_SHAPIRO", False)
    return SolarSystemShapiroDelay(
        raj_name=ctx.raj,
        decj_name=ctx.decj,
        pmra_name=ctx.pmra,
        pmdec_name=ctx.pmdec,
        posepoch_name=ctx.posepoch,
        planet_shapiro=planet_shapiro,
        obliquity_arcsec=ctx.obliquity_arcsec,
    )


def _build_solar_wind(ctx: BuildContext):
    from jaxpint.delay.solar_wind import SolarWindDispersion

    par = ctx.par
    ne_sw_names = ["NE_SW"]
    for pname in par.params.names:
        if pname.startswith("NE_SW") and pname != "NE_SW":
            val = float(par.params.values[par.params._name_to_index[pname]])
            if val != 0.0:
                ne_sw_names.append(pname)
    ne_sw_names.sort()

    ne_sw_val = float(par.params.values[par.params._name_to_index.get("NE_SW", 0)])
    if not (len(ne_sw_names) > 1 or ne_sw_val != 0.0):
        return None

    swm = par.int_params.get("SWM", 0)
    swepoch_name = "SWEPOCH" if _has_param(par, "SWEPOCH") else "PEPOCH"
    swp_name = "SWP" if swm == 1 else None

    return SolarWindDispersion(
        ne_sw_param_names=tuple(ne_sw_names),
        swepoch_name=swepoch_name,
        swm=swm,
        swp_name=swp_name,
        raj_name=ctx.raj,
        decj_name=ctx.decj,
        pmra_name=ctx.pmra,
        pmdec_name=ctx.pmdec,
        posepoch_name=ctx.posepoch,
        obliquity_arcsec=ctx.obliquity_arcsec,
    )


def _build_solar_wind_x(ctx: BuildContext):
    from jaxpint.delay.solar_wind_x import SolarWindDispersionX

    par = ctx.par
    swx_indices = _collect_prefix_indices(par, "SWXDM_")
    if not swx_indices:
        return None

    theta0_str = par.metadata.get("_SWX_THETA0_RAD")
    if theta0_str is not None:
        theta0_rad = float(theta0_str)
    else:
        theta0_rad = 0.0
        log.warning("SolarWindDispersionX theta0 not available — using 0.0")

    return SolarWindDispersionX(
        n_bins=len(swx_indices),
        swxdm_names=tuple(f"SWXDM_{i:04d}" for i in swx_indices),
        swxp_names=tuple(f"SWXP_{i:04d}" for i in swx_indices),
        swxr1_names=tuple(f"SWXR1_{i:04d}" for i in swx_indices),
        swxr2_names=tuple(f"SWXR2_{i:04d}" for i in swx_indices),
        theta0=theta0_rad,
        raj_name=ctx.raj,
        decj_name=ctx.decj,
        pmra_name=ctx.pmra,
        pmdec_name=ctx.pmdec,
        posepoch_name=ctx.posepoch,
        obliquity_arcsec=ctx.obliquity_arcsec,
    )


def _build_dispersion_dm(ctx: BuildContext):
    from jaxpint.delay.dispersion_dm import DispersionDM

    par = ctx.par
    dm_names = ["DM"]
    for pname in par.params.names:
        if pname.startswith("DM") and pname != "DM" and pname != "DMEPOCH":
            suffix = pname[2:]
            if suffix.isdigit():
                dm_names.append(pname)
    dm_names.sort(key=lambda n: int(n[2:]) if n != "DM" else 0)

    dmepoch_name = "DMEPOCH" if _has_param(par, "DMEPOCH") else "PEPOCH"

    return DispersionDM(
        dm_param_names=tuple(dm_names),
        dmepoch_name=dmepoch_name,
    )


def _build_dispersion_dmx(ctx: BuildContext):
    from jaxpint.delay.dispersion_dmx import DispersionDMX

    dmx_indices = _collect_prefix_indices(ctx.par, "DMX_")
    if not dmx_indices:
        return None
    return DispersionDMX(
        n_bins=len(dmx_indices),
        dmx_names=tuple(f"DMX_{i:04d}" for i in dmx_indices),
        dmxr1_names=tuple(f"DMXR1_{i:04d}" for i in dmx_indices),
        dmxr2_names=tuple(f"DMXR2_{i:04d}" for i in dmx_indices),
    )


def _build_dispersion_jump(ctx: BuildContext):
    from jaxpint.delay.dispersion_jump import DispersionJump

    dmjump_names = tuple(n for n in ctx.par.params.names if n.startswith("DMJUMP"))
    if not dmjump_names:
        return None
    return DispersionJump(dmjump_names=dmjump_names)


def _build_binary_comp(ctx: BuildContext):
    astro_info = {
        "raj_name": ctx.raj,
        "decj_name": ctx.decj,
        "pmra_name": ctx.pmra,
        "pmdec_name": ctx.pmdec,
        "posepoch_name": ctx.posepoch,
    }
    return _build_binary(ctx.par, astro_info=astro_info)


def _build_frequency_dependent(ctx: BuildContext):
    from jaxpint.delay.frequency_dependent import FrequencyDependent

    fd_names = sorted(
        (n for n in ctx.par.params.names if re.match(r"^FD\d+$", n)),
        key=lambda n: int(n[2:]),
    )
    if not fd_names:
        return None
    return FrequencyDependent(fd_param_names=tuple(fd_names))


def _build_fd_jump(ctx: BuildContext):
    from jaxpint.delay.fdjump import FDJump

    par = ctx.par
    fdjump_names = []
    fdjump_indices = []
    for pname in par.params.names:
        m = re.match(r"FD(\d+)JUMP\d+", pname)
        if m:
            fdjump_names.append(pname)
            fdjump_indices.append(int(m.group(1)))
    use_log = par.bool_params.get("FDJUMPLOG", True)
    if not fdjump_names:
        return None
    return FDJump(
        fdjump_param_names=tuple(fdjump_names),
        fdjump_fd_indices=tuple(fdjump_indices),
        use_log=use_log,
    )


def _build_chromatic_cm(ctx: BuildContext):
    from jaxpint.delay.chromatic_cm import ChromaticCM

    par = ctx.par
    cm_names = ["CM"]
    for pname in par.params.names:
        if pname.startswith("CM") and pname != "CM" and pname != "CMEPOCH":
            suffix = pname[2:]
            if suffix.isdigit():
                cm_names.append(pname)
    cm_names.sort(key=lambda n: int(n[2:]) if n != "CM" else 0)

    cmepoch_name = "CMEPOCH" if _has_param(par, "CMEPOCH") else "PEPOCH"
    return ChromaticCM(
        cm_param_names=tuple(cm_names),
        cmepoch_name=cmepoch_name,
        tnchromidx_name="TNCHROMIDX",
    )


def _build_chromatic_cmx(ctx: BuildContext):
    from jaxpint.delay.chromatic_cmx import ChromaticCMX

    cmx_indices = _collect_prefix_indices(ctx.par, "CMX_")
    if not cmx_indices:
        return None
    return ChromaticCMX(
        n_bins=len(cmx_indices),
        cmx_names=tuple(f"CMX_{i:04d}" for i in cmx_indices),
        cmxr1_names=tuple(f"CMXR1_{i:04d}" for i in cmx_indices),
        cmxr2_names=tuple(f"CMXR2_{i:04d}" for i in cmx_indices),
        tnchromidx_name="TNCHROMIDX",
    )


def _build_exponential_dip(ctx: BuildContext):
    from jaxpint.delay.exponential_dip import ExponentialDip

    dip_indices = _collect_prefix_indices(ctx.par, "EXPDIPEPOCH_")
    if not dip_indices:
        dip_indices = _collect_prefix_indices(ctx.par, "EXPDIPEP_")
    if not dip_indices:
        return None
    return ExponentialDip(
        n_dips=len(dip_indices),
        expdipeps_name="EXPDIPEPS",
        expdipfref_name="EXPDIPFREF",
        expdipep_names=tuple(f"EXPDIPEP_{i}" for i in dip_indices),
        expdipamp_names=tuple(f"EXPDIPAMP_{i}" for i in dip_indices),
        expdipidx_names=tuple(f"EXPDIPIDX_{i}" for i in dip_indices),
        expdiptau_names=tuple(f"EXPDIPTAU_{i}" for i in dip_indices),
    )


def _build_wave_x(ctx: BuildContext):
    from jaxpint.delay.wavex import WaveX

    wx_indices = _collect_prefix_indices(ctx.par, "WXFREQ_")
    if not wx_indices:
        return None
    wxepoch_name = "WXEPOCH" if _has_param(ctx.par, "WXEPOCH") else "PEPOCH"
    return WaveX(
        n_components=len(wx_indices),
        wxepoch_name=wxepoch_name,
        wxfreq_names=tuple(f"WXFREQ_{i:04d}" for i in wx_indices),
        wxsin_names=tuple(f"WXSIN_{i:04d}" for i in wx_indices),
        wxcos_names=tuple(f"WXCOS_{i:04d}" for i in wx_indices),
    )


def _build_dm_wave_x(ctx: BuildContext):
    from jaxpint.delay.dmwavex import DMWaveX

    dmwx_indices = _collect_prefix_indices(ctx.par, "DMWXFREQ_")
    if not dmwx_indices:
        return None
    dmwxepoch_name = "DMWXEPOCH" if _has_param(ctx.par, "DMWXEPOCH") else "PEPOCH"
    return DMWaveX(
        n_components=len(dmwx_indices),
        dmwxepoch_name=dmwxepoch_name,
        dmwxfreq_names=tuple(f"DMWXFREQ_{i:04d}" for i in dmwx_indices),
        dmwxsin_names=tuple(f"DMWXSIN_{i:04d}" for i in dmwx_indices),
        dmwxcos_names=tuple(f"DMWXCOS_{i:04d}" for i in dmwx_indices),
    )


def _build_cm_wave_x(ctx: BuildContext):
    from jaxpint.delay.cmwavex import CMWaveX

    cmwx_indices = _collect_prefix_indices(ctx.par, "CMWXFREQ_")
    if not cmwx_indices:
        return None
    cmwxepoch_name = "CMWXEPOCH" if _has_param(ctx.par, "CMWXEPOCH") else "PEPOCH"
    return CMWaveX(
        n_components=len(cmwx_indices),
        cmwxepoch_name=cmwxepoch_name,
        cmwxfreq_names=tuple(f"CMWXFREQ_{i:04d}" for i in cmwx_indices),
        cmwxsin_names=tuple(f"CMWXSIN_{i:04d}" for i in cmwx_indices),
        cmwxcos_names=tuple(f"CMWXCOS_{i:04d}" for i in cmwx_indices),
        tnchromidx_name="TNCHROMIDX",
    )


# ---- Phase components ----


def _build_spindown(ctx: BuildContext):
    from jaxpint.phase.spin import Spindown

    par = ctx.par
    spin_names = ["F0"]
    for pname in par.params.names:
        if pname.startswith("F") and pname != "F0" and pname[1:].isdigit():
            spin_names.append(pname)
    spin_names.sort(key=lambda n: int(n[1:]))
    return Spindown(spin_param_names=tuple(spin_names))


def _build_glitch(ctx: BuildContext):
    from jaxpint.phase.glitch import Glitch

    glep_indices = _collect_prefix_indices(ctx.par, "GLEP_")
    if not glep_indices:
        return None
    return Glitch(
        n_glitches=len(glep_indices),
        glep_names=tuple(f"GLEP_{i}" for i in glep_indices),
        glph_names=tuple(f"GLPH_{i}" for i in glep_indices),
        glf0_names=tuple(f"GLF0_{i}" for i in glep_indices),
        glf1_names=tuple(f"GLF1_{i}" for i in glep_indices),
        glf2_names=tuple(f"GLF2_{i}" for i in glep_indices),
        glf0d_names=tuple(f"GLF0D_{i}" for i in glep_indices),
        gltd_names=tuple(f"GLTD_{i}" for i in glep_indices),
    )


def _build_piecewise_spindown(ctx: BuildContext):
    from jaxpint.phase.piecewise_spindown import PiecewiseSpindown

    pw_indices = _collect_prefix_indices(ctx.par, "PWEP_")
    if not pw_indices:
        return None
    return PiecewiseSpindown(
        n_pieces=len(pw_indices),
        pwstart_names=tuple(f"PWSTART_{i}" for i in pw_indices),
        pwstop_names=tuple(f"PWSTOP_{i}" for i in pw_indices),
        pwep_names=tuple(f"PWEP_{i}" for i in pw_indices),
        pwph_names=tuple(f"PWPH_{i}" for i in pw_indices),
        pwf0_names=tuple(f"PWF0_{i}" for i in pw_indices),
        pwf1_names=tuple(f"PWF1_{i}" for i in pw_indices),
        pwf2_names=tuple(f"PWF2_{i}" for i in pw_indices),
    )


def _build_phase_jump(ctx: BuildContext):
    from jaxpint.phase.jump import PhaseJump

    jump_names = tuple(n for n in ctx.par.params.names if n.startswith("JUMP"))
    if not jump_names:
        return None
    return PhaseJump(jump_param_names=jump_names)


def _build_wave(ctx: BuildContext):
    from jaxpint.phase.wave import Wave

    par = ctx.par
    wave_a_names = sorted(
        (n for n in par.params.names if re.match(r"^WAVE\d+_A$", n)),
        key=lambda n: int(_match_group1(r"WAVE(\d+)_A", n)),
    )
    if not wave_a_names:
        return None
    wave_indices = [int(_match_group1(r"WAVE(\d+)_A", n)) for n in wave_a_names]
    waveepoch_name = "WAVEEPOCH" if _has_param(par, "WAVEEPOCH") else "PEPOCH"
    return Wave(
        n_terms=len(wave_indices),
        waveepoch_name=waveepoch_name,
        wave_om_name="WAVE_OM",
        wave_sin_names=tuple(f"WAVE{i}_A" for i in wave_indices),
        wave_cos_names=tuple(f"WAVE{i}_B" for i in wave_indices),
    )


def _build_ifunc(ctx: BuildContext):
    from jaxpint.phase.ifunc import IFunc

    par = ctx.par
    ifunc_a_names = sorted(
        (n for n in par.params.names if re.match(r"^IFUNC\d+_A$", n)),
        key=lambda n: int(_match_group1(r"IFUNC(\d+)_A", n)),
    )
    if not ifunc_a_names:
        return None
    interp_type = par.int_params.get("SIFUNC", 0)
    mjds = []
    delays = []
    for a_name in ifunc_a_names:
        b_name = a_name.replace("_A", "_B")
        mjd_val = float(par.params.values[par.params._name_to_index[a_name]])
        delay_val = float(par.params.values[par.params._name_to_index[b_name]])
        mjds.append(mjd_val)
        delays.append(delay_val)

    sorted_pairs = sorted(zip(mjds, delays))
    sorted_mjds, sorted_delays = zip(*sorted_pairs)
    return IFunc(
        interp_type=interp_type,
        control_mjds=tuple(float(x) for x in sorted_mjds),
        control_delays=tuple(float(x) for x in sorted_delays),
    )


# ---- Noise components ----


def _build_scale_toa_error(ctx: BuildContext):
    from jaxpint.noise.white import ScaleToaError

    par = ctx.par
    efac_names = tuple(sorted(n for n in par.params.names if n.startswith("EFAC")))
    equad_names = tuple(sorted(n for n in par.params.names if n.startswith("EQUAD")))
    return ScaleToaError(efac_names=efac_names, equad_names=equad_names)


def _build_scale_dm_error(ctx: BuildContext):
    from jaxpint.noise.dm_white import ScaleDmError

    par = ctx.par
    dmefac_names = tuple(sorted(n for n in par.params.names if n.startswith("DMEFAC")))
    dmequad_names = tuple(
        sorted(n for n in par.params.names if n.startswith("DMEQUAD"))
    )
    return ScaleDmError(dmefac_names=dmefac_names, dmequad_names=dmequad_names)


def _build_ecorr(ctx: BuildContext):
    from jaxpint.noise.ecorr import EcorrNoise

    par = ctx.par
    toa_data = ctx.toa_data
    ecorr_names = tuple(sorted(n for n in par.params.names if n.startswith("ECORR")))
    if toa_data is not None and len(ecorr_names) > 0:
        tdb_s = _tdb_seconds(toa_data)
        ecorr_masks = {}
        for ename in ecorr_names:
            if hasattr(toa_data, "flag_masks") and toa_data.flag_masks is not None:
                mask = np.asarray(
                    toa_data.flag_masks.get(
                        ename,
                        np.zeros(len(tdb_s), dtype=bool),
                    )
                )
            else:
                mask = np.ones(len(tdb_s), dtype=bool)
            ecorr_masks[ename] = mask

        U, eslices = _build_quantization_matrix(tdb_s, ecorr_masks)
        ecorr_epoch_slices = tuple(eslices[n] for n in ecorr_names)
        return EcorrNoise(
            ecorr_names=ecorr_names,
            quantization_matrix=jnp.asarray(U),
            ecorr_epoch_slices=ecorr_epoch_slices,
        )
    elif toa_data is None and len(ecorr_names) > 0:
        log.warning("EcorrNoise found but no toa_data provided — ECORR not available")
    return None


def _build_pl_red_noise(ctx: BuildContext):
    from jaxpint.noise.red_noise import PLRedNoise
    from jaxpint.utils import build_fourier_basis

    par = ctx.par
    toa_data = ctx.toa_data
    if toa_data is None:
        return None
    tdb_s = _tdb_seconds(toa_data)
    n_freqs = par.int_params.get("TNREDC", 30)
    T = _span_seconds(par, tdb_s, "TNREDTSPAN")

    F, freqs, freq_bin_widths = build_fourier_basis(tdb_s, n_freqs, T)
    return PLRedNoise(
        fourier_basis=jnp.asarray(F),
        freqs=jnp.asarray(freqs),
        freq_bin_widths=jnp.asarray(freq_bin_widths),
        tnredamp_name="TNREDAMP",
        tnredgam_name="TNREDGAM",
    )


def _build_pl_dm_noise(ctx: BuildContext):
    from jaxpint.noise.dm_noise import PLDMNoise
    from jaxpint.utils import build_fourier_basis

    par = ctx.par
    toa_data = ctx.toa_data
    if toa_data is None:
        return None
    tdb_s = _tdb_seconds(toa_data)
    n_freqs = par.int_params.get("TNDMC", 30)
    T = _span_seconds(par, tdb_s, "TNDMTSPAN")

    F, freqs, freq_bin_widths = build_fourier_basis(tdb_s, n_freqs, T)

    bary_freqs_mhz = np.asarray(toa_data.freq)
    D = (1400.0 / bary_freqs_mhz) ** 2
    F_dm = F * D[:, None]

    return PLDMNoise(
        fourier_basis=jnp.asarray(F_dm),
        freqs=jnp.asarray(freqs),
        freq_bin_widths=jnp.asarray(freq_bin_widths),
        tndmamp_name="TNDMAMP",
        tndmgam_name="TNDMGAM",
    )


def _build_pl_chrom_noise(ctx: BuildContext):
    from jaxpint.noise.chrom_noise import PLChromNoise
    from jaxpint.utils import build_fourier_basis

    par = ctx.par
    toa_data = ctx.toa_data
    if toa_data is None:
        return None
    tdb_s = _tdb_seconds(toa_data)
    n_freqs = par.int_params.get("TNCHROMC", 30)
    T = _span_seconds(par, tdb_s, "TNCHROMTSPAN")

    F, freqs, freq_bin_widths = build_fourier_basis(tdb_s, n_freqs, T)

    return PLChromNoise(
        fourier_basis=jnp.asarray(F),
        freqs=jnp.asarray(freqs),
        freq_bin_widths=jnp.asarray(freq_bin_widths),
        tnchromamp_name="TNCHROMAMP",
        tnchromgam_name="TNCHROMGAM",
        tnchromidx_name="TNCHROMIDX",
        fref=1400.0,
    )


def _build_pl_sw_noise(ctx: BuildContext):
    from jaxpint.noise.sw_noise import PLSWNoise
    from jaxpint.utils import build_fourier_basis

    par = ctx.par
    toa_data = ctx.toa_data
    if toa_data is None:
        return None
    tdb_s = _tdb_seconds(toa_data)
    n_freqs = par.int_params.get("TNSWC", 100)
    T = _span_seconds(par, tdb_s)

    F, freqs, freq_bin_widths = build_fourier_basis(tdb_s, n_freqs, T)

    swm = par.int_params.get("SWM", 0)
    swp_name = "SWP" if swm == 1 else None

    return PLSWNoise(
        fourier_basis=jnp.asarray(F),
        freqs=jnp.asarray(freqs),
        freq_bin_widths=jnp.asarray(freq_bin_widths),
        tnswamp_name="TNSWAMP",
        tnswgam_name="TNSWGAM",
        swm=swm,
        swp_name=swp_name,
        raj_name=ctx.raj,
        decj_name=ctx.decj,
        pmra_name=ctx.pmra,
        pmdec_name=ctx.pmdec,
        posepoch_name=ctx.posepoch,
        obliquity_arcsec=ctx.obliquity_arcsec,
    )


# Component -> builder.  BINARY and BINARY_BT_PIECEWISE share one builder (it
# reads par.binary_model to pick the model); a component absent here that is
# nonetheless active raises NotImplementedError in build_model.
_BUILDERS: dict[Component, Callable[[BuildContext], object]] = {
    Component.ASTROMETRY_EQUATORIAL: _build_astrometry_equatorial,
    Component.ASTROMETRY_ECLIPTIC: _build_astrometry_ecliptic,
    Component.TROPOSPHERE_DELAY: _build_troposphere,
    Component.SOLAR_SYSTEM_SHAPIRO: _build_shapiro,
    Component.SOLAR_WIND_DISPERSION: _build_solar_wind,
    Component.SOLAR_WIND_DISPERSION_X: _build_solar_wind_x,
    Component.DISPERSION_DM: _build_dispersion_dm,
    Component.DISPERSION_DMX: _build_dispersion_dmx,
    Component.DISPERSION_JUMP: _build_dispersion_jump,
    Component.BINARY: _build_binary_comp,
    Component.BINARY_BT_PIECEWISE: _build_binary_comp,
    Component.FREQUENCY_DEPENDENT: _build_frequency_dependent,
    Component.FD_JUMP: _build_fd_jump,
    Component.CHROMATIC_CM: _build_chromatic_cm,
    Component.CHROMATIC_CMX: _build_chromatic_cmx,
    Component.EXPONENTIAL_DIP: _build_exponential_dip,
    Component.WAVE_X: _build_wave_x,
    Component.DM_WAVE_X: _build_dm_wave_x,
    Component.CM_WAVE_X: _build_cm_wave_x,
    Component.SPINDOWN: _build_spindown,
    Component.GLITCH: _build_glitch,
    Component.PIECEWISE_SPINDOWN: _build_piecewise_spindown,
    Component.PHASE_JUMP: _build_phase_jump,
    Component.WAVE: _build_wave,
    Component.IFUNC: _build_ifunc,
    Component.SCALE_TOA_ERROR: _build_scale_toa_error,
    Component.SCALE_DM_ERROR: _build_scale_dm_error,
    Component.ECORR_NOISE: _build_ecorr,
    Component.PL_RED_NOISE: _build_pl_red_noise,
    Component.PL_DM_NOISE: _build_pl_dm_noise,
    Component.PL_CHROM_NOISE: _build_pl_chrom_noise,
    Component.PL_SW_NOISE: _build_pl_sw_noise,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _active_components(par: ParResult) -> set[Component]:
    """The components to build, in addition to those detected in ``par``.

    Components enabled only by special conditions (not membership in
    ``par.component_set`` alone) are added here: Shapiro rides with astrometry,
    troposphere with ``CORRECT_TROPOSPHERE``, and a binary with any
    ``BINARY`` line.  ``PHASE_OFFSET`` (a ``TimingModel`` field) and ``NONE``
    (admin params) have no builder and are excluded.
    """
    _auto_detect_only = {
        Component.NONE,
        Component.PHASE_OFFSET,
        Component.TROPOSPHERE_DELAY,
        Component.SOLAR_SYSTEM_SHAPIRO,
    }
    active = {comp for comp in par.component_set if comp not in _auto_detect_only}

    has_astrometry = (
        Component.ASTROMETRY_EQUATORIAL in active
        or Component.ASTROMETRY_ECLIPTIC in active
    )
    if has_astrometry:
        active.add(Component.SOLAR_SYSTEM_SHAPIRO)
    if par.bool_params.get("CORRECT_TROPOSPHERE", False):
        active.add(Component.TROPOSPHERE_DELAY)
    if par.binary_model is not None and Component.BINARY not in active:
        active.add(Component.BINARY)

    return active


def build_model(
    par: ParResult,
    toa_data: Optional[TOAData] = None,
):
    """Build JaxPINT TimingModel + NoiseModel from a parsed .par result.

    Parameters
    ----------
    par : ~jaxpint.par.result.ParResult
        Output of :func:`jaxpint.bridge.pint_model_to_params`.
    toa_data : TOAData, optional
        If provided, TOA-dependent components (ECORR, red noise, etc.)
        will be constructed.

    Returns
    -------
    (TimingModel, NoiseModel)
    """
    from jaxpint.model import TimingModel
    from jaxpint.components import (
        DelayComponent,
        DispersionDelayComponent,
        NoiseComponent,
        PhaseComponent,
    )
    from jaxpint.noise.noise_model import NoiseModel
    from jaxpint.noise.white import ScaleToaError
    from jaxpint.noise.dm_white import ScaleDmError

    # Astrometry names, resolved once up front (read by astrometry / Shapiro /
    # solar wind / binary builders -- so the result is independent of build order).
    raj, decj, pmra, pmdec, posepoch, obliquity_arcsec = _resolve_astrometry(par)
    ctx = BuildContext(
        par=par,
        toa_data=toa_data,
        raj=raj,
        decj=decj,
        pmra=pmra,
        pmdec=pmdec,
        posepoch=posepoch,
        obliquity_arcsec=obliquity_arcsec,
    )

    phoff_name = "PHOFF" if _param_is_set(par, "PHOFF") else None

    # Process the active components in PINT execution order and route each
    # result to its bucket by base class.  Iteration order is the priority
    # order; ``comp.value`` breaks ties identically to the old priority heap.
    active = _active_components(par)
    delay_components = []
    phase_components = []
    noise_components = []  # in priority order: white, dm_white, then correlated

    for comp in sorted(
        active, key=lambda c: (PRIORITY.get(c, len(DEFAULT_ORDER)), c.value)
    ):
        builder = _BUILDERS.get(comp)
        if builder is None:
            raise NotImplementedError(
                f"Component {comp!r} is present in the par file "
                f"but is not yet implemented in JaxPINT"
            )
        obj = builder(ctx)
        if obj is None:
            continue
        if isinstance(obj, DelayComponent):
            delay_components.append(obj)
        elif isinstance(obj, PhaseComponent):
            phase_components.append(obj)
        elif isinstance(obj, (NoiseComponent, ScaleDmError)):
            noise_components.append(obj)
        else:
            raise TypeError(
                f"Builder for {comp!r} returned an unroutable {type(obj).__name__}"
            )

    # ---- Assemble ----
    dispersion_components = tuple(
        c for c in delay_components if isinstance(c, DispersionDelayComponent)
    )

    timing_model = TimingModel(
        delay_components=tuple(delay_components),
        phase_components=tuple(phase_components),
        dispersion_components=dispersion_components,
        phoff_name=phoff_name,
    )

    # Partition noise by type.  ScaleToaError -> white slot, ScaleDmError -> DM
    # white slot, the rest -> correlated (kept in priority order, which equals
    # the historical ECORR, PLRed, PLDM, PLChrom, PLSW append sequence).
    white_noise = next(
        (c for c in noise_components if isinstance(c, ScaleToaError)), None
    )
    dm_white_noise = next(
        (c for c in noise_components if isinstance(c, ScaleDmError)), None
    )
    correlated = tuple(
        c for c in noise_components if not isinstance(c, (ScaleToaError, ScaleDmError))
    )

    combined_noise = NoiseModel(
        white_noise=white_noise,
        correlated=correlated,
        dm_white_noise=dm_white_noise,
    )

    return timing_model, combined_noise
