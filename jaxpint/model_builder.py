"""Model builder: ParResult → TimingModel + NoiseModel.

Constructs JaxPINT timing and noise components from a
:class:`~jaxpint.par.result.ParResult`.  The PINT bridge delegates to
:func:`build_model` after converting a PINT model to ``ParResult`` via
:func:`~jaxpint.bridge.pint_model_to_params`.
"""

from __future__ import annotations

import heapq
import logging
import re
from typing import Optional

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

    if bname is BinaryModel.BT:
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

    elif bname is BinaryModel.DD:
        return BinaryDD(
            **_dd_common_kwargs(par),
            m2_name=_opt_name(par, "M2"),
            sini_name=_opt_name(par, "SINI"),
            shapiro_mode="standard",
        )

    elif bname is BinaryModel.DDS:
        return BinaryDD(
            **_dd_common_kwargs(par),
            m2_name=_opt_name(par, "M2"),
            shapmax_name="SHAPMAX",
            shapiro_mode="shapmax",
        )

    elif bname is BinaryModel.DDH:
        return BinaryDD(
            **_dd_common_kwargs(par),
            h3_name="H3",
            stigma_name="STIGMA",
            shapiro_mode="h3stigma",
        )

    elif bname is BinaryModel.ELL1:
        return BinaryELL1(
            **_ell1_common_kwargs(par),
            eps1dot_name=_opt_name(par, "EPS1DOT"),
            eps2dot_name=_opt_name(par, "EPS2DOT"),
            m2_name=_opt_name(par, "M2"),
            sini_name=_opt_name(par, "SINI"),
            shapiro_mode="standard" if _param_is_set(par, "M2") else "none",
        )

    elif bname is BinaryModel.ELL1H:
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

    elif bname is BinaryModel.ELL1k:
        return BinaryELL1(
            **_ell1_common_kwargs(par),
            omdot_name=_opt_name(par, "OMDOT"),
            lnedot_name=_opt_name(par, "LNEDOT"),
            m2_name=_opt_name(par, "M2"),
            sini_name=_opt_name(par, "SINI"),
            shapiro_mode="standard" if _param_is_set(par, "M2") else "none",
        )

    elif bname is BinaryModel.DDK:
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

    elif bname is BinaryModel.DDGR:
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

    elif bname is BinaryModel.BT_PIECEWISE:
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

    else:
        raise NotImplementedError(
            f"Binary model {bname!r} is not yet ported to JaxPINT"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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
    from jaxpint.components import DispersionDelayComponent
    from jaxpint.phase.spin import Spindown
    from jaxpint.delay.dispersion_dm import DispersionDM
    from jaxpint.delay.dispersion_dmx import DispersionDMX
    from jaxpint.delay.dispersion_jump import DispersionJump
    from jaxpint.delay.astrometry import AstrometryEquatorial, AstrometryEcliptic
    from jaxpint.noise.ecorr import EcorrNoise
    from jaxpint.noise.white import ScaleToaError
    from jaxpint.noise.noise_model import NoiseModel
    from jaxpint.noise.dm_white import ScaleDmError as JaxScaleDmError
    from jaxpint.noise.red_noise import PLRedNoise
    from jaxpint.noise.dm_noise import PLDMNoise
    from jaxpint.noise.chrom_noise import PLChromNoise
    from jaxpint.noise.sw_noise import PLSWNoise
    from jaxpint.delay.shapiro import SolarSystemShapiroDelay
    from jaxpint.delay.solar_wind import SolarWindDispersion
    from jaxpint.delay.solar_wind_x import SolarWindDispersionX
    from jaxpint.delay.troposphere import TroposphereDelay
    from jaxpint.phase.jump import PhaseJump
    from jaxpint.phase.glitch import Glitch
    from jaxpint.delay.wavex import WaveX
    from jaxpint.delay.dmwavex import DMWaveX
    from jaxpint.delay.cmwavex import CMWaveX
    from jaxpint.delay.chromatic_cm import ChromaticCM
    from jaxpint.delay.chromatic_cmx import ChromaticCMX
    from jaxpint.delay.frequency_dependent import FrequencyDependent
    from jaxpint.delay.fdjump import FDJump
    from jaxpint.delay.exponential_dip import ExponentialDip
    from jaxpint.phase.piecewise_spindown import PiecewiseSpindown
    from jaxpint.phase.wave import Wave
    from jaxpint.phase.ifunc import IFunc

    delay_components = []
    phase_components = []
    white_noise = None
    dm_white_noise = None
    ecorr_noise = None
    plred_noise = None
    pldm_noise = None
    plchrom_noise = None
    plsw_noise = None

    # Detect which components are present from parsed parameter metadata
    comp_set = par.component_set

    # Astrometry names, resolved once up front (read by astrometry / Shapiro /
    # solar wind / binary builders -- so the result is independent of build order).
    (
        _astro_raj,
        _astro_decj,
        _astro_pmra,
        _astro_pmdec,
        _astro_posepoch,
        _astro_obliquity_arcsec,
    ) = _resolve_astrometry(par)

    phoff_name = None
    if _param_is_set(par, "PHOFF"):
        phoff_name = "PHOFF"

    # ---- Build set of active components ----
    # Components that are only enabled via special conditions (not comp_set
    # membership alone) are excluded from the generic loop and handled below.
    _auto_detect_only = {
        Component.NONE,
        Component.PHASE_OFFSET,
        Component.TROPOSPHERE_DELAY,
        Component.SOLAR_SYSTEM_SHAPIRO,
    }
    active = set()
    for comp in comp_set:
        if comp not in _auto_detect_only:
            active.add(comp)

    # Auto-detected components
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

    # ---- Build priority queue and process in PINT order ----
    queue = []
    for comp in active:
        pri = PRIORITY.get(comp, len(DEFAULT_ORDER))
        heapq.heappush(queue, (pri, comp.value, comp))

    while queue:
        _, _, comp = heapq.heappop(queue)
        match comp:
            # ---- Astrometry Equatorial ----
            case Component.ASTROMETRY_EQUATORIAL:
                delay_components.append(
                    AstrometryEquatorial(
                        raj_name=_astro_raj,
                        decj_name=_astro_decj,
                        pmra_name=_astro_pmra,
                        pmdec_name=_astro_pmdec,
                        px_name=_opt_name(par, "PX"),
                        posepoch_name=_astro_posepoch,
                    )
                )

            # ---- Astrometry Ecliptic ----
            case Component.ASTROMETRY_ECLIPTIC:
                delay_components.append(
                    AstrometryEcliptic(
                        elong_name=_astro_raj,
                        elat_name=_astro_decj,
                        pmelong_name=_astro_pmra,
                        pmelat_name=_astro_pmdec,
                        px_name=_opt_name(par, "PX"),
                        posepoch_name=_astro_posepoch,
                        obliquity_arcsec=_astro_obliquity_arcsec,
                    )
                )

            # ---- TroposphereDelay ----
            case Component.TROPOSPHERE_DELAY:
                delay_components.append(TroposphereDelay())

            # ---- SolarSystemShapiro ----
            case Component.SOLAR_SYSTEM_SHAPIRO:
                planet_shapiro = par.bool_params.get("PLANET_SHAPIRO", False)
                delay_components.append(
                    SolarSystemShapiroDelay(
                        raj_name=_astro_raj,
                        decj_name=_astro_decj,
                        pmra_name=_astro_pmra,
                        pmdec_name=_astro_pmdec,
                        posepoch_name=_astro_posepoch,
                        planet_shapiro=planet_shapiro,
                        obliquity_arcsec=_astro_obliquity_arcsec,
                    )
                )

            # ---- SolarWindDispersion ----
            case Component.SOLAR_WIND_DISPERSION:
                ne_sw_names = ["NE_SW"]
                for pname in par.params.names:
                    if pname.startswith("NE_SW") and pname != "NE_SW":
                        val = float(par.params.values[par.params._name_to_index[pname]])
                        if val != 0.0:
                            ne_sw_names.append(pname)
                ne_sw_names.sort()

                ne_sw_val = float(
                    par.params.values[par.params._name_to_index.get("NE_SW", 0)]
                )
                if len(ne_sw_names) > 1 or ne_sw_val != 0.0:
                    swm = par.int_params.get("SWM", 0)
                    swepoch_name = "SWEPOCH" if _has_param(par, "SWEPOCH") else "PEPOCH"
                    swp_name = "SWP" if swm == 1 else None

                    delay_components.append(
                        SolarWindDispersion(
                            ne_sw_param_names=tuple(ne_sw_names),
                            swepoch_name=swepoch_name,
                            swm=swm,
                            swp_name=swp_name,
                            raj_name=_astro_raj,
                            decj_name=_astro_decj,
                            pmra_name=_astro_pmra,
                            pmdec_name=_astro_pmdec,
                            posepoch_name=_astro_posepoch,
                            obliquity_arcsec=_astro_obliquity_arcsec,
                        )
                    )

            # ---- SolarWindDispersionX ----
            case Component.SOLAR_WIND_DISPERSION_X:
                swx_indices = _collect_prefix_indices(par, "SWXDM_")
                if swx_indices:
                    theta0_str = par.metadata.get("_SWX_THETA0_RAD")
                    if theta0_str is not None:
                        theta0_rad = float(theta0_str)
                    else:
                        theta0_rad = 0.0
                        log.warning(
                            "SolarWindDispersionX theta0 not available — using 0.0"
                        )

                    delay_components.append(
                        SolarWindDispersionX(
                            n_bins=len(swx_indices),
                            swxdm_names=tuple(f"SWXDM_{i:04d}" for i in swx_indices),
                            swxp_names=tuple(f"SWXP_{i:04d}" for i in swx_indices),
                            swxr1_names=tuple(f"SWXR1_{i:04d}" for i in swx_indices),
                            swxr2_names=tuple(f"SWXR2_{i:04d}" for i in swx_indices),
                            theta0=theta0_rad,
                            raj_name=_astro_raj,
                            decj_name=_astro_decj,
                            pmra_name=_astro_pmra,
                            pmdec_name=_astro_pmdec,
                            posepoch_name=_astro_posepoch,
                            obliquity_arcsec=_astro_obliquity_arcsec,
                        )
                    )

            # ---- DispersionDM ----
            case Component.DISPERSION_DM:
                dm_names = ["DM"]
                for pname in par.params.names:
                    if pname.startswith("DM") and pname != "DM" and pname != "DMEPOCH":
                        suffix = pname[2:]
                        if suffix.isdigit():
                            dm_names.append(pname)
                dm_names.sort(key=lambda n: int(n[2:]) if n != "DM" else 0)

                dmepoch_name = "DMEPOCH" if _has_param(par, "DMEPOCH") else "PEPOCH"

                delay_components.append(
                    DispersionDM(
                        dm_param_names=tuple(dm_names),
                        dmepoch_name=dmepoch_name,
                    )
                )

            # ---- DispersionDMX ----
            case Component.DISPERSION_DMX:
                dmx_indices = _collect_prefix_indices(par, "DMX_")
                if dmx_indices:
                    delay_components.append(
                        DispersionDMX(
                            n_bins=len(dmx_indices),
                            dmx_names=tuple(f"DMX_{i:04d}" for i in dmx_indices),
                            dmxr1_names=tuple(f"DMXR1_{i:04d}" for i in dmx_indices),
                            dmxr2_names=tuple(f"DMXR2_{i:04d}" for i in dmx_indices),
                        )
                    )

            # ---- DispersionJump ----
            case Component.DISPERSION_JUMP:
                dmjump_names = tuple(
                    n for n in par.params.names if n.startswith("DMJUMP")
                )
                if dmjump_names:
                    delay_components.append(DispersionJump(dmjump_names=dmjump_names))

            # ---- Binary ----
            case Component.BINARY | Component.BINARY_BT_PIECEWISE:
                _astro = {
                    "raj_name": _astro_raj,
                    "decj_name": _astro_decj,
                    "pmra_name": _astro_pmra,
                    "pmdec_name": _astro_pmdec,
                    "posepoch_name": _astro_posepoch,
                }
                delay_components.append(_build_binary(par, astro_info=_astro))

            # ---- FrequencyDependent ----
            case Component.FREQUENCY_DEPENDENT:
                fd_names = sorted(
                    (n for n in par.params.names if re.match(r"^FD\d+$", n)),
                    key=lambda n: int(n[2:]),
                )
                if fd_names:
                    delay_components.append(
                        FrequencyDependent(fd_param_names=tuple(fd_names))
                    )

            # ---- FDJump ----
            case Component.FD_JUMP:
                fdjump_names = []
                fdjump_indices = []
                for pname in par.params.names:
                    m = re.match(r"FD(\d+)JUMP\d+", pname)
                    if m:
                        fdjump_names.append(pname)
                        fdjump_indices.append(int(m.group(1)))
                use_log = par.bool_params.get("FDJUMPLOG", True)
                if fdjump_names:
                    delay_components.append(
                        FDJump(
                            fdjump_param_names=tuple(fdjump_names),
                            fdjump_fd_indices=tuple(fdjump_indices),
                            use_log=use_log,
                        )
                    )

            # ---- ChromaticCM ----
            case Component.CHROMATIC_CM:
                cm_names = ["CM"]
                for pname in par.params.names:
                    if pname.startswith("CM") and pname != "CM" and pname != "CMEPOCH":
                        suffix = pname[2:]
                        if suffix.isdigit():
                            cm_names.append(pname)
                cm_names.sort(key=lambda n: int(n[2:]) if n != "CM" else 0)

                cmepoch_name = "CMEPOCH" if _has_param(par, "CMEPOCH") else "PEPOCH"
                delay_components.append(
                    ChromaticCM(
                        cm_param_names=tuple(cm_names),
                        cmepoch_name=cmepoch_name,
                        tnchromidx_name="TNCHROMIDX",
                    )
                )

            # ---- ChromaticCMX ----
            case Component.CHROMATIC_CMX:
                cmx_indices = _collect_prefix_indices(par, "CMX_")
                if cmx_indices:
                    delay_components.append(
                        ChromaticCMX(
                            n_bins=len(cmx_indices),
                            cmx_names=tuple(f"CMX_{i:04d}" for i in cmx_indices),
                            cmxr1_names=tuple(f"CMXR1_{i:04d}" for i in cmx_indices),
                            cmxr2_names=tuple(f"CMXR2_{i:04d}" for i in cmx_indices),
                            tnchromidx_name="TNCHROMIDX",
                        )
                    )

            # ---- ExponentialDip ----
            case Component.EXPONENTIAL_DIP:
                dip_indices = _collect_prefix_indices(par, "EXPDIPEPOCH_")
                if not dip_indices:
                    dip_indices = _collect_prefix_indices(par, "EXPDIPEP_")
                if dip_indices:
                    delay_components.append(
                        ExponentialDip(
                            n_dips=len(dip_indices),
                            expdipeps_name="EXPDIPEPS",
                            expdipfref_name="EXPDIPFREF",
                            expdipep_names=tuple(f"EXPDIPEP_{i}" for i in dip_indices),
                            expdipamp_names=tuple(
                                f"EXPDIPAMP_{i}" for i in dip_indices
                            ),
                            expdipidx_names=tuple(
                                f"EXPDIPIDX_{i}" for i in dip_indices
                            ),
                            expdiptau_names=tuple(
                                f"EXPDIPTAU_{i}" for i in dip_indices
                            ),
                        )
                    )

            # ---- WaveX ----
            case Component.WAVE_X:
                wx_indices = _collect_prefix_indices(par, "WXFREQ_")
                if wx_indices:
                    wxepoch_name = "WXEPOCH" if _has_param(par, "WXEPOCH") else "PEPOCH"
                    delay_components.append(
                        WaveX(
                            n_components=len(wx_indices),
                            wxepoch_name=wxepoch_name,
                            wxfreq_names=tuple(f"WXFREQ_{i:04d}" for i in wx_indices),
                            wxsin_names=tuple(f"WXSIN_{i:04d}" for i in wx_indices),
                            wxcos_names=tuple(f"WXCOS_{i:04d}" for i in wx_indices),
                        )
                    )

            # ---- DMWaveX ----
            case Component.DM_WAVE_X:
                dmwx_indices = _collect_prefix_indices(par, "DMWXFREQ_")
                if dmwx_indices:
                    dmwxepoch_name = (
                        "DMWXEPOCH" if _has_param(par, "DMWXEPOCH") else "PEPOCH"
                    )
                    delay_components.append(
                        DMWaveX(
                            n_components=len(dmwx_indices),
                            dmwxepoch_name=dmwxepoch_name,
                            dmwxfreq_names=tuple(
                                f"DMWXFREQ_{i:04d}" for i in dmwx_indices
                            ),
                            dmwxsin_names=tuple(
                                f"DMWXSIN_{i:04d}" for i in dmwx_indices
                            ),
                            dmwxcos_names=tuple(
                                f"DMWXCOS_{i:04d}" for i in dmwx_indices
                            ),
                        )
                    )

            # ---- CMWaveX ----
            case Component.CM_WAVE_X:
                cmwx_indices = _collect_prefix_indices(par, "CMWXFREQ_")
                if cmwx_indices:
                    cmwxepoch_name = (
                        "CMWXEPOCH" if _has_param(par, "CMWXEPOCH") else "PEPOCH"
                    )
                    delay_components.append(
                        CMWaveX(
                            n_components=len(cmwx_indices),
                            cmwxepoch_name=cmwxepoch_name,
                            cmwxfreq_names=tuple(
                                f"CMWXFREQ_{i:04d}" for i in cmwx_indices
                            ),
                            cmwxsin_names=tuple(
                                f"CMWXSIN_{i:04d}" for i in cmwx_indices
                            ),
                            cmwxcos_names=tuple(
                                f"CMWXCOS_{i:04d}" for i in cmwx_indices
                            ),
                            tnchromidx_name="TNCHROMIDX",
                        )
                    )

            # ---- Spindown ----
            case Component.SPINDOWN:
                spin_names = ["F0"]
                for pname in par.params.names:
                    if pname.startswith("F") and pname != "F0" and pname[1:].isdigit():
                        spin_names.append(pname)
                spin_names.sort(key=lambda n: int(n[1:]))
                phase_components.append(Spindown(spin_param_names=tuple(spin_names)))

            # ---- Glitch ----
            case Component.GLITCH:
                glep_indices = _collect_prefix_indices(par, "GLEP_")
                if glep_indices:
                    phase_components.append(
                        Glitch(
                            n_glitches=len(glep_indices),
                            glep_names=tuple(f"GLEP_{i}" for i in glep_indices),
                            glph_names=tuple(f"GLPH_{i}" for i in glep_indices),
                            glf0_names=tuple(f"GLF0_{i}" for i in glep_indices),
                            glf1_names=tuple(f"GLF1_{i}" for i in glep_indices),
                            glf2_names=tuple(f"GLF2_{i}" for i in glep_indices),
                            glf0d_names=tuple(f"GLF0D_{i}" for i in glep_indices),
                            gltd_names=tuple(f"GLTD_{i}" for i in glep_indices),
                        )
                    )

            # ---- PiecewiseSpindown ----
            case Component.PIECEWISE_SPINDOWN:
                pw_indices = _collect_prefix_indices(par, "PWEP_")
                if pw_indices:
                    phase_components.append(
                        PiecewiseSpindown(
                            n_pieces=len(pw_indices),
                            pwstart_names=tuple(f"PWSTART_{i}" for i in pw_indices),
                            pwstop_names=tuple(f"PWSTOP_{i}" for i in pw_indices),
                            pwep_names=tuple(f"PWEP_{i}" for i in pw_indices),
                            pwph_names=tuple(f"PWPH_{i}" for i in pw_indices),
                            pwf0_names=tuple(f"PWF0_{i}" for i in pw_indices),
                            pwf1_names=tuple(f"PWF1_{i}" for i in pw_indices),
                            pwf2_names=tuple(f"PWF2_{i}" for i in pw_indices),
                        )
                    )

            # ---- PhaseJump ----
            case Component.PHASE_JUMP:
                jump_names = tuple(n for n in par.params.names if n.startswith("JUMP"))
                if jump_names:
                    phase_components.append(PhaseJump(jump_param_names=jump_names))

            # ---- Wave ----
            case Component.WAVE:
                wave_a_names = sorted(
                    (n for n in par.params.names if re.match(r"^WAVE\d+_A$", n)),
                    key=lambda n: int(_match_group1(r"WAVE(\d+)_A", n)),
                )
                if wave_a_names:
                    wave_indices = [
                        int(_match_group1(r"WAVE(\d+)_A", n)) for n in wave_a_names
                    ]
                    waveepoch_name = (
                        "WAVEEPOCH" if _has_param(par, "WAVEEPOCH") else "PEPOCH"
                    )
                    phase_components.append(
                        Wave(
                            n_terms=len(wave_indices),
                            waveepoch_name=waveepoch_name,
                            wave_om_name="WAVE_OM",
                            wave_sin_names=tuple(f"WAVE{i}_A" for i in wave_indices),
                            wave_cos_names=tuple(f"WAVE{i}_B" for i in wave_indices),
                        )
                    )

            # ---- IFunc ----
            case Component.IFUNC:
                ifunc_a_names = sorted(
                    (n for n in par.params.names if re.match(r"^IFUNC\d+_A$", n)),
                    key=lambda n: int(_match_group1(r"IFUNC(\d+)_A", n)),
                )
                if ifunc_a_names:
                    interp_type = par.int_params.get("SIFUNC", 0)
                    mjds = []
                    delays = []
                    for a_name in ifunc_a_names:
                        b_name = a_name.replace("_A", "_B")
                        mjd_val = float(
                            par.params.values[par.params._name_to_index[a_name]]
                        )
                        delay_val = float(
                            par.params.values[par.params._name_to_index[b_name]]
                        )
                        mjds.append(mjd_val)
                        delays.append(delay_val)

                    sorted_pairs = sorted(zip(mjds, delays))
                    sorted_mjds, sorted_delays = zip(*sorted_pairs)
                    phase_components.append(
                        IFunc(
                            interp_type=interp_type,
                            control_mjds=tuple(float(x) for x in sorted_mjds),
                            control_delays=tuple(float(x) for x in sorted_delays),
                        )
                    )

            # ---- Noise: ScaleToaError ----
            case Component.SCALE_TOA_ERROR:
                efac_names = tuple(
                    sorted(n for n in par.params.names if n.startswith("EFAC"))
                )
                equad_names = tuple(
                    sorted(n for n in par.params.names if n.startswith("EQUAD"))
                )
                white_noise = ScaleToaError(
                    efac_names=efac_names,
                    equad_names=equad_names,
                )

            # ---- Noise: ScaleDmError ----
            case Component.SCALE_DM_ERROR:
                dmefac_names = tuple(
                    sorted(n for n in par.params.names if n.startswith("DMEFAC"))
                )
                dmequad_names = tuple(
                    sorted(n for n in par.params.names if n.startswith("DMEQUAD"))
                )
                dm_white_noise = JaxScaleDmError(
                    dmefac_names=dmefac_names,
                    dmequad_names=dmequad_names,
                )

            # ---- Noise: EcorrNoise ----
            case Component.ECORR_NOISE:
                ecorr_names = tuple(
                    sorted(n for n in par.params.names if n.startswith("ECORR"))
                )
                if toa_data is not None and len(ecorr_names) > 0:
                    tdb_s = (
                        np.asarray(toa_data.tdb_int) * 86400.0
                        + np.asarray(toa_data.tdb_frac) * 86400.0
                    )
                    ecorr_masks = {}
                    for ename in ecorr_names:
                        if (
                            hasattr(toa_data, "flag_masks")
                            and toa_data.flag_masks is not None
                        ):
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
                    ecorr_noise = EcorrNoise(
                        ecorr_names=ecorr_names,
                        quantization_matrix=jnp.asarray(U),
                        ecorr_epoch_slices=ecorr_epoch_slices,
                    )
                elif toa_data is None and len(ecorr_names) > 0:
                    log.warning(
                        "EcorrNoise found but no toa_data provided"
                        " — ECORR not available"
                    )

            # ---- Noise: PLRedNoise ----
            case Component.PL_RED_NOISE:
                if toa_data is not None:
                    tdb_s = (
                        np.asarray(toa_data.tdb_int) * 86400.0
                        + np.asarray(toa_data.tdb_frac) * 86400.0
                    )
                    n_freqs = par.int_params.get("TNREDC", 30)
                    T = float(np.max(tdb_s) - np.min(tdb_s))
                    if _has_param(par, "TNREDTSPAN"):
                        tspan_days = float(
                            par.params.values[par.params._name_to_index["TNREDTSPAN"]]
                        )
                        T = tspan_days * 86400.0

                    from jaxpint.utils import build_fourier_basis

                    F, freqs, freq_bin_widths = build_fourier_basis(tdb_s, n_freqs, T)
                    plred_noise = PLRedNoise(
                        fourier_basis=jnp.asarray(F),
                        freqs=jnp.asarray(freqs),
                        freq_bin_widths=jnp.asarray(freq_bin_widths),
                        tnredamp_name="TNREDAMP",
                        tnredgam_name="TNREDGAM",
                    )

            # ---- Noise: PLDMNoise ----
            case Component.PL_DM_NOISE:
                if toa_data is not None:
                    tdb_s = (
                        np.asarray(toa_data.tdb_int) * 86400.0
                        + np.asarray(toa_data.tdb_frac) * 86400.0
                    )
                    n_freqs = par.int_params.get("TNDMC", 30)
                    T = float(np.max(tdb_s) - np.min(tdb_s))
                    if _has_param(par, "TNDMTSPAN"):
                        tspan_days = float(
                            par.params.values[par.params._name_to_index["TNDMTSPAN"]]
                        )
                        T = tspan_days * 86400.0

                    from jaxpint.utils import build_fourier_basis

                    F, freqs, freq_bin_widths = build_fourier_basis(tdb_s, n_freqs, T)

                    bary_freqs_mhz = np.asarray(toa_data.freq)
                    D = (1400.0 / bary_freqs_mhz) ** 2
                    F_dm = F * D[:, None]

                    pldm_noise = PLDMNoise(
                        fourier_basis=jnp.asarray(F_dm),
                        freqs=jnp.asarray(freqs),
                        freq_bin_widths=jnp.asarray(freq_bin_widths),
                        tndmamp_name="TNDMAMP",
                        tndmgam_name="TNDMGAM",
                    )

            # ---- Noise: PLChromNoise ----
            case Component.PL_CHROM_NOISE:
                if toa_data is not None:
                    tdb_s = (
                        np.asarray(toa_data.tdb_int) * 86400.0
                        + np.asarray(toa_data.tdb_frac) * 86400.0
                    )
                    n_freqs = par.int_params.get("TNCHROMC", 30)
                    T = float(np.max(tdb_s) - np.min(tdb_s))
                    if _has_param(par, "TNCHROMTSPAN"):
                        tspan_days = float(
                            par.params.values[par.params._name_to_index["TNCHROMTSPAN"]]
                        )
                        T = tspan_days * 86400.0

                    from jaxpint.utils import build_fourier_basis

                    F, freqs, freq_bin_widths = build_fourier_basis(tdb_s, n_freqs, T)

                    plchrom_noise = PLChromNoise(
                        fourier_basis=jnp.asarray(F),
                        freqs=jnp.asarray(freqs),
                        freq_bin_widths=jnp.asarray(freq_bin_widths),
                        tnchromamp_name="TNCHROMAMP",
                        tnchromgam_name="TNCHROMGAM",
                        tnchromidx_name="TNCHROMIDX",
                        fref=1400.0,
                    )

            # ---- Noise: PLSWNoise ----
            case Component.PL_SW_NOISE:
                if toa_data is not None:
                    tdb_s = (
                        np.asarray(toa_data.tdb_int) * 86400.0
                        + np.asarray(toa_data.tdb_frac) * 86400.0
                    )
                    n_freqs = par.int_params.get("TNSWC", 100)
                    T = float(np.max(tdb_s) - np.min(tdb_s))

                    from jaxpint.utils import build_fourier_basis

                    F, freqs, freq_bin_widths = build_fourier_basis(tdb_s, n_freqs, T)

                    swm = par.int_params.get("SWM", 0)
                    swp_name = "SWP" if swm == 1 else None

                    plsw_noise = PLSWNoise(
                        fourier_basis=jnp.asarray(F),
                        freqs=jnp.asarray(freqs),
                        freq_bin_widths=jnp.asarray(freq_bin_widths),
                        tnswamp_name="TNSWAMP",
                        tnswgam_name="TNSWGAM",
                        swm=swm,
                        swp_name=swp_name,
                        raj_name=_astro_raj,
                        decj_name=_astro_decj,
                        pmra_name=_astro_pmra,
                        pmdec_name=_astro_pmdec,
                        posepoch_name=_astro_posepoch,
                        obliquity_arcsec=_astro_obliquity_arcsec,
                    )

            # ---- Unimplemented ----
            case _:
                raise NotImplementedError(
                    f"Component {comp!r} is present in the par file "
                    f"but is not yet implemented in JaxPINT"
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

    correlated = []
    if ecorr_noise is not None:
        correlated.append(ecorr_noise)
    if plred_noise is not None:
        correlated.append(plred_noise)
    if pldm_noise is not None:
        correlated.append(pldm_noise)
    if plchrom_noise is not None:
        correlated.append(plchrom_noise)
    if plsw_noise is not None:
        correlated.append(plsw_noise)

    combined_noise = NoiseModel(
        white_noise=white_noise,
        correlated=tuple(correlated),
        dm_white_noise=dm_white_noise,
    )

    return timing_model, combined_noise
