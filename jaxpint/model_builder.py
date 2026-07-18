"""Model builder: ParResult → TimingModel + NoiseModel.

Constructs JaxPINT timing and noise components from a
:class:`~jaxpint.par.result.ParResult`.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from jaxpint.par.registry import Component
from jaxpint.par.registry_table import PRIORITY

import jax.numpy as jnp
import numpy as np

from jaxpint.par.result import ParResult
from jaxpint.types import TOAData
from jaxpint.utils import build_quantization_matrix as _build_quantization_matrix

# BuildContext and the parse-result helpers live in a neutral module so component
# ``build`` methods can reference them without importing this module (a cycle).
# Imported here (aliased to the private names the builders below use) because the
# model builder assembles the context and calls the helpers itself.
from jaxpint._build_context import (
    BuildContext,
    basis_seconds as _basis_seconds,
    opt_name as _opt_name,
    param_is_set as _param_is_set,
    span_seconds as _span_seconds,
    value as _value,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _epoch_or_pepoch(par: ParResult, name: str) -> str:
    """Epoch parameter *name* if present, else fall back to ``PEPOCH`` (PINT)."""
    return name if (name in par.params) else "PEPOCH"


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
        posepoch = _epoch_or_pepoch(par, "POSEPOCH")

    return raj, decj, pmra, pmdec, posepoch, obliquity_arcsec


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

    # An ecliptic-frame model always resolves obliquity in _resolve_astrometry.
    assert ctx.obliquity_arcsec is not None
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
    # NE_SW Taylor coefficients: base NE_SW (order 0) plus NE_SW{i} that are
    # actually set (nonzero). indexed_family gives them in numeric order, which
    # the Taylor sum requires (tuple position == derivative order).
    ne_sw_names = ["NE_SW"]
    for i in par.params.indexed_family("NE_SW"):
        name = f"NE_SW{i}"
        if _value(par, name) != 0.0:
            ne_sw_names.append(name)

    ne_sw_val = _value(par, "NE_SW") if ("NE_SW" in par.params) else 0.0
    if not (len(ne_sw_names) > 1 or ne_sw_val != 0.0):
        return None

    swm = par.int_params.get("SWM", 0)
    swepoch_name = _epoch_or_pepoch(par, "SWEPOCH")
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
    swx_indices = par.params.prefix_indices("SWXDM_")
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
    # Taylor coefficients: base DM (order 0) then DM1, DM2, ... in numeric order.
    dm_names = ["DM"] + [f"DM{i}" for i in par.params.indexed_family("DM")]

    dmepoch_name = _epoch_or_pepoch(par, "DMEPOCH")

    return DispersionDM(
        dm_param_names=tuple(dm_names),
        dmepoch_name=dmepoch_name,
    )


def _build_dispersion_jump(ctx: BuildContext):
    from jaxpint.delay.dispersion_jump import DispersionJump

    dmjump_names = ctx.par.params.names_with_prefix("DMJUMP")
    if not dmjump_names:
        return None
    return DispersionJump(dmjump_names=dmjump_names)


def _build_frequency_dependent(ctx: BuildContext):
    from jaxpint.delay.frequency_dependent import FrequencyDependent

    fd_indices = ctx.par.params.prefix_indices("FD")
    if not fd_indices:
        return None
    return FrequencyDependent(fd_param_names=tuple(f"FD{i}" for i in fd_indices))


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
    # Taylor coefficients: base CM (order 0) then CM1, CM2, ... in numeric order.
    cm_names = ["CM"] + [f"CM{i}" for i in par.params.indexed_family("CM")]

    cmepoch_name = _epoch_or_pepoch(par, "CMEPOCH")
    return ChromaticCM(
        cm_param_names=tuple(cm_names),
        cmepoch_name=cmepoch_name,
        tnchromidx_name="TNCHROMIDX",
    )


def _build_chromatic_cmx(ctx: BuildContext):
    from jaxpint.delay.chromatic_cmx import ChromaticCMX

    cmx_indices = ctx.par.params.prefix_indices("CMX_")
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

    dip_indices = ctx.par.params.prefix_indices("EXPDIPEPOCH_")
    if not dip_indices:
        dip_indices = ctx.par.params.prefix_indices("EXPDIPEP_")
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

    wx_indices = ctx.par.params.prefix_indices("WXFREQ_")
    if not wx_indices:
        return None
    wxepoch_name = _epoch_or_pepoch(ctx.par, "WXEPOCH")
    return WaveX(
        n_components=len(wx_indices),
        wxepoch_name=wxepoch_name,
        wxfreq_names=tuple(f"WXFREQ_{i:04d}" for i in wx_indices),
        wxsin_names=tuple(f"WXSIN_{i:04d}" for i in wx_indices),
        wxcos_names=tuple(f"WXCOS_{i:04d}" for i in wx_indices),
    )


def _build_dm_wave_x(ctx: BuildContext):
    from jaxpint.delay.dmwavex import DMWaveX

    dmwx_indices = ctx.par.params.prefix_indices("DMWXFREQ_")
    if not dmwx_indices:
        return None
    dmwxepoch_name = _epoch_or_pepoch(ctx.par, "DMWXEPOCH")
    return DMWaveX(
        n_components=len(dmwx_indices),
        dmwxepoch_name=dmwxepoch_name,
        dmwxfreq_names=tuple(f"DMWXFREQ_{i:04d}" for i in dmwx_indices),
        dmwxsin_names=tuple(f"DMWXSIN_{i:04d}" for i in dmwx_indices),
        dmwxcos_names=tuple(f"DMWXCOS_{i:04d}" for i in dmwx_indices),
    )


def _build_cm_wave_x(ctx: BuildContext):
    from jaxpint.delay.cmwavex import CMWaveX

    cmwx_indices = ctx.par.params.prefix_indices("CMWXFREQ_")
    if not cmwx_indices:
        return None
    cmwxepoch_name = _epoch_or_pepoch(ctx.par, "CMWXEPOCH")
    return CMWaveX(
        n_components=len(cmwx_indices),
        cmwxepoch_name=cmwxepoch_name,
        cmwxfreq_names=tuple(f"CMWXFREQ_{i:04d}" for i in cmwx_indices),
        cmwxsin_names=tuple(f"CMWXSIN_{i:04d}" for i in cmwx_indices),
        cmwxcos_names=tuple(f"CMWXCOS_{i:04d}" for i in cmwx_indices),
        tnchromidx_name="TNCHROMIDX",
    )


# ---- Phase components ----


def _build_glitch(ctx: BuildContext):
    from jaxpint.phase.glitch import Glitch

    glep_indices = ctx.par.params.prefix_indices("GLEP_")
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

    pw_indices = ctx.par.params.prefix_indices("PWEP_")
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

    jump_names = ctx.par.params.names_with_prefix("JUMP")
    if not jump_names:
        return None
    return PhaseJump(jump_param_names=jump_names)


def _build_wave(ctx: BuildContext):
    from jaxpint.phase.wave import Wave

    par = ctx.par
    wave_indices = par.params.indexed_family("WAVE", "_A")
    if not wave_indices:
        return None
    waveepoch_name = _epoch_or_pepoch(par, "WAVEEPOCH")
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
    ifunc_indices = par.params.indexed_family("IFUNC", "_A")
    if not ifunc_indices:
        return None
    interp_type = par.int_params.get("SIFUNC", 0)
    mjds = []
    delays = []
    for i in ifunc_indices:
        mjds.append(_value(par, f"IFUNC{i}_A"))
        delays.append(_value(par, f"IFUNC{i}_B"))

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
    efac_names = par.params.names_with_prefix("EFAC")
    equad_names = par.params.names_with_prefix("EQUAD")
    return ScaleToaError(efac_names=efac_names, equad_names=equad_names)


def _build_scale_dm_error(ctx: BuildContext):
    from jaxpint.noise.dm_white import ScaleDmError

    par = ctx.par
    dmefac_names = par.params.names_with_prefix("DMEFAC")
    dmequad_names = par.params.names_with_prefix("DMEQUAD")
    return ScaleDmError(dmefac_names=dmefac_names, dmequad_names=dmequad_names)


def _build_ecorr(ctx: BuildContext):
    from jaxpint.noise.ecorr import EcorrNoise

    par = ctx.par
    toa_data = ctx.toa_data
    ecorr_names = tuple(sorted(n for n in par.params.names if n.startswith("ECORR")))
    if toa_data is not None and len(ecorr_names) > 0:
        basis_s = _basis_seconds(toa_data)
        # Missing mask -> all-False (this ECORR group selects no TOAs); the
        # build-time _validate_flag_masks check flags genuinely-absent masks.
        ecorr_masks = {
            ename: np.asarray(toa_data.flag_mask(ename, default=False))
            for ename in ecorr_names
        }

        U, eslices = _build_quantization_matrix(basis_s, ecorr_masks)
        ecorr_epoch_slices = tuple(eslices[n] for n in ecorr_names)
        return EcorrNoise(
            ecorr_names=ecorr_names,
            quantization_matrix=jnp.asarray(U),
            ecorr_epoch_slices=ecorr_epoch_slices,
        )
    elif toa_data is None and len(ecorr_names) > 0:
        log.warning("EcorrNoise found but no toa_data provided — ECORR not available")
    return None


def _build_pl_dm_noise(ctx: BuildContext):
    from jaxpint.noise.dm_noise import PLDMNoise
    from jaxpint.utils import build_fourier_basis

    par = ctx.par
    toa_data = ctx.toa_data
    if toa_data is None:
        return None
    basis_s = _basis_seconds(toa_data)
    n_freqs = par.int_params.get("TNDMC", 30)
    T = _span_seconds(par, basis_s, "TNDMTSPAN")

    F, freqs, freq_bin_widths = build_fourier_basis(basis_s, n_freqs, T)

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
    basis_s = _basis_seconds(toa_data)
    n_freqs = par.int_params.get("TNCHROMC", 30)
    T = _span_seconds(par, basis_s, "TNCHROMTSPAN")

    F, freqs, freq_bin_widths = build_fourier_basis(basis_s, n_freqs, T)

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
    basis_s = _basis_seconds(toa_data)
    n_freqs = par.int_params.get("TNSWC", 100)
    T = _span_seconds(par, basis_s)

    F, freqs, freq_bin_widths = build_fourier_basis(basis_s, n_freqs, T)

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


# Component -> builder for the still-manual components.  Self-registered
# components (Spindown / DispersionDMX / PLRedNoise / the binary family) are
# merged in below from the registry.  A component absent from the merged table
# that is nonetheless active raises NotImplementedError in build_model.
_BUILDERS: dict[Component, Callable[[BuildContext], object]] = {
    Component.ASTROMETRY_EQUATORIAL: _build_astrometry_equatorial,
    Component.ASTROMETRY_ECLIPTIC: _build_astrometry_ecliptic,
    Component.TROPOSPHERE_DELAY: _build_troposphere,
    Component.SOLAR_SYSTEM_SHAPIRO: _build_shapiro,
    Component.SOLAR_WIND_DISPERSION: _build_solar_wind,
    Component.SOLAR_WIND_DISPERSION_X: _build_solar_wind_x,
    Component.DISPERSION_DM: _build_dispersion_dm,
    Component.DISPERSION_JUMP: _build_dispersion_jump,
    Component.FREQUENCY_DEPENDENT: _build_frequency_dependent,
    Component.FD_JUMP: _build_fd_jump,
    Component.CHROMATIC_CM: _build_chromatic_cm,
    Component.CHROMATIC_CMX: _build_chromatic_cmx,
    Component.EXPONENTIAL_DIP: _build_exponential_dip,
    Component.WAVE_X: _build_wave_x,
    Component.DM_WAVE_X: _build_dm_wave_x,
    Component.CM_WAVE_X: _build_cm_wave_x,
    Component.GLITCH: _build_glitch,
    Component.PIECEWISE_SPINDOWN: _build_piecewise_spindown,
    Component.PHASE_JUMP: _build_phase_jump,
    Component.WAVE: _build_wave,
    Component.IFUNC: _build_ifunc,
    Component.SCALE_TOA_ERROR: _build_scale_toa_error,
    Component.SCALE_DM_ERROR: _build_scale_dm_error,
    Component.ECORR_NOISE: _build_ecorr,
    Component.PL_DM_NOISE: _build_pl_dm_noise,
    Component.PL_CHROM_NOISE: _build_pl_chrom_noise,
    Component.PL_SW_NOISE: _build_pl_sw_noise,
}

# Self-registered components supply their builder here (Spindown, DispersionDMX,
# PLRedNoise, and the binary family — see jaxpint/binary/_build.py).
from jaxpint.par._component_registry import registered as _registered  # noqa: E402

_BUILDERS.update({rc.component: rc.build for rc in _registered().values()})


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _active_components(par: ParResult) -> set[Component]:
    """The components to build, in addition to those detected in ``par``.

    Components enabled only by special conditions (not membership in
    ``par.component_set`` alone) are added here: Shapiro rides with astrometry,
    troposphere with ``CORRECT_TROPOSPHERE``, and a binary with any
    ``BINARY`` line.
    """
    _auto_detect_only = {
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


def _validate_referenced_params(timing_model, noise_model, params) -> None:
    """Verify every parameter name a component references exists in ``params``.

    Components bind to parameters by name (static ``*_name`` / ``*_names`` fields)
    and resolve them lazily inside ``__call__`` via ``params.param_value(name)``.
    A name that is absent from the ``ParameterVector`` would otherwise only fail
    as a ``KeyError`` deep in a (possibly jitted) evaluation. This check runs once
    at build time and fails early with a message naming every offending component
    and parameter.

    Raises
    ------
    ValueError
        If any component (timing or noise) -- or the model's ``PHOFF`` binding --
        references a name not present in ``params``.
    """
    # `name in params` uses NamedVector.__contains__; epoch params (PEPOCH/T0/...)
    # are present in the vector too.
    missing: list[tuple[str, str]] = []  # (component label, parameter name)

    def _check(components, labels):
        for comp, label in zip(components, labels):
            for pname in comp.required_params():  # public API; skips unset Optionals
                if pname not in params:
                    missing.append((label, pname))

    _check(timing_model.components, timing_model.component_names)
    _check(noise_model.components, noise_model.component_names)

    # PHOFF is the one name-bearing field on TimingModel itself, not a sub-component.
    if timing_model.phoff_name is not None and timing_model.phoff_name not in params:
        missing.append(("TimingModel", timing_model.phoff_name))

    if missing:
        lines = "\n".join(
            f"  - {label} references unknown parameter {pname!r}"
            for label, pname in missing
        )
        raise ValueError(
            "build_model: component(s) reference parameter names not present in "
            f"the ParameterVector:\n{lines}\n"
            "This usually means a component's *_name field, its PARAMS schema, and "
            "the builder wiring disagree. Available parameters: "
            f"{tuple(params.names)}"
        )


def _validate_flag_masks(par: ParResult, toa_data: TOAData) -> None:
    """Verify the TOAData carries a flag mask for every masked parameter.

    The native loader builds ``toa_data.flag_masks`` with exactly the keys in
    ``par.mask_info``, so for native-loaded data this always holds. It can break
    when the TOAData comes from a different source than the par (the bridge path,
    or a hand-built TOAData): a masked parameter (EFAC/EQUAD/JUMP/...) would then
    either raise a ``KeyError`` deep in a jitted evaluation (the no-default
    ``flag_mask(name)`` consumers in white/dm_white/dispersion_jump) or silently
    contribute nothing (the ``default=False`` consumers). This check fails early
    with a clear message instead.

    Raises
    ------
    ValueError
        If any masked parameter declared in ``par`` lacks a mask in ``toa_data``.
    """
    missing = sorted(set(par.mask_info) - set(toa_data.flag_masks))
    if missing:
        raise ValueError(
            "build_model: TOAData is missing flag masks for masked parameter(s) "
            f"{missing} declared in the par. This usually means the TOAData was "
            "built from a different source than the par (e.g. the bridge path or "
            f"a hand-built TOAData). Masks present: {sorted(toa_data.flag_masks)}"
        )


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

    for comp in sorted(active, key=lambda c: (PRIORITY.get(c, len(PRIORITY)), c.value)):
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
        elif isinstance(obj, NoiseComponent):
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

    _validate_referenced_params(timing_model, combined_noise, par.params)
    if toa_data is not None:
        _validate_flag_masks(par, toa_data)
    return timing_model, combined_noise
