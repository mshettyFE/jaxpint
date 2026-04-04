"""Component builder: PINT components → JaxPINT timing model components.

Inspects a PINT model's component list and creates the corresponding
JaxPINT delay, phase, and noise components.
"""

from __future__ import annotations

import logging
from typing import Optional

import jax.numpy as jnp
import numpy as np
from pint.models.timing_model import TimingModel as PINTTimingModel
from pint.toa import TOAs

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _param_is_set(pint_model, name):
    """Check if a PINT parameter is set (non-None, non-zero)."""
    if not hasattr(pint_model, name):
        return False
    p = getattr(pint_model, name)
    return p.value is not None and p.value != 0.0


def _opt_name(pint_model, name):
    """Return parameter name if set, else None."""
    return name if _param_is_set(pint_model, name) else None


def _dd_common_kwargs(pint_model):
    """Shared keyword arguments for DD-family binary models."""
    return dict(
        pb_name="PB", t0_name="T0", a1_name="A1",
        ecc_name="ECC", om_name="OM",
        pbdot_name=_opt_name(pint_model, "PBDOT"),
        omdot_name=_opt_name(pint_model, "OMDOT"),
        edot_name=_opt_name(pint_model, "EDOT"),
        a1dot_name=_opt_name(pint_model, "A1DOT"),
        xpbdot_name=_opt_name(pint_model, "XPBDOT"),
        gamma_name=_opt_name(pint_model, "GAMMA"),
        dr_name=_opt_name(pint_model, "DR"),
        dth_name=_opt_name(pint_model, "DTH"),
        a0_name=_opt_name(pint_model, "A0"),
        b0_name=_opt_name(pint_model, "B0"),
    )


def _ell1_common_kwargs(pint_model):
    """Shared keyword arguments for ELL1-family binary models."""
    return dict(
        pb_name="PB", tasc_name="TASC", a1_name="A1",
        eps1_name="EPS1", eps2_name="EPS2",
        pbdot_name=_opt_name(pint_model, "PBDOT"),
        a1dot_name=_opt_name(pint_model, "A1DOT"),
        xpbdot_name=_opt_name(pint_model, "XPBDOT"),
    )


def _build_binary_component(comp, pint_model, astro_info=None):
    """Construct the appropriate JaxPINT binary DelayComponent from a PINT binary component."""
    from jaxpint.binary.bt import BinaryBT
    from jaxpint.binary.bt_piecewise import BinaryBTPiecewise
    from jaxpint.binary.dd import BinaryDD
    from jaxpint.binary.ddk import BinaryDDK
    from jaxpint.binary.ddgr import BinaryDDGR
    from jaxpint.binary.ell1 import BinaryELL1

    bname = comp.binary_model_name
    if astro_info is None:
        astro_info = {}

    if bname == "BT":
        return BinaryBT(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            omdot_name=_opt_name(pint_model, "OMDOT"),
            edot_name=_opt_name(pint_model, "EDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            gamma_name=_opt_name(pint_model, "GAMMA"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
        )

    elif bname == "DD":
        return BinaryDD(
            **_dd_common_kwargs(pint_model),
            m2_name=_opt_name(pint_model, "M2"),
            sini_name=_opt_name(pint_model, "SINI"),
            shapiro_mode="standard",
        )

    elif bname == "DDS":
        return BinaryDD(
            **_dd_common_kwargs(pint_model),
            m2_name=_opt_name(pint_model, "M2"),
            shapmax_name="SHAPMAX",
            shapiro_mode="shapmax",
        )

    elif bname == "DDH":
        return BinaryDD(
            **_dd_common_kwargs(pint_model),
            h3_name="H3",
            stigma_name="STIGMA",
            shapiro_mode="h3stigma",
        )

    elif bname == "ELL1":
        return BinaryELL1(
            **_ell1_common_kwargs(pint_model),
            eps1dot_name=_opt_name(pint_model, "EPS1DOT"),
            eps2dot_name=_opt_name(pint_model, "EPS2DOT"),
            m2_name=_opt_name(pint_model, "M2"),
            sini_name=_opt_name(pint_model, "SINI"),
            shapiro_mode="standard" if _param_is_set(pint_model, "M2") else "none",
        )

    elif bname == "ELL1H":
        if _param_is_set(pint_model, "STIGMA"):
            shapiro_mode = "h3stigma"
        elif _param_is_set(pint_model, "H4"):
            shapiro_mode = "h3h4"
        else:
            shapiro_mode = "h3stigma"
        return BinaryELL1(
            **_ell1_common_kwargs(pint_model),
            eps1dot_name=_opt_name(pint_model, "EPS1DOT"),
            eps2dot_name=_opt_name(pint_model, "EPS2DOT"),
            h3_name="H3",
            stigma_name=_opt_name(pint_model, "STIGMA"),
            h4_name=_opt_name(pint_model, "H4"),
            shapiro_mode=shapiro_mode,
        )

    elif bname == "ELL1k":
        return BinaryELL1(
            **_ell1_common_kwargs(pint_model),
            omdot_name=_opt_name(pint_model, "OMDOT"),
            lnedot_name=_opt_name(pint_model, "LNEDOT"),
            m2_name=_opt_name(pint_model, "M2"),
            sini_name=_opt_name(pint_model, "SINI"),
            shapiro_mode="standard" if _param_is_set(pint_model, "M2") else "none",
        )

    elif bname == "DDK":
        k96 = bool(getattr(pint_model, "K96", None) and pint_model.K96.value)
        return BinaryDDK(
            **_dd_common_kwargs(pint_model),
            m2_name=_opt_name(pint_model, "M2"),
            kin_name="KIN", kom_name="KOM",
            px_name="PX",
            raj_name=astro_info.get("raj_name", "RAJ"),
            decj_name=astro_info.get("decj_name", "DECJ"),
            pmra_name=astro_info.get("pmra_name"),
            pmdec_name=astro_info.get("pmdec_name"),
            posepoch_name=astro_info.get("posepoch_name"),
            k96=k96,
        )

    elif bname == "DDGR":
        return BinaryDDGR(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            mtot_name="MTOT", m2_name="M2",
            edot_name=_opt_name(pint_model, "EDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            xomdot_name=_opt_name(pint_model, "XOMDOT"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
            a0_name=_opt_name(pint_model, "A0"),
            b0_name=_opt_name(pint_model, "B0"),
        )

    elif bname == "BT_piecewise":
        # Scan for piecewise parameters (T0X_*, A1X_*, XR1_*, XR2_*)
        t0x_names = []
        a1x_names = []
        xr1_names = []
        xr2_names = []

        # Collect piece indices from the PINT component
        piece_indices = set()
        for pname in comp.params:
            if pname.startswith("T0X_") or pname.startswith("A1X_"):
                piece_indices.add(pname[4:])
            elif pname.startswith("XR1_") or pname.startswith("XR2_"):
                piece_indices.add(pname[4:])

        for idx in sorted(piece_indices):
            t0x_name = f"T0X_{idx}"
            a1x_name = f"A1X_{idx}"
            xr1_name = f"XR1_{idx}"
            xr2_name = f"XR2_{idx}"
            if hasattr(pint_model, t0x_name):
                t0x_names.append(t0x_name)
            if hasattr(pint_model, a1x_name):
                a1x_names.append(a1x_name)
            if hasattr(pint_model, xr1_name):
                xr1_names.append(xr1_name)
            if hasattr(pint_model, xr2_name):
                xr2_names.append(xr2_name)

        n_pieces = len(xr1_names)
        return BinaryBTPiecewise(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            omdot_name=_opt_name(pint_model, "OMDOT"),
            edot_name=_opt_name(pint_model, "EDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            gamma_name=_opt_name(pint_model, "GAMMA"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
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


def _build_quantization_matrix(
    tdb_times_s: np.ndarray,
    ecorr_masks: dict[str, np.ndarray],
    dt: float = 1.0,
    nmin: int = 2,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Build the ECORR quantization matrix (NumPy, not JIT-compatible).

    Groups TOAs within *dt* seconds into epochs and creates a binary
    matrix ``U`` mapping TOAs to epochs.  Only epochs with at least
    *nmin* TOAs are kept.  This mirrors PINT's
    ``create_ecorr_quantization_matrix``.

    Parameters
    ----------
    tdb_times_s : (n_toas,) float64
        TOA times in TDB seconds.
    ecorr_masks : dict[str, ndarray]
        Boolean masks keyed by ECORR parameter name.
    dt, nmin : float, int
        Epoch grouping threshold (seconds) and minimum TOAs per epoch.

    Returns
    -------
    U : (n_toas, n_total_epochs)
        Binary quantization matrix.
    epoch_slices : dict[str, (int, int)]
        Column-index range for each ECORR parameter.
    """
    n_toas = len(tdb_times_s)
    columns: list[np.ndarray] = []
    epoch_slices: dict[str, tuple[int, int]] = {}
    col_offset = 0

    for ecorr_name in sorted(ecorr_masks):
        mask = ecorr_masks[ecorr_name]
        subset_indices = np.where(mask)[0]
        if len(subset_indices) == 0:
            epoch_slices[ecorr_name] = (col_offset, col_offset)
            continue

        subset_times = tdb_times_s[subset_indices]
        isort = np.argsort(subset_times)
        sorted_times = subset_times[isort]
        sorted_indices = subset_indices[isort]

        # Group into epochs
        epochs: list[list[int]] = [[sorted_indices[0]]]
        ref_time = sorted_times[0]
        for j in range(1, len(sorted_times)):
            if sorted_times[j] - ref_time < dt:
                epochs[-1].append(sorted_indices[j])
            else:
                epochs.append([sorted_indices[j]])
                ref_time = sorted_times[j]

        # Keep only epochs with >= nmin TOAs
        epochs = [ep for ep in epochs if len(ep) >= nmin]

        start = col_offset
        for ep in epochs:
            col = np.zeros(n_toas, dtype=np.float64)
            col[ep] = 1.0
            columns.append(col)
        col_offset += len(epochs)
        epoch_slices[ecorr_name] = (start, col_offset)

    if columns:
        U = np.column_stack(columns)
    else:
        U = np.zeros((n_toas, 0), dtype=np.float64)

    return U, epoch_slices


def build_timing_model(
    pint_model: PINTTimingModel,
    toas: Optional[TOAs] = None,
):
    """Construct a JaxPINT :class:`~jaxpint.model.TimingModel` from a PINT model.

    Inspects the PINT model's component list and creates the corresponding
    JaxPINT delay, phase, and noise components.  Unrecognised components are
    logged as warnings and skipped.

    Parameters
    ----------
    pint_model : pint.models.TimingModel
        The PINT timing model to convert.
    toas : pint.toa.TOAs, optional
        If provided and the model has an ``EcorrNoise`` component, the
        quantization matrix is built and an :class:`EcorrNoise` instance
        is returned.

    Returns
    -------
    (TimingModel, NoiseModel)
        The timing model and a :class:`NoiseModel` that aggregates all
        noise sources (white noise and correlated components).
    """
    from pint.models.spindown import Spindown as PINTSpindown
    from pint.models.dispersion_model import DispersionDM as PINTDispersionDM
    from pint.models.dispersion_model import DispersionDMX as PINTDispersionDMX
    from pint.models.astrometry import AstrometryEquatorial as PINTAstrometryEquatorial
    from pint.models.astrometry import AstrometryEcliptic as PINTAstrometryEcliptic
    from pint.models.noise_model import ScaleToaError as PINTScaleToaError
    from pint.models.noise_model import EcorrNoise as PINTEcorrNoise
    from pint.models.noise_model import PLRedNoise as PINTPLRedNoise
    from pint.models.noise_model import PLDMNoise as PINTPLDMNoise
    from pint.models.noise_model import PLChromNoise as PINTPLChromNoise
    from pint.models.noise_model import PLSWNoise as PINTPLSWNoise
    from pint.models.pulsar_binary import PulsarBinary as PINTPulsarBinary
    from pint.models.solar_system_shapiro import SolarSystemShapiro as PINTSolarSystemShapiro
    from pint.models.solar_wind_dispersion import SolarWindDispersion as PINTSolarWindDispersion
    from pint.models.solar_wind_dispersion import SolarWindDispersionX as PINTSolarWindDispersionX
    from pint.models.troposphere_delay import TroposphereDelay as PINTTroposphereDelay
    from pint.models.jump import PhaseJump as PINTPhaseJump
    from pint.models.glitch import Glitch as PINTGlitch

    from jaxpint.model import TimingModel
    from jaxpint.phase.spin import Spindown
    from jaxpint.delay.dispersion_dm import DispersionDM
    from jaxpint.delay.dispersion_dmx import DispersionDMX
    from jaxpint.delay.astrometry import AstrometryEquatorial, AstrometryEcliptic
    from jaxpint.noise.ecorr import EcorrNoise
    from jaxpint.noise.white import ScaleToaError
    from jaxpint.noise.noise_model import NoiseModel
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

    delay_components = []
    phase_components = []
    noise_model = None
    ecorr_noise = None
    plred_noise = None
    pldm_noise = None
    plchrom_noise = None
    plsw_noise = None

    # Cached astrometry param names (reused by Shapiro component).
    _astro_raj = "RAJ"
    _astro_decj = "DECJ"
    _astro_pmra = None
    _astro_pmdec = None
    _astro_posepoch = None
    _astro_obliquity_arcsec = None

    # Components that are handled implicitly (not mapped to JaxPINT components)
    _IMPLICIT = {"AbsPhase"}

    for name, comp in pint_model.components.items():
        if name in _IMPLICIT:
            continue

        if isinstance(comp, PINTSpindown):
            spin_names = tuple(comp.F_terms)
            phase_components.append(Spindown(spin_param_names=spin_names))

        elif isinstance(comp, PINTGlitch):
            comp.setup()
            glitch_indices = sorted(set(
                getattr(pint_model, y).index
                for y in comp.params
                if y.startswith("GLEP_")
            ))
            if glitch_indices:
                phase_components.append(Glitch(
                    n_glitches=len(glitch_indices),
                    glep_names=tuple(f"GLEP_{i}" for i in glitch_indices),
                    glph_names=tuple(f"GLPH_{i}" for i in glitch_indices),
                    glf0_names=tuple(f"GLF0_{i}" for i in glitch_indices),
                    glf1_names=tuple(f"GLF1_{i}" for i in glitch_indices),
                    glf2_names=tuple(f"GLF2_{i}" for i in glitch_indices),
                    glf0d_names=tuple(f"GLF0D_{i}" for i in glitch_indices),
                    gltd_names=tuple(f"GLTD_{i}" for i in glitch_indices),
                ))

        elif isinstance(comp, PINTAstrometryEquatorial):
            if hasattr(comp, "PMRA") and comp.PMRA.value is not None and comp.PMRA.value != 0.0:
                _astro_pmra = "PMRA"
            if hasattr(comp, "PMDEC") and comp.PMDEC.value is not None and comp.PMDEC.value != 0.0:
                _astro_pmdec = "PMDEC"

            px_name = None
            if hasattr(comp, "PX") and comp.PX.value is not None and comp.PX.value != 0.0:
                px_name = "PX"

            # POSEPOCH needed only when proper motion is active
            if _astro_pmra is not None or _astro_pmdec is not None:
                _astro_posepoch = "POSEPOCH"
                if comp.POSEPOCH.value is None:
                    _astro_posepoch = "PEPOCH"

            delay_components.append(
                AstrometryEquatorial(
                    raj_name=_astro_raj,
                    decj_name=_astro_decj,
                    pmra_name=_astro_pmra,
                    pmdec_name=_astro_pmdec,
                    px_name=px_name,
                    posepoch_name=_astro_posepoch,
                )
            )

        elif isinstance(comp, PINTAstrometryEcliptic):
            from jaxpint.constants import OBLIQUITY_ARCSEC

            _astro_raj = "ELONG"
            _astro_decj = "ELAT"

            if hasattr(comp, "PMELONG") and comp.PMELONG.value is not None and comp.PMELONG.value != 0.0:
                _astro_pmra = "PMELONG"
            if hasattr(comp, "PMELAT") and comp.PMELAT.value is not None and comp.PMELAT.value != 0.0:
                _astro_pmdec = "PMELAT"

            px_name = None
            if hasattr(comp, "PX") and comp.PX.value is not None and comp.PX.value != 0.0:
                px_name = "PX"

            # POSEPOCH needed only when proper motion is active
            if _astro_pmra is not None or _astro_pmdec is not None:
                _astro_posepoch = "POSEPOCH"
                if comp.POSEPOCH.value is None:
                    _astro_posepoch = "PEPOCH"

            # Resolve obliquity from ECL parameter
            ecl_name = comp.ECL.value if comp.ECL.value else "IERS2010"
            _astro_obliquity_arcsec = OBLIQUITY_ARCSEC[ecl_name]

            delay_components.append(
                AstrometryEcliptic(
                    elong_name=_astro_raj,
                    elat_name=_astro_decj,
                    pmelong_name=_astro_pmra,
                    pmelat_name=_astro_pmdec,
                    px_name=px_name,
                    posepoch_name=_astro_posepoch,
                    obliquity_arcsec=_astro_obliquity_arcsec,
                )
            )

        elif isinstance(comp, PINTDispersionDM):
            # Collect DM Taylor terms that are set (value not None)
            dm_names = ["DM"]
            for idx in sorted(pint_model.get_prefix_mapping("DM")):
                pname = pint_model.get_prefix_mapping("DM")[idx]
                param = getattr(pint_model, pname)
                if param.value is not None and param.value != 0.0:
                    dm_names.append(pname)

            # Determine epoch name: use DMEPOCH if set, else fall back to PEPOCH
            dmepoch_name = "DMEPOCH"
            if comp.DMEPOCH.value is None:
                dmepoch_name = "PEPOCH"

            delay_components.append(
                DispersionDM(
                    dm_param_names=tuple(dm_names),
                    dmepoch_name=dmepoch_name,
                )
            )

        elif isinstance(comp, PINTDispersionDMX):
            comp.setup()
            DMX_mapping = comp.get_prefix_mapping_component("DMX_")
            dmx_indices = sorted(DMX_mapping.keys())
            if dmx_indices:
                delay_components.append(DispersionDMX(
                    n_bins=len(dmx_indices),
                    dmx_names=tuple(f"DMX_{i:04d}" for i in dmx_indices),
                    dmxr1_names=tuple(f"DMXR1_{i:04d}" for i in dmx_indices),
                    dmxr2_names=tuple(f"DMXR2_{i:04d}" for i in dmx_indices),
                ))

        elif isinstance(comp, PINTPulsarBinary):
            _astro = {
                "raj_name": _astro_raj,
                "decj_name": _astro_decj,
                "pmra_name": _astro_pmra,
                "pmdec_name": _astro_pmdec,
                "posepoch_name": _astro_posepoch,
            }
            delay_components.append(_build_binary_component(comp, pint_model, astro_info=_astro))

        elif isinstance(comp, PINTSolarSystemShapiro):
            delay_components.append(
                SolarSystemShapiroDelay(
                    raj_name=_astro_raj,
                    decj_name=_astro_decj,
                    pmra_name=_astro_pmra,
                    pmdec_name=_astro_pmdec,
                    posepoch_name=_astro_posepoch,
                    planet_shapiro=bool(comp.PLANET_SHAPIRO.value),
                    obliquity_arcsec=_astro_obliquity_arcsec,
                )
            )

        elif isinstance(comp, PINTSolarWindDispersion):
            # Skip if NE_SW is zero and no derivatives are set (no effect).
            ne_sw_names = ["NE_SW"]
            for idx in sorted(pint_model.get_prefix_mapping("NE_SW")):
                pname = pint_model.get_prefix_mapping("NE_SW")[idx]
                param = getattr(pint_model, pname)
                if param.value is not None and param.value != 0.0:
                    ne_sw_names.append(pname)

            if len(ne_sw_names) == 1 and comp.NE_SW.value == 0.0:
                log.info(
                    "Skipping SolarWindDispersion — NE_SW is zero"
                )
                continue

            swm = comp.SWM.value
            swepoch_name = "SWEPOCH"
            if comp.SWEPOCH.value is None:
                swepoch_name = "PEPOCH"
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

        elif isinstance(comp, PINTSolarWindDispersionX):
            import astropy.units as u

            swxdm_mapping = comp.get_prefix_mapping_component("SWXDM_")
            swx_indices = sorted(swxdm_mapping.keys())

            if swx_indices:
                # Precompute theta0 (elongation at conjunction) via PINT.
                theta0_rad = float(comp.theta0.to(u.rad).value)

                n_bins = len(swx_indices)
                swxdm_names = tuple(f"SWXDM_{i:04d}" for i in swx_indices)
                swxp_names = tuple(f"SWXP_{i:04d}" for i in swx_indices)
                swxr1_names = tuple(f"SWXR1_{i:04d}" for i in swx_indices)
                swxr2_names = tuple(f"SWXR2_{i:04d}" for i in swx_indices)

                delay_components.append(
                    SolarWindDispersionX(
                        n_bins=n_bins,
                        swxdm_names=swxdm_names,
                        swxp_names=swxp_names,
                        swxr1_names=swxr1_names,
                        swxr2_names=swxr2_names,
                        theta0=theta0_rad,
                        raj_name=_astro_raj,
                        decj_name=_astro_decj,
                        pmra_name=_astro_pmra,
                        pmdec_name=_astro_pmdec,
                        posepoch_name=_astro_posepoch,
                        obliquity_arcsec=_astro_obliquity_arcsec,
                    )
                )

        elif isinstance(comp, PINTTroposphereDelay):
            if comp.CORRECT_TROPOSPHERE.value:
                delay_components.append(TroposphereDelay())

        elif isinstance(comp, PINTPhaseJump):
            comp.setup()
            jump_names = tuple(
                mask_par
                for mask_par in comp.get_params_of_type("maskParameter")
                if mask_par.startswith("JUMP")
            )
            if jump_names:
                phase_components.append(PhaseJump(jump_param_names=jump_names))

        elif isinstance(comp, PINTScaleToaError):
            # Extract EFAC and EQUAD parameter names from the PINT component
            comp.setup()
            efac_names = tuple(sorted(comp.EFACs.keys()))
            equad_names = tuple(sorted(comp.EQUADs.keys()))
            noise_model = ScaleToaError(
                efac_names=efac_names,
                equad_names=equad_names,
            )

        elif isinstance(comp, PINTEcorrNoise):
            comp.setup()
            ecorr_names_sorted = tuple(sorted(comp.ECORRs.keys()))
            if toas is not None and len(ecorr_names_sorted) > 0:
                # Need TDB times in seconds for epoch grouping
                if "tdbld" not in toas.table.colnames:
                    toas.compute_TDBs()
                tdb_ld = np.asarray(toas.table["tdbld"])
                tdb_s = np.float64(tdb_ld) * 86400.0

                # Collect masks (must match flag_masks built by pint_toas_to_jax)
                ecorr_masks = {}
                for ename in ecorr_names_sorted:
                    param = getattr(pint_model, ename)
                    idx = param.select_toa_mask(toas)
                    mask = np.zeros(toas.ntoas, dtype=bool)
                    if len(idx) > 0:
                        mask[idx] = True
                    ecorr_masks[ename] = mask

                U, eslices = _build_quantization_matrix(tdb_s, ecorr_masks)
                ecorr_epoch_slices = tuple(
                    eslices[n] for n in ecorr_names_sorted
                )
                ecorr_noise = EcorrNoise(
                    ecorr_names=ecorr_names_sorted,
                    quantization_matrix=jnp.asarray(U),
                    ecorr_epoch_slices=ecorr_epoch_slices,
                )
            elif toas is None and len(ecorr_names_sorted) > 0:
                log.warning(
                    "EcorrNoise component found but no TOAs provided to "
                    "build_timing_model — ECORR will not be available"
                )

        elif isinstance(comp, PINTPLRedNoise):
            if toas is not None:
                # Need TDB times in seconds for Fourier basis
                if "tdbld" not in toas.table.colnames:
                    toas.compute_TDBs()
                tdb_ld = np.asarray(toas.table["tdbld"])
                tdb_s = np.float64(tdb_ld) * 86400.0

                # Number of frequency modes
                n_freqs = (
                    int(comp.TNREDC.value)
                    if comp.TNREDC.value is not None
                    else 30
                )

                # Time span for fundamental frequency
                if comp.TNREDTSPAN.quantity is not None:
                    import astropy.units as u
                    T = float(comp.TNREDTSPAN.quantity.to(u.s).value)
                else:
                    T = float(np.max(tdb_s) - np.min(tdb_s))

                # Build Fourier design matrix
                from jaxpint.utils import build_fourier_basis
                F, freqs, freq_bin_widths = build_fourier_basis(tdb_s, n_freqs, T)

                plred_noise = PLRedNoise(
                    fourier_basis=jnp.asarray(F),
                    freqs=jnp.asarray(freqs),
                    freq_bin_widths=jnp.asarray(freq_bin_widths),
                    tnredamp_name="TNREDAMP",
                    tnredgam_name="TNREDGAM",
                )
            else:
                log.warning(
                    "PLRedNoise component found but no TOAs provided to "
                    "build_timing_model — red noise will not be available"
                )

        elif isinstance(comp, PINTPLDMNoise):
            if toas is not None:
                if "tdbld" not in toas.table.colnames:
                    toas.compute_TDBs()
                tdb_ld = np.asarray(toas.table["tdbld"])
                tdb_s = np.float64(tdb_ld) * 86400.0

                n_freqs = (
                    int(comp.TNDMC.value)
                    if comp.TNDMC.value is not None
                    else 30
                )

                if comp.TNDMTSPAN.quantity is not None:
                    import astropy.units as u
                    T = float(comp.TNDMTSPAN.quantity.to(u.s).value)
                else:
                    T = float(np.max(tdb_s) - np.min(tdb_s))

                from jaxpint.utils import build_fourier_basis
                F, freqs, freq_bin_widths = build_fourier_basis(tdb_s, n_freqs, T)

                # Pre-compute DM scaling: (1400 / f_obs)^2
                import astropy.units as u
                bary_freqs_mhz = pint_model.barycentric_radio_freq(toas).to(u.MHz).value
                D = (1400.0 / bary_freqs_mhz) ** 2
                F_dm = F * D[:, None]

                pldm_noise = PLDMNoise(
                    fourier_basis=jnp.asarray(F_dm),
                    freqs=jnp.asarray(freqs),
                    freq_bin_widths=jnp.asarray(freq_bin_widths),
                    tndmamp_name="TNDMAMP",
                    tndmgam_name="TNDMGAM",
                )
            else:
                log.warning(
                    "PLDMNoise component found but no TOAs provided to "
                    "build_timing_model — DM noise will not be available"
                )

        elif isinstance(comp, PINTPLChromNoise):
            if toas is not None:
                if "tdbld" not in toas.table.colnames:
                    toas.compute_TDBs()
                tdb_ld = np.asarray(toas.table["tdbld"])
                tdb_s = np.float64(tdb_ld) * 86400.0

                n_freqs = (
                    int(comp.TNCHROMC.value)
                    if comp.TNCHROMC.value is not None
                    else 30
                )

                if comp.TNCHROMTSPAN.quantity is not None:
                    import astropy.units as u
                    T = float(comp.TNCHROMTSPAN.quantity.to(u.s).value)
                else:
                    T = float(np.max(tdb_s) - np.min(tdb_s))

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
            else:
                log.warning(
                    "PLChromNoise component found but no TOAs provided to "
                    "build_timing_model — chromatic noise will not be available"
                )

        elif isinstance(comp, PINTPLSWNoise):
            if toas is not None:
                if "tdbld" not in toas.table.colnames:
                    toas.compute_TDBs()
                tdb_ld = np.asarray(toas.table["tdbld"])
                tdb_s = np.float64(tdb_ld) * 86400.0

                n_freqs = (
                    int(comp.TNSWC.value)
                    if comp.TNSWC.value is not None
                    else 100
                )

                T = float(np.max(tdb_s) - np.min(tdb_s))

                from jaxpint.utils import build_fourier_basis
                F, freqs, freq_bin_widths = build_fourier_basis(tdb_s, n_freqs, T)

                # Determine SWM from companion SolarWindDispersion component
                swm = 0
                swp_name = None
                if hasattr(pint_model, "SWM"):
                    swm = int(pint_model.SWM.value) if pint_model.SWM.value is not None else 0
                if swm == 1:
                    swp_name = "SWP"

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
            else:
                log.warning(
                    "PLSWNoise component found but no TOAs provided to "
                    "build_timing_model — solar wind noise will not be available"
                )

        else:
            log.warning(
                "Skipping PINT component %r (%s) — not yet ported to JaxPINT",
                name,
                type(comp).__name__,
            )

    timing_model = TimingModel(
        delay_components=tuple(delay_components),
        phase_components=tuple(phase_components),
    )

    correlated: list[NoiseComponent] = []
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
        white_noise=noise_model,
        correlated=tuple(correlated),
    )
    return timing_model, combined_noise
