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


def _build_binary_component(comp, pint_model):
    """Construct the appropriate JaxPINT binary DelayComponent from a PINT binary component."""
    from jaxpint.binary.bt import BinaryBT
    from jaxpint.binary.dd import BinaryDD, BinaryDDS, BinaryDDH
    from jaxpint.binary.ell1 import BinaryELL1, BinaryELL1H, BinaryELL1k

    bname = comp.binary_model_name

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
            m2_name=_opt_name(pint_model, "M2"),
            sini_name=_opt_name(pint_model, "SINI"),
            shapiro_mode="standard",
        )

    elif bname == "DDS":
        return BinaryDDS(
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
            m2_name=_opt_name(pint_model, "M2"),
            shapmax_name="SHAPMAX",
        )

    elif bname == "DDH":
        return BinaryDDH(
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
            h3_name="H3",
            stigma_name="STIGMA",
        )

    elif bname == "ELL1":
        return BinaryELL1(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            eps1dot_name=_opt_name(pint_model, "EPS1DOT"),
            eps2dot_name=_opt_name(pint_model, "EPS2DOT"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
            m2_name=_opt_name(pint_model, "M2"),
            sini_name=_opt_name(pint_model, "SINI"),
            shapiro_mode="standard" if _param_is_set(pint_model, "M2") else "none",
        )

    elif bname == "ELL1H":
        # Determine Shapiro mode: H3+STIGMA or H3+H4
        if _param_is_set(pint_model, "STIGMA"):
            shapiro_mode = "h3stigma"
        elif _param_is_set(pint_model, "H4"):
            shapiro_mode = "h3h4"
        else:
            shapiro_mode = "h3stigma"
        return BinaryELL1H(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            eps1dot_name=_opt_name(pint_model, "EPS1DOT"),
            eps2dot_name=_opt_name(pint_model, "EPS2DOT"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
            h3_name="H3",
            stigma_name=_opt_name(pint_model, "STIGMA"),
            h4_name=_opt_name(pint_model, "H4"),
            shapiro_mode=shapiro_mode,
        )

    elif bname == "ELL1k":
        return BinaryELL1k(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
            omdot_name=_opt_name(pint_model, "OMDOT"),
            lnedot_name=_opt_name(pint_model, "LNEDOT"),
            m2_name=_opt_name(pint_model, "M2"),
            sini_name=_opt_name(pint_model, "SINI"),
            shapiro_mode="standard" if _param_is_set(pint_model, "M2") else "none",
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
    (TimingModel, Optional[ScaleToaError], Optional[EcorrNoise])
        The timing model, optional white noise model, and optional ECORR
        noise model.
    """
    from pint.models.spindown import Spindown as PINTSpindown
    from pint.models.dispersion_model import DispersionDM as PINTDispersionDM
    from pint.models.astrometry import AstrometryEquatorial as PINTAstrometryEquatorial
    from pint.models.astrometry import AstrometryEcliptic as PINTAstrometryEcliptic
    from pint.models.noise_model import ScaleToaError as PINTScaleToaError
    from pint.models.noise_model import EcorrNoise as PINTEcorrNoise
    from pint.models.pulsar_binary import PulsarBinary as PINTPulsarBinary
    from pint.models.solar_system_shapiro import SolarSystemShapiro as PINTSolarSystemShapiro
    from pint.models.troposphere_delay import TroposphereDelay as PINTTroposphereDelay

    from jaxpint.model import TimingModel
    from jaxpint.spin import Spindown
    from jaxpint.dispersion_dm import DispersionDM
    from jaxpint.astrometry import AstrometryEquatorial, AstrometryEcliptic
    from jaxpint.noise import EcorrNoise, ScaleToaError
    from jaxpint.shapiro import SolarSystemShapiroDelay
    from jaxpint.troposphere import TroposphereDelay

    delay_components = []
    phase_components = []
    noise_model = None
    ecorr_noise = None

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

        elif isinstance(comp, PINTPulsarBinary):
            delay_components.append(_build_binary_component(comp, pint_model))

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

        elif isinstance(comp, PINTTroposphereDelay):
            if comp.CORRECT_TROPOSPHERE.value:
                delay_components.append(TroposphereDelay())

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
    return timing_model, noise_model, ecorr_noise
