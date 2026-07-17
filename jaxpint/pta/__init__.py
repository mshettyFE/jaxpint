"""PTA likelihood module for JaxPINT.

Composes :func:`jaxpint.likelihood.single_pulsar_logL` across multiple
pulsars with shared signal injections (CW sources, GWB, etc.).  Optional
cross-pulsar correlations (e.g. Hellings-Downs GWB) are handled by the
same :func:`pta_logL` entry point via the
:class:`~jaxpint.pta.injectors.CorrelatedSignalInjector` interface.
"""

from jaxpint.types import GlobalParams
from jaxpint.pta.injectors import CorrelatedSignalInjector, SignalInjector
from jaxpint.pta.conditional import (
    ConditionalGP,
    DelayBand,
    conditional_covariance,
    conditional_gwb,
    conditional_gwb_delay_bands,
    conditional_gwb_delays,
    conditional_single_pulsar,
    sample_conditional,
)
from jaxpint.pta.likelihood import (
    GWBlocks,
    PTAConfig,
    per_pulsar_gw_blocks,
    precompute_single_pulsar_pta_factor,
    pta_logL,
    single_pulsar_pta_logL,
    single_pulsar_pta_logL_with_factor,
)
from jaxpint.pta.fisher import fisher_matrix, flatten_params, unflatten_params
from jaxpint.pta.scan import (
    GlobalScanAxis,
    PerPulsarScanAxis,
    ScanAxis,
    scan_logL,
)
from jaxpint.pta.signals import (
    CW_PARAM_DEFAULTS,
    CWInjector,
    CURN_PARAM_DEFAULTS,
    CURNInjector,
    BrokenPowerLawSpectrum,
    FreeSpectrum,
    HDCorrelatedGWBInjector,
    PowerLawSpectrum,
    SpectralModel,
    cw_delay,
    fplus_fcross,
    fourier_basis,
    gwb_covariance,
    hd_orf,
    monopole_orf,
    dipole_orf,
)
from jaxpint.pta.extraction import (
    EXTRACTION_ORIENTATIONS,
    bM2_coeffs,
    basis_quadratics,
    default_extraction_orientations,
    extract_pulsar_bM,
    extract_pulsar_blocks,
    orientation_coeffs,
    quadratic_coeffs,
)
from jaxpint.pta.cw_localization import (
    assemble_joint_fisher,
    credible_area_deg2,
    gram_at_pixel,
    gram_block_at_pair,
    h0_for_snr,
    make_logL_2sky,
    per_source_credible_areas_deg2,
)

__all__ = [
    # Core
    "GlobalParams",
    "PTAConfig",
    "SignalInjector",
    "CorrelatedSignalInjector",
    "pta_logL",
    "single_pulsar_pta_logL",
    "single_pulsar_pta_logL_with_factor",
    "precompute_single_pulsar_pta_factor",
    "per_pulsar_gw_blocks",
    "GWBlocks",
    # Conditional GP posteriors
    "ConditionalGP",
    "DelayBand",
    "conditional_single_pulsar",
    "conditional_gwb",
    "conditional_gwb_delays",
    "conditional_gwb_delay_bands",
    "conditional_covariance",
    "sample_conditional",
    # Dependency-aware grid scans
    "scan_logL",
    "PerPulsarScanAxis",
    "GlobalScanAxis",
    "ScanAxis",
    # Fisher
    "fisher_matrix",
    "flatten_params",
    "unflatten_params",
    # CW signals
    "CW_PARAM_DEFAULTS",
    "CWInjector",
    "cw_delay",
    "fplus_fcross",
    # GWB / red noise
    "CURN_PARAM_DEFAULTS",
    "CURNInjector",
    "HDCorrelatedGWBInjector",
    "fourier_basis",
    "gwb_covariance",
    # Spectral models (constructor args for the GWB/CURN injectors)
    "SpectralModel",
    "PowerLawSpectrum",
    "BrokenPowerLawSpectrum",
    "FreeSpectrum",
    # Overlap reduction functions
    "hd_orf",
    "monopole_orf",
    "dipole_orf",
    # CW (b, M) block extraction -- frequentist / localization building blocks
    "EXTRACTION_ORIENTATIONS",
    "bM2_coeffs",
    "basis_quadratics",
    "default_extraction_orientations",
    "extract_pulsar_bM",
    "extract_pulsar_blocks",
    "orientation_coeffs",
    "quadratic_coeffs",
    # CW Fisher-matrix sky localization
    "h0_for_snr",
    "make_logL_2sky",
    "gram_at_pixel",
    "gram_block_at_pair",
    "assemble_joint_fisher",
    "per_source_credible_areas_deg2",
    "credible_area_deg2",
]
