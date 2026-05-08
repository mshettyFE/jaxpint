"""PTA likelihood module for JaxPINT.

Composes :func:`jaxpint.likelihood.single_pulsar_logL` across multiple
pulsars with shared signal injections (CW sources, GWB, etc.).
"""

from jaxpint.pta.params import GlobalParams
from jaxpint.pta.likelihood import (
    PTAConfig,
    SignalInjector,
    precompute_single_pulsar_pta_factor,
    pta_logL,
    single_pulsar_pta_logL,
    single_pulsar_pta_logL_with_factor,
)
from jaxpint.pta.scan import (
    PerPulsarScanAxis,
    GlobalScanAxis,
    scan_logL,
)
from jaxpint.pta.correlated_likelihood import (
    CorrelatedPTAConfig,
    CorrelatedSignalInjector,
    pta_logL_correlated,
)
from jaxpint.pta.fisher import fisher_matrix, flatten_params, unflatten_params
from jaxpint.pta.signals import (
    CW_PARAM_DEFAULTS,
    CWInjector,
    CURN_PARAM_DEFAULTS,
    CURNInjector,
    HDCorrelatedGWBInjector,
    cw_delay,
    fplus_fcross,
    fourier_basis,
    gwb_covariance,
    powerlaw_psd,
    hd_orf,
    monopole_orf,
    dipole_orf,
)

__all__ = [
    # Core
    "GlobalParams",
    "PTAConfig",
    "SignalInjector",
    "pta_logL",
    "single_pulsar_pta_logL",
    "single_pulsar_pta_logL_with_factor",
    "precompute_single_pulsar_pta_factor",
    # Scans
    "PerPulsarScanAxis",
    "GlobalScanAxis",
    "scan_logL",
    # Correlated
    "CorrelatedPTAConfig",
    "CorrelatedSignalInjector",
    "pta_logL_correlated",
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
    "powerlaw_psd",
    # Overlap reduction functions
    "hd_orf",
    "monopole_orf",
    "dipole_orf",
]
