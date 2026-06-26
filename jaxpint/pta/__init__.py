"""PTA likelihood module for JaxPINT.

Composes :func:`jaxpint.likelihood.single_pulsar_logL` across multiple
pulsars with shared signal injections (CW sources, GWB, etc.).  Optional
cross-pulsar correlations (e.g. Hellings-Downs GWB) are handled by the
same :func:`pta_logL` entry point via the
:class:`~jaxpint.pta.likelihood.CorrelatedSignalInjector` interface.
"""

from jaxpint.types import GlobalParams
from jaxpint.pta.injectors import CorrelatedSignalInjector, SignalInjector
from jaxpint.pta.likelihood import (
    PTAConfig,
    precompute_single_pulsar_pta_factor,
    pta_logL,
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
    "CorrelatedSignalInjector",
    "pta_logL",
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
