"""WLS, GLS, and wideband fitters for JaxPINT."""

from ._common import (
    compute_chi2,
    compute_design_matrix,
    compute_phase_residuals,
    compute_time_residuals,
    _subtract_weighted_mean,
    wls_step,
)
from .wls import WLSFitResult, WLSFitter
from .gls import (
    GLSFitResult,
    GLSFitter,
    compute_gls_chi2,
    gls_step_augmented,
    gls_step_fullcov,
)
from .wideband import (
    WidebandGLSFitResult,
    WidebandGLSFitter,
    compute_dm_residuals,
    compute_wideband_design_matrix,
    compute_wideband_residuals,
)

__all__ = [
    "WLSFitResult",
    "WLSFitter",
    "GLSFitResult",
    "GLSFitter",
    "WidebandGLSFitResult",
    "WidebandGLSFitter",
    "compute_chi2",
    "compute_design_matrix",
    "compute_dm_residuals",
    "compute_gls_chi2",
    "compute_phase_residuals",
    "compute_time_residuals",
    "compute_wideband_design_matrix",
    "compute_wideband_residuals",
    "gls_step_augmented",
    "gls_step_fullcov",
    "wls_step",
    "_subtract_weighted_mean",
]
