"""WLS, GLS, and wideband fitters for JaxPINT."""

from ._base import (
    BaseFitter,
    BaseFitResult,
    compute_chi2,
    compute_design_matrix,
    compute_phase_residuals,
    compute_time_residuals,
)
from .wls import WLSFitResult, WLSFitter
from .gls import (
    GLSFitResult,
    GLSFitter,
    compute_gls_chi2,
)
from .wideband import (
    WidebandGLSFitResult,
    WidebandGLSFitter,
    compute_dm_residuals,
    compute_wideband_design_matrix,
    compute_wideband_residuals,
)

__all__ = [
    "BaseFitter",
    "BaseFitResult",
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
]
