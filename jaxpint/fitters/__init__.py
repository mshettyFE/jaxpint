"""WLS, GLS, and wideband fitters for JaxPINT."""

from ._base import (
    BaseFitter,
    BaseFitResult,
    compute_chi2,
    compute_chi2_cov,
    compute_design_matrix,
    compute_phase_residuals,
    TrackMode,
    compute_pulse_numbers,
    compute_time_residuals,
)
from .diagnostics import (
    EpochAverage,
    FTestResult,
    compute_residuals,
    ecorr_average,
    NormalityReport,
    ftest,
    ftest_results,
    normality_tests,
    whiten_residuals,
    whiten_wideband_residuals,
)
from .wls import WLSFitResult, WLSFitter
from .gls import (
    GLSFitResult,
    GLSFitter,
)
from .wideband import (
    WidebandGLSFitResult,
    WidebandGLSFitter,
    compute_dm_residuals,
    compute_wideband_design_matrix,
    compute_wideband_residuals,
)

__all__ = [
    "whiten_residuals",
    "whiten_wideband_residuals",
    "compute_residuals",
    "ecorr_average",
    "EpochAverage",
    "normality_tests",
    "NormalityReport",
    "ftest",
    "ftest_results",
    "FTestResult",
    "BaseFitter",
    "BaseFitResult",
    "WLSFitResult",
    "WLSFitter",
    "GLSFitResult",
    "GLSFitter",
    "WidebandGLSFitResult",
    "WidebandGLSFitter",
    "compute_chi2",
    "compute_chi2_cov",
    "compute_design_matrix",
    "compute_dm_residuals",
    "compute_phase_residuals",
    "TrackMode",
    "compute_pulse_numbers",
    "compute_time_residuals",
    "compute_wideband_design_matrix",
    "compute_wideband_residuals",
]
