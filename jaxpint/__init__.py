"""JaxPINT: JAX-accelerated pulsar timing (port of PINT)."""

import jax

jax.config.update("jax_enable_x64", True)

from .types import PhaseResult, TOAData, ParameterVector
from .components import PhaseComponent, DelayComponent
from .spin import Spindown
from .dispersion_dm import DispersionDM
from .bridge import pint_toas_to_jax, pint_model_to_params, params_to_pint_model, build_timing_model
from .model import TimingModel
from .noise import ScaleToaError, EcorrNoise
from .astrometry import AstrometryEcliptic
from .shapiro import SolarSystemShapiroDelay
from .solar_wind import SolarWindDispersion
from .troposphere import TroposphereDelay
from .fitter import WLSFitter, WLSFitResult, GLSFitter, GLSFitResult
from .simulation import apply_delay_to_toas, zero_residuals
from .utils import (
    taylor_horner,
    taylor_horner_deriv,
    weighted_mean,
    normalize_designmatrix,
    sherman_morrison_dot,
    woodbury_dot,
    woodbury_solve,
)

__all__ = [
    "PhaseResult",
    "TOAData",
    "ParameterVector",
    "PhaseComponent",
    "DelayComponent",
    "Spindown",
    "DispersionDM",
    "TimingModel",
    "ScaleToaError",
    "AstrometryEcliptic",
    "SolarSystemShapiroDelay",
    "SolarWindDispersion",
    "TroposphereDelay",
    "WLSFitter",
    "WLSFitResult",
    "pint_toas_to_jax",
    "pint_model_to_params",
    "params_to_pint_model",
    "build_timing_model",
    "taylor_horner",
    "taylor_horner_deriv",
    "weighted_mean",
    "normalize_designmatrix",
    "sherman_morrison_dot",
    "woodbury_dot",
    "woodbury_solve",
    "EcorrNoise",
    "GLSFitter",
    "GLSFitResult",
    "apply_delay_to_toas",
    "zero_residuals",
]
