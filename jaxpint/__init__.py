"""JaxPINT: JAX-accelerated pulsar timing (port of PINT)."""

import jax

jax.config.update("jax_enable_x64", True)

from .types import PhaseResult, TOAData, ParameterVector
from .components import PhaseComponent, DelayComponent, NoiseComponent
from .phase import Spindown, Glitch, PhaseJump
from .delay import (
    AstrometryEcliptic,
    AstrometryEquatorial,
    DispersionDM,
    DispersionDMX,
    SolarSystemShapiroDelay,
    SolarWindDispersion,
    SolarWindDispersionX,
    TroposphereDelay,
)
from .binary import (
    BinaryBT,
    BinaryBTPiecewise,
    BinaryDD,
    BinaryDDGR,
    BinaryDDH,
    BinaryDDK,
    BinaryDDS,
    BinaryELL1,
    BinaryELL1H,
    BinaryELL1k,
    solve_kepler,
)
from .noise import (
    EcorrNoise,
    NoiseModel,
    PLChromNoise,
    PLDMNoise,
    PLRedNoise,
    PLSWNoise,
    ScaleToaError,
)
from .model import TimingModel
from .fitter import (
    GLSFitResult,
    GLSFitter,
    WLSFitResult,
    WLSFitter,
    compute_design_matrix,
    compute_phase_residuals,
    compute_time_residuals,
)
from .bridge import (
    build_timing_model,
    extract_tzr_toa,
    params_to_pint_model,
    pint_model_to_params,
    pint_toas_to_jax,
)
from .simulation import apply_delay_to_toas, make_fake_toas, simulate_noise, zero_residuals
from .utils import (
    normalize_designmatrix,
    sherman_morrison_dot,
    taylor_horner,
    taylor_horner_deriv,
    taylor_horner_phase,
    weighted_mean,
    woodbury_dot,
    woodbury_solve,
)

__all__ = [
    "AstrometryEcliptic",
    "AstrometryEquatorial",
    "BinaryBT",
    "BinaryBTPiecewise",
    "BinaryDD",
    "BinaryDDGR",
    "BinaryDDH",
    "BinaryDDK",
    "BinaryDDS",
    "BinaryELL1",
    "BinaryELL1H",
    "BinaryELL1k",
    "DelayComponent",
    "DispersionDM",
    "DispersionDMX",
    "EcorrNoise",
    "GLSFitResult",
    "GLSFitter",
    "Glitch",
    "NoiseComponent",
    "NoiseModel",
    "PLChromNoise",
    "PLDMNoise",
    "PLRedNoise",
    "PLSWNoise",
    "ParameterVector",
    "PhaseComponent",
    "PhaseJump",
    "PhaseResult",
    "ScaleToaError",
    "SolarSystemShapiroDelay",
    "SolarWindDispersion",
    "SolarWindDispersionX",
    "Spindown",
    "TOAData",
    "TimingModel",
    "TroposphereDelay",
    "WLSFitResult",
    "WLSFitter",
    "apply_delay_to_toas",
    "build_timing_model",
    "compute_design_matrix",
    "compute_phase_residuals",
    "compute_time_residuals",
    "extract_tzr_toa",
    "make_fake_toas",
    "normalize_designmatrix",
    "params_to_pint_model",
    "pint_model_to_params",
    "pint_toas_to_jax",
    "sherman_morrison_dot",
    "simulate_noise",
    "solve_kepler",
    "taylor_horner",
    "taylor_horner_deriv",
    "taylor_horner_phase",
    "weighted_mean",
    "woodbury_dot",
    "woodbury_solve",
    "zero_residuals",
]
