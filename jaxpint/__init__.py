"""JaxPINT: JAX-accelerated pulsar timing (port of PINT)."""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

from .types.dual_float import DualFloat
from .types import TOAData, ParameterVector
from .components import PhaseComponent, DelayComponent, NoiseComponent
from .phase import Spindown, Glitch, IFunc, PhaseJump, PiecewiseSpindown, Wave
from .delay import (
    AstrometryEcliptic,
    AstrometryEquatorial,
    CMWaveX,
    ChromaticCM,
    ChromaticCMX,
    DMWaveX,
    DispersionDM,
    DispersionDMX,
    ExponentialDip,
    FDJump,
    FrequencyDependent,
    SolarSystemShapiroDelay,
    SolarWindDispersion,
    SolarWindDispersionX,
    TroposphereDelay,
    WaveX,
)
from .binary import (
    BinaryBT,
    BinaryBTPiecewise,
    BinaryDD,
    BinaryDDGR,
    BinaryDDK,
    BinaryELL1,
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
from .fitters import (
    GLSFitResult,
    GLSFitter,
    WidebandGLSFitResult,
    WidebandGLSFitter,
    WLSFitResult,
    WLSFitter,
    compute_design_matrix,
    compute_dm_residuals,
    compute_phase_residuals,
    compute_time_residuals,
    compute_wideband_design_matrix,
    compute_wideband_residuals,
)
from .model_builder import build_model
from .likelihood import single_pulsar_logL
from . import native
from .loaders import (
    NanogravPTA,
    PulsarRecord,
    iter_nanograv_pta,
    load_nanograv_pta,
    native_toas_to_jax,
)
from .simulation import (
    apply_delay_to_toas,
    make_fake_toas,
    simulate_noise,
    zero_residuals,
)
from .utils import (
    fourier_sum,
    normalize_designmatrix,
    sherman_morrison_dot,
    taylor_horner,
    taylor_horner_deriv,
    taylor_horner_phase,
    weighted_mean,
    weighted_mean_sdev,
    woodbury_dot,
    woodbury_dot_qr,
    woodbury_solve,
)

__all__ = [
    "AstrometryEcliptic",
    "AstrometryEquatorial",
    "BinaryBT",
    "BinaryBTPiecewise",
    "BinaryDD",
    "BinaryDDGR",
    "BinaryDDK",
    "BinaryELL1",
    "CMWaveX",
    "ChromaticCM",
    "ChromaticCMX",
    "DMWaveX",
    "DelayComponent",
    "DispersionDM",
    "DispersionDMX",
    "EcorrNoise",
    "ExponentialDip",
    "FDJump",
    "FrequencyDependent",
    "GLSFitResult",
    "GLSFitter",
    "Glitch",
    "IFunc",
    "NanogravPTA",
    "NoiseComponent",
    "NoiseModel",
    "PLChromNoise",
    "PLDMNoise",
    "PLRedNoise",
    "PLSWNoise",
    "DualFloat",
    "ParameterVector",
    "PhaseComponent",
    "PhaseJump",
    "PiecewiseSpindown",
    "ScaleToaError",
    "SolarSystemShapiroDelay",
    "SolarWindDispersion",
    "SolarWindDispersionX",
    "single_pulsar_logL",
    "Spindown",
    "TOAData",
    "TimingModel",
    "TroposphereDelay",
    "Wave",
    "WaveX",
    "WidebandGLSFitResult",
    "WidebandGLSFitter",
    "WLSFitResult",
    "WLSFitter",
    "apply_delay_to_toas",
    "compute_design_matrix",
    "compute_dm_residuals",
    "compute_phase_residuals",
    "compute_time_residuals",
    "compute_wideband_design_matrix",
    "compute_wideband_residuals",
    "fourier_sum",
    "load_nanograv_pta",
    "iter_nanograv_pta",
    "PulsarRecord",
    "make_fake_toas",
    "native",
    "normalize_designmatrix",
    "sherman_morrison_dot",
    "simulate_noise",
    "solve_kepler",
    "taylor_horner",
    "taylor_horner_deriv",
    "taylor_horner_phase",
    "weighted_mean",
    "weighted_mean_sdev",
    "woodbury_dot",
    "woodbury_dot_qr",
    "woodbury_solve",
    "zero_residuals",
    # PINT-backed (lazy; require `pip install jaxpint[pint]`)
    "build_timing_model",
    "extract_tzr_toa",
    "params_to_pint_model",
    "pint_model_to_params",
    "pint_toas_to_jax",
    # PINT-free
    "build_model",
    "native_toas_to_jax",
]

# PINT-backed top-level symbols, resolved lazily so `import jaxpint` works
# without PINT (an optional dependency).  Accessing one without PINT installed
# raises a clear ImportError via the bridge/loaders __getattr__.
_LAZY_PINT = {
    "build_timing_model": "jaxpint.bridge",
    "extract_tzr_toa": "jaxpint.bridge",
    "params_to_pint_model": "jaxpint.bridge",
    "pint_model_to_params": "jaxpint.bridge",
    "pint_toas_to_jax": "jaxpint.bridge",
}


def __getattr__(name):
    target = _LAZY_PINT.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(target), name)
