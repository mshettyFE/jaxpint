"""JaxPINT: JAX-accelerated pulsar timing (port of PINT)."""

import jax

jax.config.update("jax_enable_x64", True)

from .types import PhaseResult, TOAData, ParameterVector
from .bridge import pint_toas_to_jax, pint_model_to_params, params_to_pint_model
from .utils import (
    taylor_horner,
    taylor_horner_deriv,
    weighted_mean,
    normalize_designmatrix,
    sherman_morrison_dot,
    woodbury_dot,
)

__all__ = [
    "PhaseResult",
    "TOAData",
    "ParameterVector",
    "pint_toas_to_jax",
    "pint_model_to_params",
    "params_to_pint_model",
    "taylor_horner",
    "taylor_horner_deriv",
    "weighted_mean",
    "normalize_designmatrix",
    "sherman_morrison_dot",
    "woodbury_dot",
]
