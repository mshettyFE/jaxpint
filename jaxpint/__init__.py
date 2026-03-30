"""JaxPINT: JAX-accelerated pulsar timing (port of PINT)."""

import jax

jax.config.update("jax_enable_x64", True)

from .types import PhaseResult, TOAData, ParameterVector

__all__ = ["PhaseResult", "TOAData", "ParameterVector"]
