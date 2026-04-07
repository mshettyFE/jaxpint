"""Bridge layer: converts PINT objects to JaxPINT JAX-native types.

PINT's role is purely I/O: .par/.tim parsing, observatory database, clock
corrections, ephemeris lookups, and coordinate transforms.  JaxPINT owns all
numerical computation.  This module is the boundary -- the **only** place that
touches astropy units.  It runs once per fit setup; after conversion everything
is convention-based float64 arrays (see Plans/Units.md for the unit contract).
"""

from jaxpint.bridge.toa_conversion import extract_tzr_toa, pint_toas_to_jax
from jaxpint.bridge.model_conversion import (
    pint_model_to_params,
    params_to_pint_model,
)
from jaxpint.bridge.component_builder import (
    build_timing_model,
    _build_quantization_matrix,
)
from jaxpint.bridge._param_builder import ParResult

__all__ = [
    "_build_quantization_matrix",
    "build_timing_model",
    "extract_tzr_toa",
    "params_to_pint_model",
    "pint_model_to_params",
    "pint_toas_to_jax",
    "ParResult",
]
