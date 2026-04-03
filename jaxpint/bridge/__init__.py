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

__all__ = [
    "extract_tzr_toa",
    "pint_toas_to_jax",
    "pint_model_to_params",
    "params_to_pint_model",
    "build_timing_model",
    "_build_quantization_matrix",
]
