"""Core data types for JaxPINT.

Defines the foundational types:
- DualFloat: Extended-precision value as integer + fractional parts
  (defined in :mod:`jaxpint.types.dual_float`, re-exported here for convenience)
- TOAData: Pre-extracted TOA data as JAX arrays
- ParameterVector: Timing model parameters as a flat JAX array with metadata

"""

from jaxpint.types.dual_float import DualFloat
from jaxpint.types.parameter_vector import ParameterVector
from jaxpint.types.toa_data import TOAData

__all__ = ["DualFloat", "TOAData", "ParameterVector"]
