"""Bridge layer: converts PINT objects to JaxPINT JAX-native types.

PINT's role is purely I/O: .par/.tim parsing, observatory database, clock
corrections, ephemeris lookups, and coordinate transforms.  JaxPINT owns all
numerical computation.  This module is the boundary -- the **only** place that
touches PINT.  It runs once per fit setup; after conversion everything is
convention-based float64 arrays (see Plans/Units.md for the unit contract).

PINT is an **optional** dependency (``pip install jaxpint[pint]``).  The symbols
below are imported lazily so ``import jaxpint`` works without PINT; accessing a
PINT-backed converter without PINT raises a clear :class:`ImportError`.  The
native, PINT-free pipeline lives in :mod:`jaxpint.native`.
"""

from jaxpint.par.result import ParResult  # PINT-free

__all__ = [
    "ParResult",
    "build_timing_model",
    "extract_tzr_toa",
    "params_to_pint_model",
    "pint_model_to_params",
    "pint_toas_to_jax",
]

# name -> (submodule, attribute)
_LAZY = {
    "extract_tzr_toa": ("jaxpint.bridge.toa_conversion", "extract_tzr_toa"),
    "pint_toas_to_jax": ("jaxpint.bridge.toa_conversion", "pint_toas_to_jax"),
    "pint_model_to_params": ("jaxpint.bridge.model_conversion", "pint_model_to_params"),
    "params_to_pint_model": ("jaxpint.bridge.model_conversion", "params_to_pint_model"),
    "build_timing_model": ("jaxpint.bridge.component_builder", "build_timing_model"),
}

def __getattr__(name):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        mod = importlib.import_module(target[0])
    except ImportError as exc:  # PINT (or a PINT dep) missing
        raise ImportError(
            f"jaxpint.bridge.{name} requires PINT, which is an optional "
            f"dependency. Install it with: pip install jaxpint[pint]"
        ) from exc
    return getattr(mod, target[1])

def __dir__():
    return sorted(__all__)
