"""Dataset loaders.

The native loader (:func:`jaxpint.loaders.native.native_toas_to_jax`) is
PINT-free.  The NANOGrav PTA loader is a thin wrapper around the PINT bridge and
therefore requires PINT (``pip install jaxpint[pint]``); it is imported lazily so
``import jaxpint`` works without PINT.
"""

from jaxpint.loaders.native import native_toas_to_jax  # PINT-free

__all__ = ["NanogravPTA", "load_nanograv_pta", "native_toas_to_jax"]

_LAZY = {
    "NanogravPTA": "NanogravPTA",
    "load_nanograv_pta": "load_nanograv_pta",
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        mod = importlib.import_module("jaxpint.loaders.nanograv")
    except ImportError as exc:
        raise ImportError(
            f"jaxpint.loaders.{name} requires PINT, which is an optional "
            f"dependency. Install it with: pip install jaxpint[pint]"
        ) from exc
    return getattr(mod, _LAZY[name])


def __dir__():
    return sorted(__all__)
