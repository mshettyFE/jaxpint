"""Dataset loaders that turn published PTA archives into JaxPINT objects.

Each loader is a thin wrapper around the existing PINT bridge
(:mod:`jaxpint.bridge`): it discovers ``.par``/``.tim`` pairs on disk, hands
each pair to PINT, then runs ``pint_toas_to_jax`` / ``pint_model_to_params`` /
``build_timing_model`` to produce the per-pulsar tuples that
:class:`jaxpint.pta.likelihood.PTAConfig` consumes.
"""

from jaxpint.loaders.nanograv import NanogravPTA, load_nanograv_pta

__all__ = ["NanogravPTA", "load_nanograv_pta"]
