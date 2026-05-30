"""Dataset loaders.

Both loaders are PINT-free: :func:`jaxpint.loaders.native_toas_to_jax` converts
a single ``.tim`` into a :class:`~jaxpint.types.TOAData`, and
:func:`jaxpint.loaders.load_nanograv_pta` ingests a whole NANOGrav narrowband
PTA dataset (each ``.par`` / ``.tim`` pair parsed natively).
"""

from jaxpint.loaders.nanograv import NanogravPTA, load_nanograv_pta
from jaxpint.loaders.native import native_toas_to_jax

__all__ = ["NanogravPTA", "load_nanograv_pta", "native_toas_to_jax"]
