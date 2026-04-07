"""Timing model data structures and component builder.

Provides :class:`ParResult` as the canonical API boundary between PINT
bridge conversion and JaxPINT's model builder, and :func:`build_model`
to construct a :class:`~jaxpint.model.TimingModel` and
:class:`~jaxpint.noise.noise_model.NoiseModel` from a ``ParResult``.
"""

from jaxpint.parfile._model_builder import build_model
from jaxpint.parfile._param_builder import ParResult

__all__ = ["build_model", "ParResult"]
