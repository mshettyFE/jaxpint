"""Standalone .par file parser for JaxPINT.

Provides :func:`parse_par` to parse a .par file into a
:class:`~jaxpint.types.ParameterVector` (plus metadata), and
:func:`build_model` to construct a :class:`~jaxpint.model.TimingModel`
and :class:`~jaxpint.noise.noise_model.NoiseModel` without PINT.
"""

from jaxpint.parfile._model_builder import build_model
from jaxpint.parfile._param_builder import ParResult, parse_par

__all__ = ["parse_par", "build_model", "ParResult"]
