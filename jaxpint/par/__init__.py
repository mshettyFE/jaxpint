"""PINT-free shared core for parameter conversion.

This subpackage holds the adapter-neutral parameter logic shared between the
PINT bridge (:mod:`jaxpint.bridge`) and the future native ``.par`` parser:

- :class:`~jaxpint.par.raw_params.RawParam` -- the normalized parsed-parameter
  record produced by either adapter,
- :func:`~jaxpint.par.core.raw_params_to_result` -- turns a ``list[RawParam]``
  (+ detected components / binary model) into a
  :class:`~jaxpint.par.result.ParResult`,
- :class:`~jaxpint.par.result.ParResult` / :class:`~jaxpint.par.result.MaskInfo`
  -- the contract consumed by ``jaxpint.bridge._model_builder.build_model``,
- :class:`~jaxpint.par.registry.Component` / ``BinaryModel`` enums.

Nothing here imports PINT.
"""

from jaxpint.par.core import raw_params_to_result
from jaxpint.par.parser import get_model
from jaxpint.par.raw_params import ParamKind, RawParam
from jaxpint.par.registry import BinaryModel, Component
from jaxpint.par.result import MaskInfo, ParResult

__all__ = [
    "BinaryModel",
    "Component",
    "MaskInfo",
    "ParResult",
    "ParamKind",
    "RawParam",
    "get_model",
    "raw_params_to_result",
]
