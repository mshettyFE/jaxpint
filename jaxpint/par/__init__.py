"""
This subpackage holds the parameter logic shared between the
native ``.par`` parser (:mod:`jaxpint.par.parser`) and the PINT bridge
(:mod:`jaxpint.bridge`):
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
