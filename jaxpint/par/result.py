"""Data structures for parsed timing model parameters.

Defines :class:`ParResult` and :class:`MaskInfo`, the canonical contract
between a parameter adapter (the native ``.par`` parser or the PINT bridge)
and JaxPINT's model builder (:func:`jaxpint.model_builder.build_model`).

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from jaxpint.par.registry import BinaryModel, Component
from jaxpint.types import ParameterVector


@dataclass
class MaskInfo:
    """Metadata for a mask parameter (JUMP, EFAC, etc.) needed for TOA matching."""

    name: str  # e.g. "JUMP1"
    key: str  # e.g. "-fe" or "-sys"
    key_value: str  # e.g. "Rcvr_800" or "430"
    key_value2: Optional[str] = None  # Second value for range-type keys (mjd, freq)


@dataclass
class ParResult:
    """Complete result of converting a timing model to JaxPINT's internal format.

    Produced by a parameter adapter (e.g.
    :func:`jaxpint.bridge.pint_model_to_params`) via
    :func:`jaxpint.par.core.raw_params_to_result`, and consumed by
    :func:`jaxpint.model_builder.build_model`.
    """

    params: ParameterVector  # JIT-able values that jax can trace
    component_set: set[Component] = field(
        default_factory=set
    )  # What components need to be built
    binary_model: Optional[BinaryModel] = None  # What binary model you are assuming
    metadata: dict[str, str] = field(
        default_factory=dict
    )  # Non-numeric .par parameters
    mask_info: dict[str, MaskInfo] = field(default_factory=dict)
    int_params: dict[str, int] = field(
        default_factory=dict
    )  # Non-jittable integer parameters
    bool_params: dict[str, bool] = field(
        default_factory=dict
    )  # Non-jittable boolean parameters
