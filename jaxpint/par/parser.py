"""Native ``.par`` -> :class:`ParResult` entry point (PINT-free).

Composes the tokenizer, text adapter, and component detector, then hands the
result to the shared :func:`jaxpint.par.core.raw_params_to_result` -- the same
core the PINT bridge uses.
"""

from __future__ import annotations

from pathlib import Path

from jaxpint.par.components import detect_components
from jaxpint.par.core import raw_params_to_result
from jaxpint.par.parfile import tokenize
from jaxpint.par.result import ParResult
from jaxpint.par.text_adapter import to_raw_params


def get_model(par_path: str | Path) -> ParResult:
    """Parse a ``.par`` file into a JaxPINT :class:`ParResult`, without PINT.

    Mirror of ``pint.models.get_model`` + ``pint_model_to_params``: tokenize the
    file, adapt each line into a ``RawParam``, detect the active components and
    binary model, then assemble via the shared core.

    The SolarWindDispersionX ``theta0`` metadata is deferred to the ephemeris
    phase (the model builder falls back to ``theta0=0.0``); everything else is
    produced natively.
    """
    parsed = to_raw_params(tokenize(par_path))
    component_set, binary_model = detect_components(parsed.templates, parsed.binary_value)
    return raw_params_to_result(parsed.raw_params, component_set, binary_model)
