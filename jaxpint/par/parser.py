"""Native ``.par`` -> :class:`~jaxpint.par.result.ParResult` entry point (PINT-free).

Composes the tokenizer, text adapter, and component detector, then hands the
result to the shared :func:`jaxpint.par.core.raw_params_to_result` -- the same
core the PINT bridge uses.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

from jaxpint.par._tcb_tables import convert_raw_params_tcb_to_tdb
from jaxpint.par.components import detect_components
from jaxpint.par.core import raw_params_to_result
from jaxpint.par.parfile import tokenize
from jaxpint.par.result import ParResult
from jaxpint.par.text_adapter import to_raw_params

log = logging.getLogger(__name__)


def get_model(par_path: str | Path) -> ParResult:
    """Parse a ``.par`` file into a JaxPINT :class:`~jaxpint.par.result.ParResult`, without PINT.

    Mirror of ``pint.models.get_model`` + ``pint_model_to_params``: tokenize the
    file, adapt each line into a ``RawParam``, detect the active components and
    binary model, then assemble via the shared core.

    The SolarWindDispersionX ``theta0`` metadata is deferred to the ephemeris
    phase (the model builder falls back to ``theta0=0.0``); everything else is
    produced natively.
    """
    parsed = to_raw_params(tokenize(par_path))

    # A TCB par is converted to TDB here, before any downstream code sees it,
    # so the rest of the stack only ever handles TDB. ``validate_units`` in the
    # core then sees UNITS TDB and passes. See ``_tcb_tables`` for what the
    # conversion does and does not cover, and why TZRMJD is refused.
    raw_params = parsed.raw_params
    if _units_of(raw_params) == "TCB":
        log.warning(
            "Converting this timing model from TCB to TDB. The conversion is "
            "approximate -- the model was fitted in TCB, and rescaling is not "
            "the same as re-minimizing -- so re-fit before trusting the result."
        )
        raw_params = convert_raw_params_tcb_to_tdb(raw_params)
        raw_params = [
            dataclasses.replace(rp, str_value="TDB") if rp.name == "UNITS" else rp
            for rp in raw_params
        ]

    component_set, binary_model = detect_components(
        parsed.templates, parsed.binary_value
    )
    return raw_params_to_result(raw_params, component_set, binary_model)


def _units_of(raw_params) -> Optional[str]:
    for rp in raw_params:
        if rp.name == "UNITS" and rp.str_value is not None:
            return rp.str_value.strip().upper()
    return None
