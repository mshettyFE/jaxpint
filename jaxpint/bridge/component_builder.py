"""Component builder: PINT components → JaxPINT timing model components.

Delegates to :mod:`jaxpint.bridge._model_builder` via the unified
:class:`ParResult` interface.  The bridge converts a PINT model to
``ParResult`` (see :func:`pint_model_to_params`), then
``build_model()`` constructs the JaxPINT components.
"""

from __future__ import annotations

import logging
from typing import Optional

from pint.models.timing_model import TimingModel as PINTTimingModel
from pint.toa import TOAs

from jaxpint.utils import build_quantization_matrix as _build_quantization_matrix

log = logging.getLogger(__name__)


def build_timing_model(
    pint_model: PINTTimingModel,
    toas: Optional[TOAs] = None,
):
    """Construct a JaxPINT :class:`~jaxpint.model.TimingModel` from a PINT model.

    Converts the PINT model to a :class:`ParResult` and optional
    :class:`TOAData`, then delegates to
    :func:`jaxpint.bridge._model_builder.build_model`.

    Parameters
    ----------
    pint_model : pint.models.TimingModel
        The PINT timing model to convert.
    toas : pint.toa.TOAs, optional
        If provided, TOA-dependent components (ECORR, red noise, etc.)
        will be constructed.

    Returns
    -------
    (TimingModel, NoiseModel)
        The timing model and a :class:`NoiseModel` that aggregates all
        noise sources (white noise and correlated components).
    """
    from jaxpint.bridge.model_conversion import pint_model_to_params
    from jaxpint.bridge.toa_conversion import pint_toas_to_jax
    from jaxpint.bridge._model_builder import build_model

    par = pint_model_to_params(pint_model)
    toa_data = pint_toas_to_jax(toas, model=pint_model) if toas is not None else None
    return build_model(par, toa_data)
