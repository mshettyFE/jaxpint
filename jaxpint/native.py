"""Native (PINT-free) public API, mirroring PINT's ``get_model`` / ``get_TOAs`` /
``get_model_and_toas``.

Swapping ``from pint.models import get_model_and_toas`` for
``from jaxpint.native import get_model_and_toas`` is ~a one-line change for the
*loading* step.  (Downstream usage still differs: JaxPINT's residual/phase verbs
and objects are not PINT's.)

Composes the already-built native pieces:
``jaxpint.par.get_model`` (.par -> ParResult) +
``jaxpint.loaders.native.native_toas_to_jax`` (.tim + ParResult -> TOAData) +
``jaxpint.bridge._model_builder.build_model`` (ParResult [+ TOAData] ->
(TimingModel, NoiseModel)).  All PINT-free at runtime.
"""

from __future__ import annotations

from typing import Optional, Union

from .model_builder import build_model
from .loaders.native import native_toas_to_jax
from .par import get_model as _parse_par
from .par.result import ParResult

__all__ = ["get_model", "get_TOAs", "get_model_and_toas"]


def get_model(par_path):
    """Parse a ``.par`` and build a (TimingModel, NoiseModel).

    .. note::
       Without TOAs, **TOA-dependent noise components are omitted** (ECORR and
       the power-law red/DM/chromatic/solar-wind noise need ``TOAData`` to build
       their Fourier/quantization bases).  The returned model therefore carries
       only the deterministic components (spin, astrometry, dispersion, binary,
       ...).  Use :func:`get_model_and_toas` for a fully-built model.
    """
    return build_model(_parse_par(par_path), None)


def get_TOAs(
    tim_path,
    par: Optional[Union[ParResult, str]] = None,
    *,
    ephem: Optional[str] = None,
    include_bipm: Optional[bool] = None,
    bipm_version: Optional[str] = None,
    planets: Optional[bool] = None,
    limits: str = "warn",
):
    """Load a ``.tim`` into a :class:`~jaxpint.types.TOAData`.

    ``par`` may be a parsed :class:`ParResult` or a ``.par`` path (parsed here);
    it supplies astrometry (barycentric freq), TZR/abs-phase, flag-mask
    selectors, troposphere config, and the default ephem/clock/planet settings.
    """
    par_result = _parse_par(par) if isinstance(par, str) else par
    return native_toas_to_jax(
        tim_path, par_result, ephem=ephem, include_bipm=include_bipm,
        bipm_version=bipm_version, planets=planets, limits=limits,
    )


def get_model_and_toas(
    par_path,
    tim_path,
    *,
    ephem: Optional[str] = None,
    include_bipm: Optional[bool] = None,
    bipm_version: Optional[str] = None,
    planets: Optional[bool] = None,
    limits: str = "warn",
):
    """Native equivalent of PINT's ``get_model_and_toas``.

    Returns ``(model, noise, toa_data)`` -- the ``NoiseModel`` is returned
    explicitly because the GLS/likelihood path consumes it directly (avoids a
    second ``build_model``).  TOAs are built first so the model can include the
    TOA-dependent noise components.
    """
    par_result = _parse_par(par_path)
    toa_data = native_toas_to_jax(
        tim_path, par_result, ephem=ephem, include_bipm=include_bipm,
        bipm_version=bipm_version, planets=planets, limits=limits,
    )
    model, noise = build_model(par_result, toa_data)
    return model, noise, toa_data
