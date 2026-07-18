"""Shared build inputs and helpers for component ``build`` methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from jaxpint.par.result import ParResult
    from jaxpint.types import TOAData


def param_is_set(par: "ParResult", name: str) -> bool:
    """Check whether *name* exists in the ParameterVector and is non-zero."""
    return name in par.params and float(par.params.param_value(name)) != 0.0


def opt_name(par: "ParResult", name: str) -> Optional[str]:
    return name if param_is_set(par, name) else None


def value(par: "ParResult", name: str) -> float:
    """Float value of parameter *name* (caller guarantees it exists)."""
    return float(par.params.param_value(name))


def basis_seconds(toa_data) -> np.ndarray:
    """Time coordinate for GP bases / ECORR quantization, as float64 numpy.

    Raises when the producer of the TOAData never chose one — see
    :attr:`jaxpint.types.TOAData.basis_seconds` for the conventions.
    """
    return np.asarray(toa_data.require_basis_seconds(), dtype=np.float64)


def span_seconds(par: "ParResult", basis_s, tspan_param: Optional[str] = None) -> float:
    """Observation span in seconds.

    The default is ``max - min`` of the supplied basis times; an explicit
    ``T...TSPAN`` parameter (in days), when present, overrides it (matches
    PINT / enterprise's per-pulsar-span default).
    """
    T = float(np.max(basis_s) - np.min(basis_s))
    if tspan_param is not None and (tspan_param in par.params):
        tspan_days = value(par, tspan_param)
        T = tspan_days * 86400.0
    return T


@dataclass(frozen=True)
class BuildContext:
    """Shared inputs threaded to every component ``build`` method.

    Bundles the parse result, optional TOA data, and the astrometry names
    resolved once up front (see ``model_builder._resolve_astrometry``).
    """

    par: "ParResult"
    toa_data: Optional["TOAData"]
    raj: str
    decj: str
    pmra: Optional[str]
    pmdec: Optional[str]
    posepoch: Optional[str]
    obliquity_arcsec: Optional[float]
