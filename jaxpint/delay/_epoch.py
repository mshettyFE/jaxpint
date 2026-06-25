"""Shared epoch-time helper for Taylor-in-years delay components.

``DispersionDM``, ``ChromaticCM`` and ``SolarWind`` all expand a measure as a
Taylor series in *Julian years* about a named epoch parameter; this centralises
the (previously triplicated) ``dt`` computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxtyping import Array, Float

from ..constants import DAYS_PER_JULIAN_YEAR

if TYPE_CHECKING:
    from ..types import ParameterVector, TOAData


def dt_years_from_epoch(
    toa_data: "TOAData",
    params: "ParameterVector",
    epoch_name: str,
) -> Float[Array, " n_toas"]:
    """Time from the named epoch to each TOA, in Julian years.

    ``(tdb - epoch)`` is formed in extended precision (``DualFloat``) before
    collapsing to float64, so the year value stays precise despite the large
    absolute MJD.
    """
    epoch = params.epoch_dual(epoch_name)
    dt_days = (toa_data.tdb - epoch).total
    return dt_days / DAYS_PER_JULIAN_YEAR
