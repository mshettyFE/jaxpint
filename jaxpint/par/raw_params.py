"""Adapter-neutral parsed-parameter record.

:class:`RawParam` is the shared seam between parameter adapters and the
:func:`jaxpint.par.core.raw_params_to_result` core.  A *source* adapter (the
PINT bridge today; the native ``.par`` text parser later) does only the
source-specific, precision-critical extraction and emits a ``list[RawParam]``;
the shared core then performs all the unit-algebra and structural work
(deg->rad, us->s, pair splitting, classification, alias synthesis,
``ParameterVector`` assembly).

The division of labour:

- **Adapter owns precision-critical, representation-dependent steps.**  An MJD
  (~60000) overflows float64 precision well short of nanoseconds, so the adapter
  splits it at full precision *its own way* (PINT via the astropy ``jd1/jd2``
  pair; the native parser via a longdouble/string parse) and hands over the
  ``(int_day, frac_day)`` pair in :attr:`RawParam.mjd_split`.  Angles fit float64
  comfortably, so the adapter also resolves the source-specific bit (sexagesimal
  parse vs. ``Quantity.to(rad)``) and emits radians in :attr:`RawParam.value`.
- **Core owns everything unit-algebraic / structural** (see
  :func:`jaxpint.par.core.raw_params_to_result`).

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ParamKind(Enum):
    """How :func:`jaxpint.par.core.raw_params_to_result` routes/coerces a param."""

    FLOAT = "float"  # plain float in native unit; core applies deg->rad algebra if unit has deg
    ANGLE = "angle"  # adapter already produced radians (sexagesimal parse / .to(rad))
    MJD = "mjd"  # epoch; adapter already split into (int_day, frac_day) for precision
    PAIR = "pair"  # two-valued (WAVEn, IFUNC); core splits into _A / _B entries
    MASK = "mask"  # JUMP/EFAC/EQUAD/ECORR/...; contributes BOTH a value and a mask_info entry
    STR = "str"  # -> ParResult.metadata
    BOOL = "bool"  # -> ParResult.bool_params
    INT = "int"  # -> ParResult.int_params (incl. int-valued floats: TNREDC, SWM, ...)


@dataclass(frozen=True)
class RawParam:
    """Adapter-neutral parsed parameter.

    Exactly one numeric/value payload field is meaningful per :attr:`kind`
    (see :class:`ParamKind`).  Produced by an adapter, consumed by
    :func:`jaxpint.par.core.raw_params_to_result`.
    """

    name: str  # canonical name, post-alias-resolution
    kind: ParamKind
    frozen: bool = True  # ``not fit_flag``

    # --- numeric payloads (pick by kind) ---
    value: Optional[float] = None  # FLOAT (native unit), ANGLE (radians), MASK scalar
    mjd_split: Optional[tuple[float, float]] = (
        None  # MJD: (int_day, frac_day) -- adapter pre-splits
    )
    value_pair: Optional[tuple[float, float]] = None  # PAIR: (a, b)

    # 1-sigma fit uncertainty in the SAME native unit as ``value`` (core applies the
    # same unit scaling, e.g. deg->rad / us->s, that it applies to ``value``).
    # ``None`` when the source did not report one (frozen/value-only line, or a
    # non-FLOAT kind for which an uncertainty is not tracked).
    uncertainty: Optional[float] = None

    # --- non-numeric payloads ---
    str_value: Optional[str] = None  # STR / metadata-only floats (e.g. TZRFRQ=inf)
    bool_value: Optional[bool] = None  # BOOL
    int_value: Optional[int] = None  # INT

    # --- unit + mask metadata ---
    unit: str = ""  # native unit string; core converts (deg->rad, us->s)
    mask_key: Optional[str] = None  # MASK only: "-fe" / "mjd" / "freq" / "tel" / ...
    mask_key_value: Optional[str] = None  # MASK only
    mask_key_value2: Optional[str] = (
        None  # MASK only: 2nd value for range keys (mjd/freq)
    )
