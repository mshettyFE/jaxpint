"""TOA selection for masked parameters.

Given a :class:`~jaxpint.par.result.MaskInfo` selector (e.g. ``JUMP -fe Rcvr_800`` or
``DMX ... mjd 55000 55100``) and the parsed TOAs, return a boolean mask over the
TOAs the parameter applies to.  Keyed by parameter name, these populate
``TOAData.flag_masks``.

* the key is matched after stripping a leading ``-`` and lower-casing;
* ``mjd``/``freq``/``tel`` map to TOA columns (clock-corrected MJD, topocentric
  observing frequency, canonical observatory name), everything else matches a
  per-TOA flag value;
* one key-value -> exact equality; two -> inclusive range ``[lo, hi]`` (numeric);
  key-values are sorted; zero -> matches nothing.

"""

from __future__ import annotations

import numpy as np

from ..par.result import MaskInfo
from .raw_toa import RawTOA


def select_toa_mask(
    info: MaskInfo,
    raw_toas: list[RawTOA],
    *,
    obs_canonical: list[str],
    mjd_corrected: np.ndarray,
) -> np.ndarray:
    """Boolean mask (shape ``(n_toas,)``) of TOAs selected by ``info``.

    Parameters
    ----------
    info:
        The selector for one masked parameter.
    raw_toas:
        The parsed ``RawTOA`` list (provides per-TOA ``flags`` and ``freq_mhz``).
    obs_canonical:
        Per-TOA canonical observatory names (for the ``tel`` key); parallel to
        ``raw_toas``.
    mjd_corrected:
        Per-TOA clock-corrected MJD (matches PINT's ``mjd_float`` column, used by
        the ``mjd`` key).
    """
    n = len(raw_toas)
    # The per-TOA column arrays must be parallel to raw_toas
    if len(mjd_corrected) != n:
        raise ValueError(f"mjd_corrected length {len(mjd_corrected)} != n_toas {n}")
    if len(obs_canonical) != n:
        raise ValueError(f"obs_canonical length {len(obs_canonical)} != n_toas {n}")

    key = info.key[1:] if info.key.startswith("-") else info.key
    klow = key.lower()

    # Collect 1-2 key-values; drop empties.  No values -> selects nothing.
    values = [v for v in (info.key_value, info.key_value2) if v not in (None, "")]
    if not values:
        return np.zeros(n, dtype=bool)

    # Build the comparison column (mirrors PINT's column_match).
    if klow == "mjd":
        column = np.asarray(mjd_corrected, dtype=np.float64)
        numeric = True
    elif klow == "freq":
        column = np.array([t.freq_mhz for t in raw_toas], dtype=np.float64)
        numeric = True
    elif klow == "tel":
        column = np.asarray(obs_canonical, dtype=object)
        numeric = False
    else:  # open-vocabulary flag key
        column = np.array([t.flags.get(klow) for t in raw_toas], dtype=object)
        numeric = False

    if numeric:
        parsed = sorted(float(v) for v in values)
    else:
        parsed = sorted(str(v) for v in values)

    if len(parsed) == 1:
        target = parsed[0]
        if numeric:
            return column == target
        # string/flag equality; missing flags (None) never match
        present = np.array([v is not None for v in column])
        return present & (column == target)

    # Two values -> inclusive range (numeric, as in PINT for mjd/freq).
    assert numeric
    lo, hi = float(parsed[0]), float(parsed[1])
    col_num = np.asarray(column, dtype=np.float64)
    return (col_num >= lo) & (col_num <= hi)
