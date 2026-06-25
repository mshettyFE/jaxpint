"""
PINT stores topocentric frequency in its TOA table and applies the barycentric
Doppler correction on demand (``model.barycentric_radio_freq``):

    f_bary = f_topo * (1 - v_obs . Lhat / c)

where ``v_obs`` is the observatory velocity wrt the SSB (``toa_data.ssb_obs_vel``,
km/s) and ``Lhat`` is the unit vector toward the pulsar (from astrometry).

JaxPINT instead **precomputes** this once at TOAData-build time and stores the
barycentric value as ``TOAData.freq`` (see :mod:`jaxpint.loaders.native`), rather
than recomputing it per component on demand.

- *Lazy (PINT):* ``Lhat`` (and thus the Doppler factor) is recomputed from the
  current astrometry, so the barycentric freq stays exact while RAJ/DECJ/proper
  motion are being fit.
- *Precompute (JaxPINT):* the build-time astrometry is baked in, keeping
  ``TOAData`` a static, parameter-independent data container and keeping the
  Doppler/astrometry work off the (jitted) per-call likelihood path.  The cost is
  that the stored freq is not refreshed if the astrometry is later refit.

That staleness is negligible in practice: real ``.par`` astrometry
uncertainties are sub-arcsec (~1e-4" for good MSPs, <~ 1" even for
poorly-constrained ones), and the Doppler factor is only first order in
``v/c`` (~1e-4).  So the worst-case fit-step error in the frequency-dependent
delays (dispersion, FD, chromatic) is <~ 50 ps -- far below ns-level timing
precision (guarded by
``tests/test_timescale.py::test_precompute_staleness_below_ns``).
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from ..constants import C_KM_PER_S
from ..types import ParameterVector, TOAData
from ..utils import compute_pulsar_direction, compute_pulsar_direction_ecl


def doppler_shift_freq(
    freq_topo: Float[Array, " n_toas"],
    ssb_obs_vel: Float[Array, "n_toas 3"],
    l_hat: Float[Array, "n_toas 3"],
) -> Float[Array, " n_toas"]:
    """Apply the first-order Doppler factor to a topocentric frequency.

    ``f_bary = f_topo * (1 - v.Lhat/c)``, with ``v`` in km/s.
    """
    v_dot_l = jnp.sum(ssb_obs_vel * l_hat, axis=1)  # km/s
    return freq_topo * (1.0 - v_dot_l / C_KM_PER_S)


def barycentric_radio_freq(
    toa_data: TOAData,
    params: ParameterVector,
    *,
    raj_name: str = "RAJ",
    decj_name: str = "DECJ",
    pmra_name: Optional[str] = None,
    pmdec_name: Optional[str] = None,
    posepoch_name: Optional[str] = None,
    ecliptic: bool = False,
    elong_name: str = "ELONG",
    elat_name: str = "ELAT",
    pmelong_name: Optional[str] = None,
    pmelat_name: Optional[str] = None,
    obliquity_arcsec: float = 0.0,
) -> Float[Array, " n_toas"]:
    """Barycentric (Doppler-corrected) radio frequency for each TOA.

    Computes ``Lhat`` from the model astrometry (equatorial by default, or
    ecliptic when ``ecliptic=True``) and applies the Doppler factor to
    ``toa_data.freq`` (which must be topocentric).
    """
    if ecliptic:
        l_hat = compute_pulsar_direction_ecl(
            toa_data,
            params,
            elong_name=elong_name,
            elat_name=elat_name,
            pmelong_name=pmelong_name,
            pmelat_name=pmelat_name,
            posepoch_name=posepoch_name,
            obliquity_arcsec=obliquity_arcsec,
        )
    else:
        l_hat = compute_pulsar_direction(
            toa_data,
            params,
            raj_name=raj_name,
            decj_name=decj_name,
            pmra_name=pmra_name,
            pmdec_name=pmdec_name,
            posepoch_name=posepoch_name,
        )
    return doppler_shift_freq(toa_data.freq, toa_data.ssb_obs_vel, l_hat)
