"""Barycentric radio frequency (native; model/consumption side).

PINT stores topocentric frequency in its TOA table and applies the barycentric
Doppler correction on demand (``model.barycentric_radio_freq``):

    f_bary = f_topo * (1 - v_obs . Lhat / c)

where ``v_obs`` is the observatory velocity wrt the SSB (``toa_data.ssb_obs_vel``,
km/s) and ``Lhat`` is the unit vector toward the pulsar (from astrometry).  This
mirrors that, reusing the native pulsar-direction code already in
:mod:`jaxpint.utils`.

(Applied once at TOAData-build time today so ``TOAData.freq`` stays barycentric
and matches PINT; available for a future per-component refactor.)
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
