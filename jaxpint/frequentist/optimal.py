"""Frequentist optimal statistic: the GWB cross-correlation detector.

The optimal statistic (OS) estimates the squared amplitude ``Â²`` of a
stochastic gravitational-wave background (GWB) from the Hellings-Downs-weighted
cross-correlation of pulsar *pairs*.  Unlike the continuous-wave F-statistics
in :mod:`jaxpint.frequentist.detection` (which live in each pulsar's own
matched filter), the OS lives in the *off-diagonal* pulsar-pair terms and is
the standard GWB-detection statistic of NANOGrav / EPTA / PPTA
([os_abcps09]_, [os_c15]_).

It consumes the per-pulsar blocks from
:func:`jaxpint.pta.per_pulsar_gw_blocks` — ``kv = FᵀC⁻¹r``, ``km = FᵀC⁻¹F``,
the shared PSD  and the ORF matrix  — and contracts them into
pairwise estimates::

    ρ_ab = ts_ab / bs_ab ,   σ_ab = 1 / √bs_ab
      ts_ab = (√Φ·kv_a) · (√Φ·kv_b)              # cross-correlation numerator
      bs_ab = tr(D_a D_b) ,  D_a = √Φ·km_a·√Φ    # normalization

    Â²   = Σ_{a<b} ρ_ab Γ_ab / σ_ab²  /  Σ_{a<b} Γ_ab² / σ_ab²
    σ_Â² = ( Σ_{a<b} Γ_ab² / σ_ab² )^(-1/2)
    SNR  = Â² / σ_Â²

Ported from ``discovery.optimal.OS``.
This is the single-component OS, matching discovery.

References
----------
.. [os_abcps09] Anholm et al. (2009), "Optimal strategies for gravitational
   wave stochastic background searches in pulsar timing data", Phys. Rev. D 79,
   084030 (arXiv:0809.0701).  Original derivation of the pulsar-timing
   cross-correlation optimal statistic.
.. [os_c15] Chamberlin et al. (2015), "Time-domain implementation of the
   optimal cross-correlation statistic for stochastic gravitational-wave
   background searches in pulsar timing data", Phys. Rev. D 91, 044048
   (arXiv:1410.8256).  The time-domain ρ_ab / σ_ab / Â² estimator this module
   implements.
.. [os_v18] Vigeland et al. (2018), "Noise-marginalized optimal statistic: A
   robust hybrid frequentist-Bayesian statistic for the stochastic
   gravitational-wave background in pulsar timing arrays", Phys. Rev. D 98,
   044003 (arXiv:1805.12188).  The noise-marginalized OS (NMOS) — the
   ``jax.vmap``-over-noise-draws use case this API is built for.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from jaxpint.pta.likelihood import GWBlocks

__all__ = ["OptimalStatistic", "optimal_statistic"]


class OptimalStatistic(NamedTuple):
    """Result of the optimal-statistic contraction.

    Attributes
    ----------
    a_squared : scalar
        ``Â²``, the GWB squared-amplitude point estimate (discovery's ``os``).
    a_squared_sigma : scalar
        Its 1σ uncertainty ``σ_Â²`` (discovery's ``os_sigma``); the ``Â² = 0``
        null spread against which ``Â²`` is measured.
    snr : scalar
        ``Â² / σ_Â²`` — the detection signal-to-noise ratio.  Invariant to the amplitude convention.
    """

    a_squared: Float[Array, ""]
    a_squared_sigma: Float[Array, ""]
    snr: Float[Array, ""]


def optimal_statistic(blocks: GWBlocks, log10_A: ArrayLike) -> OptimalStatistic:
    """Single-component optimal statistic from per-pulsar GW blocks.

    A pure, ``jit``/``vmap``-friendly contraction of :class:`GWBlocks`;

    Parameters
    ----------
    blocks : GWBlocks
        Per-pulsar ``(kv, km)`` plus shared PSD and ORF from
        :func:`jaxpint.pta.per_pulsar_gw_blocks`.
    log10_A : scalar
        The fiducial log10 GWB amplitude used to build ``blocks`` (sets
        ``gwnorm = 10^(2 log10_A)``; see the module amplitude-convention note).

    Returns
    -------
    OptimalStatistic
        ``(a_squared, a_squared_sigma, snr)``.
    """
    kv = blocks.basis_proj_residual  # (n_psr, n_basis)
    km = blocks.basis_overlap  # (n_psr, n_basis, n_basis)
    orf_matrix = blocks.orf_matrix  # (n_psr, n_psr)
    sqrt_phi = jnp.sqrt(blocks.psd)  # (n_basis,)

    # Whiten the GW basis by the (amplitude-baked) PSD
    # ts/bs are then simple contractions over pairs.
    uk = sqrt_phi[None, :] * kv  # (n_psr, n_basis)
    D = sqrt_phi[None, :, None] * km * sqrt_phi[None, None, :]

    n_psr = kv.shape[0]
    ia, ib = jnp.triu_indices(n_psr, k=1)  # unordered pairs a<b

    ts = jnp.sum(uk[ia] * uk[ib], axis=1)  # (n_pairs,)
    # bs_ab = tr(D_a D_b) = Σ_kl D_a[k,l] D_b[l,k].
    bs = jnp.sum(D[ia] * jnp.swapaxes(D[ib], -1, -2), axis=(1, 2))

    gwnorm = 10.0 ** (2.0 * jnp.asarray(log10_A))
    rho = gwnorm * ts / bs
    sigma = gwnorm / jnp.sqrt(bs)
    orf = orf_matrix[ia, ib]

    inv_var = 1.0 / sigma**2
    denom = jnp.sum(orf**2 * inv_var)
    a_squared = jnp.sum(rho * orf * inv_var) / denom
    a_squared_sigma = 1.0 / jnp.sqrt(denom)
    snr = a_squared / a_squared_sigma

    return OptimalStatistic(a_squared, a_squared_sigma, snr)
