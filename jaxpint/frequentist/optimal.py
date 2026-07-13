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

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.integrate
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from jaxpint.frequentist.nulls import isotropic_positions
from jaxpint.pta.likelihood import GWBlocks
from jaxpint.pta.signals.orf import hd_orf

__all__ = [
    "OptimalStatistic",
    "optimal_statistic",
    # Empirical null distributions
    "sky_scramble",
    "sky_scramble_snrs",
    "phase_shift",
    "phase_shift_snrs",
    # Analytic generalized-χ² null
    "os_quadratic_form",
    "gx2_cdf",
    "os_null_cdf",
]


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


# ---------------------------------------------------------------------------
# Shared contraction helpers
# ---------------------------------------------------------------------------


def _pair_indices(n_psr: int) -> tuple[Array, Array]:
    """Row/column indices of the unordered pulsar pairs ``a < b``."""
    return jnp.triu_indices(n_psr, k=1)


def _whitened(blocks: GWBlocks) -> tuple[Array, Array]:
    """PSD-whitened per-pulsar products ``uk = √Φ·kv`` and ``D = √Φ·km·√Φ``."""
    sqrt_phi = jnp.sqrt(blocks.psd)
    uk = sqrt_phi[None, :] * blocks.basis_proj_residual
    D = sqrt_phi[None, :, None] * blocks.basis_overlap * sqrt_phi[None, None, :]
    return uk, D


def _real_pair_products(blocks: GWBlocks) -> tuple[Array, Array, Array, Array]:
    """Per-pair ``ts_ab = uk_a·uk_b``, ``bs_ab = tr(D_a D_b)`` and pair indices."""
    uk, D = _whitened(blocks)
    ia, ib = _pair_indices(uk.shape[0])
    ts = jnp.sum(uk[ia] * uk[ib], axis=1)
    # bs_ab = tr(D_a D_b) = Σ_kl D_a[k,l] D_b[l,k].
    bs = jnp.sum(D[ia] * jnp.swapaxes(D[ib], -1, -2), axis=(1, 2))
    return ts, bs, ia, ib


def _complex_pair_products(
    blocks: GWBlocks,
) -> tuple[Array, Array, Array, Array]:
    """Per-pair, per-frequency complex ``ts`` and per-pair real ``bs``.

    On the interleaved basis, frequency ``f`` occupies columns
    ``[2f, 2f+1] = (sin, cos)``, so its complex per-frequency amplitude is
    ``tsf_a[f] = √Φ_f (kv_sin + i kv_cos)`` and
    ``ts_complex[ab, f] = tsf_a[f] · conj(tsf_b[f])``.  Summing ``Re(ts_complex)``
    over frequency recovers the real ``ts`` of :func:`_real_pair_products`.
    """
    kv = blocks.basis_proj_residual
    sqrt_phi = jnp.sqrt(blocks.psd)
    tsf = sqrt_phi[0::2] * (kv[:, 0::2] + 1j * kv[:, 1::2])  # (n_psr, n_freq)
    ia, ib = _pair_indices(kv.shape[0])
    ts_complex = tsf[ia] * jnp.conj(tsf[ib])  # (n_pairs, n_freq)
    _, D = _whitened(blocks)
    bs = jnp.sum(D[ia] * jnp.swapaxes(D[ib], -1, -2), axis=(1, 2))
    return ts_complex, bs, ia, ib


def _orf_pairs_from_positions(
    positions: Float[Array, "n_psr 3"],
    ia: Array,
    ib: Array,
    orf_func: Callable,
) -> Float[Array, " n_pairs"]:
    """ORF weight per pulsar pair from sky ``positions``."""
    return jax.vmap(lambda a, b: orf_func(positions[a], positions[b]))(ia, ib)


def _combine(ts: Array, bs: Array, orf_pairs: Array, gwnorm: Array) -> OptimalStatistic:
    """OS combination from per-pair ``ts``, ``bs``, ORF weights and ``gwnorm``."""
    rho = gwnorm * ts / bs
    sigma = gwnorm / jnp.sqrt(bs)
    inv_var = 1.0 / sigma**2
    denom = jnp.sum(orf_pairs**2 * inv_var)
    a_squared = jnp.sum(rho * orf_pairs * inv_var) / denom
    a_squared_sigma = 1.0 / jnp.sqrt(denom)
    return OptimalStatistic(a_squared, a_squared_sigma, a_squared / a_squared_sigma)


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
    ts, bs, ia, ib = _real_pair_products(blocks)
    gwnorm = 10.0 ** (2.0 * jnp.asarray(log10_A))
    return _combine(ts, bs, blocks.orf_matrix[ia, ib], gwnorm)


def sky_scramble(
    blocks: GWBlocks,
    log10_A: ArrayLike,
    positions: Float[Array, "n_psr 3"],
    orf_func: Callable = hd_orf,
) -> OptimalStatistic:
    """OS with the ORF recomputed from (scrambled) sky ``positions``.

    A sky scramble destroys the geometric ORF correlation between the array
    and a correlated signal while preserving every per-pulsar data product:
    ``ts`` and ``bs`` are unchanged, only the pairwise ORF weights change.
    """
    ts, bs, ia, ib = _real_pair_products(blocks)
    gwnorm = 10.0 ** (2.0 * jnp.asarray(log10_A))
    orf = _orf_pairs_from_positions(positions, ia, ib, orf_func)
    return _combine(ts, bs, orf, gwnorm)


def sky_scramble_snrs(
    blocks: GWBlocks,
    log10_A: ArrayLike,
    key: Array,
    n_scrambles: int,
    orf_func: Callable = hd_orf,
) -> Float[Array, " n_scrambles"]:
    """Sky-scramble null: SNR under ``n_scrambles`` isotropic position draws.

    Score an observed SNR against this background with
    :func:`jaxpint.frequentist.pvalue`.  ``ts``/``bs`` are scramble-independent,
    so they are computed once and only the ORF is redrawn per scramble.
    """
    n_psr = blocks.basis_proj_residual.shape[0]
    ts, bs, ia, ib = _real_pair_products(blocks)
    gwnorm = 10.0 ** (2.0 * jnp.asarray(log10_A))

    def one(k: Array) -> Float[Array, ""]:
        pos = isotropic_positions(k, n_psr)
        orf = _orf_pairs_from_positions(pos, ia, ib, orf_func)
        return _combine(ts, bs, orf, gwnorm).snr

    return jax.vmap(one)(jax.random.split(key, n_scrambles))


def phase_shift(
    blocks: GWBlocks,
    log10_A: ArrayLike,
    phases: Float[Array, "n_psr n_freq"],
) -> OptimalStatistic:
    """OS with per-pulsar, per-frequency phase rotations.

    ``phases`` has shape ``(n_psr, n_freq)``.  Each pulsar's per-frequency
    complex amplitude is rotated by ``exp(i φ)``, destroying inter-pulsar phase
    coherence while preserving each pulsar's spectrum (``bs`` is unchanged).
    ``phases = 0`` reproduces :func:`optimal_statistic`.
    """
    ts_complex, bs, ia, ib = _complex_pair_products(blocks)
    phaseprod = jnp.exp(1j * (phases[ia] - phases[ib]))  # (n_pairs, n_freq)
    ts = jnp.sum(jnp.real(ts_complex * phaseprod), axis=1)
    gwnorm = 10.0 ** (2.0 * jnp.asarray(log10_A))
    return _combine(ts, bs, blocks.orf_matrix[ia, ib], gwnorm)


def phase_shift_snrs(
    blocks: GWBlocks,
    log10_A: ArrayLike,
    key: Array,
    n_shifts: int,
) -> Float[Array, " n_shifts"]:
    """Phase-shift null: SNR under ``n_shifts`` random per-pulsar per-frequency
    phase draws.  The NANOGrav phase-shift background; score with
    :func:`jaxpint.frequentist.pvalue`."""
    n_psr, n_col = blocks.basis_proj_residual.shape
    n_freq = n_col // 2
    ts_complex, bs, ia, ib = _complex_pair_products(blocks)
    orf = blocks.orf_matrix[ia, ib]
    gwnorm = 10.0 ** (2.0 * jnp.asarray(log10_A))

    def one(k: Array) -> Float[Array, ""]:
        phases = jax.random.uniform(k, (n_psr, n_freq), minval=0.0, maxval=2.0 * jnp.pi)
        phaseprod = jnp.exp(1j * (phases[ia] - phases[ib]))
        ts = jnp.sum(jnp.real(ts_complex * phaseprod), axis=1)
        return _combine(ts, bs, orf, gwnorm).snr

    return jax.vmap(one)(jax.random.split(key, n_shifts))


# ---------------------------------------------------------------------------
# analytic generalized-χ² null
# ---------------------------------------------------------------------------


def os_quadratic_form(blocks: GWBlocks) -> Float[Array, "cnt cnt"]:
    """The matrix ``Q`` for which the OS detection statistic is ``xᵀ Q x``.

    Under the null (whitened data ``x ~ N(0, I)``) the single-component OS is a
    quadratic form ``xᵀ Q x`` whose distribution is a generalized χ² fixed by
    the eigenvalues of ``Q`` — the analytic alternative to the empirical
    scramble / phase-shift nulls.  ``Q`` is amplitude-independent (the
    ``gwnorm`` factors cancel; cf. discovery ``OS.Q``).

    ``km`` is ill-conditioned, so its Cholesky uses the same ridge
    (``1e-10·tr(km)/n``) discovery applies; the Phase-0 symmetrization keeps it
    a valid symmetric input.
    """
    km = blocks.basis_overlap
    sqrt_phi = jnp.sqrt(blocks.psd)
    n_psr, n_basis, _ = km.shape

    ridge = 1e-10 * jnp.trace(km, axis1=1, axis2=2) / n_basis
    eye = jnp.eye(n_basis)
    chol = jnp.linalg.cholesky(km + ridge[:, None, None] * eye[None])
    a_scaled = sqrt_phi[None, :, None] * chol  # √Φ · chol(km_i), per pulsar

    D = sqrt_phi[None, :, None] * km * sqrt_phi[None, None, :]
    ia, ib = _pair_indices(n_psr)
    bs = jnp.sum(D[ia] * jnp.swapaxes(D[ib], -1, -2), axis=(1, 2))
    orf_pairs = blocks.orf_matrix[ia, ib]
    denom = 2.0 * jnp.sqrt(jnp.sum(orf_pairs**2 * bs))

    cnt = n_psr * n_basis
    Q = jnp.zeros((cnt, cnt))
    for i, j, w in zip([int(a) for a in ia], [int(b) for b in ib], orf_pairs):
        block = w * (a_scaled[i].T @ a_scaled[j])
        si = slice(i * n_basis, (i + 1) * n_basis)
        sj = slice(j * n_basis, (j + 1) * n_basis)
        Q = Q.at[si, sj].add(block)
        Q = Q.at[sj, si].add(block.T)
    return Q / denom


@jax.jit
def _imhof_integrand(
    u: Float[Array, ""], x: Float[Array, ""], eigs: Float[Array, " k"]
) -> Float[Array, ""]:
    """Imhof (1961) integrand for the generalized-χ² CDF of ``Σ_k eigs_k z_k²``.
    https://www.rdocumentation.org/packages/CompQuadForm/versions/1.4.4/topics/imhof"""
    theta = 0.5 * jnp.sum(jnp.arctan(eigs * u)) - 0.5 * x * u
    rho = jnp.prod((1.0 + (eigs * u) ** 2) ** 0.25)
    return jnp.sin(theta) / (u * rho)


def gx2_cdf(
    values: ArrayLike,
    eigs: Float[Array, " cnt"],
    cutoff: float = 1e-6,
    limit: int = 100,
    epsabs: float = 1e-6,
) -> np.ndarray:
    """Generalized-χ² CDF ``P(xᵀ Q x ≤ value)`` from ``Q``'s eigenvalues.

    Imhof's method via host-side ``scipy.integrate.quad`` (one integral per
    value; **not** jit-traceable).  ``eigs`` are typically
    ``jnp.linalg.eigvalsh(os_quadratic_form(blocks))``; near-zero eigenvalues
    (below ``cutoff``) are dropped for a well-behaved integrand.
    """
    eigs = jnp.asarray(eigs)
    eigs = eigs[jnp.abs(eigs) > cutoff]
    out = []
    for v in np.atleast_1d(np.asarray(values, dtype=float)):
        integral, _ = scipy.integrate.quad(
            lambda u: float(_imhof_integrand(u, v, eigs)),
            0.0,
            np.inf,
            limit=limit,
            epsabs=epsabs,
        )
        out.append(0.5 - integral / np.pi)
    return np.array(out)


def os_null_cdf(blocks: GWBlocks, values: ArrayLike, **kwargs) -> np.ndarray:
    """Analytic null CDF of the OS statistic at ``values`` (convenience).

    ``os_null_cdf(blocks, snr) ≈ P(OS ≤ snr | no signal)``; the one-sided
    p-value is ``1 - os_null_cdf(...)``.  Combines
    :func:`os_quadratic_form` + :func:`gx2_cdf`.
    """
    eigs = jnp.linalg.eigvalsh(os_quadratic_form(blocks))
    return gx2_cdf(values, eigs, **kwargs)
