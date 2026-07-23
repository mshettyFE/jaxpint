"""Phase 3/4 of the optimal-statistic port: empirical + analytic null distros.

The key fixture builds *proper noise-only* blocks — per-pulsar residuals drawn
``~ N(0, C)`` — so the OS null is genuinely ``N(0, 1)`` and the analytic
generalized-χ² null  can be checked against the empirical OS null and
the empirical scramble / phase-shift nulls .

Optimal statistic: Anholm et al. (2009, arXiv:0809.0701); Chamberlin et al.
(2015, arXiv:1410.8256).  Imhof's generalized-χ² CDF: Imhof (1961), Biometrika
48, 419.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxpint.pta.likelihood import GWBlocks
from jaxpint._psd import powerlaw_psd
from jaxpint.pta.signals.gwb import fourier_basis
from jaxpint.pta.signals.orf import hd_orf
from jaxpint.frequentist.optimal import (
    gx2_cdf,
    optimal_statistic,
    os_quadratic_form,
    phase_shift,
    phase_shift_snrs,
    sky_scramble,
)
from jaxpint.frequentist import pvalue


jax.config.update("jax_enable_x64", True)

T_SPAN = 365.25 * 86400.0
LOG10_A = -14.0


def _null_model(seed=1, n_psr=5, n_freq=4, n_toa=60):
    """A fixed noise-only OS model + a ``draw(key)`` for one null realization.

    ``C_p = diag(c)`` (white), shared across pulsars; residuals ``n_p ~ N(0, C)``
    give ``kv_p = FᵀC⁻¹ n_p`` with ``Cov(kv_p) = FᵀC⁻¹F = km`` — so the OS built
    from these blocks is a genuine draw from the null.  ``km``, ``Φ``, ``Γ`` (and
    hence ``Q``) are the same across draws; only ``kv`` (the noise) varies.
    """
    kp, kc = jax.random.split(jax.random.PRNGKey(seed))
    pos = jax.random.normal(kp, (n_psr, 3))
    pos = pos / jnp.linalg.norm(pos, axis=1, keepdims=True)

    tt = jnp.linspace(0.0, T_SPAN, n_toa)
    F, freqs = fourier_basis(tt, n_freq, T_SPAN)
    c = (1e-6**2) * (1.0 + jax.random.uniform(kc, (n_toa,)))  # white var / TOA
    c_inv = 1.0 / c
    km = F.T @ (c_inv[:, None] * F)
    km = 0.5 * (km + km.T)
    Phi = jnp.repeat(powerlaw_psd(freqs, LOG10_A, 4.33) / T_SPAN, 2)
    Gamma = jnp.array(
        [[float(hd_orf(pos[a], pos[b])) for b in range(n_psr)] for a in range(n_psr)]
    )
    km_batched = jnp.broadcast_to(km, (n_psr,) + km.shape)

    def draw(key: jax.Array) -> GWBlocks:
        z = jax.random.normal(key, (n_psr, n_toa))
        noise = jnp.sqrt(c)[None, :] * z  # Cov = diag(c)
        kv = jnp.einsum("tf,pt->pf", F, noise * c_inv[None, :])
        return GWBlocks(
            basis_proj_residual=kv,
            basis_overlap=km_batched,
            psd=Phi,
            orf_matrix=Gamma,
        )

    return draw, pos


# ---------------------------------------------------------------------------
# Consistency identities (data-agnostic)
# ---------------------------------------------------------------------------


def test_sky_scramble_true_positions_is_the_os():
    """Scrambling with the ACTUAL positions reproduces the plain OS."""
    draw, pos = _null_model()
    blocks = draw(jax.random.PRNGKey(3))
    ref = optimal_statistic(blocks, LOG10_A)
    out = sky_scramble(blocks, LOG10_A, pos, orf_func=hd_orf)
    np.testing.assert_allclose(float(out.snr), float(ref.snr), rtol=1e-10)
    np.testing.assert_allclose(
        float(out.a_squared), float(ref.a_squared), rtol=1e-10
    )


def test_phase_shift_zero_is_the_os():
    """Zero phase shift reproduces the plain OS (complex path ≡ real path)."""
    draw, _ = _null_model()
    blocks = draw(jax.random.PRNGKey(4))
    ref = optimal_statistic(blocks, LOG10_A)
    n_psr, n_col = blocks.basis_proj_residual.shape
    out = phase_shift(blocks, LOG10_A, jnp.zeros((n_psr, n_col // 2)))
    np.testing.assert_allclose(float(out.snr), float(ref.snr), rtol=1e-10)


# ---------------------------------------------------------------------------
# The OS null really is N(0, 1) under proper noise
# ---------------------------------------------------------------------------


def test_os_null_is_standard_normal():
    """OS SNR over many noise draws has mean ≈ 0, std ≈ 1."""
    draw, _ = _null_model()
    keys = jax.random.split(jax.random.PRNGKey(11), 4000)
    snrs = np.array([float(optimal_statistic(draw(k), LOG10_A).snr) for k in keys])
    assert abs(snrs.mean()) < 0.05
    assert abs(snrs.std() - 1.0) < 0.05


# ---------------------------------------------------------------------------
# the analytic generalized-χ² null
# ---------------------------------------------------------------------------


def test_gx2_matches_monte_carlo_quadratic_form():
    """``gx2_cdf`` is the CDF of ``xᵀQx`` for ``x ~ N(0, I)`` (pure Imhof check).

    Independent of any OS semantics: draw Gaussian ``x``, evaluate the quadratic
    form directly, and compare the empirical CDF to the analytic one.
    """
    draw, _ = _null_model()
    Q = np.asarray(os_quadratic_form(draw(jax.random.PRNGKey(0))))
    eigs = np.linalg.eigvalsh(Q)

    x = np.asarray(jax.random.normal(jax.random.PRNGKey(21), (6000, Q.shape[0])))
    q = np.einsum("mi,ij,mj->m", x, Q, x)
    grid = np.percentile(q, [5, 25, 50, 75, 95])
    emp = np.array([(q <= v).mean() for v in grid])
    ana = gx2_cdf(grid, jnp.asarray(eigs))
    assert np.max(np.abs(emp - ana)) < 0.03


def test_gx2_matches_empirical_os_null():
    """Analytic GX2 CDF matches the empirical OS null distribution.

    Ties Imhof to the actual OS statistic: the OS SNR over
    noise draws should follow ``gx2_cdf(eigvalsh(Q))``.
    """
    draw, _ = _null_model()
    keys = jax.random.split(jax.random.PRNGKey(13), 4000)
    snrs = np.array([float(optimal_statistic(draw(k), LOG10_A).snr) for k in keys])
    eigs = jnp.linalg.eigvalsh(os_quadratic_form(draw(keys[0])))
    grid = np.percentile(snrs, [5, 25, 50, 75, 95])
    emp = np.array([(snrs <= v).mean() for v in grid])
    ana = gx2_cdf(grid, eigs)
    assert np.max(np.abs(emp - ana)) < 0.03


# ---------------------------------------------------------------------------
#  empirical nulls agree with the analytic null
# ---------------------------------------------------------------------------


def test_phase_shift_null_matches_gx2():
    """The phase-shift null (one noise draw) matches the analytic GX2 null.

    The phase-shift background resamples inter-pulsar phase coherence; on
    noise-only data its SNR distribution is the OS null, so it must track
    ``gx2_cdf(eigvalsh(Q))``.
    """
    draw, _ = _null_model()
    blocks = draw(jax.random.PRNGKey(15))
    shifted = np.asarray(phase_shift_snrs(blocks, LOG10_A, jax.random.PRNGKey(2), 4000))
    eigs = jnp.linalg.eigvalsh(os_quadratic_form(blocks))
    grid = np.percentile(shifted, [5, 25, 50, 75, 95])
    emp = np.array([(shifted <= v).mean() for v in grid])
    ana = gx2_cdf(grid, eigs)
    assert np.max(np.abs(emp - ana)) < 0.05


def test_pvalue_of_injected_hd_signal_is_small():
    """An HD-correlated injection reads as a small phase-shift-null p-value.

    A monopole (identical common mode) would be *cancelled* by the HD weights
    (Σ Γ_ab ≈ 0), so inject a genuinely HD-correlated component: per basis
    column, add per-pulsar amplitudes with cross-pulsar covariance ∝ Γ (via
    ``chol(Γ)``), scaled to the ``kv`` magnitude.  Its coherent HD structure
    lifts the OS SNR into the tail of its own phase-shift null.
    """
    draw, _ = _null_model(n_psr=8)
    blocks = draw(jax.random.PRNGKey(17))
    n_psr, n_col = blocks.basis_proj_residual.shape

    # Cross-pulsar amplitudes with covariance ∝ Γ: s[:, col] = chol(Γ) @ z_col.
    gamma = np.asarray(blocks.orf_matrix)
    w, V = np.linalg.eigh(gamma)
    chol_gamma = V @ np.diag(np.sqrt(np.clip(w, 1e-6, None)))  # Γ ≈ L Lᵀ
    z = np.asarray(jax.random.normal(jax.random.PRNGKey(23), (n_col, n_psr)))
    hd = jnp.asarray((z @ chol_gamma.T).T)  # (n_psr, n_col), Cov_pulsars ∝ Γ

    kv_rms = jnp.sqrt(jnp.mean(blocks.basis_proj_residual**2))
    kv_sig = blocks.basis_proj_residual + 1.5 * kv_rms * hd
    signal = blocks._replace(basis_proj_residual=kv_sig)

    obs = float(optimal_statistic(signal, LOG10_A).snr)
    background = phase_shift_snrs(signal, LOG10_A, jax.random.PRNGKey(1), 2000)
    p = pvalue(obs, background)
    assert obs > 0
    assert p < 0.05, f"p={p}"
