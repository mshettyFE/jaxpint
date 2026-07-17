"""Per-pair outputs of the optimal statistic: rho_ab, angles, HD binning.

Three layers:

1. Exact identities — ``pair_correlations`` recombines into
   ``optimal_statistic`` bit-for-bit; angles and binning match hand
   computations.
2. Pure-function unit tests of ``bin_pair_correlations`` on synthetic
   values (weighted means, empty bins).
3. A signal-injection recovery test: with an HD-correlated signal drawn at
   ``κ×`` the fiducial power, ``E[Â²] = κ·gwnorm`` and the draw-averaged
   ``rho_ab`` traces ``κ·gwnorm·Γ(ξ_ab)`` — pairwise and binned.  This is
   exact for the *mean* (no weak-signal approximation), so the tolerances
   are pure Monte-Carlo standard errors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.pta.likelihood import GWBlocks
from jaxpint._psd import powerlaw_psd
from jaxpint.pta.signals.gwb import fourier_basis
from jaxpint.pta.signals.orf import hd_orf
from jaxpint.frequentist.optimal import (
    BinnedPairCorrelations,
    PairCorrelations,
    bin_pair_correlations,
    combine_pair_correlations,
    optimal_statistic,
    pair_angles,
    pair_correlations,
)

jax.config.update("jax_enable_x64", True)

T_SPAN = 365.25 * 86400.0
LOG10_A = -14.0
GWNORM = 10.0 ** (2.0 * LOG10_A)


def _model(seed=1, n_psr=6, n_freq=4, n_toa=60, kappa=0.0):
    """Noise-only (or signal-injected) OS model, as in ``test_optimal_nulls``.

    ``kappa`` scales the injected HD-correlated signal power relative to the
    fiducial PSD ``Φ``: the signal Fourier coefficients are drawn with
    ``Cov(s_a[k], s_b[l]) = κ Γ_ab Φ_k δ_kl``, so ``E[Â²] = κ·gwnorm``
    exactly.  ``kappa = 0`` reproduces the pure null model.
    """
    kp, kc = jax.random.split(jax.random.PRNGKey(seed))
    pos = jax.random.normal(kp, (n_psr, 3))
    pos = pos / jnp.linalg.norm(pos, axis=1, keepdims=True)

    tt = jnp.linspace(0.0, T_SPAN, n_toa)
    F, freqs = fourier_basis(tt, n_freq, T_SPAN)
    c = (1e-6**2) * (1.0 + jax.random.uniform(kc, (n_toa,)))
    c_inv = 1.0 / c
    km = F.T @ (c_inv[:, None] * F)
    km = 0.5 * (km + km.T)
    Phi = jnp.repeat(powerlaw_psd(freqs, LOG10_A, 4.33) / T_SPAN, 2)
    Gamma = jnp.array(
        [[float(hd_orf(pos[a], pos[b])) for b in range(n_psr)] for a in range(n_psr)]
    )
    km_batched = jnp.broadcast_to(km, (n_psr,) + km.shape)
    n_basis = 2 * n_freq
    # Cross-covariances come out as Γ_ab exactly even with the diagonal
    # jitter (it only inflates the auto terms, which the OS never uses).
    L = jnp.linalg.cholesky(Gamma + 1e-10 * jnp.eye(n_psr))

    def draw(key: jax.Array) -> GWBlocks:
        kn, ks = jax.random.split(key)
        noise = jnp.sqrt(c)[None, :] * jax.random.normal(kn, (n_psr, n_toa))
        resid = noise
        if kappa > 0.0:
            z = jax.random.normal(ks, (n_psr, n_basis))
            s = jnp.sqrt(kappa * Phi)[None, :] * (L @ z)  # (n_psr, n_basis)
            resid = resid + s @ F.T
        kv = jnp.einsum("tf,pt->pf", F, resid * c_inv[None, :])
        return GWBlocks(
            basis_proj_residual=kv,
            basis_overlap=km_batched,
            psd=Phi,
            orf_matrix=Gamma,
        )

    return draw, pos, Gamma


# ---------------------------------------------------------------------------
# Exact identities
# ---------------------------------------------------------------------------


def test_pair_recombination_matches_os():
    """combine_pair_correlations ∘ pair_correlations ≡ optimal_statistic."""
    draw, _, _ = _model()
    blocks = draw(jax.random.PRNGKey(2))
    pairs = pair_correlations(blocks, LOG10_A)
    combined = combine_pair_correlations(pairs)
    ref = optimal_statistic(blocks, LOG10_A)
    # Identical operations on identical values — no tolerance.
    npt.assert_array_equal(np.asarray(combined.a_squared), np.asarray(ref.a_squared))
    npt.assert_array_equal(
        np.asarray(combined.a_squared_sigma), np.asarray(ref.a_squared_sigma)
    )
    npt.assert_array_equal(np.asarray(combined.snr), np.asarray(ref.snr))


def test_pair_fields_shapes_and_order():
    """Pair indexing is triu order; orf matches the blocks' ORF matrix."""
    n_psr = 6
    draw, _, Gamma = _model(n_psr=n_psr)
    pairs = pair_correlations(draw(jax.random.PRNGKey(5)), LOG10_A)
    n_pairs = n_psr * (n_psr - 1) // 2
    for field in pairs:
        assert field.shape == (n_pairs,)
    ia, ib = np.asarray(pairs.pulsar_a), np.asarray(pairs.pulsar_b)
    assert (ia < ib).all()
    ref_ia, ref_ib = np.triu_indices(n_psr, k=1)
    npt.assert_array_equal(ia, ref_ia)
    npt.assert_array_equal(ib, ref_ib)
    npt.assert_array_equal(np.asarray(pairs.orf), np.asarray(Gamma)[ia, ib])
    assert np.all(np.asarray(pairs.sigma) > 0)


def test_pair_angles_known_geometry():
    """Angles for hand-picked positions, in pair order (0,1),(0,2),(1,2)."""
    pos = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],  # ⊥ to the first
            [-1.0, 0.0, 0.0],  # antipodal to the first
        ]
    )
    xi = np.asarray(pair_angles(pos))
    npt.assert_allclose(xi, [np.pi / 2, np.pi, np.pi / 2], atol=1e-12)


def test_pair_angles_match_orf():
    """hd_orf evaluated at pair_angles reproduces the pair ORF weights."""
    draw, pos, _ = _model()
    pairs = pair_correlations(draw(jax.random.PRNGKey(6)), LOG10_A)
    xi = pair_angles(pos)
    # HD as a function of angle: x = (1 - cos ξ) / 2.
    x = (1.0 - jnp.cos(xi)) / 2.0
    hd = 1.5 * x * jnp.log(x) - x / 4.0 + 0.5
    npt.assert_allclose(np.asarray(pairs.orf), np.asarray(hd), rtol=1e-10)


# ---------------------------------------------------------------------------
# Binning (pure function)
# ---------------------------------------------------------------------------


def _pairs_from(rho, sigma):
    n = len(rho)
    ia = jnp.zeros(n, dtype=jnp.int32)
    return PairCorrelations(
        ia, ia + 1, jnp.asarray(rho), jnp.asarray(sigma), jnp.zeros(n)
    )

def test_bin_weighted_means_exact():
    """Two bins over [0, π]: hand-computed inverse-variance weighted means."""
    angles = jnp.array([0.1, 0.8, 2.0])
    rho = [1.0, 3.0, 5.0]
    sigma = [1.0, 0.5, 2.0]
    out = bin_pair_correlations(angles, _pairs_from(rho, sigma), n_bins=2)
    assert isinstance(out, BinnedPairCorrelations)
    # Bin 0 = [0, π/2): pairs 0 and 1, weights 1 and 4.
    npt.assert_allclose(float(out.rho[0]), (1.0 * 1 + 4 * 3.0) / 5.0, rtol=1e-14)
    npt.assert_allclose(float(out.angle[0]), (1 * 0.1 + 4 * 0.8) / 5.0, rtol=1e-14)
    npt.assert_allclose(float(out.sigma[0]), 1.0 / np.sqrt(5.0), rtol=1e-14)
    # Bin 1 = [π/2, π]: pair 2 alone.
    npt.assert_allclose(float(out.rho[1]), 5.0, rtol=1e-14)
    npt.assert_allclose(float(out.sigma[1]), 2.0, rtol=1e-14)
    npt.assert_array_equal(np.asarray(out.n_pairs), [2, 1])


def test_bin_empty_bins_and_edges():
    """Empty bins are nan/inf/0; ξ = 0 and ξ = π land in the end bins."""
    angles = jnp.array([0.0, jnp.pi])
    out = bin_pair_correlations(
        angles, _pairs_from([2.0, 4.0], [1.0, 1.0]), n_bins=4
    )
    npt.assert_array_equal(np.asarray(out.n_pairs), [1, 0, 0, 1])
    assert np.isnan(np.asarray(out.rho)[1:3]).all()
    assert np.isnan(np.asarray(out.angle)[1:3]).all()
    assert np.isinf(np.asarray(out.sigma)[1:3]).all()
    npt.assert_allclose(float(out.rho[0]), 2.0)
    npt.assert_allclose(float(out.rho[3]), 4.0)


# ---------------------------------------------------------------------------
# Signal recovery: E[Â²] and the HD curve
# ---------------------------------------------------------------------------


KAPPA = 100.0
N_DRAWS = 3000


@pytest.fixture(scope="module")
def signal_draws():
    """Shared Monte-Carlo ensemble: the three signal tests use the same draws."""
    draw, pos, Gamma = _model(kappa=KAPPA)
    keys = jax.random.split(jax.random.PRNGKey(30), N_DRAWS)

    def one(k):
        b = draw(k)
        return pair_correlations(b, LOG10_A), optimal_statistic(b, LOG10_A)

    pairs_all, os_all = jax.jit(jax.vmap(one))(keys)
    return pairs_all, os_all, pos, Gamma


def test_signal_a_squared_unbiased(signal_draws):
    """mean(Â²) over draws = κ·gwnorm within Monte-Carlo error (exact identity)."""
    _, os_all, _, _ = signal_draws
    a2 = np.asarray(os_all.a_squared)
    se = a2.std(ddof=1) / np.sqrt(N_DRAWS)
    expected = KAPPA * GWNORM
    assert expected > 8 * se, "signal too weak for a meaningful mean test"
    assert abs(a2.mean() - expected) < 4 * se, (
        f"mean Â² = {a2.mean():.3e}, expected {expected:.3e} ± {se:.1e}"
    )


def test_signal_pair_rho_traces_hd(signal_draws):
    """Draw-averaged rho_ab = κ·gwnorm·Γ_ab within 5 SE, per pair."""
    pairs_all, _, _, Gamma = signal_draws
    rho = np.asarray(pairs_all.rho)  # (N_DRAWS, n_pairs)
    mean_rho = rho.mean(axis=0)
    se = rho.std(axis=0, ddof=1) / np.sqrt(N_DRAWS)
    ia, ib = np.asarray(pairs_all.pulsar_a[0]), np.asarray(pairs_all.pulsar_b[0])
    expected = KAPPA * GWNORM * np.asarray(Gamma)[ia, ib]
    dev = np.abs(mean_rho - expected) / se
    assert dev.max() < 5.0, f"pair deviations (SE units): {dev}"
    # Non-vacuous: the HD signal must actually be resolved in most pairs.
    assert (np.abs(expected) / se > 3).mean() > 0.5


def test_signal_binned_rho_traces_hd(signal_draws):
    """Binned draw-averaged rho matches the identically-binned κ·gwnorm·Γ(ξ)."""
    pairs_all, _, pos, Gamma = signal_draws
    rho = np.asarray(pairs_all.rho)
    mean_rho = jnp.asarray(rho.mean(axis=0))
    se = jnp.asarray(rho.std(axis=0, ddof=1) / np.sqrt(N_DRAWS))
    ia = pairs_all.pulsar_a[0]
    ib = pairs_all.pulsar_b[0]
    orf = pairs_all.orf[0]

    xi = pair_angles(pos)
    averaged = PairCorrelations(ia, ib, mean_rho, se, orf)
    out = bin_pair_correlations(xi, averaged, n_bins=4)

    # Expected bin values: the same inverse-variance weighting applied to
    # the exact per-pair means κ·gwnorm·Γ_ab.
    expected_pairs = PairCorrelations(ia, ib, KAPPA * GWNORM * orf, se, orf)
    expected = bin_pair_correlations(xi, expected_pairs, n_bins=4)

    valid = np.asarray(out.n_pairs) > 0
    assert valid.sum() >= 3, "fixture should populate at least 3 angle bins"
    dev = np.abs(np.asarray(out.rho) - np.asarray(expected.rho))[valid]
    tol = 4.0 * np.asarray(out.sigma)[valid]
    assert (dev < tol).all(), f"binned deviations {dev} vs tolerances {tol}"
