"""Conditional GP posteriors: dense references, sampling, reconstruction.

The dense references use the *dual* Gaussian-conditioning form,

    \hat{a} = A F^{T} C_tot^{-1} r,   \Sigma = A − A F^{T} C_tot^{-1} F A,
    C_tot = C_noise + F A F^{T},   A = prior coefficient covariance,

which is algebraically different from the implementation's precision form
``P = A^{-1} + F^{T} C_noise^{-1} F``

see https://scoste.fr/posts/schur/ for math
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.fitters import compute_time_residuals
from jaxpint.noise import NoiseModel, PLRedNoise, ScaleToaError
from jaxpint.pta import (
    CURNInjector,
    HDCorrelatedGWBInjector,
    PTAConfig,
    conditional_covariance,
    conditional_gwb,
    conditional_gwb_delays,
    conditional_single_pulsar,
    sample_conditional,
)
from jaxpint.types import GlobalParams
from jaxpint.simulation import apply_delay_to_toas, make_fake_toas

from tests.helpers import make_params, make_simple_pulsar

jax.config.update("jax_enable_x64", True)

T_SPAN = 1e8
N_IRN = 8  # per-pulsar red-noise bins
N_GW = 5  # correlated-signal bins
LOG10_A = -13.0
GAMMA = 4.33


def _interleaved_basis(toas_seconds, n_freq):
    freqs = np.arange(1, n_freq + 1) / T_SPAN
    phase = 2.0 * np.pi * np.asarray(toas_seconds)[:, None] * freqs[None, :]
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1).reshape(
        -1, 2 * n_freq
    )


@functools.lru_cache(maxsize=None)
def _pulsar_with_red_noise(p: int):
    """One pulsar with white + 8-bin power-law red noise (asymmetric per i).

    ``make_simple_pulsar`` puts all TOAs inside one day; here they are
    respread over the full ``T_SPAN`` so the 1..N/T Fourier modes are
    actually resolved (required by the reconstruction test; harmless for
    the algebraic ones).
    """
    import equinox as eqx

    td, tm, _nm, pp = make_simple_pulsar(
        n_toas=25 + 5 * p, f0=200.0 + 10.0 * p, f1=-1e-15, seed=61 + p
    )
    t_days = np.linspace(0.0, T_SPAN / 86400.0, td.n_toas)
    t_int = jnp.asarray(59000.0 + np.floor(t_days))
    t_frac = jnp.asarray(t_days - np.floor(t_days))
    td = eqx.tree_at(
        lambda t: (t.tdb_int, t.tdb_frac, t.mjd_int, t.mjd_frac),
        td,
        (t_int, t_frac, t_int, t_frac),
    )
    F_irn = _interleaved_basis(td.tdb_seconds, N_IRN)
    freqs_irn = np.arange(1, N_IRN + 1) / T_SPAN
    comp = PLRedNoise(
        fourier_basis=jnp.asarray(F_irn),
        freqs=jnp.asarray(freqs_irn),
        freq_bin_widths=jnp.full(N_IRN, 1.0 / T_SPAN),
        tnredamp_name="TNREDAMP",
        tnredgam_name="TNREDGAM",
    )
    pp = make_params(
        names=pp.names + ("TNREDAMP", "TNREDGAM"),
        values=list(np.asarray(pp.values)) + [-13.2 - 0.3 * p, 3.0 + 0.8 * p],
        frozen_mask=pp.frozen_mask + (True, True),
        epoch_int_values=pp.epoch_int_values,
    )
    white = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
    nm = NoiseModel(white_noise=white, correlated=(comp,))
    return td, tm, nm, pp


@functools.lru_cache(maxsize=None)
def _hd_config(log10_A: float = LOG10_A):
    """Two-pulsar HD-correlated setup (memoized; see _pulsar_with_red_noise)."""
    tds, tms, nms, pps = zip(*[_pulsar_with_red_noise(i) for i in range(2)])
    rng = np.random.default_rng(5)
    pos = rng.normal(size=(2, 3))
    pos = jnp.asarray(pos / np.linalg.norm(pos, axis=1, keepdims=True))
    inj = HDCorrelatedGWBInjector(
        pos, N_GW, T_SPAN, initial_values={"log10_A": log10_A, "gamma": GAMMA}
    )
    config = PTAConfig(
        toa_data_list=tuple(tds),
        timing_models=tuple(tms),
        noise_models=tuple(nms),
        signal_injectors=(),
        correlated_injectors=(inj,),
    )
    gp = inj.register_params(GlobalParams.empty())
    return gp, tuple(pps), config, inj


def _dense_dual_conditional(A, F, C_noise, r):
    """Dual-form dense reference: \hat{a} and \Sigma from ``C_tot = C_noise + F A F^{T}``."""
    C_tot = C_noise + F @ A @ F.T
    AFt = A @ F.T
    mean = AFt @ np.linalg.solve(C_tot, r)
    cov = A - AFt @ np.linalg.solve(C_tot, AFt.T)
    return mean, cov


# ---------------------------------------------------------------------------
# Dense references
# ---------------------------------------------------------------------------


def test_single_pulsar_matches_dual_dense():
    """Per-pulsar conditional (red noise + external CURN block) vs dual form."""
    td, tm, nm, pp = _pulsar_with_red_noise(0)
    curn = CURNInjector(
        N_GW, T_SPAN, initial_values={"log10_A": LOG10_A, "gamma": GAMMA}
    )
    gp = curn.register_params(GlobalParams.empty())
    ext_cov = curn.covariance(0, td, pp, gp)

    cond = conditional_single_pulsar(td, tm, nm, pp, external_cov=ext_cov)

    # Dual-form dense reference from the same raw ingredients.
    r = np.asarray(compute_time_residuals(tm, td, pp))
    Ndiag, U_n, Phi_n = nm.covariance(td, pp)
    U = np.concatenate([np.asarray(U_n), np.asarray(ext_cov[0])], axis=1)
    Phi = np.concatenate([np.asarray(Phi_n), np.asarray(ext_cov[1])])
    mean_d, cov_d = _dense_dual_conditional(
        np.diag(Phi), U, np.diag(np.asarray(Ndiag)), r
    )

    assert cond.mean.shape == (2 * N_IRN + 2 * N_GW,)
    npt.assert_allclose(np.asarray(cond.mean), mean_d, rtol=1e-9)
    npt.assert_allclose(
        np.asarray(conditional_covariance(cond)), cov_d, rtol=1e-8, atol=0.0
    )


def test_gwb_conditional_matches_dual_dense():
    """Joint GWB conditional vs the dual form on the full 2-pulsar system."""
    gp, pps, config, inj = _hd_config()
    cond = conditional_gwb(gp, pps, config)

    # Dense ingredients: stacked residuals, block-diagonal per-pulsar noise
    # (white + IRN, densified), block-diagonal GW basis, ORF-coupled prior.
    rs, C_blocks, F_blocks = [], [], []
    for td, tm, nm, pp in zip(
        config.toa_data_list, config.timing_models, config.noise_models, pps
    ):
        rs.append(np.asarray(compute_time_residuals(tm, td, pp)))
        Ndiag, U_n, Phi_n = nm.covariance(td, pp)
        C_blocks.append(
            np.diag(np.asarray(Ndiag))
            + np.asarray(U_n) @ np.diag(np.asarray(Phi_n)) @ np.asarray(U_n).T
        )
        F_blocks.append(np.asarray(inj.get_fourier_basis(td)))

    r = np.concatenate(rs)
    C_noise = np.block(
        [
            [C_blocks[0], np.zeros((len(rs[0]), len(rs[1])))],
            [np.zeros((len(rs[1]), len(rs[0]))), C_blocks[1]],
        ]
    )
    F = np.block(
        [
            [F_blocks[0], np.zeros_like(F_blocks[0])],
            [np.zeros_like(F_blocks[1]), F_blocks[1]],
        ]
    )
    A = np.kron(np.asarray(inj.get_orf_matrix()), np.diag(np.asarray(inj.get_psd(gp))))
    mean_d, cov_d = _dense_dual_conditional(A, F, C_noise, r)

    assert cond.mean.shape == (2 * 2 * N_GW,)
    npt.assert_allclose(np.asarray(cond.mean), mean_d, rtol=1e-9)
    npt.assert_allclose(
        np.asarray(conditional_covariance(cond)), cov_d, rtol=1e-7, atol=0.0
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sample_conditional_moments():
    """Draw ensemble reproduces the mean and the FULL covariance.

    The diagonal (per-component variance) is checked directly; the full
    covariance — including the cross-pulsar correlations that carry the
    HD structure — is checked by whitening: since ``x - mean = L^{-T} z``
    with ``P = L L^T``, left-multiplying by ``L^T`` recovers ``z ~ N(0, I)``,
    so ``Cov((draws - mean) @ L)`` must be the identity.  This is what
    pins :func:`sample_conditional`'s exactness claim, not just its marginals.
    """
    gp, pps, config, _ = _hd_config()
    cond = conditional_gwb(gp, pps, config)
    cov = np.asarray(conditional_covariance(cond))

    # 1500 draws: variance-estimator SE ~ sqrt(2/n) ~ 3.7% vs rtol 0.15
    # (~4 sigma); off-diagonal sample-cov SE ~ 1/sqrt(n) ~ 0.026 vs
    # atol 0.15 (~5.8 sigma).  
    n_draws = 1500
    draws = np.asarray(
        sample_conditional(jax.random.PRNGKey(2), cond, n_draws=n_draws)
    )
    assert draws.shape == (n_draws, cond.mean.shape[0])

    se = np.sqrt(np.diag(cov) / n_draws)
    npt.assert_array_less(
        np.abs(draws.mean(axis=0) - np.asarray(cond.mean)), 5.0 * se
    )
    npt.assert_allclose(draws.var(axis=0, ddof=1), np.diag(cov), rtol=0.15)

    # Full covariance, off-diagonals included: whitened draws must be N(0, I).
    n = cond.mean.shape[0]
    Z = (draws - np.asarray(cond.mean)) @ np.asarray(cond.precision_chol)
    npt.assert_allclose(np.cov(Z, rowvar=False), np.eye(n), atol=0.15)

    single = sample_conditional(jax.random.PRNGKey(3), cond)
    assert single.shape == cond.mean.shape


# ---------------------------------------------------------------------------
# End-to-end: reconstruction of an injected GWB realization
# ---------------------------------------------------------------------------


def test_gwb_reconstruction_recovers_injection():
    """Conditional mean tracks injected coefficients; subtraction whitens.

    Inject a loud HD-correlated realization drawn from the prior; the
    conditional mean must correlate strongly with the injected
    coefficients (sign included), and subtracting the reconstructed
    delays must reduce residual power in every pulsar.
    """
    gp, pps, config, inj = _hd_config(log10_A=-12.5)  # loud: coeffs >> white

    A = np.kron(
        np.asarray(inj.get_orf_matrix()), np.diag(np.asarray(inj.get_psd(gp)))
    )
    L = np.linalg.cholesky(A + 1e-30 * np.eye(A.shape[0]))
    rng = np.random.default_rng(11)
    a_true = L @ rng.standard_normal(A.shape[0])  # (p, b) layout

    white = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
    tds = []
    nb = 2 * N_GW
    for p, (td, tm, pp) in enumerate(
        zip(config.toa_data_list, config.timing_models, pps)
    ):
        fake = make_fake_toas(tm, td, pp, jax.random.PRNGKey(20 + p), [white])
        delay = np.asarray(inj.get_fourier_basis(td)) @ a_true[p * nb : (p + 1) * nb]
        tds.append(apply_delay_to_toas(fake, jnp.asarray(delay)))
    config = PTAConfig(
        toa_data_list=tuple(tds),
        timing_models=config.timing_models,
        noise_models=config.noise_models,
        signal_injectors=(),
        correlated_injectors=(inj,),
    )

    cond = conditional_gwb(gp, pps, config)
    mean = np.asarray(cond.mean)
    corr = np.corrcoef(mean, a_true)[0, 1]
    assert corr > 0.7, f"conditional mean vs injected coefficients: corr={corr:.3f}"

    recon = conditional_gwb_delays(config, cond.mean)
    for p, (td, tm, pp) in enumerate(
        zip(config.toa_data_list, config.timing_models, pps)
    ):
        r = np.asarray(compute_time_residuals(tm, td, pp))
        cleaned = r - np.asarray(recon[p])
        assert cleaned.var() < 0.5 * r.var(), f"pulsar {p} not whitened"


# ---------------------------------------------------------------------------
# Reconstruction bands and arbitrary-time evaluation
# ---------------------------------------------------------------------------


def test_delay_bands_match_dense_and_draws():
    """Band mean/std vs hand-indexed dense propagation AND a draw ensemble.

    The dense check re-derives the (p, b) coefficient indexing by hand
    (K = 1: pulsar p owns the contiguous slice ``p·nb:(p+1)·nb``); the
    ensemble check goes through the completely independent sampling path.
    """
    from jaxpint.pta import conditional_gwb_delay_bands

    gp, pps, config, inj = _hd_config()
    cond = conditional_gwb(gp, pps, config)
    bands = conditional_gwb_delay_bands(config, cond)

    cov = np.asarray(conditional_covariance(cond))
    mean = np.asarray(cond.mean)
    nb = 2 * N_GW
    for p, td in enumerate(config.toa_data_list):
        F = np.asarray(inj.get_fourier_basis(td))
        sl = slice(p * nb, (p + 1) * nb)
        npt.assert_allclose(np.asarray(bands[p].mean), F @ mean[sl], rtol=1e-12)
        var_dense = np.einsum("tb,bc,tc->t", F, cov[sl, sl], F)
        npt.assert_allclose(
            np.asarray(bands[p].std), np.sqrt(var_dense), rtol=1e-10
        )

    # Independent route: pointwise std of delays over posterior draws.
    # 1500 draws: SE(std) ~ 1/sqrt(2n) ~ 1.8% vs rtol 0.15 (~8 sigma).
    draws = sample_conditional(jax.random.PRNGKey(7), cond, n_draws=1500)
    for p in range(2):
        delay_p = jax.vmap(lambda a: conditional_gwb_delays(config, a)[p])(draws)
        npt.assert_allclose(
            np.asarray(delay_p).std(axis=0, ddof=1),
            np.asarray(bands[p].std),
            rtol=0.15,
        )


def test_delay_bands_on_time_grid():
    """Arbitrary-time evaluation: smooth grid, consistent with TOA-epoch path."""
    from jaxpint.pta import conditional_gwb_delay_bands

    gp, pps, config, _ = _hd_config()
    cond = conditional_gwb(gp, pps, config)

    grid = jnp.linspace(0.0, T_SPAN, 200)
    bands = conditional_gwb_delay_bands(config, cond, times_seconds=grid)
    for band in bands:
        assert band.mean.shape == (200,)
        assert band.std.shape == (200,)
        assert bool(jnp.all(jnp.isfinite(band.mean)))
        assert bool(jnp.all(band.std > 0.0))

    # Evaluating explicitly AT the TOA epochs must reproduce the default path.
    default = conditional_gwb_delay_bands(config, cond)
    explicit = conditional_gwb_delay_bands(
        config, cond,
        times_seconds=[td.tdb_seconds for td in config.toa_data_list],
    )
    for d, e in zip(default, explicit):
        npt.assert_allclose(np.asarray(e.mean), np.asarray(d.mean), rtol=1e-12)
        npt.assert_allclose(np.asarray(e.std), np.asarray(d.std), rtol=1e-12)

    # Delays of a coefficient vector on the grid work too.
    delays = jax.tree.map(
        np.asarray, conditional_gwb_delays(config, cond.mean, times_seconds=grid)
    )
    assert all(d.shape == (200,) for d in delays)


# ---------------------------------------------------------------------------
# Transform-compatibility and errors
# ---------------------------------------------------------------------------


def test_conditional_gwb_jit_and_grad():
    """jit agrees with eager; the mean is differentiable w.r.t. the GWB amp."""
    gp, pps, config, _ = _hd_config()

    eager = conditional_gwb(gp, pps, config)
    jitted = jax.jit(lambda g, p: conditional_gwb(g, p, config))(gp, pps)
    npt.assert_allclose(
        np.asarray(jitted.mean), np.asarray(eager.mean), rtol=1e-9
    )

    def summary(v):
        cond = conditional_gwb(gp.with_values(v), pps, config)
        return jnp.sum(cond.mean**2)

    grad = np.asarray(jax.grad(summary)(gp.values))
    assert np.isfinite(grad).all()
    assert grad[gp.param_index("gwb_log10_A")] != 0.0


def test_conditional_gwb_requires_correlated_injector():
    gp, pps, config, _ = _hd_config()
    bare = PTAConfig(
        toa_data_list=config.toa_data_list,
        timing_models=config.timing_models,
        noise_models=config.noise_models,
        signal_injectors=(),
        correlated_injectors=(),
    )
    with pytest.raises(ValueError, match="correlated injector"):
        conditional_gwb(gp, pps, bare)
