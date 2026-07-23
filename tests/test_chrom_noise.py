"""Tests for power-law chromatic noise (PLChromNoise)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.noise import NoiseModel, ScaleToaError
from jaxpint.noise.chrom_noise import PLChromNoise
from tests.helpers import (
    make_fourier_basis,
    make_params,
    make_toa_data,
    run_gls_whitening,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FREF = 1400.0  # MHz


def _make_plchrom(n_toas=100, n_freqs=5, T=3.0 * 365.25 * 86400.0, alpha=4.0):
    """Build a PLChromNoise component with multi-frequency TOAs.

    Uses alternating 800 MHz and 1400 MHz TOAs.
    """
    F_raw, freqs, df, t = make_fourier_basis(n_toas, n_freqs, T)

    obs_freqs = np.where(np.arange(n_toas) % 2 == 0, 800.0, 1400.0)

    plchrom = PLChromNoise(
        fourier_basis=F_raw,
        freqs=freqs,
        freq_bin_widths=df,
        tnchromamp_name="TNCHROMAMP",
        tnchromgam_name="TNCHROMGAM",
        tnchromidx_name="TNCHROMIDX",
        fref=FREF,
    )

    params = make_params(
        ("TNCHROMAMP", "TNCHROMGAM", "TNCHROMIDX"),
        [-13.0, 3.5, alpha],
        units=("", "", ""),
    )
    toa_data = make_toa_data(n_toas=n_toas, freq=obs_freqs)

    return plchrom, params, toa_data, F_raw, freqs, df, obs_freqs


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestPLChromNoiseBasic:
    """Chrom-noise-specific tests; shared shape/PSD/generate tests live in
    ``test_correlated_noise_common.py``.
    """

    @pytest.mark.parametrize("alpha", [2.0, 4.0])
    def test_runtime_scaling(self, alpha):
        """covariance() basis equals raw basis × (fref/f)^alpha at runtime.

        alpha=2 doubles as the reduces-to-DM-scaling case.  The exact
        rtol-1e-12 pin at two distinct alphas also implies alpha
        sensitivity and the 800-vs-1400 MHz variance ordering.
        """
        plchrom, params, toa_data, F_raw, _, _, obs_freqs = _make_plchrom(
            n_toas=20, alpha=alpha
        )
        _, U, _ = plchrom.covariance(toa_data, params)

        D = (FREF / obs_freqs[:20]) ** alpha
        expected = F_raw * jnp.asarray(D)[:, None]
        npt.assert_allclose(np.array(U), np.array(expected), rtol=1e-12)

    def test_differentiable_through_alpha(self):
        """jax.grad through covariance w.r.t. TNCHROMIDX should work."""
        plchrom, params, toa_data, _, _, _, _ = _make_plchrom(n_toas=20)

        def loss_fn(values):
            p = eqx.tree_at(lambda x: x.values, params, values)
            _, U, Phi = plchrom.covariance(toa_data, p)
            return jnp.sum(U ** 2 * Phi[None, :])

        grad = jax.grad(loss_fn)(params.values)
        # TNCHROMIDX is at index 2
        assert jnp.isfinite(grad[2])
        assert grad[2] != 0.0


# ---------------------------------------------------------------------------
# Covariance-generation consistency (whitening)
# ---------------------------------------------------------------------------


class TestPLChromNoiseWhitening:
    """Validate that generate() is consistent with covariance()."""

    @pytest.mark.slow
    def test_chrom_noise_whitening(self):
        """Empirical variance matches analytic covariance diagonal."""
        n_toas = 60
        n_freqs = 5
        T = 3.0 * 365.25 * 86400.0
        plchrom, params, toa_data, _, _, _, _ = _make_plchrom(
            n_toas=n_toas, n_freqs=n_freqs, T=T, alpha=4.0,
        )

        _, U, Phidiag = plchrom.covariance(toa_data, params)
        C_analytic = U @ jnp.diag(Phidiag) @ U.T
        analytic_var = jnp.diag(C_analytic)

        n_draws = 4_000
        keys = jax.random.split(jax.random.PRNGKey(123), n_draws)
        draws = jax.vmap(lambda k: plchrom.generate(toa_data, params, k))(keys)
        empirical_var = jnp.var(draws, axis=0)

        npt.assert_allclose(
            np.array(empirical_var),
            np.array(analytic_var),
            rtol=0.06,
            err_msg="Empirical variance doesn't match analytic covariance diagonal",
        )


# ---------------------------------------------------------------------------
# NoiseModel integration
# ---------------------------------------------------------------------------


class TestNoiseModelWithChromNoise:
    """Test that PLChromNoise works correctly inside a NoiseModel."""

    def test_noise_model_covariance(self):
        n_toas = 40
        plchrom, _, _, _, _, _, obs_freqs = _make_plchrom(n_toas=n_toas, n_freqs=5)

        mask = np.ones(n_toas, dtype=bool)
        toa_data = make_toa_data(
            n_toas=n_toas,
            error=1e-6,
            freq=obs_freqs[:n_toas],
            flag_masks={"EFAC1": mask},
        )
        params = make_params(
            ("EFAC1", "TNCHROMAMP", "TNCHROMGAM", "TNCHROMIDX"),
            [1.2, -13.0, 3.5, 4.0],
            units=("", "", "", ""),
        )

        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        nm = NoiseModel(white_noise=white, correlated=(plchrom,))

        Ndiag, U, Phidiag = nm.covariance(toa_data, params)

        assert Ndiag.shape == (n_toas,)
        assert U.shape == (n_toas, 10)
        assert Phidiag.shape == (10,)
        assert nm.has_correlated

    def test_noise_model_no_white(self):
        n_toas = 30
        plchrom, _, _, _, _, _, obs_freqs = _make_plchrom(n_toas=n_toas, n_freqs=3)

        toa_data = make_toa_data(n_toas=n_toas, error=1e-6, freq=obs_freqs[:n_toas])
        params = make_params(
            ("TNCHROMAMP", "TNCHROMGAM", "TNCHROMIDX"),
            [-13.0, 3.5, 4.0],
            units=("", "", ""),
        )

        nm = NoiseModel(white_noise=None, correlated=(plchrom,))
        Ndiag, U, Phidiag = nm.covariance(toa_data, params)

        npt.assert_allclose(
            np.array(Ndiag), np.array(toa_data.error ** 2),
        )
        assert U.shape == (n_toas, 6)


# ---------------------------------------------------------------------------
# GLS fitter integration
# ---------------------------------------------------------------------------


class TestGLSWithChromNoise:
    """End-to-end: generate chromatic noise, fit with GLS, whiten residuals."""

    @pytest.fixture(scope="class")
    def gls_fit_result(self):
        n_toas = 200
        F_raw, freqs, df, t = make_fourier_basis(n_toas, 10, 3.0 * 365.25 * 86400.0)
        obs_freqs = np.where(np.arange(n_toas) % 2 == 0, 800.0, 1400.0)

        plchrom = PLChromNoise(
            fourier_basis=F_raw,
            freqs=freqs,
            freq_bin_widths=df,
            tnchromamp_name="TNCHROMAMP",
            tnchromgam_name="TNCHROMGAM",
            tnchromidx_name="TNCHROMIDX",
            fref=FREF,
        )
        return run_gls_whitening(
            plchrom,
            param_names=("TNCHROMAMP", "TNCHROMGAM", "TNCHROMIDX"),
            param_values=(-13.0, 3.5, 4.0),
            param_units=("", "", ""),
            freq=obs_freqs,
        )

    @pytest.mark.slow
    def test_whitened_std(self, gls_fit_result):
        whitened, _ = gls_fit_result
        assert np.isclose(np.std(whitened), 1.0, atol=0.15), (
            f"std = {np.std(whitened):.4f}"
        )

    @pytest.mark.slow
    def test_whitened_mean(self, gls_fit_result):
        whitened, _ = gls_fit_result
        assert np.isclose(np.mean(whitened), 0.0, atol=0.1), (
            f"mean = {np.mean(whitened):.4f}"
        )

    @pytest.mark.slow
    def test_reduced_chi2_near_one(self, gls_fit_result):
        _, result = gls_fit_result
        assert np.isclose(result.reduced_chi2, 1.0, atol=0.3), (
            f"reduced chi2 = {result.reduced_chi2:.4f}"
        )
