"""Tests for power-law chromatic noise (PLChromNoise)."""

from __future__ import annotations

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.constants import FYR
from jaxpint.noise import NoiseModel, ScaleToaError
from jaxpint.noise.chrom_noise import PLChromNoise
from jaxpint.simulation import simulate_noise
from jaxpint.utils import build_fourier_basis
from tests.helpers import make_params, make_toa_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FREF = 1400.0  # MHz


def _make_fourier_basis(n_toas, n_freqs, T):
    """Build a raw Fourier basis for tests."""
    t = np.linspace(0.0, T, n_toas)
    F, freqs, df = build_fourier_basis(t, n_freqs, T)
    return jnp.asarray(F), jnp.asarray(freqs), jnp.asarray(df), t


def _make_plchrom(n_toas=100, n_freqs=5, T=3.0 * 365.25 * 86400.0, alpha=4.0):
    """Build a PLChromNoise component with multi-frequency TOAs.

    Uses alternating 800 MHz and 1400 MHz TOAs.
    """
    F_raw, freqs, df, t = _make_fourier_basis(n_toas, n_freqs, T)

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
    """Basic shape and value tests for PLChromNoise."""

    def test_covariance_shape(self):
        n_toas, n_freqs = 50, 5
        plchrom, params, toa_data, _, _, _, _ = _make_plchrom(n_toas=n_toas, n_freqs=n_freqs)

        Ndiag, U, Phidiag = plchrom.covariance(toa_data, params)

        assert Ndiag.shape == (n_toas,)
        assert U.shape == (n_toas, 2 * n_freqs)
        assert Phidiag.shape == (2 * n_freqs,)
        npt.assert_array_equal(Ndiag, jnp.zeros(n_toas))

    def test_psd_weights_positive(self):
        plchrom, params, _, _, _, _, _ = _make_plchrom()
        weights = plchrom.psd_weights(params)
        assert jnp.all(weights > 0)

    def test_psd_weights_values(self):
        """Verify PSD formula against manual computation."""
        n_freqs = 3
        T = 5.0 * 365.25 * 86400.0
        plchrom, params, _, _, freqs, df, _ = _make_plchrom(
            n_toas=20, n_freqs=n_freqs, T=T
        )

        log10_A = -13.0
        gamma = 3.5
        A = 10.0 ** log10_A

        expected_psd = (
            A ** 2 / (12.0 * np.pi ** 2)
            * FYR ** (gamma - 3.0)
            * np.array(freqs) ** (-gamma)
        )
        expected_weights = np.repeat(expected_psd * np.array(df), 2)

        weights = plchrom.psd_weights(params)
        npt.assert_allclose(np.array(weights), expected_weights, rtol=1e-12)

    def test_generate_shape(self):
        plchrom, params, toa_data, _, _, _, _ = _make_plchrom(n_toas=50)
        key = jax.random.PRNGKey(42)
        draws = plchrom.generate(toa_data, params, key)
        assert draws.shape == (50,)

    def test_generate_reproducible(self):
        plchrom, params, toa_data, _, _, _, _ = _make_plchrom()
        key = jax.random.PRNGKey(42)
        d1 = plchrom.generate(toa_data, params, key)
        d2 = plchrom.generate(toa_data, params, key)
        npt.assert_array_equal(d1, d2)

    def test_generate_different_keys(self):
        plchrom, params, toa_data, _, _, _, _ = _make_plchrom()
        d1 = plchrom.generate(toa_data, params, jax.random.PRNGKey(0))
        d2 = plchrom.generate(toa_data, params, jax.random.PRNGKey(1))
        assert not np.allclose(d1, d2)

    def test_runtime_scaling(self):
        """covariance() basis should equal raw basis × (fref/f)^alpha."""
        plchrom, params, toa_data, F_raw, _, _, obs_freqs = _make_plchrom(
            n_toas=20, alpha=4.0
        )
        _, U, _ = plchrom.covariance(toa_data, params)

        D = (FREF / obs_freqs[:20]) ** 4.0
        expected = F_raw * jnp.asarray(D)[:, None]
        npt.assert_allclose(np.array(U), np.array(expected), rtol=1e-12)

    def test_alpha_2_matches_dm_scaling(self):
        """When alpha=2, the scaling should match DM's (fref/f)^2."""
        plchrom, params, toa_data, F_raw, _, _, obs_freqs = _make_plchrom(
            n_toas=20, alpha=2.0
        )
        _, U, _ = plchrom.covariance(toa_data, params)

        D = (FREF / obs_freqs[:20]) ** 2.0
        expected = F_raw * jnp.asarray(D)[:, None]
        npt.assert_allclose(np.array(U), np.array(expected), rtol=1e-12)

    def test_alpha_sensitivity(self):
        """Different alpha values should produce different scaled bases."""
        plchrom, params, toa_data, _, _, _, _ = _make_plchrom(n_toas=20, alpha=2.0)

        _, U_alpha2, _ = plchrom.covariance(toa_data, params)

        params_alpha4 = make_params(
            ("TNCHROMAMP", "TNCHROMGAM", "TNCHROMIDX"),
            [-13.0, 3.5, 4.0],
            units=("", "", ""),
        )
        _, U_alpha4, _ = plchrom.covariance(toa_data, params_alpha4)

        assert not np.allclose(U_alpha2, U_alpha4)

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

    def test_chrom_scaling_frequency_dependence(self):
        """Lower-frequency TOAs should have larger noise amplitude."""
        plchrom, params, toa_data, _, _, _, _ = _make_plchrom(n_toas=100, alpha=4.0)
        _, U, Phidiag = plchrom.covariance(toa_data, params)
        C_diag = jnp.sum(U ** 2 * Phidiag[None, :], axis=1)

        var_800 = C_diag[0::2].mean()
        var_1400 = C_diag[1::2].mean()
        assert var_800 > var_1400


# ---------------------------------------------------------------------------
# Covariance-generation consistency (whitening)
# ---------------------------------------------------------------------------


class TestPLChromNoiseWhitening:
    """Validate that generate() is consistent with covariance()."""

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

        n_draws = 10_000
        keys = jax.random.split(jax.random.PRNGKey(123), n_draws)
        draws = jnp.stack([plchrom.generate(toa_data, params, k) for k in keys])
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
        from jaxpint.fitter import GLSFitter
        from jaxpint.model import TimingModel
        from jaxpint.phase.spin import Spindown

        n_toas = 200
        n_freqs = 10
        T = 3.0 * 365.25 * 86400.0

        F_raw, freqs, df, t = _make_fourier_basis(n_toas, n_freqs, T)

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

        efac_val = 1.0
        error = 1e-6
        mask = np.ones(n_toas, dtype=bool)

        t_mjd = np.linspace(53000.0, 53000.0 + T / 86400.0, n_toas)
        toa_data = make_toa_data(
            t_mjd=t_mjd,
            error=error,
            freq=obs_freqs,
            flag_masks={"EFAC1": mask},
            tzr_tdb_int=53000.0,
            tzr_tdb_frac=0.0,
            tzr_freq=1400.0,
            tzr_ssb_obs_pos=np.zeros(3),
        )

        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        noise_model = NoiseModel(white_noise=white, correlated=(plchrom,))

        spin = Spindown(spin_param_names=("F0", "F1"))
        timing_model = TimingModel(
            delay_components=(),
            phase_components=(spin,),
        )

        params = make_params(
            ("F0", "F1", "PEPOCH", "EFAC1", "TNCHROMAMP", "TNCHROMGAM", "TNCHROMIDX"),
            [100.0, -1e-15, 0.0, efac_val, -13.0, 3.5, 4.0],
            units=("Hz", "Hz/s", "day", "", "", "", ""),
            frozen_mask=(False, False, True, True, True, True, True),
            epoch_int_values={"PEPOCH": 53000.0},
        )

        from jaxpint.simulation import make_fake_toas
        key = jax.random.PRNGKey(2024)
        fake_toa_data = make_fake_toas(
            timing_model, toa_data, params, key,
            noise_components=[white, plchrom],
        )

        fit_params = copy.deepcopy(params)
        fitter = GLSFitter(
            timing_model, fake_toa_data, fit_params,
            noise_model=noise_model,
        )
        result = fitter.fit_toas(maxiter=3)
        sigma = noise_model.scaled_sigma(fake_toa_data, result.params)

        if noise_model.has_correlated and result.noise_realizations is not None:
            _, U, _ = noise_model.covariance(fake_toa_data, result.params)
            rc = U @ result.noise_realizations
            whitened = (result.residuals - rc) / sigma
        else:
            whitened = result.residuals / sigma

        return whitened, result

    def test_whitened_std(self, gls_fit_result):
        whitened, _ = gls_fit_result
        assert np.isclose(np.std(whitened), 1.0, atol=0.15), (
            f"std = {np.std(whitened):.4f}"
        )

    def test_whitened_mean(self, gls_fit_result):
        whitened, _ = gls_fit_result
        assert np.isclose(np.mean(whitened), 0.0, atol=0.1), (
            f"mean = {np.mean(whitened):.4f}"
        )

    def test_reduced_chi2_near_one(self, gls_fit_result):
        _, result = gls_fit_result
        assert np.isclose(result.reduced_chi2, 1.0, atol=0.3), (
            f"reduced chi2 = {result.reduced_chi2:.4f}"
        )
