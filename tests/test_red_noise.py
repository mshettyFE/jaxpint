"""Tests for power-law red noise (PLRedNoise)."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.noise import NoiseModel, ScaleToaError
from jaxpint.noise.red_noise import PLRedNoise
from jaxpint.simulation import simulate_noise
from tests.helpers import make_fourier_basis, make_params, make_toa_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plred(n_toas=100, n_freqs=5, T=3.0 * 365.25 * 86400.0):
    """Build a PLRedNoise component and matching params for tests."""
    F, freqs, df, t = make_fourier_basis(n_toas, n_freqs, T)

    plred = PLRedNoise(
        fourier_basis=F,
        freqs=freqs,
        freq_bin_widths=df,
        tnredamp_name="TNREDAMP",
        tnredgam_name="TNREDGAM",
    )

    # Typical red noise parameters: log10(A) = -13, gamma = 3.5
    params = make_params(
        ("TNREDAMP", "TNREDGAM"),
        [-13.0, 3.5],
        units=("", ""),
    )
    toa_data = make_toa_data(n_toas=n_toas)

    return plred, params, toa_data, F, freqs, df


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestPLRedNoiseBasic:
    """Red-noise-specific tests; shared shape/PSD/generate tests live in
    ``test_correlated_noise_common.py``.
    """

    def test_psd_weights_red_spectrum(self):
        """Lower frequencies should have higher PSD (red spectrum)."""
        plred, params, _, _, _, _ = _make_plred(n_freqs=10)
        weights = plred.psd_weights(params)
        # Even indices are sin weights; compare consecutive frequencies
        for i in range(0, 16, 2):
            assert weights[i] > weights[i + 2]

    def test_basis_is_fourier_matrix(self):
        """The stored basis should match the Fourier design matrix."""
        plred, _, _, F, _, _ = _make_plred()
        npt.assert_array_equal(plred.fourier_basis, F)


# ---------------------------------------------------------------------------
# Covariance-generation consistency (whitening)
# ---------------------------------------------------------------------------


class TestPLRedNoiseWhitening:
    """Validate that generate() is consistent with covariance()."""

    @pytest.mark.slow
    def test_red_noise_only_whitening(self):
        """Empirical variance of red noise draws matches analytic covariance.

        Red noise covariance F @ diag(w) @ F^T is rank-deficient
        (rank 2*n_freqs < n_toas), so we verify consistency by
        checking that the empirical per-TOA variance matches the
        diagonal of the analytic covariance.

        With 10,000 draws the SE of each per-element variance estimate
        is sqrt(2/N) * sigma^2 ~ 1.4% of the true value.  Across 60
        elements, the worst-case excursion is ~3.5*SE ~ 5%, so
        rtol=0.06 gives comfortable margin.
        """
        n_toas = 60
        n_freqs = 5
        T = 3.0 * 365.25 * 86400.0
        plred, params, toa_data, F, _, _ = _make_plred(
            n_toas=n_toas, n_freqs=n_freqs, T=T,
        )

        _, U, Phidiag = plred.covariance(toa_data, params)
        C_analytic = U @ jnp.diag(Phidiag) @ U.T
        analytic_var = jnp.diag(C_analytic)

        n_draws = 10_000
        keys = jax.random.split(jax.random.PRNGKey(123), n_draws)
        draws = jnp.stack([plred.generate(toa_data, params, k) for k in keys])
        empirical_var = jnp.var(draws, axis=0)

        # SE(var) = sigma^2 * sqrt(2/N) ~ 1.4%.  With 60 elements,
        # Bonferroni-corrected ~3.5 sigma tail => ~5% max deviation.
        npt.assert_allclose(
            np.array(empirical_var),
            np.array(analytic_var),
            rtol=0.06,
            err_msg="Empirical variance doesn't match analytic covariance diagonal",
        )

    @pytest.mark.slow
    def test_combined_white_red_whitening(self):
        """White + red noise: Cholesky-whiten combined draws."""
        n_toas = 60
        n_freqs = 5
        T = 3.0 * 365.25 * 86400.0

        plred, _, _, F, freqs, df = _make_plred(
            n_toas=n_toas, n_freqs=n_freqs, T=T,
        )

        efac_val = 1.0
        error = 1e-6  # 1 microsecond

        mask = np.ones(n_toas, dtype=bool)
        toa_data = make_toa_data(
            n_toas=n_toas,
            error=error,
            flag_masks={"EFAC1": mask},
        )
        params = make_params(
            ("EFAC1", "TNREDAMP", "TNREDGAM"),
            [efac_val, -13.0, 3.5],
            units=("", "", ""),
        )

        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())

        # Full covariance: C = diag(sigma^2) + F @ diag(w) @ F^T
        Ndiag_w, _, _ = white.covariance(toa_data, params)
        _, U_rn, Phi_rn = plred.covariance(toa_data, params)
        C = jnp.diag(Ndiag_w) + U_rn @ jnp.diag(Phi_rn) @ U_rn.T
        L = jnp.linalg.cholesky(C)

        # 5000 draws x 60 elements = 300,000 whitened samples.
        # SE(std) ~ 1/sqrt(2*300000) ~ 0.0013.  atol=0.02 gives ~15 sigma.
        # SE(mean) ~ 1/sqrt(300000) ~ 0.0018.  atol=0.02 gives ~11 sigma.
        n_draws = 5000
        keys = jax.random.split(jax.random.PRNGKey(456), n_draws)
        whitened_all = []
        for k in keys:
            delays = simulate_noise(toa_data, params, k, [white, plred])
            w = jax.scipy.linalg.solve_triangular(L, delays, lower=True)
            whitened_all.append(w)

        whitened = jnp.stack(whitened_all)

        assert np.isclose(np.std(whitened), 1.0, atol=0.02), (
            f"std = {np.std(whitened):.4f}, expected ~1.0"
        )
        assert np.isclose(np.mean(whitened), 0.0, atol=0.02), (
            f"mean = {np.mean(whitened):.4f}, expected ~0.0"
        )


# ---------------------------------------------------------------------------
# NoiseModel integration
# ---------------------------------------------------------------------------


class TestNoiseModelWithRedNoise:
    """Test that PLRedNoise works correctly inside a NoiseModel."""

    def test_noise_model_covariance(self):
        """NoiseModel wrapping PLRedNoise produces correct covariance."""
        n_toas = 40
        plred, _, _, F, _, _ = _make_plred(n_toas=n_toas, n_freqs=5)

        mask = np.ones(n_toas, dtype=bool)
        toa_data = make_toa_data(
            n_toas=n_toas,
            error=1e-6,
            flag_masks={"EFAC1": mask},
        )
        params = make_params(
            ("EFAC1", "TNREDAMP", "TNREDGAM"),
            [1.2, -13.0, 3.5],
            units=("", "", ""),
        )

        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        nm = NoiseModel(white_noise=white, correlated=(plred,))

        Ndiag, U, Phidiag = nm.covariance(toa_data, params)

        assert Ndiag.shape == (n_toas,)
        assert U.shape == (n_toas, 10)
        assert Phidiag.shape == (10,)
        assert nm.has_correlated

        # Ndiag should be sigma^2 from white noise
        sigma = white.scaled_sigma(toa_data, params)
        npt.assert_allclose(np.array(Ndiag), np.array(sigma ** 2))

        # U should be the Fourier basis
        npt.assert_array_equal(U, F)

    def test_noise_model_no_white(self):
        """NoiseModel with red noise only (no white noise)."""
        n_toas = 30
        plred, _, _, _, _, _ = _make_plred(n_toas=n_toas, n_freqs=3)

        toa_data = make_toa_data(n_toas=n_toas, error=1e-6)
        params = make_params(
            ("TNREDAMP", "TNREDGAM"),
            [-13.0, 3.5],
            units=("", ""),
        )

        nm = NoiseModel(white_noise=None, correlated=(plred,))
        Ndiag, U, Phidiag = nm.covariance(toa_data, params)

        # Ndiag should be raw errors squared
        npt.assert_allclose(
            np.array(Ndiag), np.array(toa_data.error ** 2),
        )
        assert U.shape == (n_toas, 6)


# ---------------------------------------------------------------------------
# GLS fitter integration
# ---------------------------------------------------------------------------


class TestGLSWithRedNoise:
    """End-to-end: generate red noise, fit with GLS, whiten residuals."""

    @pytest.fixture(scope="class")
    def gls_fit_result(self):
        """Generate fake TOAs with white + red noise, fit with GLS."""
        from jaxpint.fitters import GLSFitter
        from jaxpint.model import TimingModel
        from jaxpint.phase.spin import Spindown

        # Set up a simple spindown model
        n_toas = 200
        n_freqs = 10
        T = 3.0 * 365.25 * 86400.0  # 3 years in seconds

        # Build Fourier basis
        F, freqs, df, t = make_fourier_basis(n_toas, n_freqs, T)

        plred = PLRedNoise(
            fourier_basis=F,
            freqs=freqs,
            freq_bin_widths=df,
            tnredamp_name="TNREDAMP",
            tnredgam_name="TNREDGAM",
        )

        # White noise
        efac_val = 1.0
        error = 1e-6  # 1 microsecond

        mask = np.ones(n_toas, dtype=bool)

        # TOA data: spread over 3 years
        t_mjd = np.linspace(53000.0, 53000.0 + T / 86400.0, n_toas)
        toa_data = make_toa_data(
            t_mjd=t_mjd,
            error=error,
            freq=1400.0,
            flag_masks={"EFAC1": mask},
            tzr_tdb_int=53000.0,
            tzr_tdb_frac=0.0,
            tzr_freq=1400.0,
            tzr_ssb_obs_pos=np.zeros(3),
        )

        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        noise_model = NoiseModel(white_noise=white, correlated=(plred,))

        # Parameters: F0 + F1 (spindown) + PEPOCH + EFAC + TNREDAMP + TNREDGAM
        spin = Spindown(spin_param_names=("F0", "F1"))
        timing_model = TimingModel(
            delay_components=(),
            phase_components=(spin,),
        )

        params = make_params(
            ("F0", "F1", "PEPOCH", "EFAC1", "TNREDAMP", "TNREDGAM"),
            [100.0, -1e-15, 0.0, efac_val, -13.0, 3.5],
            units=("Hz", "Hz/s", "day", "", "", ""),
            frozen_mask=(False, False, True, True, True, True),
            epoch_int_values={"PEPOCH": 53000.0},
        )

        # Generate fake TOAs with noise
        from jaxpint.simulation import make_fake_toas
        key = jax.random.PRNGKey(2024)
        fake_toa_data = make_fake_toas(
            timing_model, toa_data, params, key,
            noise_components=[white, plred],
        )

        # Fit with GLS
        fit_params = copy.deepcopy(params)
        fitter = GLSFitter(
            timing_model, fake_toa_data, fit_params,
            noise_model=noise_model,
        )
        result = fitter.fit_toas(maxiter=3)
        sigma = noise_model.scaled_sigma(fake_toa_data, result.params)

        # Whiten: subtract correlated noise, divide by scaled sigma
        if noise_model.has_correlated and result.noise_realizations is not None:
            _, U, _ = noise_model.covariance(fake_toa_data, result.params)
            rc = U @ result.noise_realizations
            whitened = (result.residuals - rc) / sigma
        else:
            whitened = result.residuals / sigma

        return whitened, result

    @pytest.mark.slow
    def test_whitened_std(self, gls_fit_result):
        """Whitened residuals should have std ~ 1.

        Single realization with 200 TOAs.  For chi-distributed
        residuals, SE(std) ~ 1/sqrt(2*N) ~ 0.05.  atol=0.15
        gives ~3 sigma margin.
        """
        whitened, _ = gls_fit_result
        assert np.isclose(np.std(whitened), 1.0, atol=0.15), (
            f"std = {np.std(whitened):.4f}"
        )

    @pytest.mark.slow
    def test_whitened_mean(self, gls_fit_result):
        """Whitened residuals should have mean ~ 0.

        SE(mean) ~ 1/sqrt(200) ~ 0.07.  atol=0.1 is ~1.4 sigma
        — tight enough to catch gross errors while allowing for
        the single-realization variance.
        """
        whitened, _ = gls_fit_result
        assert np.isclose(np.mean(whitened), 0.0, atol=0.1), (
            f"mean = {np.mean(whitened):.4f}"
        )

    @pytest.mark.slow
    def test_reduced_chi2_near_one(self, gls_fit_result):
        """Reduced chi-squared should be near 1.

        With ~198 dof, std(chi2/dof) = sqrt(2/dof) ~ 0.10.
        atol=0.3 gives ~3 sigma margin.
        """
        _, result = gls_fit_result
        assert np.isclose(result.reduced_chi2, 1.0, atol=0.3), (
            f"reduced chi2 = {result.reduced_chi2:.4f}"
        )
