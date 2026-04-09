"""Tests for power-law DM noise (PLDMNoise)."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.constants import FYR
from jaxpint.noise import NoiseModel, ScaleToaError
from jaxpint.noise.dm_noise import PLDMNoise
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


def _make_pldm(n_toas=100, n_freqs=5, T=3.0 * 365.25 * 86400.0):
    """Build a PLDMNoise component with multi-frequency TOAs.

    Uses alternating 800 MHz and 1400 MHz TOAs to exercise the
    ``(1400/f)^2`` DM scaling.
    """
    F_raw, freqs, df, t = _make_fourier_basis(n_toas, n_freqs, T)

    # Alternate between 800 MHz and 1400 MHz
    obs_freqs = np.where(np.arange(n_toas) % 2 == 0, 800.0, 1400.0)
    D = (FREF / obs_freqs) ** 2  # DM scaling per TOA
    F_dm = F_raw * jnp.asarray(D)[:, None]

    pldm = PLDMNoise(
        fourier_basis=F_dm,
        freqs=freqs,
        freq_bin_widths=df,
        tndmamp_name="TNDMAMP",
        tndmgam_name="TNDMGAM",
    )

    params = make_params(
        ("TNDMAMP", "TNDMGAM"),
        [-13.0, 3.5],
        units=("", ""),
    )
    toa_data = make_toa_data(n_toas=n_toas, freq=obs_freqs)

    return pldm, params, toa_data, F_raw, F_dm, freqs, df, obs_freqs


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestPLDMNoiseBasic:
    """Basic shape and value tests for PLDMNoise."""

    def test_covariance_shape(self):
        n_toas, n_freqs = 50, 5
        pldm, params, toa_data, _, _, _, _, _ = _make_pldm(n_toas=n_toas, n_freqs=n_freqs)

        Ndiag, U, Phidiag = pldm.covariance(toa_data, params)

        assert Ndiag.shape == (n_toas,)
        assert U.shape == (n_toas, 2 * n_freqs)
        assert Phidiag.shape == (2 * n_freqs,)
        npt.assert_array_equal(Ndiag, jnp.zeros(n_toas))

    def test_psd_weights_positive(self):
        pldm, params, _, _, _, _, _, _ = _make_pldm()
        weights = pldm.psd_weights(params)
        assert jnp.all(weights > 0)

    def test_psd_weights_values(self):
        """Verify PSD formula against manual computation."""
        n_freqs = 3
        T = 5.0 * 365.25 * 86400.0
        pldm, params, _, _, _, freqs, df, _ = _make_pldm(
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

        weights = pldm.psd_weights(params)
        npt.assert_allclose(np.array(weights), expected_weights, rtol=1e-12)

    def test_psd_weights_red_spectrum(self):
        """Lower frequencies should have higher PSD."""
        pldm, params, _, _, _, _, _, _ = _make_pldm(n_freqs=10)
        weights = pldm.psd_weights(params)
        for i in range(0, 16, 2):
            assert weights[i] > weights[i + 2]

    def test_generate_shape(self):
        pldm, params, toa_data, _, _, _, _, _ = _make_pldm(n_toas=50)
        key = jax.random.PRNGKey(42)
        draws = pldm.generate(toa_data, params, key)
        assert draws.shape == (50,)

    def test_generate_reproducible(self):
        pldm, params, toa_data, _, _, _, _, _ = _make_pldm()
        key = jax.random.PRNGKey(42)
        d1 = pldm.generate(toa_data, params, key)
        d2 = pldm.generate(toa_data, params, key)
        npt.assert_array_equal(d1, d2)

    def test_generate_different_keys(self):
        pldm, params, toa_data, _, _, _, _, _ = _make_pldm()
        d1 = pldm.generate(toa_data, params, jax.random.PRNGKey(0))
        d2 = pldm.generate(toa_data, params, jax.random.PRNGKey(1))
        assert not np.allclose(d1, d2)

    def test_basis_includes_dm_scaling(self):
        """Pre-computed basis should differ from raw Fourier by (1400/f)^2."""
        pldm, _, _, F_raw, F_dm, _, _, obs_freqs = _make_pldm(n_toas=20)
        D = (FREF / obs_freqs) ** 2
        expected = F_raw * jnp.asarray(D)[:, None]
        npt.assert_allclose(np.array(pldm.fourier_basis), np.array(expected), rtol=1e-14)

    def test_dm_scaling_frequency_dependence(self):
        """Lower-frequency TOAs should have larger noise amplitude."""
        pldm, params, toa_data, _, _, _, _, _ = _make_pldm(n_toas=100)
        _, U, Phidiag = pldm.covariance(toa_data, params)
        C_diag = jnp.sum(U ** 2 * Phidiag[None, :], axis=1)

        # 800 MHz TOAs (even indices) should have larger variance than 1400 MHz (odd)
        var_800 = C_diag[0::2].mean()
        var_1400 = C_diag[1::2].mean()
        assert var_800 > var_1400


# ---------------------------------------------------------------------------
# Covariance-generation consistency (whitening)
# ---------------------------------------------------------------------------


class TestPLDMNoiseWhitening:
    """Validate that generate() is consistent with covariance()."""

    @pytest.mark.slow
    def test_dm_noise_whitening(self):
        """Empirical variance matches analytic covariance diagonal."""
        n_toas = 60
        n_freqs = 5
        T = 3.0 * 365.25 * 86400.0
        pldm, params, toa_data, _, _, _, _, _ = _make_pldm(
            n_toas=n_toas, n_freqs=n_freqs, T=T,
        )

        _, U, Phidiag = pldm.covariance(toa_data, params)
        C_analytic = U @ jnp.diag(Phidiag) @ U.T
        analytic_var = jnp.diag(C_analytic)

        n_draws = 10_000
        keys = jax.random.split(jax.random.PRNGKey(123), n_draws)
        draws = jnp.stack([pldm.generate(toa_data, params, k) for k in keys])
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


class TestNoiseModelWithDMNoise:
    """Test that PLDMNoise works correctly inside a NoiseModel."""

    def test_noise_model_covariance(self):
        n_toas = 40
        pldm, _, _, _, F_dm, _, _, obs_freqs = _make_pldm(n_toas=n_toas, n_freqs=5)

        mask = np.ones(n_toas, dtype=bool)
        toa_data = make_toa_data(
            n_toas=n_toas,
            error=1e-6,
            freq=obs_freqs[:n_toas],
            flag_masks={"EFAC1": mask},
        )
        params = make_params(
            ("EFAC1", "TNDMAMP", "TNDMGAM"),
            [1.2, -13.0, 3.5],
            units=("", "", ""),
        )

        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        nm = NoiseModel(white_noise=white, correlated=(pldm,))

        Ndiag, U, Phidiag = nm.covariance(toa_data, params)

        assert Ndiag.shape == (n_toas,)
        assert U.shape == (n_toas, 10)
        assert Phidiag.shape == (10,)
        assert nm.has_correlated

    def test_noise_model_no_white(self):
        n_toas = 30
        pldm, _, _, _, _, _, _, obs_freqs = _make_pldm(n_toas=n_toas, n_freqs=3)

        toa_data = make_toa_data(n_toas=n_toas, error=1e-6, freq=obs_freqs[:n_toas])
        params = make_params(
            ("TNDMAMP", "TNDMGAM"),
            [-13.0, 3.5],
            units=("", ""),
        )

        nm = NoiseModel(white_noise=None, correlated=(pldm,))
        Ndiag, U, Phidiag = nm.covariance(toa_data, params)

        npt.assert_allclose(
            np.array(Ndiag), np.array(toa_data.error ** 2),
        )
        assert U.shape == (n_toas, 6)


# ---------------------------------------------------------------------------
# GLS fitter integration
# ---------------------------------------------------------------------------


class TestGLSWithDMNoise:
    """End-to-end: generate DM noise, fit with GLS, whiten residuals."""

    @pytest.fixture(scope="class")
    def gls_fit_result(self):
        from jaxpint.fitters import GLSFitter
        from jaxpint.model import TimingModel
        from jaxpint.phase.spin import Spindown
        from jaxpint.delay.dispersion_dm import DispersionDM

        n_toas = 200
        n_freqs = 10
        T = 3.0 * 365.25 * 86400.0

        F_raw, freqs, df, t = _make_fourier_basis(n_toas, n_freqs, T)

        # Multi-frequency TOAs
        obs_freqs = np.where(np.arange(n_toas) % 2 == 0, 800.0, 1400.0)
        D = (FREF / obs_freqs) ** 2
        F_dm = F_raw * jnp.asarray(D)[:, None]

        pldm = PLDMNoise(
            fourier_basis=F_dm,
            freqs=freqs,
            freq_bin_widths=df,
            tndmamp_name="TNDMAMP",
            tndmgam_name="TNDMGAM",
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
        noise_model = NoiseModel(white_noise=white, correlated=(pldm,))

        spin = Spindown(spin_param_names=("F0", "F1"))
        dm_delay = DispersionDM(dm_param_names=("DM",), dmepoch_name="PEPOCH")
        timing_model = TimingModel(
            delay_components=(dm_delay,),
            phase_components=(spin,),
        )

        params = make_params(
            ("F0", "F1", "PEPOCH", "DM", "EFAC1", "TNDMAMP", "TNDMGAM"),
            [100.0, -1e-15, 0.0, 15.0, efac_val, -13.0, 3.5],
            units=("Hz", "Hz/s", "day", "pc/cm^3", "", "", ""),
            frozen_mask=(False, False, True, False, True, True, True),
            epoch_int_values={"PEPOCH": 53000.0},
        )

        from jaxpint.simulation import make_fake_toas
        key = jax.random.PRNGKey(2024)
        fake_toa_data = make_fake_toas(
            timing_model, toa_data, params, key,
            noise_components=[white, pldm],
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
