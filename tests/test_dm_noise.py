"""Tests for power-law DM noise (PLDMNoise)."""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.noise import NoiseModel, ScaleToaError
from jaxpint.noise.dm_noise import PLDMNoise
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


def _make_pldm(n_toas=100, n_freqs=5, T=3.0 * 365.25 * 86400.0):
    """Build a PLDMNoise component with multi-frequency TOAs.

    Uses alternating 800 MHz and 1400 MHz TOAs to exercise the
    ``(1400/f)^2`` DM scaling.
    """
    F_raw, freqs, df, t = make_fourier_basis(n_toas, n_freqs, T)

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
    """DM-noise-specific tests; shared shape/PSD/generate tests live in
    ``test_correlated_noise_common.py``.
    """

    def test_psd_weights_red_spectrum(self):
        """Lower frequencies should have higher PSD."""
        pldm, params, _, _, _, _, _, _ = _make_pldm(n_freqs=10)
        weights = pldm.psd_weights(params)
        for i in range(0, weights.shape[0] - 2, 2):
            assert weights[i] > weights[i + 2]

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
        draws = jax.vmap(lambda k: pldm.generate(toa_data, params, k))(keys)
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
        from jaxpint.delay.dispersion_dm import DispersionDM

        n_toas = 200
        n_freqs = 10
        T = 3.0 * 365.25 * 86400.0

        F_raw, freqs, df, _ = make_fourier_basis(n_toas, n_freqs, T)

        # Multi-frequency TOAs; scale the Fourier basis to DM (nu^-2).
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

        # Reuse the shared GLS driver, adding a DispersionDM delay + free DM.
        dm_delay = DispersionDM(dm_param_names=("DM",), dmepoch_name="PEPOCH")
        return run_gls_whitening(
            pldm,
            param_names=("TNDMAMP", "TNDMGAM"),
            param_values=(-13.0, 3.5),
            param_units=("", ""),
            freq=obs_freqs,
            n_toas=n_toas,
            seed=2024,
            delay_components=(dm_delay,),
            extra_free_params=(("DM", 15.0, "pc/cm^3"),),
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
