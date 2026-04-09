"""Tests for power-law solar wind noise (PLSWNoise)."""

from __future__ import annotations

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.constants import AU_KM, DMCONST, FYR
from jaxpint.delay.solar_wind import _solar_wind_geometry_swm0, _sun_angle_and_distance
from jaxpint.noise import NoiseModel, ScaleToaError
from jaxpint.noise.sw_noise import PLSWNoise
from jaxpint.simulation import simulate_noise
from jaxpint.utils import build_fourier_basis, compute_pulsar_direction
from tests.helpers import make_params, make_toa_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pulsar at RA=0h, DEC=+45deg (well away from the ecliptic plane)
PSR_RAJ = 0.0  # radians
PSR_DECJ = np.pi / 4.0  # radians


def _make_fourier_basis(n_toas, n_freqs, T):
    """Build a raw Fourier basis for tests."""
    t = np.linspace(0.0, T, n_toas)
    F, freqs, df = build_fourier_basis(t, n_freqs, T)
    return jnp.asarray(F), jnp.asarray(freqs), jnp.asarray(df), t


def _make_obs_sun_pos(n_toas):
    """Create synthetic observer-Sun position vectors.

    Simulates an Earth-like orbit at 1 AU, with the Sun at varying
    angles through the year.
    """
    # Phase of Earth around the Sun
    phases = np.linspace(0, 2 * np.pi, n_toas, endpoint=False)
    obs_sun = np.zeros((n_toas, 3))
    obs_sun[:, 0] = AU_KM * np.cos(phases)
    obs_sun[:, 1] = AU_KM * np.sin(phases)
    obs_sun[:, 2] = 0.0
    return obs_sun


def _make_plsw(n_toas=100, n_freqs=5, T=3.0 * 365.25 * 86400.0):
    """Build a PLSWNoise component with realistic geometry.

    Uses alternating 800 MHz and 1400 MHz TOAs with synthetic
    observer-Sun positions for a pulsar at RA=0, DEC=+45deg.
    """
    F_raw, freqs, df, t = _make_fourier_basis(n_toas, n_freqs, T)

    obs_freqs = np.where(np.arange(n_toas) % 2 == 0, 800.0, 1400.0)
    obs_sun_pos = _make_obs_sun_pos(n_toas)

    plsw = PLSWNoise(
        fourier_basis=F_raw,
        freqs=freqs,
        freq_bin_widths=df,
        tnswamp_name="TNSWAMP",
        tnswgam_name="TNSWGAM",
        swm=0,
        swp_name=None,
        raj_name="RAJ",
        decj_name="DECJ",
        pmra_name=None,
        pmdec_name=None,
        posepoch_name=None,
        obliquity_arcsec=None,
    )

    params = make_params(
        ("TNSWAMP", "TNSWGAM", "RAJ", "DECJ"),
        [-13.0, 3.5, PSR_RAJ, PSR_DECJ],
        units=("", "", "rad", "rad"),
    )

    # Build TOAData with realistic obs_sun_pos
    t_mjd = np.linspace(53000.0, 53000.0 + T / 86400.0, n_toas)
    toa_data = make_toa_data(
        t_mjd=t_mjd,
        freq=obs_freqs,
    )
    # Replace obs_sun_pos with realistic values
    toa_data = eqx.tree_at(lambda t: t.obs_sun_pos, toa_data, jnp.asarray(obs_sun_pos))

    return plsw, params, toa_data, F_raw, freqs, df, obs_freqs, obs_sun_pos


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestPLSWNoiseBasic:
    """Basic shape and value tests for PLSWNoise."""

    def test_covariance_shape(self):
        n_toas, n_freqs = 50, 5
        plsw, params, toa_data, _, _, _, _, _ = _make_plsw(n_toas=n_toas, n_freqs=n_freqs)

        Ndiag, U, Phidiag = plsw.covariance(toa_data, params)

        assert Ndiag.shape == (n_toas,)
        assert U.shape == (n_toas, 2 * n_freqs)
        assert Phidiag.shape == (2 * n_freqs,)
        npt.assert_array_equal(Ndiag, jnp.zeros(n_toas))

    def test_psd_weights_positive(self):
        plsw, params, _, _, _, _, _, _ = _make_plsw()
        weights = plsw.psd_weights(params)
        assert jnp.all(weights > 0)

    def test_psd_weights_values(self):
        """Verify PSD formula against manual computation."""
        n_freqs = 3
        T = 5.0 * 365.25 * 86400.0
        plsw, params, _, _, freqs, df, _, _ = _make_plsw(
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

        weights = plsw.psd_weights(params)
        npt.assert_allclose(np.array(weights), expected_weights, rtol=1e-12)

    def test_generate_shape(self):
        plsw, params, toa_data, _, _, _, _, _ = _make_plsw(n_toas=50)
        key = jax.random.PRNGKey(42)
        draws = plsw.generate(toa_data, params, key)
        assert draws.shape == (50,)

    def test_generate_reproducible(self):
        plsw, params, toa_data, _, _, _, _, _ = _make_plsw()
        key = jax.random.PRNGKey(42)
        d1 = plsw.generate(toa_data, params, key)
        d2 = plsw.generate(toa_data, params, key)
        npt.assert_array_equal(d1, d2)

    def test_generate_different_keys(self):
        plsw, params, toa_data, _, _, _, _, _ = _make_plsw()
        d1 = plsw.generate(toa_data, params, jax.random.PRNGKey(0))
        d2 = plsw.generate(toa_data, params, jax.random.PRNGKey(1))
        # SW noise amplitudes are very small; use relative tolerance
        assert not np.allclose(d1, d2, atol=0, rtol=0.01)

    def test_sw_scaling_with_geometry(self):
        """Verify scaled basis matches manual geometry × DMCONST / f² computation."""
        n_toas = 20
        plsw, params, toa_data, F_raw, _, _, obs_freqs, obs_sun_pos = _make_plsw(
            n_toas=n_toas
        )

        # Manually compute the scaling
        psr_dir = compute_pulsar_direction(
            toa_data, params,
            raj_name="RAJ", decj_name="DECJ",
            pmra_name=None, pmdec_name=None, posepoch_name=None,
        )
        theta, r_km = _sun_angle_and_distance(toa_data, psr_dir)
        geometry_pc = _solar_wind_geometry_swm0(theta, r_km)
        D_expected = geometry_pc * DMCONST / jnp.asarray(obs_freqs[:n_toas]) ** 2

        # Get the actual scaled basis
        _, U, _ = plsw.covariance(toa_data, params)
        expected_basis = F_raw * D_expected[:, None]

        npt.assert_allclose(np.array(U), np.array(expected_basis), rtol=1e-10)

    def test_swm0_geometry_nonzero(self):
        """SWM=0 geometry should produce non-zero scaling with realistic positions."""
        plsw, params, toa_data, _, _, _, _, _ = _make_plsw(n_toas=20)
        scaling = plsw._sw_scaling(toa_data, params)
        assert jnp.all(jnp.isfinite(scaling))
        assert jnp.all(scaling != 0.0)

    def test_sw_scaling_frequency_dependence(self):
        """Lower-frequency TOAs should have larger SW noise amplitude."""
        plsw, params, toa_data, _, _, _, _, _ = _make_plsw(n_toas=100)
        _, U, Phidiag = plsw.covariance(toa_data, params)
        C_diag = jnp.sum(U ** 2 * Phidiag[None, :], axis=1)

        var_800 = C_diag[0::2].mean()
        var_1400 = C_diag[1::2].mean()
        assert var_800 > var_1400


# ---------------------------------------------------------------------------
# Covariance-generation consistency (whitening)
# ---------------------------------------------------------------------------


class TestPLSWNoiseWhitening:
    """Validate that generate() is consistent with covariance()."""

    @pytest.mark.slow
    def test_sw_noise_whitening(self):
        """Empirical variance matches analytic covariance diagonal."""
        n_toas = 60
        n_freqs = 5
        T = 3.0 * 365.25 * 86400.0
        plsw, params, toa_data, _, _, _, _, _ = _make_plsw(
            n_toas=n_toas, n_freqs=n_freqs, T=T,
        )

        _, U, Phidiag = plsw.covariance(toa_data, params)
        C_analytic = U @ jnp.diag(Phidiag) @ U.T
        analytic_var = jnp.diag(C_analytic)

        n_draws = 10_000
        keys = jax.random.split(jax.random.PRNGKey(123), n_draws)
        draws = jnp.stack([plsw.generate(toa_data, params, k) for k in keys])
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


class TestNoiseModelWithSWNoise:
    """Test that PLSWNoise works correctly inside a NoiseModel."""

    def test_noise_model_covariance(self):
        n_toas = 40
        plsw, _, _, _, _, _, obs_freqs, obs_sun_pos = _make_plsw(n_toas=n_toas, n_freqs=5)

        mask = np.ones(n_toas, dtype=bool)
        toa_data = make_toa_data(
            n_toas=n_toas,
            error=1e-6,
            freq=obs_freqs[:n_toas],
            flag_masks={"EFAC1": mask},
        )
        toa_data = eqx.tree_at(lambda t: t.obs_sun_pos, toa_data, jnp.asarray(obs_sun_pos[:n_toas]))

        params = make_params(
            ("EFAC1", "TNSWAMP", "TNSWGAM", "RAJ", "DECJ"),
            [1.2, -13.0, 3.5, PSR_RAJ, PSR_DECJ],
            units=("", "", "", "rad", "rad"),
        )

        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        nm = NoiseModel(white_noise=white, correlated=(plsw,))

        Ndiag, U, Phidiag = nm.covariance(toa_data, params)

        assert Ndiag.shape == (n_toas,)
        assert U.shape == (n_toas, 10)
        assert Phidiag.shape == (10,)
        assert nm.has_correlated


# ---------------------------------------------------------------------------
# GLS fitter integration
# ---------------------------------------------------------------------------


class TestGLSWithSWNoise:
    """End-to-end: generate SW noise, fit with GLS, whiten residuals."""

    @pytest.fixture(scope="class")
    def gls_fit_result(self):
        from jaxpint.fitters import GLSFitter
        from jaxpint.model import TimingModel
        from jaxpint.phase.spin import Spindown

        n_toas = 200
        n_freqs = 10
        T = 3.0 * 365.25 * 86400.0

        F_raw, freqs, df, t = _make_fourier_basis(n_toas, n_freqs, T)
        obs_freqs = np.where(np.arange(n_toas) % 2 == 0, 800.0, 1400.0)
        obs_sun_pos = _make_obs_sun_pos(n_toas)

        plsw = PLSWNoise(
            fourier_basis=F_raw,
            freqs=freqs,
            freq_bin_widths=df,
            tnswamp_name="TNSWAMP",
            tnswgam_name="TNSWGAM",
            swm=0,
            swp_name=None,
            raj_name="RAJ",
            decj_name="DECJ",
            pmra_name=None,
            pmdec_name=None,
            posepoch_name=None,
            obliquity_arcsec=None,
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
        toa_data = eqx.tree_at(lambda t: t.obs_sun_pos, toa_data, jnp.asarray(obs_sun_pos))

        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        noise_model = NoiseModel(white_noise=white, correlated=(plsw,))

        spin = Spindown(spin_param_names=("F0", "F1"))
        timing_model = TimingModel(
            delay_components=(),
            phase_components=(spin,),
        )

        params = make_params(
            ("F0", "F1", "PEPOCH", "EFAC1", "TNSWAMP", "TNSWGAM", "RAJ", "DECJ"),
            [100.0, -1e-15, 0.0, efac_val, -13.0, 3.5, PSR_RAJ, PSR_DECJ],
            units=("Hz", "Hz/s", "day", "", "", "", "rad", "rad"),
            frozen_mask=(False, False, True, True, True, True, True, True),
            epoch_int_values={"PEPOCH": 53000.0},
        )

        from jaxpint.simulation import make_fake_toas
        key = jax.random.PRNGKey(2024)
        fake_toa_data = make_fake_toas(
            timing_model, toa_data, params, key,
            noise_components=[white, plsw],
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
