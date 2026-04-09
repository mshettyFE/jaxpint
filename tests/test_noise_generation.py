"""Tests for noise generation (NoiseComponent.generate and simulate_noise)."""

from __future__ import annotations

import copy
import io
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.noise import EcorrNoise, ScaleToaError
from jaxpint.simulation import make_fake_toas, simulate_noise
from tests.helpers import make_params, make_toa_data


# ---------------------------------------------------------------------------
# White noise generation (ScaleToaError)
# ---------------------------------------------------------------------------


class TestWhiteNoiseGenerate:
    """Tests for ScaleToaError.generate."""

    @pytest.fixture
    def white_noise_setup(self):
        n_toas = 5000
        efac_val = 1.3
        equad_val = 0.8e-6  # seconds

        mask = np.ones(n_toas, dtype=bool)
        toa_data = make_toa_data(
            n_toas=n_toas,
            error=1e-6,
            flag_masks={"EFAC1": mask, "EQUAD1": mask},
        )
        params = make_params(
            ("EFAC1", "EQUAD1"),
            [efac_val, equad_val],
            units=("", "s"),
        )
        noise = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
        return toa_data, params, noise

    def test_whitened_std(self, white_noise_setup):
        """Noise divided by scaled_sigma should have std ~ 1."""
        toa_data, params, noise = white_noise_setup
        key = jax.random.PRNGKey(42)

        draws = noise.generate(toa_data, params, key)
        sigma = noise.scaled_sigma(toa_data, params)
        whitened = draws / sigma

        assert np.isclose(np.std(whitened), 1.0, atol=0.05)

    def test_whitened_mean(self, white_noise_setup):
        """Whitened noise should have mean ~ 0."""
        toa_data, params, noise = white_noise_setup
        key = jax.random.PRNGKey(42)

        draws = noise.generate(toa_data, params, key)
        sigma = noise.scaled_sigma(toa_data, params)
        whitened = draws / sigma

        assert np.isclose(np.mean(whitened), 0.0, atol=0.05)

    def test_covariance_consistent_with_generate(self):
        """Empirical variance from many draws should match covariance diagonal."""
        n_toas = 200
        mask = np.ones(n_toas, dtype=bool)
        toa_data = make_toa_data(
            n_toas=n_toas,
            error=1e-6,
            flag_masks={"EFAC1": mask, "EQUAD1": mask},
        )
        params = make_params(
            ("EFAC1", "EQUAD1"), [1.3, 0.8e-6], units=("", "s"),
        )
        noise = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))

        Ndiag, U, Phidiag = noise.covariance(toa_data, params)
        assert U is None
        assert Phidiag is None

        n_draws = 5000
        keys = jax.random.split(jax.random.PRNGKey(0), n_draws)
        draws = jnp.stack([noise.generate(toa_data, params, k) for k in keys])
        empirical_var = jnp.var(draws, axis=0)

        npt.assert_allclose(
            np.array(empirical_var), np.array(Ndiag),
            rtol=0.15, err_msg="Empirical variance doesn't match covariance",
        )


# ---------------------------------------------------------------------------
# ECORR noise generation
# ---------------------------------------------------------------------------


class TestEcorrNoiseGenerate:
    """Tests for EcorrNoise.generate."""

    @pytest.fixture
    def ecorr_setup(self):
        # 12 TOAs in 3 epochs of 4
        n_toas = 12
        n_epochs = 3
        ecorr_val = 2e-6  # seconds

        U_np = np.zeros((n_toas, n_epochs))
        for i in range(n_epochs):
            U_np[i * 4 : (i + 1) * 4, i] = 1.0
        U = jnp.array(U_np)

        ecorr = EcorrNoise(
            ecorr_names=("ECORR1",),
            quantization_matrix=U,
            ecorr_epoch_slices=((0, n_epochs),),
        )
        params = make_params(("ECORR1",), [ecorr_val], units=("s",))
        toa_data = make_toa_data(n_toas=n_toas)

        return toa_data, params, ecorr, U

    def test_intra_epoch_correlation(self, ecorr_setup):
        """TOAs within the same epoch should get identical noise."""
        toa_data, params, ecorr, _ = ecorr_setup
        key = jax.random.PRNGKey(7)

        draws = ecorr.generate(toa_data, params, key)

        # Epoch 0: TOAs 0-3 should be identical
        assert np.allclose(draws[0], draws[1])
        assert np.allclose(draws[0], draws[2])
        assert np.allclose(draws[0], draws[3])

        # Epoch 1: TOAs 4-7
        assert np.allclose(draws[4], draws[5])
        assert np.allclose(draws[4], draws[6])

    def test_inter_epoch_independence(self, ecorr_setup):
        """Different epochs should have different noise (with high probability)."""
        toa_data, params, ecorr, _ = ecorr_setup
        key = jax.random.PRNGKey(7)

        draws = ecorr.generate(toa_data, params, key)

        # Epoch 0 vs epoch 1 should differ
        assert not np.allclose(draws[0], draws[4])

    def test_epoch_variance(self, ecorr_setup):
        """Variance of per-epoch noise should match ECORR²."""
        toa_data, params, ecorr, U = ecorr_setup
        ecorr_val = 2e-6

        n_draws = 5000
        keys = jax.random.split(jax.random.PRNGKey(0), n_draws)

        # Collect the noise value for one TOA per epoch
        epoch_samples = {0: [], 1: [], 2: []}
        for k in keys:
            draw = ecorr.generate(toa_data, params, k)
            epoch_samples[0].append(float(draw[0]))   # epoch 0
            epoch_samples[1].append(float(draw[4]))   # epoch 1
            epoch_samples[2].append(float(draw[8]))   # epoch 2

        for ep in range(3):
            samples = np.array(epoch_samples[ep])
            npt.assert_allclose(
                np.var(samples), ecorr_val ** 2,
                rtol=0.15,
                err_msg=f"Epoch {ep} variance doesn't match ECORR²",
            )
            assert np.isclose(np.mean(samples), 0.0, atol=ecorr_val * 0.1)


# ---------------------------------------------------------------------------
# Combined noise: simulate_noise
# ---------------------------------------------------------------------------


class TestSimulateNoise:
    """Tests for simulate_noise with multiple NoiseComponents."""

    def test_combined_whitening(self):
        """Whitening combined white + ECORR noise via Cholesky should yield std ~ 1."""
        n_toas = 40
        n_epochs = 5
        efac_val = 1.2
        equad_val = 0.5e-6
        ecorr_val = 1.5e-6

        # Quantization matrix: 8 TOAs per epoch
        U_np = np.zeros((n_toas, n_epochs))
        for i in range(n_epochs):
            U_np[i * 8 : (i + 1) * 8, i] = 1.0
        U = jnp.array(U_np)

        mask = np.ones(n_toas, dtype=bool)
        toa_data = make_toa_data(
            n_toas=n_toas,
            error=1e-6,
            flag_masks={"EFAC1": mask, "EQUAD1": mask},
        )
        params = make_params(
            ("EFAC1", "EQUAD1", "ECORR1"),
            [efac_val, equad_val, ecorr_val],
            units=("", "s", "s"),
        )

        white = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
        ecorr = EcorrNoise(
            ecorr_names=("ECORR1",),
            quantization_matrix=U,
            ecorr_epoch_slices=((0, n_epochs),),
        )

        # Build full covariance: C = diag(sigma²) + U diag(ECORR²) U^T
        Ndiag_w, _, _ = white.covariance(toa_data, params)
        Ndiag_e, U_e, Phi_e = ecorr.covariance(toa_data, params)
        C = jnp.diag(Ndiag_w + Ndiag_e) + U_e @ jnp.diag(Phi_e) @ U_e.T
        L = jnp.linalg.cholesky(C)

        n_draws = 2000
        keys = jax.random.split(jax.random.PRNGKey(123), n_draws)
        whitened_all = []
        for k in keys:
            delays = simulate_noise(toa_data, params, k, [white, ecorr])
            w = jax.scipy.linalg.solve_triangular(L, delays, lower=True)
            whitened_all.append(w)

        whitened = jnp.stack(whitened_all)

        assert np.isclose(np.std(whitened), 1.0, atol=0.1)
        assert np.isclose(np.mean(whitened), 0.0, atol=0.05)

    def test_different_keys_give_different_noise(self):
        """Two different keys should produce different noise realizations."""
        n_toas = 20
        mask = np.ones(n_toas, dtype=bool)
        toa_data = make_toa_data(
            n_toas=n_toas,
            error=1e-6,
            flag_masks={"EFAC1": mask},
        )
        params = make_params(("EFAC1",), [1.0], units=("",))
        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())

        d1 = simulate_noise(toa_data, params, jax.random.PRNGKey(0), [white])
        d2 = simulate_noise(toa_data, params, jax.random.PRNGKey(1), [white])

        assert not np.allclose(d1, d2)

    def test_same_key_is_reproducible(self):
        """Same key should produce identical noise."""
        n_toas = 20
        mask = np.ones(n_toas, dtype=bool)
        toa_data = make_toa_data(
            n_toas=n_toas,
            error=1e-6,
            flag_masks={"EFAC1": mask},
        )
        params = make_params(("EFAC1",), [1.0], units=("",))
        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())

        key = jax.random.PRNGKey(42)
        d1 = simulate_noise(toa_data, params, key, [white])
        d2 = simulate_noise(toa_data, params, key, [white])

        npt.assert_array_equal(d1, d2)

    def test_empty_components_gives_zeros(self):
        """No noise components should produce zero delays."""
        toa_data = make_toa_data(n_toas=10)
        params = make_params((), [])
        key = jax.random.PRNGKey(0)

        delays = simulate_noise(toa_data, params, key, [])
        npt.assert_array_equal(delays, jnp.zeros(10))


# ---------------------------------------------------------------------------
# Integration: GLS fitter whitening
# ---------------------------------------------------------------------------


class TestGLSWhitening:
    """End-to-end test: generate noise, fit with GLS, whiten residuals."""

    @pytest.fixture(scope="class")
    def gls_fit_result(self):
        """Generate fake TOAs with noise, fit with GLS, return whitened residuals."""
        import pint.models as pm
        from pint.simulation import make_fake_toas_uniform

        from jaxpint.bridge import (
            build_timing_model,
            pint_model_to_params,
            pint_toas_to_jax,
        )
        from jaxpint.fitters import GLSFitter

        # Use PINT's B1855+09 example data (has EFAC, EQUAD, ECORR)
        pint_data = Path(pm.__file__).parent.parent / "data" / "examples"
        par_file = pint_data / "B1855+09_NANOGrav_9yv1.gls.par"

        if not par_file.exists():
            pytest.skip("PINT example data not found")

        pint_model = pm.get_model(str(par_file))

        # Remove components that need flags not present on uniform fake TOAs
        pint_model.remove_component("DispersionDMX")
        pint_model.remove_component("PhaseJump")
        pint_model.remove_component("FD")
        pint_model.PLANET_SHAPIRO.value = False

        # Create fake TOAs with PINT (no noise yet — we add it via JaxPINT)
        import astropy.units as u

        pint_toas = make_fake_toas_uniform(
            pint_model.START.value,
            pint_model.FINISH.value,
            500,
            pint_model,
            error=1 * u.us,
            add_noise=False,
        )

        # Convert to JaxPINT
        toa_data = pint_toas_to_jax(pint_toas, model=pint_model)
        params = pint_model_to_params(pint_model).params
        jax_model, noise_model = build_timing_model(
            pint_model, pint_toas
        )

        # Build noise components list for simulation
        noise_components = []
        if noise_model.white_noise is not None:
            noise_components.append(noise_model.white_noise)
        noise_components.extend(noise_model.correlated)

        # Generate fake TOAs with JaxPINT noise
        key = jax.random.PRNGKey(2024)
        fake_toa_data = make_fake_toas(
            jax_model, toa_data, params, key,
            noise_components=noise_components,
        )

        # Fit with GLS (augmented mode to get noise_realizations)
        fit_params = copy.deepcopy(params)
        fitter = GLSFitter(
            jax_model, fake_toa_data, fit_params,
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
        """Whitened residuals should have std ~ 1."""
        whitened, _ = gls_fit_result
        assert np.isclose(np.std(whitened), 1.0, atol=0.2)

    @pytest.mark.slow
    def test_whitened_mean(self, gls_fit_result):
        """Whitened residuals should have mean ~ 0."""
        whitened, _ = gls_fit_result
        assert np.isclose(np.mean(whitened), 0.0, atol=0.05)

    @pytest.mark.slow
    def test_reduced_chi2_near_one(self, gls_fit_result):
        """Reduced chi-squared should be near 1 for correctly generated noise."""
        _, result = gls_fit_result
        assert np.isclose(result.reduced_chi2, 1.0, atol=0.3)
