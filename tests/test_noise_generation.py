"""Tests for noise generation (NoiseComponent.generate and simulate_noise)."""

from __future__ import annotations

import copy
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

    def test_whitened_moments(self, white_noise_setup):
        """Noise divided by scaled_sigma is ~N(0, 1): both moments of ONE
        realization .

        At n=5000: SE(std) ~ 1/sqrt(2n) ~ 0.01 and SE(mean) ~ 1/sqrt(n)
        ~ 0.014, so atol=0.05 sits at ~4-5 sigma.
        """
        toa_data, params, noise = white_noise_setup
        key = jax.random.PRNGKey(42)

        draws = noise.generate(toa_data, params, key)
        sigma = noise.scaled_sigma(toa_data, params)
        whitened = draws / sigma

        assert np.isclose(np.std(whitened), 1.0, atol=0.05)
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
        assert U.shape == (n_toas, 0)
        assert Phidiag.shape == (0,)

        # Variance-estimator relative error is sqrt(2/n) ~ 3.2% at 2000
        # draws; rtol=0.15 is ~4.7 sigma per element across 200
        # independent (white) elements.
        n_draws = 2000
        keys = jax.random.split(jax.random.PRNGKey(0), n_draws)
        draws = jax.vmap(lambda k: noise.generate(toa_data, params, k))(keys)
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

        # One draw per key (vmapped), then read one TOA per epoch. Columns
        # 0/4/8 index epoch 0/1/2.
        draws = jax.vmap(lambda k: ecorr.generate(toa_data, params, k))(keys)
        epoch_cols = {0: 0, 1: 4, 2: 8}

        for ep in range(3):
            samples = np.array(draws[:, epoch_cols[ep]])
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

        # 1000 draws x 40 elements = 40k whitened samples: SE(std) ~
        # 0.0035 and SE(mean) ~ 0.005, so the atols sit at ~10-28 sigma.
        n_draws = 1000
        keys = jax.random.split(jax.random.PRNGKey(123), n_draws)
        def _whiten(k):
            delays = simulate_noise(toa_data, params, k, [white, ecorr])
            return jax.scipy.linalg.solve_triangular(L, delays, lower=True)

        whitened = jax.vmap(_whiten)(keys)

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
        """Generate fake TOAs with noise, fit with GLS, return whitened residuals.
        """
        from io import StringIO

        import jaxpint.par as jpar
        from jaxpint import build_model
        from jaxpint.fitters import GLSFitter
        from jaxpint.simulation import make_uniform_toa_data

        par_file = (
            Path(__file__).resolve().parent
            / "data"
            / "pint_inputs"
            / "B1855+09_NANOGrav_9yv1.gls.par"
        )
        drop = ("DMX", "JUMP", "FD", "PLANET_SHAPIRO")
        par_text = "\n".join(
            line
            for line in par_file.read_text().splitlines()
            if line.strip() and not line.split()[0].startswith(drop)
        )
        par_result = jpar.get_model(StringIO(par_text))
        # START/FINISH are epoch-type parameters: the integer MJD day lives in
        # epoch_int_values (static) and param_value returns only the
        # fractional day, so the two halves must be recombined.
        p = par_result.params
        start = p.epoch_int_values["START"] + float(p.param_value("START"))
        finish = p.epoch_int_values["FINISH"] + float(p.param_value("FINISH"))

        # Bare epochs only -- make_fake_toas below does its own zeroing, so
        # the model-realizing generator would zero residuals twice for nothing.
        toa_data = make_uniform_toa_data(
            start, finish, 500, par_result,
            obs="ao", freq_mhz=1400.0, error_us=1.0,
        )
        params = par_result.params
        jax_model, noise_model = build_model(par_result, toa_data)

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

        # Whitening via the public API (jaxpint.fitters.whiten_residuals);
        # this fixture's earlier hand-rolled version is what that API was
        # extracted from, so it now doubles as the API's end-to-end consumer.
        from jaxpint.fitters import whiten_residuals

        whitened = whiten_residuals(
            result.residuals,
            fake_toa_data,
            result.params,
            noise_model,
            noise_realizations=result.noise_realizations,
        )
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
