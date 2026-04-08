"""Tests for single_pulsar_logL."""

from __future__ import annotations

import io

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import pint.models as models
from pint.simulation import make_fake_toas_uniform

from jaxpint.bridge import build_timing_model, pint_model_to_params, pint_toas_to_jax
from jaxpint.fitter import compute_time_residuals
from jaxpint.likelihood import single_pulsar_logL
from jaxpint.noise import NoiseModel, ScaleToaError
from jaxpint.noise.red_noise import PLRedNoise
from jaxpint.utils import build_fourier_basis, woodbury_dot


# ---------------------------------------------------------------------------
# Synthetic pulsar (Spindown + DM, white noise only)
# ---------------------------------------------------------------------------

_SYNTH_PAR = """\
PSR           J0000+0000
EPHEM         DE421
CLK           TT(BIPM2019)
UNITS         TDB
START         53000 1
FINISH        55000 1
PEPOCH        54000
F0            100.0 1
F1            -1e-15 1
DM            15.0 1
TZRMJD        54000
TZRFRQ        1400
TZRSITE       @
"""


@pytest.fixture(scope="module")
def synth_objects():
    """Timing model, noise model, TOA data, and params from a synthetic pulsar."""
    np.random.seed(42)
    m_true = models.get_model(io.StringIO(_SYNTH_PAR))
    toas = make_fake_toas_uniform(
        53000, 55000, 30, m_true,
        error=10 * u.us, add_noise=True, freq=1400 * u.MHz,
    )
    toa_data = pint_toas_to_jax(toas, model=m_true)
    params = pint_model_to_params(m_true).params
    jax_model, noise_model = build_timing_model(m_true)
    return jax_model, noise_model, toa_data, params


# ---------------------------------------------------------------------------
# Synthetic pulsar with red noise
# ---------------------------------------------------------------------------

_SYNTH_PAR_RN = """\
PSR           J0000+0000
EPHEM         DE421
CLK           TT(BIPM2019)
UNITS         TDB
START         53000 1
FINISH        55000 1
PEPOCH        54000
F0            100.0 1
F1            -1e-15 1
DM            15.0 1
TZRMJD        54000
TZRFRQ        1400
TZRSITE       @
TNREDAMP      -13.0
TNREDGAM      3.5
TNREDC        5
"""


@pytest.fixture(scope="module")
def synth_objects_rn():
    """Synthetic pulsar with power-law red noise."""
    np.random.seed(42)
    m_true = models.get_model(io.StringIO(_SYNTH_PAR_RN))
    toas = make_fake_toas_uniform(
        53000, 55000, 60, m_true,
        error=10 * u.us, add_noise=True, freq=1400 * u.MHz,
    )
    toa_data = pint_toas_to_jax(toas, model=m_true)
    params = pint_model_to_params(m_true).params
    jax_model, noise_model = build_timing_model(m_true)
    return jax_model, noise_model, toa_data, params


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoInjection:
    """Verify the basic log-likelihood (no external injections)."""

    def test_matches_chi2_plus_normalization(self, synth_objects):
        """logL should equal -0.5*(chi2 + logdet + n*log(2pi))."""
        jax_model, noise_model, toa_data, params = synth_objects

        logL = single_pulsar_logL(toa_data, jax_model, noise_model, params)

        # Compute the same thing manually
        r = compute_time_residuals(jax_model, toa_data, params)
        Ndiag, U, Phi = noise_model.covariance(toa_data, params)
        rCr, logdetC = woodbury_dot(Ndiag, U, Phi, r, r)
        n = r.shape[0]
        expected = -0.5 * rCr - 0.5 * logdetC - 0.5 * n * jnp.log(2 * jnp.pi)

        npt.assert_allclose(float(logL), float(expected), rtol=1e-12)

    def test_returns_scalar(self, synth_objects):
        jax_model, noise_model, toa_data, params = synth_objects
        logL = single_pulsar_logL(toa_data, jax_model, noise_model, params)
        assert logL.shape == ()

    def test_finite_value(self, synth_objects):
        jax_model, noise_model, toa_data, params = synth_objects
        logL = single_pulsar_logL(toa_data, jax_model, noise_model, params)
        assert jnp.isfinite(logL)

    def test_with_correlated_noise(self, synth_objects_rn):
        """logL with red noise should still match manual Woodbury computation."""
        jax_model, noise_model, toa_data, params = synth_objects_rn

        logL = single_pulsar_logL(toa_data, jax_model, noise_model, params)

        r = compute_time_residuals(jax_model, toa_data, params)
        Ndiag, U, Phi = noise_model.covariance(toa_data, params)
        rCr, logdetC = woodbury_dot(Ndiag, U, Phi, r, r)
        n = r.shape[0]
        expected = -0.5 * rCr - 0.5 * logdetC - 0.5 * n * jnp.log(2 * jnp.pi)

        npt.assert_allclose(float(logL), float(expected), rtol=1e-12)


class TestJIT:
    """JIT compilation should work without errors."""

    def test_jit_produces_same_result(self, synth_objects):
        jax_model, noise_model, toa_data, params = synth_objects
        logL_eager = single_pulsar_logL(toa_data, jax_model, noise_model, params)
        logL_jit = jax.jit(single_pulsar_logL)(toa_data, jax_model, noise_model, params)
        npt.assert_allclose(float(logL_jit), float(logL_eager), rtol=1e-12)


class TestAutodiff:
    """Autodiff w.r.t. timing parameters should produce finite gradients."""

    def test_grad_wrt_params(self, synth_objects):
        jax_model, noise_model, toa_data, params = synth_objects

        def logL_of_values(values):
            p = params.with_free_values(values)
            return single_pulsar_logL(toa_data, jax_model, noise_model, p)

        grad = jax.grad(logL_of_values)(params.free_values())
        assert jnp.all(jnp.isfinite(grad))
        # At least some gradients should be non-zero
        assert jnp.any(grad != 0.0)


class TestExternalDelay:
    """Verify external delay injection."""

    def test_delay_changes_likelihood(self, synth_objects):
        """Injecting a non-zero delay should change the log-likelihood."""
        jax_model, noise_model, toa_data, params = synth_objects

        logL_base = single_pulsar_logL(toa_data, jax_model, noise_model, params)

        # Sinusoidal delay: A * sin(2 * pi * f * t)
        t = toa_data.tdb_frac  # fractional day within each TOA
        A = 1e-6  # 1 microsecond amplitude
        f = 2.0   # cycles per day
        delay = A * jnp.sin(2 * jnp.pi * f * t)

        logL_delayed = single_pulsar_logL(
            toa_data, jax_model, noise_model, params,
            external_delay=delay,
        )

        assert jnp.isfinite(logL_delayed)
        assert float(logL_base) != float(logL_delayed)

    def test_zero_delay_matches_base(self, synth_objects):
        """A zero external delay should give the same result as no delay."""
        jax_model, noise_model, toa_data, params = synth_objects

        logL_base = single_pulsar_logL(toa_data, jax_model, noise_model, params)
        logL_zero = single_pulsar_logL(
            toa_data, jax_model, noise_model, params,
            external_delay=jnp.zeros(toa_data.n_toas),
        )
        npt.assert_allclose(float(logL_zero), float(logL_base), rtol=1e-12)

    def test_delay_is_deterministic(self, synth_objects):
        """Same delay should give the same result every time."""
        jax_model, noise_model, toa_data, params = synth_objects
        t = toa_data.tdb_frac
        delay = 1e-6 * jnp.sin(2 * jnp.pi * 2.0 * t)

        logL_1 = single_pulsar_logL(
            toa_data, jax_model, noise_model, params, external_delay=delay,
        )
        logL_2 = single_pulsar_logL(
            toa_data, jax_model, noise_model, params, external_delay=delay,
        )
        assert float(logL_1) == float(logL_2)


class TestExternalCovariance:
    """Verify external covariance injection."""

    def test_augmented_cov_matches_brute_force(self, synth_objects):
        """Injecting (U_ext, Phi_ext) should match brute-force dense computation."""
        jax_model, noise_model, toa_data, params = synth_objects

        r = compute_time_residuals(jax_model, toa_data, params)
        Ndiag, U, Phi = noise_model.covariance(toa_data, params)

        # Small random external covariance
        key = jax.random.PRNGKey(0)
        n_ext = 3
        U_ext = jax.random.normal(key, (toa_data.n_toas, n_ext)) * 1e-6
        Phi_ext = jnp.array([1e-12, 2e-12, 3e-12])

        # Via single_pulsar_logL
        logL = single_pulsar_logL(
            toa_data, jax_model, noise_model, params,
            external_cov=(U_ext, Phi_ext),
        )

        # Via brute-force dense computation
        U_all = jnp.concatenate([U, U_ext], axis=1)
        Phi_all = jnp.concatenate([Phi, Phi_ext])
        C = jnp.diag(Ndiag) + U_all @ jnp.diag(Phi_all) @ U_all.T
        n = r.shape[0]
        sign, logdetC = jnp.linalg.slogdet(C)
        rCr = r @ jnp.linalg.solve(C, r)
        expected = -0.5 * rCr - 0.5 * logdetC - 0.5 * n * jnp.log(2 * jnp.pi)

        npt.assert_allclose(float(logL), float(expected), rtol=1e-8)


class TestAutodiffExternalSignal:
    """Autodiff through external signal parameters."""

    def test_grad_wrt_external_delay_frequency(self, synth_objects):
        """Gradient of logL w.r.t. sinusoidal delay frequency should be finite and non-zero."""
        jax_model, noise_model, toa_data, params = synth_objects
        t = toa_data.tdb_frac
        A = 1e-6

        def logL_of_freq(f):
            delay = A * jnp.sin(2 * jnp.pi * f * t)
            return single_pulsar_logL(
                toa_data, jax_model, noise_model, params,
                external_delay=delay,
            )

        f0 = 2.0
        grad_f = jax.grad(logL_of_freq)(f0)
        assert jnp.isfinite(grad_f)
        assert float(grad_f) != 0.0
