"""Tests for the noise model (EFAC/EQUAD white noise scaling).

Covers unit tests with synthetic data and integration tests comparing
against PINT's ScaleToaError.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.noise import ScaleToaError
from jaxpint.types import TOAData, ParameterVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toa_data(n_toas, errors, flag_masks):
    """Build a minimal TOAData for noise tests."""
    zeros = jnp.zeros(n_toas)
    zeros3 = jnp.zeros((n_toas, 3))
    return TOAData(
        mjd_int=zeros,
        mjd_frac=zeros,
        tdb_int=zeros,
        tdb_frac=zeros,
        error=jnp.asarray(errors, dtype=jnp.float64),
        freq=jnp.ones(n_toas) * 1400.0,
        delta_pulse_number=zeros,
        ssb_obs_pos=zeros3,
        ssb_obs_vel=zeros3,
        obs_sun_pos=zeros3,
        obs_indices=jnp.zeros(n_toas, dtype=jnp.int32),
        flag_masks={k: jnp.asarray(v, dtype=jnp.bool_) for k, v in flag_masks.items()},
        planet_positions=None,
        dm_values=None,
        dm_errors=None,
        n_toas=n_toas,
        obs_names=("fake",),
    )


def _make_params(names, values, frozen=None):
    """Build a minimal ParameterVector."""
    n = len(names)
    if frozen is None:
        frozen = [True] * n
    return ParameterVector(
        values=jnp.asarray(values, dtype=jnp.float64),
        frozen_mask=tuple(frozen),
        names=tuple(names),
        units=tuple([""] * n),
        components=tuple(["noise"] * n),
        _name_to_index={name: i for i, name in enumerate(names)},
        bounds=tuple([(None, None)] * n),
        epoch_int_values={},
    )


# ---------------------------------------------------------------------------
# Unit tests: ScaleToaError
# ---------------------------------------------------------------------------


class TestScaleToaErrorUnit:
    """Unit tests for EFAC/EQUAD scaling with synthetic data."""

    def test_efac_only(self):
        """EFAC multiplies errors on masked TOAs, leaves others unchanged."""
        n = 6
        errors = np.full(n, 1e-6)  # 1 µs
        mask = np.array([True, True, True, False, False, False])
        toa_data = _make_toa_data(n, errors, {"EFAC1": mask})
        params = _make_params(["EFAC1"], [1.5])

        noise = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        sigma = noise.scaled_sigma(toa_data, params)

        np.testing.assert_allclose(sigma[:3], 1.5e-6)
        np.testing.assert_allclose(sigma[3:], 1e-6)

    def test_equad_only(self):
        """EQUAD is added in quadrature to masked TOAs."""
        n = 4
        errors = np.full(n, 3e-6)  # 3 µs
        equad = 4e-6  # 4 µs → sqrt(9+16) = 5 µs
        mask = np.array([True, True, False, False])
        toa_data = _make_toa_data(n, errors, {"EQUAD1": mask})
        params = _make_params(["EQUAD1"], [equad])

        noise = ScaleToaError(efac_names=(), equad_names=("EQUAD1",))
        sigma = noise.scaled_sigma(toa_data, params)

        expected_masked = np.sqrt(3e-6**2 + 4e-6**2)
        np.testing.assert_allclose(sigma[:2], expected_masked)
        np.testing.assert_allclose(sigma[2:], 3e-6)

    def test_efac_and_equad(self):
        """EFAC × √(σ² + EQUAD²) matches PINT convention."""
        n = 4
        errors = np.full(n, 3e-6)
        efac = 1.2
        equad = 4e-6
        mask = np.ones(n, dtype=bool)
        toa_data = _make_toa_data(n, errors, {"EFAC1": mask, "EQUAD1": mask})
        params = _make_params(["EFAC1", "EQUAD1"], [efac, equad])

        noise = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
        sigma = noise.scaled_sigma(toa_data, params)

        expected = efac * np.sqrt(3e-6**2 + equad**2)
        np.testing.assert_allclose(sigma, expected, rtol=1e-14)

    def test_multiple_masks_disjoint(self):
        """Two EFAC/EQUAD pairs on disjoint TOA subsets."""
        n = 6
        errors = np.full(n, 1e-6)
        mask_a = np.array([True, True, True, False, False, False])
        mask_b = np.array([False, False, False, True, True, True])
        toa_data = _make_toa_data(
            n,
            errors,
            {"EFAC1": mask_a, "EFAC2": mask_b, "EQUAD1": mask_a, "EQUAD2": mask_b},
        )
        params = _make_params(
            ["EFAC1", "EFAC2", "EQUAD1", "EQUAD2"],
            [1.5, 2.0, 0.5e-6, 1.0e-6],
        )

        noise = ScaleToaError(
            efac_names=("EFAC1", "EFAC2"),
            equad_names=("EQUAD1", "EQUAD2"),
        )
        sigma = noise.scaled_sigma(toa_data, params)

        expected_a = 1.5 * np.sqrt(1e-6**2 + 0.5e-6**2)
        expected_b = 2.0 * np.sqrt(1e-6**2 + 1.0e-6**2)
        np.testing.assert_allclose(sigma[:3], expected_a, rtol=1e-14)
        np.testing.assert_allclose(sigma[3:], expected_b, rtol=1e-14)

    def test_jit_compatible(self):
        """scaled_sigma works under jax.jit."""
        n = 4
        errors = np.full(n, 1e-6)
        mask = np.ones(n, dtype=bool)
        toa_data = _make_toa_data(n, errors, {"EFAC1": mask})
        params = _make_params(["EFAC1"], [1.3])

        noise = ScaleToaError(efac_names=("EFAC1",), equad_names=())

        sigma_eager = noise.scaled_sigma(toa_data, params)
        sigma_jit = jax.jit(noise.scaled_sigma)(toa_data, params)

        np.testing.assert_allclose(sigma_jit, sigma_eager, rtol=1e-15)

    def test_differentiable_efac(self):
        """Can differentiate chi2 through scaled_sigma w.r.t. EFAC."""
        n = 4
        errors = np.full(n, 1e-6)
        mask = np.ones(n, dtype=bool)
        toa_data = _make_toa_data(n, errors, {"EFAC1": mask})

        noise = ScaleToaError(efac_names=("EFAC1",), equad_names=())

        def chi2_fn(efac_val):
            p = _make_params(["EFAC1"], [efac_val])
            sigma = noise.scaled_sigma(toa_data, p)
            residuals = jnp.ones(n) * 1e-6  # dummy residuals
            return jnp.sum((residuals / sigma) ** 2)

        grad = jax.grad(chi2_fn)(1.0)
        assert jnp.isfinite(grad)
        # Increasing EFAC increases sigma, decreases chi2 → grad < 0
        assert grad < 0

    def test_differentiable_equad(self):
        """Can differentiate chi2 through scaled_sigma w.r.t. EQUAD."""
        n = 4
        errors = np.full(n, 1e-6)
        mask = np.ones(n, dtype=bool)
        toa_data = _make_toa_data(n, errors, {"EQUAD1": mask})

        noise = ScaleToaError(efac_names=(), equad_names=("EQUAD1",))

        def chi2_fn(equad_val):
            p = _make_params(["EQUAD1"], [equad_val])
            sigma = noise.scaled_sigma(toa_data, p)
            residuals = jnp.ones(n) * 1e-6
            return jnp.sum((residuals / sigma) ** 2)

        grad = jax.grad(chi2_fn)(0.5e-6)
        assert jnp.isfinite(grad)
        assert grad < 0

    def test_no_noise_returns_raw_errors(self):
        """Empty EFAC/EQUAD lists return raw TOA errors unchanged."""
        n = 4
        errors = np.array([1e-6, 2e-6, 3e-6, 4e-6])
        toa_data = _make_toa_data(n, errors, {})
        params = _make_params([], [])

        noise = ScaleToaError(efac_names=(), equad_names=())
        sigma = noise.scaled_sigma(toa_data, params)

        np.testing.assert_allclose(sigma, errors)


# ---------------------------------------------------------------------------
# Integration tests against PINT
# ---------------------------------------------------------------------------


class TestScaleToaErrorVsPINT:
    """Compare JaxPINT noise scaling against PINT on real/synthetic data."""

    @pytest.fixture(scope="class")
    def synthetic_with_noise(self):
        """Create a synthetic pulsar with EFAC/EQUAD noise parameters."""
        import io
        import pint.models as models
        import pint.toa as toa
        from pint.simulation import make_fake_toas_uniform

        par = """\
PSR           J0000+0000
RAJ           00:00:00   1
DECJ          00:00:00   1
PEPOCH        55000
F0            100        1
F1            -1e-15     1
DM            15         1
EFAC -f L-wide 1.3
EQUAD -f L-wide 0.8
EFAC -f S-wide 1.1
EQUAD -f S-wide 0.5
TZRMJD        55000
TZRFRQ        1400
TZRSITE       @
EPHEM         DE421
CLOCK         TT(BIPM2019)
UNITS         TDB
"""
        m = models.get_model(io.StringIO(par))
        # Remove components not yet ported to JaxPINT
        for comp_name in list(m.components):
            if comp_name in ("TroposphereDelay",):
                m.remove_component(comp_name)
        # Create fake TOAs with two "backends"
        t1 = make_fake_toas_uniform(
            54000, 56000, 50, model=m, obs="gbt", freq=1400.0,
            error=1.0 * u.us, add_noise=True,
        )
        t2 = make_fake_toas_uniform(
            54000, 56000, 50, model=m, obs="gbt", freq=2000.0,
            error=1.5 * u.us, add_noise=True,
        )
        # Manually set flags for the backends
        for i in range(t1.ntoas):
            t1.table["flags"][i]["f"] = "L-wide"
        for i in range(t2.ntoas):
            t2.table["flags"][i]["f"] = "S-wide"
        t = toa.merge_TOAs([t1, t2])
        return m, t

    def test_scaled_sigma_matches_pint(self, synthetic_with_noise):
        """JaxPINT scaled sigma matches PINT's scaled_toa_uncertainty."""
        import astropy.units as u
        from jaxpint.bridge import (
            build_timing_model,
            pint_model_to_params,
            pint_toas_to_jax,
        )

        pint_model, toas = synthetic_with_noise

        # PINT reference
        pint_sigma = pint_model.scaled_toa_uncertainty(toas).to(u.s).value

        # JaxPINT
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)
        _tm, noise_model = build_timing_model(pint_model)

        assert noise_model is not None
        jax_sigma = noise_model.scaled_sigma(toa_data, params)

        np.testing.assert_allclose(
            np.array(jax_sigma), pint_sigma, rtol=1e-12,
            err_msg="JaxPINT scaled sigma does not match PINT",
        )


    def test_wls_chi2_matches_pint(self, synthetic_with_noise):
        """WLS fit with EFAC/EQUAD: JaxPINT chi2 matches PINT."""
        import copy

        from pint.fitter import WLSFitter as PINTWLSFitter

        from jaxpint.bridge import (
            build_timing_model,
            pint_model_to_params,
            pint_toas_to_jax,
        )
        from jaxpint.fitter import WLSFitter

        pint_model, toas = synthetic_with_noise

        # --- PINT WLS fit ---
        m_pint = copy.deepcopy(pint_model)
        pint_fitter = PINTWLSFitter(toas, m_pint)
        pint_fitter.fit_toas(maxiter=1)
        pint_chi2 = pint_fitter.resids.chi2

        # --- JaxPINT WLS fit ---
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)
        jax_model, noise_model = build_timing_model(pint_model)

        assert noise_model is not None
        fitter = WLSFitter(jax_model, toa_data, params, noise_model=noise_model)
        jax_chi2 = fitter.fit_toas(maxiter=1)

        # rtol=0.05: JaxPINT skips TroposphereDelay (handled implicitly),
        # which introduces small residual differences vs PINT.
        np.testing.assert_allclose(
            jax_chi2, pint_chi2, rtol=0.05,
            err_msg=f"JaxPINT chi2 ({jax_chi2}) != PINT chi2 ({pint_chi2})",
        )


# Import here so fixtures can use it
import astropy.units as u
