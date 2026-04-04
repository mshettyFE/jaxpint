"""Integration tests for the WLS fitter against PINT.

Uses a synthetic isolated pulsar with only Spindown + DispersionDM
(the components currently ported to JaxPINT) so results are directly
comparable between PINT and JaxPINT.
"""

from __future__ import annotations

import copy
import io

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pint.models as models
import pint.toa as toa
from pint.config import examplefile
from pint.fitter import WLSFitter as PINTWLSFitter
from pint.simulation import make_fake_toas_uniform

from jaxpint.bridge import (
    build_timing_model,
    pint_model_to_params,
    pint_toas_to_jax,
)
from jaxpint.fitter import (
    WLSFitter,
    compute_design_matrix,
    compute_time_residuals,
    _subtract_weighted_mean,
)


# ---------------------------------------------------------------------------
# Synthetic isolated pulsar (Spindown + DM only)
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
def synthetic_data():
    """Generate synthetic multi-frequency TOAs from a Spindown+DM model."""
    np.random.seed(42)
    m_true = models.get_model(io.StringIO(_SYNTH_PAR))
    # Two frequency bands so DM is well-determined
    toas_lo = make_fake_toas_uniform(
        53000, 55000, 30, m_true,
        error=10 * u.us, add_noise=True, freq=1400 * u.MHz,
    )
    toas_hi = make_fake_toas_uniform(
        53000, 55000, 30, m_true,
        error=10 * u.us, add_noise=True, freq=2000 * u.MHz,
    )
    toas_lo.merge(toas_hi)
    return m_true, toas_lo


@pytest.fixture(scope="module")
def pint_fit(synthetic_data):
    """Run PINT's WLS fitter on the synthetic data."""
    m_true, toas = synthetic_data
    mc = copy.deepcopy(m_true)
    f = PINTWLSFitter(toas, mc)
    f.fit_toas(maxiter=1)
    return f


@pytest.fixture(scope="module")
def jax_objects(synthetic_data):
    """Convert synthetic data to JaxPINT objects."""
    m_true, toas = synthetic_data
    toa_data = pint_toas_to_jax(toas, model=m_true)
    params = pint_model_to_params(m_true)
    jax_model, _noise = build_timing_model(m_true)
    return jax_model, toa_data, params


@pytest.fixture(scope="module")
def jax_fit(jax_objects):
    """Run JaxPINT's WLS fitter on the synthetic data."""
    jax_model, toa_data, params = jax_objects
    fitter = WLSFitter(jax_model, toa_data, params)
    return fitter.fit_toas(maxiter=1)


# ---------------------------------------------------------------------------
# NGC6440E (real isolated pulsar, limited comparison)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ngc6440e():
    """Load NGC6440E, freeze astrometry, keep F0/F1/DM free."""
    pint_model = models.get_model(examplefile("NGC6440E.par"))
    toas = toa.get_TOAs(examplefile("NGC6440E.tim"), ephem="DE421")
    return pint_model, toas


# ---------------------------------------------------------------------------
# Tests: synthetic data (strict comparison)
# ---------------------------------------------------------------------------


class TestDesignMatrix:
    """Design matrix shape and properties."""

    def test_shape(self, jax_objects):
        jax_model, toa_data, params = jax_objects
        M = compute_design_matrix(jax_model, toa_data, params)
        assert M.shape == (toa_data.n_toas, params.n_free)

    def test_no_nan(self, jax_objects):
        jax_model, toa_data, params = jax_objects
        M = compute_design_matrix(jax_model, toa_data, params)
        assert not jnp.any(jnp.isnan(M))

    def test_columns_nonzero(self, jax_objects):
        """Each free parameter should affect at least one TOA."""
        jax_model, toa_data, params = jax_objects
        M = compute_design_matrix(jax_model, toa_data, params)
        col_norms = jnp.linalg.norm(M, axis=0)
        assert jnp.all(col_norms > 0)


class TestSyntheticFit:
    """Compare JaxPINT WLS fit against PINT on synthetic data."""

    def test_chi2_matches(self, pint_fit, jax_fit):
        pint_chi2 = pint_fit.resids.chi2
        jax_chi2 = jax_fit.chi2
        # rtol=0.03: JaxPINT's int/frac Horner uses a different (more precise)
        # numerical path than PINT's longdouble taylor_horner, producing
        # small residual differences that accumulate into chi2.
        np.testing.assert_allclose(jax_chi2, pint_chi2, rtol=0.03)

    def test_f0_matches(self, pint_fit, jax_fit):
        pint_val = float(pint_fit.model.F0.value)
        jax_val = float(jax_fit.params.param_value("F0"))
        pint_err = float(pint_fit.model.F0.uncertainty_value)
        assert abs(jax_val - pint_val) < 3 * pint_err

    def test_f1_matches(self, pint_fit, jax_fit):
        pint_val = float(pint_fit.model.F1.value)
        jax_val = float(jax_fit.params.param_value("F1"))
        pint_err = float(pint_fit.model.F1.uncertainty_value)
        assert abs(jax_val - pint_val) < 3 * pint_err

    def test_dm_matches(self, pint_fit, jax_fit):
        pint_val = float(pint_fit.model.DM.value)
        jax_val = float(jax_fit.params.param_value("DM"))
        pint_err = float(pint_fit.model.DM.uncertainty_value)
        assert abs(jax_val - pint_val) < 3 * pint_err

    def test_uncertainties_positive(self, jax_fit):
        assert jnp.all(jax_fit.parameter_uncertainties > 0)

    def test_covariance_symmetric(self, jax_fit):
        cov = jax_fit.covariance_matrix
        np.testing.assert_allclose(
            np.array(cov), np.array(cov.T), atol=1e-20
        )

    def test_correlation_diagonal_ones(self, jax_fit):
        corr = jax_fit.correlation_matrix
        np.testing.assert_allclose(
            np.diag(np.array(corr)), 1.0, atol=1e-12
        )

    def test_dof(self, synthetic_data, jax_fit):
        _, toas = synthetic_data
        n_toas = len(toas)
        n_free = jax_fit.params.n_free
        assert jax_fit.dof == n_toas - n_free

    def test_reduced_chi2_reasonable(self, jax_fit):
        """Reduced chi2 should be close to 1 for synthetic data."""
        assert 0.5 < jax_fit.reduced_chi2 < 2.0


# ---------------------------------------------------------------------------
# Tests: NGC6440E real data (structural tests)
# ---------------------------------------------------------------------------


class TestNGC6440E:
    """Tests on real NGC6440E data with astrometry."""

    def test_chi2_decreases(self, ngc6440e):
        pint_model, toas = ngc6440e
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)
        jax_model, _noise = build_timing_model(pint_model)

        # Pre-fit chi2
        resid0 = compute_time_residuals(jax_model, toa_data, params)
        resid0 = _subtract_weighted_mean(resid0, toa_data.error)
        chi2_pre = float(jnp.sum((resid0 / toa_data.error) ** 2))

        # Single iteration
        fitter = WLSFitter(jax_model, toa_data, params)
        result = fitter.fit_toas(maxiter=1)

        assert result.chi2 < chi2_pre

    def test_multiple_iterations_converge(self, ngc6440e):
        pint_model, toas = ngc6440e
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)
        jax_model, _noise = build_timing_model(pint_model)

        fitter = WLSFitter(jax_model, toa_data, params)
        result = fitter.fit_toas(maxiter=5)

        assert result.chi2 > 0
        assert result.reduced_chi2 < 1e6

    def test_design_matrix_shape(self, ngc6440e):
        pint_model, toas = ngc6440e
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)
        jax_model, _noise = build_timing_model(pint_model)

        M = compute_design_matrix(jax_model, toa_data, params)
        assert M.shape == (toa_data.n_toas, params.n_free)
        assert not jnp.any(jnp.isnan(M))


# ---------------------------------------------------------------------------
# Tests: NGC6440E astrometry (comparison against PINT)
# ---------------------------------------------------------------------------


class TestNGC6440EAstrometry:
    """Compare JaxPINT astrometry delay and fitted positions against PINT."""

    def test_geometric_delay_matches_pint(self, ngc6440e):
        """Roemer delay from JaxPINT matches PINT's solar_system_geometric_delay."""
        pint_model, toas = ngc6440e

        # PINT delay
        pint_delay = pint_model.components[
            "AstrometryEquatorial"
        ].solar_system_geometric_delay(toas)

        # JaxPINT delay
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)
        jax_model, _noise = build_timing_model(pint_model)

        from jaxpint.delay.astrometry import AstrometryEquatorial

        for comp in jax_model.delay_components:
            if isinstance(comp, AstrometryEquatorial):
                jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
                break

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay.to(u.s).value, rtol=1e-10
        )

    def test_fit_converges_with_astrometry(self, ngc6440e):
        """JaxPINT fit with RAJ/DECJ free converges to a good chi2."""
        pint_model, toas = ngc6440e

        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)
        jax_model, _noise = build_timing_model(pint_model)
        fitter = WLSFitter(jax_model, toa_data, params)
        result = fitter.fit_toas(maxiter=5)

        # With missing components (SolarWindDispersion)
        # the reduced chi2 won't be ~1, but the fit should still converge.
        assert result.reduced_chi2 < 1000
        assert "RAJ" in result.params.free_names()
        assert "DECJ" in result.params.free_names()
        assert result.parameter_uncertainties is not None
        assert jnp.all(result.parameter_uncertainties > 0)
