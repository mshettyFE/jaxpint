"""Finite-difference verification of the implicit fit gradients.

:meth:`BaseFitter.fit_params` wraps the Gauss-Newton iteration in a
``jax.custom_vjp`` whose backward pass is derived from the implicit
function theorem (see ``docs/guides/differentiable_fitting.rst``).
These tests check that machinery end-to-end for all three fitters:
``jax.jacrev`` of the fitted free parameters -- with respect to a
frozen timing parameter, a frozen noise parameter, and an
``external_delay`` -- must match central finite differences of the
full fit.

All problems are synthetic (no PINT needed) and small; each fixture
simulates data at the true parameters so the fit converges in a few
Gauss-Newton steps and the fixed-point assumption of the IFT holds
(asserted via ``fit_gap``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.delay.dispersion_dm import DispersionDM
from jaxpint.fitters import GLSFitter, WLSFitter, WidebandGLSFitter
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel, PLRedNoise, ScaleToaError
from jaxpint.phase.spin import Spindown
from jaxpint.simulation import make_fake_toas
from tests.helpers import make_fourier_basis, make_params, make_toa_data

MAXITER = 4


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _assert_fit_gradient_matches_fd(
    f, x0, h, J_ad, rtol=5e-3, c_diff=3.0, ulp_safety=4.0
):
    """Check the reverse-mode gradient of a fit against central differences.

    ``f`` maps a scalar to the fitted free-parameter vector; ``J_ad`` is
    its reverse-mode derivative at ``x0``.  The finite-difference
    uncertainty is estimated *empirically* per component: the spread
    between the ``h`` and ``h/2`` central differences captures both
    truncation and solve-roundoff noise, and an ulp term bounds the
    quantization of each fitted value (a step that moves a parameter by
    less than ``eps * |y|`` is invisible to FD -- e.g. F0 ~ 100 with
    sub-1e-14 sensitivity).  Components whose sensitivity FD cannot
    resolve are only checked loosely; a final assertion guarantees the
    comparison is not vacuous (at least one component carries real FD
    signal).
    """
    eps = np.finfo(np.float64).eps
    y0 = np.asarray(f(x0), dtype=np.float64)
    J1 = np.asarray((f(x0 + h) - f(x0 - h)) / (2.0 * h), dtype=np.float64)
    J2 = np.asarray((f(x0 + h / 2) - f(x0 - h / 2)) / h, dtype=np.float64)
    J_ad = np.asarray(J_ad, dtype=np.float64).ravel()

    fd_unc = c_diff * np.abs(J1 - J2) + ulp_safety * eps * np.abs(y0) / h
    err = np.abs(J_ad - J1)
    tol = rtol * np.abs(J1) + fd_unc
    assert (err <= tol).all(), (
        f"AD/FD fit-gradient mismatch:\n"
        f"  J_ad    = {J_ad}\n"
        f"  J_fd(h) = {J1}\n"
        f"  J_fd(h/2) = {J2}\n"
        f"  err     = {err}\n"
        f"  tol     = {tol}\n"
        f"  ratio   = {err / tol}"
    )
    assert (np.abs(J1) > 5.0 * fd_unc).any(), (
        "FD signal below its noise floor for every component -- the "
        "comparison is vacuous; increase the step or the coupling."
    )


def _assert_converged(gap, uncertainties, frac=1e-3):
    """One further GN step must move each free param << its sigma."""
    gap = np.abs(np.asarray(gap))
    sigma = np.asarray(uncertainties)
    assert (gap < frac * sigma).all(), f"fit not converged: gap/sigma = {gap / sigma}"


# ---------------------------------------------------------------------------
# WLS: two-band Spindown + DM pulsar, DM frozen
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wls_problem():
    """Synthetic two-band pulsar; F0/F1 free, DM frozen (the IFT theta)."""
    n_toas = 60
    rng = np.random.default_rng(0)
    t_mjd = np.linspace(53000.0, 54000.0, n_toas) + rng.uniform(0.0, 0.01, n_toas)
    freq = np.where(np.arange(n_toas) % 2 == 0, 800.0, 1400.0)
    mask = np.ones(n_toas, dtype=bool)

    toa_data = make_toa_data(
        t_mjd=t_mjd,
        error=1e-6,
        freq=freq,
        flag_masks={"EFAC1": mask},
        tzr_tdb_int=53000.0,
        tzr_tdb_frac=0.5,
        tzr_freq=jnp.inf,
        tzr_ssb_obs_pos=np.zeros(3),
        tzr_obs_sun_pos=np.zeros(3),
    )

    spin = Spindown(spin_param_names=("F0", "F1"), pepoch_name="PEPOCH")
    disp = DispersionDM(dm_param_names=("DM",))
    model = TimingModel(delay_components=(disp,), phase_components=(spin,))

    params = make_params(
        ("F0", "F1", "PEPOCH", "DM", "DMEPOCH", "EFAC1"),
        (100.0, -1e-15, 0.0, 15.0, 0.0, 1.0),
        frozen_mask=(False, False, True, True, True, True),
        units=("Hz", "Hz/s", "day", "pc cm^-3", "day", ""),
        epoch_int_values={"PEPOCH": 53000.0, "DMEPOCH": 53000.0},
    )

    white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
    fake = make_fake_toas(
        model, toa_data, params, jax.random.PRNGKey(7), noise_components=[white]
    )
    fitter = WLSFitter(model, fake, params)
    sigma = fitter.fit_toas(maxiter=MAXITER).parameter_uncertainties
    return fitter, params, sigma


class TestWLSImplicitGradients:
    def test_fit_converged(self, wls_problem):
        fitter, params, sigma = wls_problem
        fitted = fitter.fit_params(maxiter=MAXITER)
        _assert_converged(fitter.fit_gap(fitted), sigma)

    @pytest.mark.slow
    def test_frozen_dm_gradient_matches_fd(self, wls_problem):
        fitter, params, _sigma = wls_problem
        idx = params.names.index("DM")

        def fitted_free(dm):
            p = params.with_values(params.values.at[idx].set(dm))
            return fitter.fit_params(p, maxiter=MAXITER).free_values()

        dm0 = params.values[idx]
        J_ad = jax.jacrev(fitted_free)(dm0)
        _assert_fit_gradient_matches_fd(fitted_free, dm0, 2e-3, J_ad)

    @pytest.mark.slow
    def test_external_delay_gradient_matches_fd(self, wls_problem):
        fitter, params, _sigma = wls_problem
        n = fitter.toa_data.n_toas
        phase = np.linspace(0.0, 4 * np.pi, n)
        delay0 = jnp.asarray(5e-7 * np.sin(phase))

        def fitted_free(delay):
            return fitter.fit_params(params, delay, maxiter=MAXITER).free_values()

        J_ad = jax.jacrev(fitted_free)(delay0)  # (n_free, n_toas)

        rng = np.random.default_rng(3)
        d = rng.standard_normal(n)
        d /= np.linalg.norm(d)
        d_jnp = jnp.asarray(d)
        _assert_fit_gradient_matches_fd(
            lambda s: fitted_free(delay0 + s * d_jnp),
            0.0,
            1e-6,
            np.asarray(J_ad) @ d,
        )

    def test_free_seed_gradient_is_zero(self, wls_problem):
        """The free entries of the input are only the iteration seed."""
        fitter, params, _sigma = wls_problem

        def fitted_free(values):
            return fitter.fit_params(
                params.with_values(values), maxiter=MAXITER
            ).free_values()

        J = jax.jacrev(fitted_free)(params.values)  # (n_free, n_params)
        free_idx = np.asarray(params.free_indices_array())
        npt.assert_array_equal(np.asarray(J)[:, free_idx], 0.0)


# ---------------------------------------------------------------------------
# GLS: Spindown + power-law red noise, gradients w.r.t. frozen noise params
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gls_problem():
    """Synthetic pulsar with red noise; theta enters only through C."""
    n_toas = 80
    T = 3.0 * 365.25 * 86400.0
    F, freqs, df, _t = make_fourier_basis(n_toas, 5, T)
    plred = PLRedNoise(
        fourier_basis=F,
        freqs=freqs,
        freq_bin_widths=df,
        tnredamp_name="TNREDAMP",
        tnredgam_name="TNREDGAM",
    )

    t_mjd = np.linspace(53000.0, 53000.0 + T / 86400.0, n_toas)
    mask = np.ones(n_toas, dtype=bool)
    toa_data = make_toa_data(
        t_mjd=t_mjd,
        error=1e-6,
        freq=1400.0,
        flag_masks={"EFAC1": mask},
        tzr_tdb_int=53000.0,
        tzr_tdb_frac=0.5,
        tzr_freq=jnp.inf,
        tzr_ssb_obs_pos=np.zeros(3),
        tzr_obs_sun_pos=np.zeros(3),
    )

    spin = Spindown(spin_param_names=("F0", "F1"), pepoch_name="PEPOCH")
    model = TimingModel(delay_components=(), phase_components=(spin,))

    params = make_params(
        ("F0", "F1", "PEPOCH", "EFAC1", "TNREDAMP", "TNREDGAM"),
        (100.0, -1e-15, 0.0, 1.0, -13.0, 3.5),
        frozen_mask=(False, False, True, True, True, True),
        units=("Hz", "Hz/s", "day", "", "", ""),
        epoch_int_values={"PEPOCH": 53000.0},
    )

    white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
    noise_model = NoiseModel(white_noise=white, correlated=(plred,))
    fake = make_fake_toas(
        model,
        toa_data,
        params,
        jax.random.PRNGKey(11),
        noise_components=[white, plred],
    )
    fitter = GLSFitter(model, fake, params, noise_model=noise_model)
    sigma = fitter.fit_toas(maxiter=MAXITER).parameter_uncertainties
    return fitter, params, sigma


class TestGLSImplicitGradients:
    @pytest.mark.parametrize("full_cov", [False, True])
    def test_fit_converged(self, gls_problem, full_cov):
        fitter, params, sigma = gls_problem
        fitted = fitter.fit_params(maxiter=MAXITER, full_cov=full_cov)
        _assert_converged(fitter.fit_gap(fitted, full_cov=full_cov), sigma)

    @pytest.mark.parametrize("name", ["TNREDAMP", "TNREDGAM"])
    @pytest.mark.parametrize("full_cov", [False, True])
    @pytest.mark.slow
    def test_frozen_noise_gradient_matches_fd(self, gls_problem, name, full_cov):
        """d(fit)/d(noise param) flows only through C^{-1} in the IFT."""
        fitter, params, _sigma = gls_problem
        idx = params.names.index(name)

        def fitted_free(theta):
            p = params.with_values(params.values.at[idx].set(theta))
            return fitter.fit_params(
                p, maxiter=MAXITER, full_cov=full_cov
            ).free_values()

        theta0 = params.values[idx]
        J_ad = jax.jacrev(fitted_free)(theta0)
        _assert_fit_gradient_matches_fd(fitted_free, theta0, 0.04, J_ad)

    @pytest.mark.slow
    def test_external_delay_gradient_matches_fd(self, gls_problem):
        fitter, params, _sigma = gls_problem
        n = fitter.toa_data.n_toas
        phase = np.linspace(0.0, 4 * np.pi, n)
        delay0 = jnp.asarray(5e-7 * np.sin(phase))

        def fitted_free(delay):
            return fitter.fit_params(params, delay, maxiter=MAXITER).free_values()

        J_ad = jax.jacrev(fitted_free)(delay0)

        rng = np.random.default_rng(13)
        d = rng.standard_normal(n)
        d /= np.linalg.norm(d)
        d_jnp = jnp.asarray(d)
        _assert_fit_gradient_matches_fd(
            lambda s: fitted_free(delay0 + s * d_jnp),
            0.0,
            1e-6,
            np.asarray(J_ad) @ d,
        )


# ---------------------------------------------------------------------------
# Wideband GLS: stacked [time; dm] residuals, DM free, DM1 frozen
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wideband_problem():
    """Synthetic wideband pulsar; F0/F1/DM free, DM1 frozen (the IFT theta)."""
    n_toas = 50
    rng = np.random.default_rng(5)
    t_mjd = np.linspace(53000.0, 54000.0, n_toas)
    freq = np.where(np.arange(n_toas) % 2 == 0, 800.0, 1400.0)
    mask = np.ones(n_toas, dtype=bool)

    toa_data = make_toa_data(
        t_mjd=t_mjd,
        error=1e-6,
        freq=freq,
        flag_masks={"EFAC1": mask},
        dm_values=jnp.zeros(n_toas),  # placeholder, filled after simulation
        dm_errors=jnp.full(n_toas, 1e-4),
        tzr_tdb_int=53000.0,
        tzr_tdb_frac=0.5,
        tzr_freq=jnp.inf,
        tzr_ssb_obs_pos=np.zeros(3),
        tzr_obs_sun_pos=np.zeros(3),
    )

    spin = Spindown(spin_param_names=("F0", "F1"), pepoch_name="PEPOCH")
    disp = DispersionDM(dm_param_names=("DM", "DM1"))
    model = TimingModel(
        delay_components=(disp,),
        phase_components=(spin,),
        dispersion_components=(disp,),
    )

    params = make_params(
        ("F0", "F1", "PEPOCH", "DM", "DM1", "DMEPOCH", "EFAC1"),
        (100.0, -1e-15, 0.0, 15.0, 0.0, 0.0, 1.0),
        frozen_mask=(False, False, True, False, True, True, True),
        units=("Hz", "Hz/s", "day", "pc cm^-3", "pc cm^-3/yr", "day", ""),
        epoch_int_values={"PEPOCH": 53000.0, "DMEPOCH": 53000.0},
    )

    white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
    fake = make_fake_toas(
        model, toa_data, params, jax.random.PRNGKey(17), noise_components=[white]
    )
    dm_true = model.compute_dm(fake, params)
    dm_values = dm_true + jnp.asarray(rng.normal(0.0, 1e-4, n_toas))
    fake = eqx.tree_at(lambda t: t.dm_values, fake, dm_values)

    fitter = WidebandGLSFitter(model, fake, params, noise_model=None)
    sigma = fitter.fit_toas(maxiter=MAXITER).parameter_uncertainties
    return fitter, params, sigma


class TestWidebandImplicitGradients:
    def test_fit_converged(self, wideband_problem):
        fitter, params, sigma = wideband_problem
        fitted = fitter.fit_params(maxiter=MAXITER)
        _assert_converged(fitter.fit_gap(fitted), sigma)

    @pytest.mark.slow
    def test_frozen_dm1_gradient_matches_fd(self, wideband_problem):
        fitter, params, _sigma = wideband_problem
        idx = params.names.index("DM1")

        def fitted_free(dm1):
            p = params.with_values(params.values.at[idx].set(dm1))
            return fitter.fit_params(p, maxiter=MAXITER).free_values()

        dm1_0 = params.values[idx]
        J_ad = jax.jacrev(fitted_free)(dm1_0)
        _assert_fit_gradient_matches_fd(fitted_free, dm1_0, 2e-4, J_ad)

    @pytest.mark.slow
    def test_external_delay_gradient_matches_fd(self, wideband_problem):
        """The delay hits only the time half of the stacked residuals."""
        fitter, params, _sigma = wideband_problem
        n = fitter.toa_data.n_toas
        phase = np.linspace(0.0, 4 * np.pi, n)
        delay0 = jnp.asarray(5e-7 * np.sin(phase))

        def fitted_free(delay):
            return fitter.fit_params(params, delay, maxiter=MAXITER).free_values()

        J_ad = jax.jacrev(fitted_free)(delay0)

        rng = np.random.default_rng(23)
        d = rng.standard_normal(n)
        d /= np.linalg.norm(d)
        d_jnp = jnp.asarray(d)
        _assert_fit_gradient_matches_fd(
            lambda s: fitted_free(delay0 + s * d_jnp),
            0.0,
            1e-6,
            np.asarray(J_ad) @ d,
        )
