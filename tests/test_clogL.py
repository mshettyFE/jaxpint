"""Conditional (coefficient-explicit) log-likelihoods ``clogL``.

``clogL`` keeps the GP coefficients ``a`` explicit and evaluates the joint
density ``p(r | a) p(a)``; the marginal ``logL`` integrates them out.  The
two are tied by the exact Gaussian marginalization identity

    logL(theta) = clogL(theta, a_hat) - 0.5 logdet(P) + 0.5 n_coeff log(2 pi),

with ``a_hat`` the conditional mean and ``P = L L^T`` the posterior
precision (``conditional_*(...).precision_chol``).  Because ``clogL`` is
exactly quadratic in ``a`` the integral is exact, so this identity — tested
below against the already-verified ``single_pulsar_logL`` / ``pta_logL`` and
``conditional_*`` — pins ``clogL`` without a fresh reference implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.stats import multivariate_normal

from jaxpint.fitters import compute_time_residuals
from jaxpint.likelihood import single_pulsar_clogL, single_pulsar_logL
from jaxpint.pta import (
    PTAConfig,
    conditional_gwb,
    conditional_single_pulsar,
    pta_clogL,
    pta_logL,
    pta_logL_and_clogL,
    sample_conditional,
)
from jaxpint.utils import concat_woodbury_blocks

# Reuse the conditional suite's fixtures verbatim so clogL is tested against
# the identical setups its conditional dual is (importing also enables x64).
from tests.test_conditional import (
    N_GW,
    _hd_config,
    _pulsar_with_red_noise,
)

LOG2PI = np.log(2.0 * np.pi)


def _logdet_precision(cond) -> float:
    """``log|P|`` from the posterior precision Cholesky ``L`` (``P = L L^T``)."""
    return 2.0 * np.sum(np.log(np.abs(np.diag(np.asarray(cond.precision_chol)))))


# ---------------------------------------------------------------------------
# Single pulsar
# ---------------------------------------------------------------------------


def test_single_pulsar_clogL_matches_dense_gaussian():
    """clogL == logpdf(r | Ua, N) + logpdf(a | 0, Phi) via dense MVN."""
    td, tm, nm, pp = _pulsar_with_red_noise(0)
    r = np.asarray(compute_time_residuals(tm, td, pp))
    Ndiag, U_n, Phi_n = nm.covariance(td, pp)
    woodbury = concat_woodbury_blocks((U_n, Phi_n), None)
    assert woodbury is not None
    U, Phi = np.asarray(woodbury[0]), np.asarray(woodbury[1])

    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal(U.shape[1]))

    got = float(single_pulsar_clogL(td, tm, nm, pp, a))
    data_ref = multivariate_normal.logpdf(
        r, mean=U @ np.asarray(a), cov=np.diag(np.asarray(Ndiag))
    )
    prior_ref = multivariate_normal.logpdf(
        np.asarray(a), mean=np.zeros(U.shape[1]), cov=np.diag(Phi)
    )
    npt.assert_allclose(got, data_ref + prior_ref, rtol=1e-9)


def test_single_pulsar_marginalization_identity():
    """logL == clogL(a_hat) - 0.5 logdet(P) + 0.5 k log(2 pi)."""
    td, tm, nm, pp = _pulsar_with_red_noise(0)
    cond = conditional_single_pulsar(td, tm, nm, pp)
    k = cond.mean.shape[0]

    clogL_at_mean = float(single_pulsar_clogL(td, tm, nm, pp, cond.mean))
    reconstructed = clogL_at_mean - 0.5 * _logdet_precision(cond) + 0.5 * k * LOG2PI
    logL = float(single_pulsar_logL(td, tm, nm, pp))
    npt.assert_allclose(reconstructed, logL, rtol=1e-9)


def test_single_pulsar_clogL_grad_zero_at_mean():
    """clogL peaks at the conditional mean: d/da clogL(a_hat) == 0."""
    td, tm, nm, pp = _pulsar_with_red_noise(0)
    cond = conditional_single_pulsar(td, tm, nm, pp)

    clogL_fn = lambda a: single_pulsar_clogL(td, tm, nm, pp, a)  # noqa: E731
    grad_fn = jax.grad(clogL_fn, argnums=0)

    bumped = cond.mean + 0.1 * jnp.std(cond.mean)
    # Scale-free: the gradient at the mean must be vanishing relative to the
    # (linearly growing) gradient a finite step away — an absolute tol is
    # meaningless when clogL itself is O(1e2).
    g_mean = float(jnp.linalg.norm(grad_fn(cond.mean)))
    g_bumped = float(jnp.linalg.norm(grad_fn(bumped)))
    assert g_mean < 1e-6 * g_bumped

    # ... and it is a maximum: perturbing the coefficients lowers clogL.
    assert float(clogL_fn(bumped)) < float(clogL_fn(cond.mean))


def test_single_pulsar_clogL_jit_matches_eager():
    td, tm, nm, pp = _pulsar_with_red_noise(0)
    a = sample_conditional(
        jax.random.PRNGKey(1), conditional_single_pulsar(td, tm, nm, pp)
    )
    eager = single_pulsar_clogL(td, tm, nm, pp, a)
    jitted = jax.jit(lambda aa: single_pulsar_clogL(td, tm, nm, pp, aa))(a)
    assert np.isfinite(float(eager))
    npt.assert_allclose(float(jitted), float(eager), rtol=1e-9)


# ---------------------------------------------------------------------------
# PTA (correlated)
# ---------------------------------------------------------------------------


def test_pta_marginalization_identity():
    """pta_logL == pta_clogL(a_hat) - 0.5 logdet(P) + 0.5 n log(2 pi)."""
    gp, pps, config, _ = _hd_config()
    cond = conditional_gwb(gp, pps, config)
    n_joint = cond.mean.shape[0]

    clogL_at_mean = float(pta_clogL(gp, pps, config, cond.mean))
    reconstructed = (
        clogL_at_mean - 0.5 * _logdet_precision(cond) + 0.5 * n_joint * LOG2PI
    )
    npt.assert_allclose(reconstructed, float(pta_logL(gp, pps, config)), rtol=1e-9)


def test_pta_clogL_grad_zero_at_mean():
    """Joint clogL peaks at the conditional_gwb mean."""
    gp, pps, config, _ = _hd_config()
    cond = conditional_gwb(gp, pps, config)
    grad_fn = jax.grad(lambda a: pta_clogL(gp, pps, config, a))

    bumped = cond.mean + 0.1 * jnp.std(cond.mean)
    g_mean = float(jnp.linalg.norm(grad_fn(cond.mean)))
    g_bumped = float(jnp.linalg.norm(grad_fn(bumped)))
    assert g_mean < 1e-6 * g_bumped


def test_pta_logL_and_clogL_matches_separate_calls():
    """The one-solve wrapper returns exactly the two independent results."""
    gp, pps, config, _ = _hd_config()
    a = sample_conditional(jax.random.PRNGKey(4), conditional_gwb(gp, pps, config))

    logL_both, clogL_both = pta_logL_and_clogL(gp, pps, config, a)
    npt.assert_allclose(float(logL_both), float(pta_logL(gp, pps, config)), rtol=1e-12)
    npt.assert_allclose(
        float(clogL_both), float(pta_clogL(gp, pps, config, a)), rtol=1e-12
    )


def test_pta_clogL_jit_and_finite():
    gp, pps, config, _ = _hd_config()
    a = sample_conditional(jax.random.PRNGKey(6), conditional_gwb(gp, pps, config))
    eager = pta_clogL(gp, pps, config, a)
    jitted = jax.jit(lambda aa: pta_clogL(gp, pps, config, aa))(a)
    assert np.isfinite(float(eager))
    npt.assert_allclose(float(jitted), float(eager), rtol=1e-9)


# ---------------------------------------------------------------------------
# Shape guards
# ---------------------------------------------------------------------------


def test_single_pulsar_clogL_wrong_length_raises():
    td, tm, nm, pp = _pulsar_with_red_noise(0)
    with pytest.raises(ValueError, match="length 7.*expects 16"):
        single_pulsar_clogL(td, tm, nm, pp, jnp.zeros(7))


def test_pta_clogL_wrong_length_raises():
    gp, pps, config, _ = _hd_config()
    with pytest.raises(ValueError, match="length 9.*expects 20"):
        pta_clogL(gp, pps, config, jnp.zeros(9))


def test_pta_clogL_requires_correlated_injector():
    gp, pps, config, _ = _hd_config()
    bare = PTAConfig(
        toa_data_list=config.toa_data_list,
        timing_models=config.timing_models,
        noise_models=config.noise_models,
        signal_injectors=(),
        correlated_injectors=(),
    )
    a = jnp.zeros(2 * 2 * N_GW)
    with pytest.raises(ValueError, match="correlated injector"):
        pta_clogL(gp, tuple(pps), bare, a)
