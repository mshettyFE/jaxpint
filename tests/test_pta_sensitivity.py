"""Tests for jaxpint.pta.sensitivity (the F-stat noncentrality producer)."""

import jax
import jax.numpy as jnp
import pytest

from jaxpint.pta.signals.cw import cw_delay_from_array
from jaxpint.pta.cw_upper_limit import (
    quadratic_coeffs, orientation_coeffs, _default_extraction_orientations,
)
from jaxpint.pta.sensitivity import earth_term_gram, unit_noncentrality
from jaxpint.sensitivity import chi2_threshold, h0_min_from_lambda
from tests.helpers import make_simple_pulsar

# A GW frequency resolved over the synthetic pulsars' ~1-day span: an unresolved
# (real ~nHz) CW leaves the sin/cos quadratures degenerate, so the Earth-term
# orientation basis collapses below full rank 4.  The producer / F-stat math is
# frequency-agnostic; this only makes the network Gram M well-conditioned rank-4.
LOG10_FGW = -5.0
_CT_SKY, _GP_SKY = 0.3, 1.2
# Three well-separated sky directions -> diverse (F+, Fx), so the *network* Earth-term
# Gram is full rank 4 (a single pulsar's is only rank 2: F+, Fx are scalars, leaving
# just the sin/cos quadratures).
_POSITIONS = [
    jnp.array([0.2, 0.5, -0.84]),
    jnp.array([-0.6, 0.3, 0.7]),
    jnp.array([0.8, -0.5, 0.1]),
]


@pytest.fixture(scope="module")
def network():
    """A 3-pulsar network: per-pulsar likelihoods + the summed rank-4 Earth-term Gram M."""
    from jaxpint.bayes import marginalize_single_pulsar, ImproperPrior

    pulsars, M = [], jnp.zeros((4, 4))
    for i, p in enumerate(_POSITIONS):
        td, tm, nm, pp = make_simple_pulsar(200, f0=100.0, f1=-1e-14, seed=i)
        over = {n for n in pp.free_names() if n in ("F0", "F1")}
        g, _, skel = marginalize_single_pulsar(
            over=over, priors={n: ImproperPrior() for n in over},
            toa_data=td, timing_model=tm, noise_model=nm, fiducial_params=pp,
            allow_nonlinear=True, validate_linearity=False)
        pos = p / jnp.linalg.norm(p)
        pulsars.append((g, skel, td, pos))
        M = M + earth_term_gram(g, skel, td, pos, 1.0, _CT_SKY, _GP_SKY, LOG10_FGW)
    return {"pulsars": pulsars, "M": M}


def test_unit_noncentrality_matches_direct_signal_power(network):
    # lambda_1(theta) = c(theta)^T M c(theta) must equal the network signal power
    # sum_i (s_i|s_i) -- computed independently via quadratic_coeffs on each pulsar's
    # full unit-strain signal, not the basis Gram.
    M, pulsars = network["M"], network["pulsars"]
    for ci, psi, ph0 in [(0.4, 0.6, 0.9), (-0.7, 1.5, 2.0), (1.0, 0.0, 0.0)]:
        lam = unit_noncentrality(M, jnp.array([[ci, psi, ph0]]))[0]
        direct = 0.0
        for g, skel, td, pos in pulsars:
            s = cw_delay_from_array(
                td, pos, 1.0, jnp.array([1.0, _CT_SKY, _GP_SKY, LOG10_FGW, ci, psi, ph0]),
                earth_term_only=True, linear_amplitude=True)
            _, Y = quadratic_coeffs(lambda a: g(skel, external_delay=a * s))
            direct += Y
        assert jnp.isclose(lam, direct, rtol=1e-6)


def test_h0_min_matches_injection_recovery_fraction(network):
    # Monte-Carlo the F-statistic detection FRACTION with Gaussian matched-
    # filter noise and 2F = b^T M^-1 b directly -- no ncx2.sf (the consumer's tool).
    # b = h0*M c(theta) + noise, noise ~ N(0, M)  =>  2F ~ ncx2_4(h0^2 c^T M c);
    # the fraction with 2F > threshold must equal beta.
    M = network["M"]
    orientations = _default_extraction_orientations(64, seed=1)
    thr, beta = chi2_threshold(1e-3, 4), 0.9
    h0 = float(h0_min_from_lambda(
        thr, unit_noncentrality(M, orientations), dof=4, beta=beta))

    C = jax.vmap(lambda o: orientation_coeffs(o[0], o[1], o[2]))(orientations)  # (64, 4)
    Minv, L = jnp.linalg.inv(M), jnp.linalg.cholesky(M)  # L L^T = M
    n = 60_000
    k_idx, k_z = jax.random.split(jax.random.PRNGKey(0))
    idx = jax.random.randint(k_idx, (n,), 0, len(C))  # sample the orientation ensemble h0_min averaged
    b = h0 * (C[idx] @ M) + jax.random.normal(k_z, (n, 4)) @ L.T  # signal + N(0, M) noise
    two_f = jnp.einsum("ta,ab,tb->t", b, Minv, b)  # 2F = b^T M^-1 b
    frac = (two_f > thr).mean()
    assert jnp.isclose(frac, beta, atol=1e-2)
