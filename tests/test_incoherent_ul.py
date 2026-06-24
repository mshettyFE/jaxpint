"""Tests for the incoherent (distance-marginalized) CW upper-limit machinery."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.pta.signals.cw import cw_delay_from_array, _KPC_TO_M, _C
from jaxpint.pta.incoherent_ul import (
    bM2_coeffs, extract_pulsar_bM, flat_phase_grid, distance_phase_grid,
    logL_pulsar_marg, h0_95_grid, _A_of_phase,
)
from tests.helpers import make_toa_data, make_simple_pulsar

LOG10_FGW = float(np.log10(27e-9))


# ----------------------------------------------------------- pure-math building blocks
def test_bM2_coeffs_recovers_known():
    b_true = jnp.array([0.7, -0.3])
    M_true = jnp.array([[2.0, 0.4], [0.4, 1.5]])
    f = lambda Ae, As: (jnp.array([Ae, As]) @ b_true
                        - 0.5 * jnp.array([Ae, As]) @ M_true @ jnp.array([Ae, As]))
    b, M = bM2_coeffs(f)
    assert np.allclose(b, b_true) and np.allclose(M, M_true)
    assert np.allclose(M, M.T)


def test_phase_marginal_matches_quadrature():
    """logL_pulsar_marg == log mean_Delta exp(logL) by independent numpy quadrature."""
    b = jnp.array([0.7, -0.3]); M = jnp.array([[2.0, 0.4], [0.4, 1.5]])
    h0 = 0.8
    got = float(logL_pulsar_marg(h0, b, M, flat_phase_grid(2048)))
    D = np.linspace(0, 2 * np.pi, 200001)
    A = np.stack([1 - np.cos(D), np.sin(D)], -1)
    integ = np.exp(h0 * (A @ np.array(b)) - 0.5 * h0**2
                   * np.einsum("ni,ij,nj->n", A, np.array(M), A))
    ref = np.log(np.trapezoid(integ, D) / (2 * np.pi))
    assert abs(got - ref) < 1e-9


def test_h0_95_grid_interior_and_sensitivity():
    # Use >=2 pulsars: the phase-marginalized posterior tail is ~h0^{-N}, so a
    # single null pulsar is improper (the Delta~0 phases give vanishing signal
    # power). With 2 pulsars it is proper and the 95% point is interior.
    M = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    g = jnp.broadcast_to(flat_phase_grid(256), (2, 256))
    b = jnp.broadcast_to(jnp.array([0.5, 0.2]), (2, 2))
    ul = float(h0_95_grid(b, M, g, jnp.float64(60.0), n_h0=6000))
    assert 0.0 < ul < 60.0
    # sensitivity: 4x more signal power (M) -> tighter (smaller) upper limit
    ul_more = float(h0_95_grid(b, 4.0 * M, g, jnp.float64(60.0), n_h0=6000))
    assert ul_more < ul


def test_matched_filter_sign_single_phase():
    """At a fixed phase (degenerate grid) the UL is a standard truncated-Gaussian
    limit, so a larger positive matched filter pushes it UP -- validates the sign."""
    M = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    g = jnp.broadcast_to(jnp.array([jnp.pi]), (2, 1))     # single phase: A=(2,0)
    weak = jnp.broadcast_to(jnp.array([0.2, 0.0]), (2, 2))
    strong = jnp.broadcast_to(jnp.array([1.0, 0.0]), (2, 2))
    ul_w = float(h0_95_grid(weak, M, g, jnp.float64(40.0), n_h0=6000))
    ul_s = float(h0_95_grid(strong, M, g, jnp.float64(40.0), n_h0=6000))
    assert ul_s > ul_w


# -------------------------------------------------------------- cw.py template algebra
def test_cw_pulsar_quadratures():
    td = make_toa_data(t_mjd=55000.0 + np.linspace(0, 3000, 300))
    pos = jnp.array([0.3, -0.5, 0.8]); pos = pos / jnp.linalg.norm(pos)
    PX = 0.7
    cw = jnp.array([1.0, 0.4, 1.1, LOG10_FGW, 0.6, 0.9, 0.3])
    res = lambda **k: cw_delay_from_array(td, pos, PX, cw, linear_amplitude=True, **k)
    e = res(earth_term_only=True)
    pc = res(pulsar_term_only=True, pulsar_term_phase=0.0)
    ps = res(pulsar_term_only=True, pulsar_term_phase=float(np.pi / 2))
    full = res()
    # pc == -e exactly
    assert np.allclose(pc, -e, atol=0, rtol=0)
    # full earth+pulsar = e*(1-cosD) + ps*sinD at the PX-derived phase D
    cos_th = cw[1]; sin_th = jnp.sqrt(1 - cos_th**2); gwphi = cw[2]; f0 = 10**cw[3]
    omhat = jnp.array([-sin_th * jnp.cos(gwphi), -sin_th * jnp.sin(gwphi), -cos_th])
    cosmu = jnp.dot(omhat, pos)
    D = 2 * np.pi * f0 * (1.0 / PX) * _KPC_TO_M * (1 + cosmu) / _C
    recon = e * (1 - jnp.cos(D)) + ps * jnp.sin(D)
    # exact to float64 relative to the peak (near-zero elements trip allclose rtol)
    assert np.max(np.abs(recon - full)) < 1e-9 * np.max(np.abs(full))
    # earth term linear in h0
    e3 = cw_delay_from_array(td, pos, PX, cw.at[0].set(3.0), linear_amplitude=True,
                             earth_term_only=True)
    assert np.allclose(e3, 3.0 * e, rtol=1e-9)


# --------------------------------------------------------- distance vs flat-phase prior
def test_distance_phase_grid_values():
    f, cosmu, L0, sig, k, n = 27e-9, 0.5, 1.0, 0.1, 3.0, 8
    g = np.asarray(distance_phase_grid(L0, sig, k, cosmu, f, n))
    lo, hi = L0 - k * sig, L0 + k * sig
    for L, expect in ((lo, g[0]), (hi, g[-1])):
        assert np.isclose(expect, 2 * np.pi * f * (L * _KPC_TO_M) * (1 + cosmu) / _C)


def test_narrow_prior_localizes_vs_flat():
    """A sub-cycle phase grid (informative distance) gives a DIFFERENT marginal
    than the flat-phase (broad-prior) limit -- i.e. the parallax is actually used."""
    b = jnp.array([0.6, -0.4]); M = jnp.array([[1.8, 0.3], [0.3, 1.2]]); h0 = 1.0
    flat = float(logL_pulsar_marg(h0, b, M, flat_phase_grid(512)))
    narrow = jnp.linspace(1.00, 1.02, 64)         # tight cluster of phases (<<1 cycle)
    loc = float(logL_pulsar_marg(h0, b, M, narrow))
    # localized marginal ~ the single-phase logL at Delta~1.01, far from flat avg
    A = _A_of_phase(jnp.array([1.01]))[0]
    single = float(h0 * (A @ b) - 0.5 * h0**2 * (A @ M @ A))
    assert abs(loc - single) < 1e-3
    assert abs(loc - flat) > 1e-2


# ------------------------------------------------------------- real-data extraction
def test_extract_pulsar_bM_self_consistent():
    """The recovered (b, M) reproduce the actual marginalized g at arbitrary
    amplitudes (validates the real-mode timing-marginalized GLS extraction)."""
    from jaxpint.likelihood import single_pulsar_logL
    from jaxpint.bayes import marginalize, ImproperPrior

    td, tm, nm, pp = make_simple_pulsar(200, f0=100.0, f1=-1e-14)
    over = {n for n in pp.free_names() if n in ("F0", "F1")}
    g, _, skel = marginalize(
        single_pulsar_logL, over=over,
        priors={n: ImproperPrior() for n in over},
        toa_data=td, timing_model=tm, noise_model=nm, fiducial_params=pp,
        allow_nonlinear=True, validate_linearity=False,
    )
    pos = jnp.array([0.2, 0.5, -0.84]); pos = pos / jnp.linalg.norm(pos)
    cw = jnp.array([1.0, 0.3, 1.2, LOG10_FGW, 1.0, 0.0, 0.0])
    e = cw_delay_from_array(td, pos, 1.0, cw, linear_amplitude=True, earth_term_only=True)
    ps = cw_delay_from_array(td, pos, 1.0, cw, linear_amplitude=True,
                             pulsar_term_only=True, pulsar_term_phase=float(np.pi / 2))
    b, M = extract_pulsar_bM(g, skel, e, ps)
    # M symmetric PSD
    assert np.allclose(M, M.T)
    assert np.all(np.linalg.eigvalsh(np.asarray(M)) >= -1e-6 * np.max(np.abs(np.asarray(M))))
    # self-consistency at random (small) amplitudes
    g0 = float(g(skel, external_delay=0.0 * e))
    rng = np.random.default_rng(0)
    scale = 1.0 / float(jnp.sqrt(jnp.max(jnp.abs(M))))   # ~ strain scale
    for Ae, As in rng.normal(size=(5, 2)) * scale:
        direct = float(g(skel, external_delay=Ae * e + As * ps)) - g0
        A = jnp.array([Ae, As])
        quad = float(A @ b - 0.5 * A @ M @ A)
        assert abs(direct - quad) <= 1e-6 * (abs(quad) + 1.0)
