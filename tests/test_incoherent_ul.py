"""Tests for the incoherent (distance-marginalized) CW upper-limit machinery."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.pta.signals.cw import cw_delay_from_array, _KPC_TO_M, _C
from jaxpint.pta.incoherent_ul import condition_on_statics, flat_phase_grid, distance_phase_grid, logL_pulsar_marg, total_logL_marg, total_logL_profile, h0_95_grid, mixed_phase_A
from jaxpint.pta.extraction import bM2_coeffs, extract_pulsar_bM, extract_pulsar_blocks
from tests.helpers import make_toa_data, make_simple_pulsar

LOG10_FGW = float(np.log10(27e-9))


def _A_of_phase(phase):
    """Test helper: the A(Δ) = (1 − cosΔ, sinΔ) coefficient vectors of a phase grid."""
    return jnp.stack([1.0 - jnp.cos(phase), jnp.sin(phase)], axis=-1)


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
    got = float(logL_pulsar_marg(h0, b, M, _A_of_phase(flat_phase_grid(2048))))
    D = np.linspace(0, 2 * np.pi, 200001)
    A = np.stack([1 - np.cos(D), np.sin(D)], -1)
    integ = np.exp(h0 * (A @ np.array(b)) - 0.5 * h0**2
                   * np.einsum("ni,ij,nj->n", A, np.array(M), A))
    ref = np.log(np.trapezoid(integ, D) / (2 * np.pi))
    assert abs(got - ref) < 1e-9


def test_earth_only_baseline_is_fixed_truncated_gaussian():
    """Earth-term-only baseline A = (1, 0): no marginalization, a plain
    truncated-Gaussian logL s = h0*e with X=b[0], Y=M[0,0]."""
    A = jnp.array([[1.0, 0.0]])
    b = jnp.array([0.7, -0.3]); M = jnp.array([[2.0, 0.4], [0.4, 1.5]]); h0 = 0.9
    got = float(logL_pulsar_marg(h0, b, M, A))
    assert np.isclose(got, h0 * b[0] - 0.5 * h0**2 * M[0, 0])


def test_h0_95_grid_interior_and_sensitivity():
    # Use >=2 pulsars: the phase-marginalized posterior tail is ~h0^{-N}, so a
    # single null pulsar is improper (the Delta~0 phases give vanishing signal
    # power). With 2 pulsars it is proper and the 95% point is interior.
    M = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    A = jnp.broadcast_to(_A_of_phase(flat_phase_grid(256)), (2, 256, 2))
    b = jnp.broadcast_to(jnp.array([0.5, 0.2]), (2, 2))
    ul = float(h0_95_grid(b, M, A, jnp.float64(60.0), n_h0=6000))
    assert 0.0 < ul < 60.0
    # sensitivity: 4x more signal power (M) -> tighter (smaller) upper limit
    ul_more = float(h0_95_grid(b, 4.0 * M, A, jnp.float64(60.0), n_h0=6000))
    assert ul_more < ul


def test_matched_filter_sign_single_phase():
    """At a fixed phase (degenerate grid) the UL is a standard truncated-Gaussian
    limit, so a larger positive matched filter pushes it UP -- validates the sign."""
    M = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    A = jnp.broadcast_to(_A_of_phase(jnp.array([jnp.pi])), (2, 1, 2))  # A=(2,0)
    weak = jnp.broadcast_to(jnp.array([0.2, 0.0]), (2, 2))
    strong = jnp.broadcast_to(jnp.array([1.0, 0.0]), (2, 2))
    ul_w = float(h0_95_grid(weak, M, A, jnp.float64(40.0), n_h0=6000))
    ul_s = float(h0_95_grid(strong, M, A, jnp.float64(40.0), n_h0=6000))
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


def test_mixed_phase_A():
    """Tight pulsars get the distance grid (localized); flat pulsars keep [0,2pi)."""
    L0 = jnp.array([1.0, 2.0]); cosmu = jnp.array([0.3, -0.2]); n = 64
    flatA = _A_of_phase(flat_phase_grid(n))
    # all-flat -> flat broadcast
    A0 = mixed_phase_A(jnp.array([False, False]), L0, 1e-5, 3.0, cosmu, 27e-9, n)
    assert np.allclose(A0, np.broadcast_to(flatA, (2, n, 2)))
    # one tight (sub-cycle) -> its row is the distance grid, differs from flat;
    # the flat pulsar is unchanged
    A1 = mixed_phase_A(jnp.array([True, False]), L0, 1e-6, 3.0, cosmu, 27e-9, n)
    assert np.allclose(A1[0], _A_of_phase(distance_phase_grid(1.0, 1e-6, 3.0, 0.3, 27e-9, n)))
    assert np.allclose(A1[1], flatA)
    assert not np.allclose(A1[0], flatA)
    # adaptive: a tight pulsar with a LOOSE prior (many cycles > grid resolves)
    # falls back to the exact flat-phase limit (no aliasing)
    A2 = mixed_phase_A(jnp.array([True, True]), L0, 2e-3, 5.0, cosmu, 27e-9, n)
    assert np.allclose(A2[0], flatA) and np.allclose(A2[1], flatA)


def test_narrow_prior_localizes_vs_flat():
    """A sub-cycle phase grid (informative distance) gives a DIFFERENT marginal
    than the flat-phase (broad-prior) limit -- i.e. the parallax is actually used."""
    b = jnp.array([0.6, -0.4]); M = jnp.array([[1.8, 0.3], [0.3, 1.2]]); h0 = 1.0
    flat = float(logL_pulsar_marg(h0, b, M, _A_of_phase(flat_phase_grid(512))))
    narrow = jnp.linspace(1.00, 1.02, 64)         # tight cluster of phases (<<1 cycle)
    loc = float(logL_pulsar_marg(h0, b, M, _A_of_phase(narrow)))
    # localized marginal ~ the single-phase logL at Delta~1.01, far from flat avg
    A = _A_of_phase(jnp.array([1.01]))[0]
    single = float(h0 * (A @ b) - 0.5 * h0**2 * (A @ M @ A))
    assert abs(loc - single) < 1e-3
    assert abs(loc - flat) > 1e-2


# ------------------------------------------------------------- real-data extraction
def test_extract_pulsar_bM_self_consistent():
    """The recovered (b, M) reproduce the actual marginalized g at arbitrary
    amplitudes (validates the real-mode timing-marginalized GLS extraction)."""
    from jaxpint.bayes import marginalize_single_pulsar

    td, tm, nm, pp = make_simple_pulsar(200, f0=100.0, f1=-1e-14)
    over = {n for n in pp.free_names() if n in ("F0", "F1")}
    g, _, skel = marginalize_single_pulsar(over=over,
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


# ------------------------------------------------------- profile (max) reduction twin
def test_total_logL_profile_reduction():
    """Profile = Σ_a max_Δ logL: equals an explicit per-pulsar max, exceeds the
    marginal, and collapses to the marginal for a singleton (one-phase) grid."""
    h0 = 1.0
    b = jnp.array([[0.6, -0.4], [0.5, 0.2]])
    M = jnp.stack([jnp.array([[1.8, 0.3], [0.3, 1.2]]), jnp.eye(2)])
    A = jnp.broadcast_to(_A_of_phase(flat_phase_grid(48)), (2, 48, 2))
    prof = float(total_logL_profile(h0, b, M, A))
    # explicit per-pulsar max of the quadratic-form grid (independent of the impl)
    Anp = np.asarray(A); ref = 0.0
    for i in range(2):
        bA = Anp[i] @ np.asarray(b[i])
        AMA = np.einsum("ni,ij,nj->n", Anp[i], np.asarray(M[i]), Anp[i])
        ref += np.max(h0 * bA - 0.5 * h0**2 * AMA)
    assert np.isclose(prof, ref)
    assert prof >= float(total_logL_marg(h0, b, M, A))          # profile >= marginal
    A1 = jnp.broadcast_to(jnp.array([1.0, 0.0]), (2, 1, 2))     # singleton -> equal
    assert np.isclose(float(total_logL_profile(h0, b, M, A1)),
                      float(total_logL_marg(h0, b, M, A1)))


def test_total_logL_marg_matches_bruteforce_likelihood():
    """End-to-end ground truth: total_logL_marg (via extract_pulsar_bM + the A(Δ)
    form) equals a NAIVE phase marginalization that evaluates the actual
    timing-marginalized GLS likelihood g at each phase -- no (b, M) shortcut."""
    from jaxpint.bayes import marginalize_single_pulsar

    td, tm, nm, pp = make_simple_pulsar(200, f0=100.0, f1=-1e-14)
    over = {n for n in pp.free_names() if n in ("F0", "F1")}
    g, _, skel = marginalize_single_pulsar(
        over=over,
        toa_data=td, timing_model=tm, noise_model=nm, fiducial_params=pp,
        allow_nonlinear=True, validate_linearity=False,
    )
    pos = jnp.array([0.2, 0.5, -0.84]); pos = pos / jnp.linalg.norm(pos)
    cw = jnp.array([1.0, 0.3, 1.2, LOG10_FGW, 1.0, 0.0, 0.0])   # cw[0] = h0 = 1 (linear)
    e = cw_delay_from_array(td, pos, 1.0, cw, linear_amplitude=True, earth_term_only=True)
    ps = cw_delay_from_array(td, pos, 1.0, cw, linear_amplitude=True,
                             pulsar_term_only=True, pulsar_term_phase=float(np.pi / 2))
    b, M = extract_pulsar_bM(g, skel, e, ps)

    h0 = float(1.0 / jnp.sqrt(jnp.max(jnp.abs(M))))            # ~strain scale -> O(1) logL
    phases = flat_phase_grid(32)
    marg = float(total_logL_marg(jnp.asarray(h0), b[None], M[None], _A_of_phase(phases)[None]))

    # naive: full Earth+pulsar signal at each phase via cw_delay (NOT e/ps or b/M),
    # eval the real g, then average by hand (numpy log-mean-exp).
    g0 = float(g(skel, external_delay=0.0 * e))
    sigs = jnp.stack([
        cw_delay_from_array(td, pos, 1.0, cw, linear_amplitude=True, pulsar_term_phase=float(ph))
        for ph in np.asarray(phases)
    ])
    logL = np.asarray(jax.vmap(lambda s: g(skel, external_delay=h0 * s))(sigs)) - g0
    marg_naive = float(np.log(np.mean(np.exp(logL - logL.max()))) + logL.max())
    assert np.isclose(marg, marg_naive, atol=1e-5)


# ------------------------------------------- multi-source conditioned scan (Tier-2)
def _cw(ct, gp):  # face-on, unit h0
    return jnp.array([1.0, ct, gp, LOG10_FGW, 1.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def two_source_blocks():
    """Per-pulsar 4x4 (b, G) for two CW sources on one synthetic pulsar."""
    from jaxpint.bayes import marginalize_single_pulsar

    td, tm, nm, pp = make_simple_pulsar(200, f0=100.0, f1=-1e-14)
    over = {n for n in pp.free_names() if n in ("F0", "F1")}
    g, _, skel = marginalize_single_pulsar(
        over=over,
        toa_data=td, timing_model=tm, noise_model=nm, fiducial_params=pp,
        allow_nonlinear=True, validate_linearity=False)
    pos = jnp.array([0.2, 0.5, -0.84]); pos = pos / jnp.linalg.norm(pos)

    def tmpl(ct, gp):
        e = cw_delay_from_array(td, pos, 1.0, _cw(ct, gp), linear_amplitude=True,
                                earth_term_only=True)
        ps = cw_delay_from_array(td, pos, 1.0, _cw(ct, gp), linear_amplitude=True,
                                 pulsar_term_only=True, pulsar_term_phase=float(np.pi / 2))
        return e, ps

    e0, ps0 = tmpl(0.3, 2.0)    # source 0 (scanned)
    e1, ps1 = tmpl(-0.4, 4.5)   # source 1 (static)
    b, G = extract_pulsar_blocks(g, skel, jnp.stack([e0, ps0, e1, ps1]))
    return dict(g=g, skel=skel, e0=e0, ps0=ps0, e1=e1, ps1=ps1, b=b, G=G)


def test_extract_pulsar_blocks_recovers_known_quadratic():
    # analytic oracle: a g exactly quadratic in its external_delay, with known
    # linear vector q and noise matrix N, gives logL(A) = g(.., A @ basis) with
    # b = basis @ q and G = basis @ N @ basis^T -- recovered to machine precision.
    n_toas, m = 6, 4
    k1, k2, k3 = jax.random.split(jax.random.key(2), 3)
    q = jax.random.normal(k1, (n_toas,))
    R = jax.random.normal(k2, (n_toas, n_toas))
    N = R @ R.T + n_toas * jnp.eye(n_toas)  # symmetric positive-definite
    basis = jax.random.normal(k3, (m, n_toas))

    def quadratic_g(reduced_params, external_delay):
        d = external_delay
        return q @ d - 0.5 * d @ N @ d

    b, G = extract_pulsar_blocks(quadratic_g, None, basis)
    assert jnp.allclose(b, basis @ q)
    assert jnp.allclose(G, basis @ N @ basis.T)


def test_self_blocks_match_independent_single_source(two_source_blocks):
    # each source's diagonal block must equal its standalone (b, M); the
    # off-diagonal cross-block is the genuinely-new inter-source coupling.
    d = two_source_blocks
    b, G = d["b"], d["G"]
    b0, M0 = extract_pulsar_bM(d["g"], d["skel"], d["e0"], d["ps0"])
    b1, M1 = extract_pulsar_bM(d["g"], d["skel"], d["e1"], d["ps1"])
    assert jnp.allclose(b[0:2], b0) and jnp.allclose(G[0:2, 0:2], M0)
    assert jnp.allclose(b[2:4], b1) and jnp.allclose(G[2:4, 2:4], M1)
    assert not jnp.allclose(G[0:2, 2:4], 0.0)  # sources couple through the noise metric


def test_conditioning_identity_holds(two_source_blocks):
    # logL_full(a0, a1_fixed) == logL_cond(a0) + const for all a0 (the O(1)-in-S core)
    d = two_source_blocks
    b, G = d["b"], d["G"]
    a1 = jnp.array([0.7e-13, -1.1e-13])
    b_eff, G_scan = condition_on_statics(b, G, a1, n_scan=2)
    const = a1 @ b[2:4] - 0.5 * a1 @ G[2:4, 2:4] @ a1

    def full(a0):
        a = jnp.concatenate([a0, a1])
        return a @ b - 0.5 * a @ G @ a

    def cond(a0):
        return a0 @ b_eff - 0.5 * a0 @ G_scan @ a0

    a0_grid = jax.random.normal(jax.random.key(1), (30, 2)) * 1e-13
    lhs = jax.vmap(cond)(a0_grid)
    rhs = jax.vmap(lambda a0: full(a0) - const)(a0_grid)
    rel = jnp.max(jnp.abs(lhs - rhs)) / jnp.mean(jnp.abs(rhs))
    assert rel < 1e-9
