"""Tests for jaxpint.frequentist.detection (F-stat detection + empirical backgrounds)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import chi2

from jaxpint.pta.cw_upper_limit import basis_quadratics
from jaxpint.pta.signals.cw import cw_delay_from_array
from jaxpint.frequentist.detection import (
    fstat,
    quadrature_blocks,
    fstat_skymap,
    fstat_p,
    fstat_p_pvalue,
    phase_shift_background,
    sky_scramble_background,
    _antenna_grid,
)
from jaxpint.frequentist.nulls import pvalue
from tests.helpers import make_simple_pulsar

LOG10_FGW = -5.0  # resolved over the synthetic span -> full-rank network Gram
_POSITIONS = [
    jnp.array([0.2, 0.5, -0.84]),
    jnp.array([-0.6, 0.3, 0.7]),
    jnp.array([0.8, -0.5, 0.1]),
]


@pytest.fixture(scope="module")
def real_blocks():
    """3 pulsars through the real marginalized likelihood: per-pulsar (S,C)/G + geometry."""
    from jaxpint.bayes import marginalize_single_pulsar, ImproperPrior

    pulsars, sc_l, gram_l = [], [], []
    for i, p in enumerate(_POSITIONS):
        td, tm, nm, pp = make_simple_pulsar(200, f0=100.0, f1=-1e-14, seed=i)
        over = {n for n in pp.free_names() if n in ("F0", "F1")}
        g, _, skel = marginalize_single_pulsar(
            over=over,
            priors={n: ImproperPrior() for n in over},
            toa_data=td,
            timing_model=tm,
            noise_model=nm,
            fiducial_params=pp,
            allow_nonlinear=True,
            validate_linearity=False,
        )
        _ = g(skel)  # warm up the noise model's cached device basis
        pos = p / jnp.linalg.norm(p)
        sc, gram = quadrature_blocks(g, skel, td, LOG10_FGW)
        pulsars.append((g, skel, td, pos))
        sc_l.append(sc)
        gram_l.append(gram)
    return {
        "pulsars": pulsars,
        "positions": jnp.stack([x[3] for x in pulsars]),
        "sc_all": jnp.stack(sc_l),
        "gram_all": jnp.stack(gram_l),
    }


def test_factorized_2f_matches_fstat(real_blocks):
    # The antenna-folded network 2F (extract each pulsar's 2x2 once, fold analytically)
    # must equal the direct F-statistic (per-pulsar 4x4 basis_quadratics summed), pixel
    # by pixel -- confirming the sky-independent-quadrature factorization.
    pulsars, positions = real_blocks["pulsars"], real_blocks["positions"]
    ct = jnp.array([0.35, 0.1, -0.5, 0.8])
    gp = jnp.array([1.1, 2.0, 4.0, 0.3])
    mine = fstat_skymap(
        real_blocks["sc_all"], real_blocks["gram_all"], positions, ct, gp
    )

    direct = []
    for c, p in zip(ct.tolist(), gp.tolist()):
        M, b = jnp.zeros((4, 4)), jnp.zeros(4)
        for g, skel, td, pos in pulsars:

            def logL(
                amp, cos_inc, psi, phase0, g=g, skel=skel, td=td, pos=pos, c=c, p=p
            ):
                cw = jnp.array([amp, c, p, LOG10_FGW, cos_inc, psi, phase0])
                d = cw_delay_from_array(
                    td, pos, 1.0, cw, earth_term_only=True, linear_amplitude=True
                )
                return g(skel, external_delay=d)

            Ma, ba = basis_quadratics(logL)
            M, b = M + Ma, b + ba
        direct.append(fstat(M, b))
    assert jnp.allclose(mine, jnp.array(direct), rtol=1e-5)


def test_backgrounds_reject_a_coherent_signal():
    # On a *controlled* coherent signal -- (S,C)_a = G_a (antenna_a . c_true) at a known
    # sky, no likelihood/noise artifacts -- the sky-max 2F must (a) peak at the truth
    # pixel and (b) sit far out in the tail of both empirical nulls, which destroy the
    # inter-pulsar coherence (phase shifts) and the geometry (sky scrambles).
    npsr, ct_truth, gp_truth = 24, 0.3, 1.2
    kp, kg, kc = jax.random.split(jax.random.PRNGKey(0), 3)
    positions = jax.random.normal(kp, (npsr, 3))
    positions = positions / jnp.linalg.norm(positions, axis=1, keepdims=True)
    L = jax.random.normal(kg, (npsr, 2, 2))
    gram_all = jax.vmap(lambda quad: quad @ quad.T + jnp.eye(2))(
        L
    )  # SPD quadrature Grams
    c_true = jax.random.normal(kc, (4,))

    F = _antenna_grid(positions, jnp.array([ct_truth]), jnp.array([gp_truth]))[
        0
    ]  # (npsr, 2)
    ab = jnp.stack(  # (alpha, beta)_a = (fp c1 + fc c3, fp c2 + fc c4)
        [
            F[:, 0] * c_true[0] + F[:, 1] * c_true[2],
            F[:, 0] * c_true[1] + F[:, 1] * c_true[3],
        ],
        axis=1,
    )
    sc_all = jnp.einsum("aij,aj->ai", gram_all, ab)  # (S, C) = G (alpha, beta)

    v = jax.random.normal(jax.random.PRNGKey(7), (60, 3))
    v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    ct = jnp.concatenate([jnp.array([ct_truth]), v[:, 2]])
    gp = jnp.concatenate([jnp.array([gp_truth]), jnp.arctan2(v[:, 1], v[:, 0])])

    m = fstat_skymap(sc_all, gram_all, positions, ct, gp)
    assert int(m.argmax()) == 0  # localizes to the truth pixel
    stat = float(m.max())
    bp = phase_shift_background(
        sc_all, gram_all, positions, ct, gp, 500, jax.random.PRNGKey(1)
    )
    bs = sky_scramble_background(sc_all, gram_all, ct, gp, 500, jax.random.PRNGKey(2))
    assert pvalue(stat, bp) < 0.01
    assert pvalue(stat, bs) < 0.01


def test_fstat_p_null_is_chi2_2n():
    # Under noise-only matched filters (S,C)_a ~ N(0, G_a), 2F_p = sum_a (S,C) G^-1 (S,C)
    # must follow chi^2(2 n_psr): its mean is 2 n_psr and the analytic p-values are
    # Uniform(0,1). This is what makes the chi^2(2 n_psr) threshold the correct null.
    npsr = 8
    kg, kz = jax.random.split(jax.random.PRNGKey(0))
    gram_all = jax.vmap(lambda quad: quad @ quad.T + jnp.eye(2))(
        jax.random.normal(kg, (npsr, 2, 2))
    )
    chol = jnp.linalg.cholesky(gram_all)  # (S,C) = chol @ z has covariance G

    def stat(k):
        z = jax.random.normal(k, (npsr, 2))
        return fstat_p(jnp.einsum("aij,aj->ai", chol, z), gram_all)

    stats = np.asarray(jax.vmap(stat)(jax.random.split(kz, 600)))
    ps = chi2.sf(stats, 2 * npsr)  # analytic p-values under the null
    assert abs(float(stats.mean()) - 2 * npsr) < 1.0  # mean of chi^2(2N) is 2N
    assert abs(float(ps.mean()) - 0.5) < 0.05  # p-values are Uniform(0,1)
    # fstat_p_pvalue matches the scipy survival function it wraps
    assert np.isclose(fstat_p_pvalue(float(stats[0]), npsr), float(ps[0]))


def test_quadrature_blocks_affine_in_injected_strain():
    # The detection example driver extracts each pulsar's block ONCE and recombines
    # with the network-calibrated h0 via sc(h0) = sc_data + h0 * sc_sig, with G
    # independent of h0 -- a single-pass, bounded-memory alternative to holding every
    # pulsar's likelihood and injecting at the final h0. That refactor is exact ONLY
    # if the matched filter is affine in the injected strain and the Gram is
    # injection-independent; verify both directly against the real likelihood.
    from jaxpint.bayes import marginalize_single_pulsar, ImproperPrior

    td, tm, nm, pp = make_simple_pulsar(200, f0=100.0, f1=-1e-14, seed=0)
    over = {n for n in pp.free_names() if n in ("F0", "F1")}
    g, _, skel = marginalize_single_pulsar(
        over=over,
        priors={n: ImproperPrior() for n in over},
        toa_data=td,
        timing_model=tm,
        noise_model=nm,
        fiducial_params=pp,
        allow_nonlinear=True,
        validate_linearity=False,
    )
    pos = jnp.array([0.2, 0.5, -0.84])
    pos = pos / jnp.linalg.norm(pos)
    cw_unit = jnp.array([1.0, 0.3, 1.2, LOG10_FGW, 0.5, 0.6, 0.9])
    s_unit = cw_delay_from_array(
        td, pos, 1.0, cw_unit, earth_term_only=True, linear_amplitude=True
    )

    def blocks_at(h0):  # extract (S,C), G with an h0-strain signal injected
        g_inj = lambda rp, external_delay=0.0: g(
            rp, external_delay=external_delay - h0 * s_unit
        )
        return quadrature_blocks(g_inj, skel, td, LOG10_FGW)

    sc0, G0 = blocks_at(0.0)
    sc1, G1 = blocks_at(1.0)
    sc2, G2 = blocks_at(2.0)
    # affine in h0: sc(2) - sc(0) == 2 * (sc(1) - sc(0))
    assert jnp.allclose(sc2 - sc0, 2.0 * (sc1 - sc0), rtol=1e-6)
    # Gram is injection-independent
    assert jnp.allclose(G0, G1, rtol=1e-8)
    assert jnp.allclose(G0, G2, rtol=1e-8)
