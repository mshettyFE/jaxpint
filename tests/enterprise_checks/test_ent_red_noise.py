"""Power-law red-noise (Fourier GP) cross-validation vs enterprise.

The par file carries TNREDAMP/TNREDGAM/TNREDC; JaxPINT's bridge builds
``PLRedNoise`` from it, enterprise gets an equivalent constant-parameter
``FourierBasisGP``.  The enterprise GP is constructed with an explicit
``Tspan`` equal to JaxPINT's *basis* span so the frequency grids coincide;
``test_basis_span_matches_enterprise_span`` cross-pins that span against
enterprise's own barycentered TOAs so handing JaxPINT's span to enterprise
does not close the comparison loop.  Both stacks now evaluate the basis at
identical barycentric times (test_ent_building_blocks.py), so the kernel
comparison is exact and the end-to-end difference is residual-parity-limited
(observed ~8e-5 in logL).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.enterprise_checks._ent_helpers import (
    build_pulsar,
    clustered_mjds,
    dense_logL,
    make_par,
)

LOG10_A = -13.5
GAMMA = 3.5
N_COMP = 10
EFAC = 1.2


@pytest.fixture(scope="module")
def red_bundle(tmp_path_factory):
    par = make_par(efac=EFAC, red=(LOG10_A, GAMMA, N_COMP))
    tmp = tmp_path_factory.mktemp("ent_red")
    return build_pulsar(tmp, par, clustered_mjds(n_epochs=50, per_epoch=2), seed=7)


def _jax_red(bundle):
    from jaxpint.noise.red_noise import PLRedNoise

    return next(c for c in bundle.noise_model.correlated if isinstance(c, PLRedNoise))


def _basis_span(bundle) -> float:
    """Span of the coordinate JaxPINT actually evaluates its GP basis at.

    Must track ``TOAData.basis_seconds``, *not* ``tdb_seconds``: the basis is
    barycentric (``basis_coord``), and the TDB span differs by ~2e-6 relative
    (the differential Roemer delay).  Since ``phi ~ f**(-gamma) * df`` and both
    scale with ``1/T``, pinning enterprise to the TDB span would offset every
    weight by ``(T_jax/T_ent)**(gamma-1)`` -- invisible at logL tolerances but
    fatal to the tight PSD-weight comparison below.

    Because this span is handed to enterprise, it is a shared input, not an
    independently-verified one -- ``test_basis_span_matches_enterprise_span``
    cross-pins it against enterprise's own barycentered ``psr.toas`` so a
    frame/span bug in ``basis_seconds`` cannot silently drag both stacks to
    the same wrong frequency grid.  (Sibling helpers take the opposite route:
    ``shared_tspan`` derives the span from enterprise's ``psr.toas``.)
    """
    basis = np.asarray(bundle.toa_data.basis_seconds)
    return float(basis.max() - basis.min())


def test_basis_span_matches_enterprise_span(red_bundle):
    """Guard: JaxPINT's basis span == enterprise's barycentered TOA span.

    Everything downstream hands ``_basis_span`` (a JaxPINT quantity) to
    enterprise's ``Tspan``; without this cross-pin a basis_seconds bug would
    move both stacks' frequency grids together and the PSD/logL comparisons
    would pass vacuously.  Both spans are barycentric, so the agreement is
    exact (observed 0 s).
    """
    b = red_bundle
    span_ent = float(np.asarray(b.psr.toas).max() - np.asarray(b.psr.toas).min())
    assert abs(_basis_span(b) - span_ent) < 1e-6, (
        f"JaxPINT basis span {_basis_span(b)!r} != enterprise span "
        f"{span_ent!r}; basis_seconds frame drifted -- the shared-Tspan "
        "comparisons below are no longer trustworthy"
    )


@pytest.fixture(scope="module")
def ent_red_pta(red_bundle):
    """Enterprise MeasurementNoise + constant-powerlaw FourierBasisGP."""
    from enterprise.signals import gp_signals, parameter, signal_base, utils, white_signals

    mn = white_signals.MeasurementNoise(efac=parameter.Constant(EFAC))
    rn = gp_signals.FourierBasisGP(
        spectrum=utils.powerlaw(
            log10_A=parameter.Constant(LOG10_A), gamma=parameter.Constant(GAMMA)
        ),
        components=N_COMP,
        Tspan=_basis_span(red_bundle),
    )
    return signal_base.PTA([(mn + rn)(red_bundle.psr)])


def test_red_psd_weights(red_bundle, ent_red_pta):
    """PLRedNoise psd_weights == enterprise get_phi at a shared Tspan."""
    red = _jax_red(red_bundle)
    phi_jax = np.asarray(red.psd_weights(red_bundle.params))
    # The PTA has exactly one basis signal (the red GP); its phi is the
    # concatenated basis prior for that pulsar.
    phi_ent = ent_red_pta.get_phi({})[0]
    npt.assert_allclose(
        phi_jax,
        np.asarray(phi_ent),
        rtol=1e-9,
        err_msg="power-law PSD weights disagree (TNREDAMP/TNREDGAM conventions?)",
    )


def test_red_kernel_logL(red_bundle, ent_red_pta):
    """Enterprise PTA logL vs dense logL from JaxPINT (Ndiag, U, Phi).

    Kernel-level (enterprise residuals both sides).  With both stacks'
    bases at identical barycentric times, this is exact up to solver float
    noise (observed diff 0.0 on |logL| ~ 1.3e3).
    """
    b = red_bundle
    logL_ent = ent_red_pta.get_lnlikelihood({})
    Ndiag, U, Phi = b.noise_model.covariance(b.toa_data, b.params)
    logL_jax = dense_logL(b.psr.residuals, Ndiag, U, Phi)
    npt.assert_allclose(
        logL_jax,
        logL_ent,
        rtol=1e-10,
        err_msg="red-noise kernel logL mismatch",
    )


def test_red_logL_grid_delta(red_bundle):
    """Delta-logL across a (log10_A, gamma) grid matches (constants cancel).

    Enterprise side re-parameterized with Uniform red-noise params; JaxPINT
    side re-evaluates single_pulsar_logL with TNREDAMP/TNREDGAM overridden.
    """
    from enterprise.signals import gp_signals, parameter, signal_base, utils, white_signals

    from jaxpint.likelihood import single_pulsar_logL

    b = red_bundle
    mn = white_signals.MeasurementNoise(efac=parameter.Constant(EFAC))
    rn = gp_signals.FourierBasisGP(
        spectrum=utils.powerlaw(
            log10_A=parameter.Uniform(-18, -11), gamma=parameter.Uniform(0, 7)
        ),
        components=N_COMP,
        Tspan=_basis_span(b),
    )
    pta = signal_base.PTA([(mn + rn)(b.psr)])
    name_A = next(p for p in pta.param_names if p.endswith("log10_A"))
    name_g = next(p for p in pta.param_names if p.endswith("gamma"))

    grid = [(-14.0, 4.33), (-13.5, 3.0), (-13.0, 5.0)]
    logLs_ent, logLs_jax = [], []
    for log10_A, gamma in grid:
        logLs_ent.append(pta.get_lnlikelihood({name_A: log10_A, name_g: gamma}))
        params = b.params.with_value("TNREDAMP", log10_A).with_value("TNREDGAM", gamma)
        logLs_jax.append(
            float(single_pulsar_logL(b.toa_data, b.timing_model, b.noise_model, params))
        )
    d_ent = np.diff(logLs_ent)
    d_jax = np.diff(logLs_jax)
    npt.assert_allclose(
        d_jax,
        d_ent,
        atol=5e-3,
        rtol=1e-4,
        err_msg="red-noise delta-logL over parameter grid disagrees",
    )


def test_red_logL_end_to_end_absolute(red_bundle, ent_red_pta):
    """Full single_pulsar_logL vs enterprise PTA (same normalization).

    Residual-parity-limited (observed ~8.5e-5 in logL); the tolerance
    carries ~10x margin over that.
    """
    from jaxpint.likelihood import single_pulsar_logL

    b = red_bundle
    logL_ent = ent_red_pta.get_lnlikelihood({})
    logL_jax = float(
        single_pulsar_logL(b.toa_data, b.timing_model, b.noise_model, b.params)
    )
    npt.assert_allclose(
        logL_jax,
        logL_ent,
        atol=1e-3,
        rtol=0,
        err_msg="end-to-end red-noise logL mismatch beyond residual-parity budget",
    )
