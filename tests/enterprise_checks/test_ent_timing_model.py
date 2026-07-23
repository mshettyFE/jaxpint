"""Analytic timing-model marginalization vs enterprise's TimingModel GP.

Both stacks integrate the free timing parameters out against an (effectively)
flat prior implemented as a GP with weight 1e40 on the design-matrix columns.
The free set here is {F0, F1, DM, PHOFF} (sky position frozen in the par so
enterprise's design matrix and JaxPINT's `over` set span the same subspace;
PHOFF free also makes PINT skip implicit mean subtraction).

Normalization: with the 1e40 flat prior, rescaling design-matrix columns
(M -> M B) shifts logL by the parameter-independent constant 2 log|det B|.
Enterprise's default ``TimingModel(normed=True)`` column-normalizes M, so

- vs ``normed=False`` the absolute values agree (B = identity between PINT's
  and JaxPINT's design matrices; observed ~3e-6 agreement),
- vs ``normed=True`` only the *offset constancy* across a noise grid is
  asserted (the offset is the norm-product constant).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.enterprise_checks._ent_helpers import (
    build_pulsar,
    clustered_mjds,
    make_par,
)

EFAC = 1.1
OVER = frozenset({"F0", "F1", "DM", "PHOFF"})


@pytest.fixture(scope="module")
def timing_bundle(tmp_path_factory):
    par = make_par(efac=EFAC, fit_spin=True)
    tmp = tmp_path_factory.mktemp("ent_timing")
    return build_pulsar(tmp, par, clustered_mjds(n_epochs=25, per_epoch=2), seed=3)


def _ent_marg_logL(bundle, efac_value, normed):
    from enterprise.signals import gp_signals, parameter, signal_base, white_signals

    mn = white_signals.MeasurementNoise(efac=parameter.Constant(efac_value))
    tm = gp_signals.TimingModel(normed=normed)
    pta = signal_base.PTA([(mn + tm)(bundle.psr)])
    return pta.get_lnlikelihood({})


def _jax_marg_logL(bundle, efac_value):
    from jaxpint.bayes.marginal import marginalize_single_pulsar

    efac_name = next(n for n in bundle.params.names if n.startswith("EFAC"))
    params = bundle.params.with_value(efac_name, efac_value)
    g, _, skeleton = marginalize_single_pulsar(
        over=OVER,
        toa_data=bundle.toa_data,
        timing_model=bundle.timing_model,
        noise_model=bundle.noise_model,
        fiducial_params=params,
    )
    return float(g(skeleton))


def test_enterprise_marginalizes_expected_params(timing_bundle):
    """Guard: enterprise's design matrix spans exactly the params JaxPINT
    marginalizes — a span mismatch would invalidate every comparison below."""
    assert set(timing_bundle.psr.fitpars) == set(OVER)


def test_timing_marg_absolute_unnormalized(timing_bundle):
    """Absolute marginalized logL matches TimingModel(normed=False).

    Agreement here means PINT's and JaxPINT's design-matrix columns agree up
    to sign (|det B| = 1) on top of the shared 1e40 flat-prior convention.
    Observed ~3e-6 on |logL| ~ 4e2.
    """
    logL_ent = _ent_marg_logL(timing_bundle, EFAC, normed=False)
    logL_jax = _jax_marg_logL(timing_bundle, EFAC)
    npt.assert_allclose(
        logL_jax,
        logL_ent,
        atol=1e-4,
        rtol=0,
        err_msg="marginalized timing-model logL mismatch (normed=False)",
    )


def test_timing_marg_delta_over_efac_grid(timing_bundle):
    """Delta-logL across an EFAC grid matches to ~1e-5 (constants cancel)."""
    grid = [0.9, 1.3, 2.0]
    logLs_ent = [_ent_marg_logL(timing_bundle, e, normed=True) for e in grid]
    logLs_jax = [_jax_marg_logL(timing_bundle, e) for e in grid]
    npt.assert_allclose(
        np.diff(logLs_jax),
        np.diff(logLs_ent),
        atol=1e-4,
        rtol=1e-5,
        err_msg="marginalized delta-logL over EFAC grid disagrees",
    )


def test_timing_marg_normed_offset_is_constant(timing_bundle):
    """TimingModel(normed=True) differs from JaxPINT by a constant only.

    The constant is 2 log|det B| for enterprise's column-norm rescaling B; it
    must not depend on noise parameters.
    """
    grid = [0.9, 1.3, 2.0]
    offsets = [
        _jax_marg_logL(timing_bundle, e)
         - _ent_marg_logL(timing_bundle, e, normed=True)
        for e in grid
    ]
    assert np.ptp(offsets) < 1e-4, (
        f"normed=True offset varies across EFAC grid: {offsets} — "
        "normalization is leaking into parameter dependence"
    )
