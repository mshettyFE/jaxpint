"""CURN (uncorrelated common red noise) pta_logL vs enterprise, 3 pulsars.

Enterprise models CURN as per-pulsar FourierBasisGPs sharing the same Uniform
parameter objects; JaxPINT as a CURNInjector in PTAConfig.signal_injectors.
Timing parameters are frozen in the par files, so no timing-model signal /
marginalization is involved and the comparison isolates the common-signal
machinery on top of the white-noise + residual parity established by the
faster modules.

Both stacks receive the same explicit Tspan (derived from enterprise's own
barycentered toas via ``shared_tspan``) and evaluate their GP bases at
identical barycentric times (see test_ent_building_blocks.py), so absolute
agreement is residual-parity-limited; observed ~2.6e-5 over the grid.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.enterprise_checks._ent_helpers import shared_tspan

pytestmark = pytest.mark.slow

N_COMP = 8
GRID = [(-14.0, 4.33), (-13.5, 3.0), (-14.5, 5.0)]


def _ent_curn_pta(bundles, tspan):
    from enterprise.signals import gp_signals, parameter, signal_base, utils, white_signals

    gw_log10_A = parameter.Uniform(-18, -11)("gw_log10_A")
    gw_gamma = parameter.Uniform(0, 7)("gw_gamma")
    crn = gp_signals.FourierBasisGP(
        spectrum=utils.powerlaw(log10_A=gw_log10_A, gamma=gw_gamma),
        components=N_COMP,
        Tspan=tspan,
        name="gw",
    )
    models = []
    for b in bundles:
        efac_name = next(n for n in b.params.names if n.startswith("EFAC"))
        efac_val = float(b.params.param_value(efac_name))
        mn = white_signals.MeasurementNoise(efac=parameter.Constant(efac_val))
        models.append((mn + crn)(b.psr))
    return signal_base.PTA(models)


def _jax_curn(bundles, tspan):
    from jaxpint.pta.likelihood import PTAConfig
    from jaxpint.pta.signals.gwb import CURNInjector
    from jaxpint.types import GlobalParams

    curn = CURNInjector(n_components=N_COMP, T_span=tspan, prefix="gw_")
    global_params = curn.register_params(GlobalParams.empty())
    config = PTAConfig(
        toa_data_list=tuple(b.toa_data for b in bundles),
        timing_models=tuple(b.timing_model for b in bundles),
        noise_models=tuple(b.noise_model for b in bundles),
        signal_injectors=(curn,),
        correlated_injectors=(),
    )
    return global_params, config


def test_curn_logL_grid(pta_bundles):
    """Absolute and delta logL parity over a (log10_A, gamma) grid.

    Residual-parity-limited (observed ~2.6e-5); tolerance carries ~10x margin.
    """
    from jaxpint.pta.likelihood import pta_logL

    tspan = shared_tspan(pta_bundles)
    pta = _ent_curn_pta(pta_bundles, tspan)
    global_params, config = _jax_curn(pta_bundles, tspan)
    pulsar_params = tuple(b.params for b in pta_bundles)

    logLs_ent, logLs_jax = [], []
    for log10_A, gamma in GRID:
        logLs_ent.append(
            pta.get_lnlikelihood({"gw_log10_A": log10_A, "gw_gamma": gamma})
        )
        gp = global_params.with_value("gw_log10_A", log10_A).with_value(
            "gw_gamma", gamma
        )
        logLs_jax.append(float(pta_logL(gp, pulsar_params, config)))

    logLs_jax = np.asarray(logLs_jax)
    npt.assert_allclose(
        logLs_jax,
        logLs_ent,
        atol=3e-4,
        rtol=0,
        err_msg="CURN absolute logL mismatch beyond residual-parity budget",
    )
    npt.assert_allclose(
        np.diff(logLs_jax),
        np.diff(logLs_ent),
        atol=3e-4,
        rtol=1e-4,
        err_msg="CURN delta-logL over parameter grid disagrees",
    )


def test_curn_equals_sum_of_single_pulsar_ptas(pta_bundles):
    """Fixture sanity: with no cross-correlations the joint enterprise logL is
    the sum of per-pulsar logLs (guards against accidental common signals)."""
    from enterprise.signals import signal_base

    tspan = shared_tspan(pta_bundles)
    pta = _ent_curn_pta(pta_bundles, tspan)
    params = {"gw_log10_A": -14.0, "gw_gamma": 4.33}
    total = pta.get_lnlikelihood(params)

    singles = 0.0
    full_models = _ent_curn_pta(pta_bundles, tspan)._signalcollections
    for sc in full_models:
        singles += signal_base.PTA([sc]).get_lnlikelihood(params)
    npt.assert_allclose(total, singles, rtol=1e-12)
