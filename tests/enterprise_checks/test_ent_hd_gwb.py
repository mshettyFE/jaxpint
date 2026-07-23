"""HD-correlated GWB pta_logL vs enterprise's FourierBasisCommonGP, 3 pulsars.

This is the deepest end-to-end check: JaxPINT's two-tier Woodbury for
cross-pulsar-correlated GPs against enterprise's block phi-matrix likelihood.

Two deliberate wrinkles:

- enterprise 3.4.4's default sparse-Cholesky likelihood is incompatible with
  scikit-sparse 0.5.0 (the cholmod API was rewritten; ``cholesky()`` no longer
  returns a callable Factor), so the PTA is built with
  ``lnlikelihood=LogLikelihoodDenseCholesky`` — numerically equivalent, just
  a dense solve.
- the ORF diagonal: both stacks now put 1.0 on the auto-term (pulsar term
  included), pinned at unit level by
  ``test_ent_building_blocks.test_hd_orf_diagonal_convention``, so the logL
  comparison below runs fully native
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.enterprise_checks._ent_helpers import shared_tspan
pytestmark = pytest.mark.slow

N_COMP = 8
GRID = [(-14.0, 4.33), (-13.5, 3.0)]


def _ent_hd_pta(bundles, tspan):
    from enterprise.signals import gp_signals, parameter, signal_base, utils, white_signals

    gw_log10_A = parameter.Uniform(-18, -11)("gw_log10_A")
    gw_gamma = parameter.Uniform(0, 7)("gw_gamma")
    hd = gp_signals.FourierBasisCommonGP(
        spectrum=utils.powerlaw(log10_A=gw_log10_A, gamma=gw_gamma),
        orf=utils.hd_orf(),
        components=N_COMP,
        Tspan=tspan,
        name="gw",
    )
    models = []
    for b in bundles:
        efac_name = next(n for n in b.params.names if n.startswith("EFAC"))
        efac_val = float(b.params.param_value(efac_name))
        mn = white_signals.MeasurementNoise(efac=parameter.Constant(efac_val))
        models.append((mn + hd)(b.psr))
    # Dense Cholesky: the default sparse path needs the pre-0.5 scikit-sparse
    # Factor API and crashes with "'tuple' object is not callable".
    return signal_base.PTA(models, lnlikelihood=signal_base.LogLikelihoodDenseCholesky)


def _jax_hd(bundles, tspan, orf_func=None):
    import jax.numpy as jnp

    from jaxpint.pta.likelihood import PTAConfig
    from jaxpint.pta.signals.correlated_gwb import HDCorrelatedGWBInjector
    from jaxpint.types import GlobalParams

    positions = jnp.asarray(np.vstack([b.psr.pos for b in bundles]))
    kwargs = {} if orf_func is None else {"orf_func": orf_func}
    inj = HDCorrelatedGWBInjector(
        pulsar_positions=positions, n_components=N_COMP, T_span=tspan,
        prefix="gw_", **kwargs,
    )
    global_params = inj.register_params(GlobalParams.empty())
    config = PTAConfig(
        toa_data_list=tuple(b.toa_data for b in bundles),
        timing_models=tuple(b.timing_model for b in bundles),
        noise_models=tuple(b.noise_model for b in bundles),
        signal_injectors=(),
        correlated_injectors=(inj,),
    )
    return global_params, config


def _grid_logLs(pta, global_params, config, bundles):
    from jaxpint.pta.likelihood import pta_logL

    pulsar_params = tuple(b.params for b in bundles)
    logLs_ent, logLs_jax = [], []
    for log10_A, gamma in GRID:
        logLs_ent.append(
            pta.get_lnlikelihood({"gw_log10_A": log10_A, "gw_gamma": gamma})
        )
        gp = global_params.with_value("gw_log10_A", log10_A).with_value(
            "gw_gamma", gamma
        )
        logLs_jax.append(float(pta_logL(gp, pulsar_params, config)))
    return np.asarray(logLs_ent), np.asarray(logLs_jax)


def test_hd_logL_matches_enterprise(pta_bundles):
    """Two-tier Woodbury == enterprise, fully native (no ORF override).

    """
    tspan = shared_tspan(pta_bundles)
    pta = _ent_hd_pta(pta_bundles, tspan)
    global_params, config = _jax_hd(pta_bundles, tspan)  # native hd_orf
    logLs_ent, logLs_jax = _grid_logLs(pta, global_params, config, pta_bundles)

    npt.assert_allclose(
        logLs_jax,
        logLs_ent,
        atol=5e-4,
        rtol=0,
        err_msg="native HD-correlated logL mismatch beyond residual-parity budget",
    )
    npt.assert_allclose(
        np.diff(logLs_jax),
        np.diff(logLs_ent),
        atol=5e-4,
        rtol=1e-4,
        err_msg="HD delta-logL over parameter grid disagrees",
    )
