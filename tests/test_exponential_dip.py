"""Tests for ExponentialDip delay component against PINT."""

from __future__ import annotations

from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model
from jaxpint.delay.exponential_dip import ExponentialDip
from tests.helpers import make_params, make_toa_data

_BASE_PAR = """\
PSR           J1234+5678
RAJ           12:34:56.789
DECJ          +56:07:08.12
F0            100.0
F1            -1e-15
PEPOCH        55000
DM            15.0
EPHEM         DE421
CLK           UTC(NIST)
UNITS         TDB
CORRECT_TROPOSPHERE  N
PLANET_SHAPIRO       N
"""


class TestExponentialDipvsPINT:

    @pytest.fixture(scope="class")
    def pint_setup(self):
        from pint.models import get_model
        from pint.simulation import make_fake_toas_uniform

        par = _BASE_PAR + """\
EXPDIPEPS     0.5
EXPDIPFREF    1400
EXPDIPEP_1    54200
EXPDIPAMP_1   1e-8
EXPDIPIDX_1   0.0
EXPDIPTAU_1   100
"""
        model = get_model(StringIO(par))
        toas = make_fake_toas_uniform(
            54500, 55500, 40, model, freq=1400.0, add_noise=False,
        )
        toas.compute_TDBs()
        toas.compute_posvels()

        pint_delay = np.array(
            model.components["SimpleExponentialDip"].expdip_delay(toas).to("s").value,
            dtype=np.float64,
        )

        toa_data = pint_toas_to_jax(toas, model)
        params = pint_model_to_params(model).params

        comp = ExponentialDip(
            n_dips=1,
            expdipep_names=("EXPDIPEP_1",),
            expdipamp_names=("EXPDIPAMP_1",),
            expdipidx_names=("EXPDIPIDX_1",),
            expdiptau_names=("EXPDIPTAU_1",),
            expdipeps_name="EXPDIPEPS",
            expdipfref_name="EXPDIPFREF",
        )
        return toa_data, params, pint_delay, model, comp

    @pytest.mark.slow
    def test_delay_matches_pint(self, pint_setup):
        toa_data, params, pint_delay, _, comp = pint_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    @pytest.mark.slow
    def test_jit_compatible(self, pint_setup):
        toa_data, params, _, _, comp = pint_setup
        delay = jnp.zeros(toa_data.n_toas)
        eager = comp(toa_data, params, delay)
        jitted = jax.jit(comp)(toa_data, params, delay)
        np.testing.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-14)

    @pytest.mark.slow
    def test_grad_finite_for_amplitudes(self, pint_setup):
        toa_data, params, _, _, comp = pint_setup

        def loss(p):
            return comp(toa_data, p, jnp.zeros(toa_data.n_toas)).sum()

        grads = jax.grad(loss)(params)
        # Check that amplitude gradients are finite (the normalization
        # factor can produce NaN gradients for frozen eps/tau params,
        # which is acceptable since those are not fitted).
        amp_idx = params._name_to_index["EXPDIPAMP_1"]
        assert jnp.isfinite(grads.values[amp_idx])

    @pytest.mark.slow
    def test_bridge_builds_expdip(self, pint_setup):
        _, _, _, model, _ = pint_setup
        tm, _ = build_timing_model(model)
        assert any(isinstance(c, ExponentialDip) for c in tm.delay_components)


class TestGradientSafety:
    """NaN-gradient regressions: the former where(dt >= 0, ...) expfac
    overflowed exp(±dt/eps) to inf/inf = nan for TOAs far from the dip
    (|dt| > ~709*tau suffices), and although where() masked the forward
    value, the nan leaked into gradients of every dip parameter.  The
    log-space form (exp(-dt/tau + log_sigmoid(dt/eps))) cannot overflow."""

    def _setup(self, *, eps=0.05, tau=1.0, span=(50000.0, 57000.0)):
        # Dip at 55000 with a short tau: TOAs thousands of days on either
        # side put |dt/tau| and |dt/eps| far beyond the exp overflow point.
        toa_data = make_toa_data(t_mjd=np.linspace(span[0], span[1], 40))
        params = make_params(
            names=(
                "EXPDIPEPS", "EXPDIPFREF", "EXPDIPEP_1",
                "EXPDIPAMP_1", "EXPDIPIDX_1", "EXPDIPTAU_1",
            ),
            values=(eps, 1400.0, 0.0, 1e-6, -1.0, tau),
            units=("day", "MHz", "day", "s", "", "day"),
            epoch_int_values={"EXPDIPEP_1": 55000.0},
        )
        comp = ExponentialDip(
            n_dips=1,
            expdipep_names=("EXPDIPEP_1",),
            expdipamp_names=("EXPDIPAMP_1",),
            expdipidx_names=("EXPDIPIDX_1",),
            expdiptau_names=("EXPDIPTAU_1",),
        )
        return toa_data, params, comp

    def test_forward_finite_and_zero_far_from_dip(self):
        toa_data, params, comp = self._setup()
        delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        assert jnp.all(jnp.isfinite(delay))
        # Pre-dip TOAs (thousands of days before the onset) contribute ~0.
        pre = toa_data.tdb.total < 54990.0
        np.testing.assert_allclose(np.array(delay)[np.array(pre)], 0.0, atol=1e-30)

    def test_grads_finite_for_all_params_far_from_dip(self):
        toa_data, params, comp = self._setup()

        def loss(p):
            return comp(toa_data, p, jnp.zeros(toa_data.n_toas)).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values)), (
            f"non-finite gradients: "
            f"{[n for n, g in zip(params.names, grads.values) if not jnp.isfinite(g)]}"
        )

    def test_norm_finite_at_tau_equals_eps(self):
        """tau == eps: the (tau/(tau-eps))^((tau-eps)/tau) factor divided by
        zero; its limit is exactly 1, which the log-space form now takes."""
        toa_data, params, comp = self._setup(eps=1.0, tau=1.0, span=(54990.0, 55050.0))
        delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        assert jnp.all(jnp.isfinite(delay))

        def loss(p):
            return comp(toa_data, p, jnp.zeros(toa_data.n_toas)).sum()

        amp_idx = params.param_index("EXPDIPAMP_1")
        assert jnp.isfinite(jax.grad(loss)(params).values[amp_idx])

    def test_eps_greater_than_tau_stays_finite(self):
        """eps > tau (outside the model domain) formerly produced a negative
        base to a fractional power -> nan; now clamps to the s = 0 boundary."""
        toa_data, params, comp = self._setup(eps=2.0, tau=1.0, span=(54990.0, 55050.0))
        delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        assert jnp.all(jnp.isfinite(delay))
