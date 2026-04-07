"""Tests for WaveX, DMWaveX, and CMWaveX delay components against PINT."""

from __future__ import annotations

from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model
from jaxpint.delay.wavex import WaveX
from jaxpint.delay.dmwavex import DMWaveX
from jaxpint.delay.cmwavex import CMWaveX

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


# =========================================================================
# WaveX
# =========================================================================

class TestWaveXvsPINT:

    @pytest.fixture(scope="class")
    def pint_setup(self):
        import astropy.units as u
        from pint.models import get_model
        from pint.simulation import make_fake_toas_uniform

        par = _BASE_PAR + """\
WXEPOCH       55000
WXFREQ_0001   0.01
WXSIN_0001    1e-6
WXCOS_0001    -0.5e-6
WXFREQ_0002   0.02
WXSIN_0002    0.3e-6
WXCOS_0002    0.8e-6
"""
        model = get_model(StringIO(par))
        toas = make_fake_toas_uniform(
            54500, 55500, 40, model, freq=1400.0, add_noise=False,
        )
        toas.compute_TDBs()
        toas.compute_posvels()

        pint_delay = np.array(
            model.components["WaveX"].wavex_delay(
                toas, np.zeros(toas.ntoas) * u.s,
            ).to("s").value,
            dtype=np.float64,
        )

        toa_data = pint_toas_to_jax(toas, model)
        params = pint_model_to_params(model).params

        comp = WaveX(
            n_components=2,
            wxfreq_names=("WXFREQ_0001", "WXFREQ_0002"),
            wxsin_names=("WXSIN_0001", "WXSIN_0002"),
            wxcos_names=("WXCOS_0001", "WXCOS_0002"),
            wxepoch_name="WXEPOCH",
        )
        return toa_data, params, pint_delay, model, comp

    def test_delay_matches_pint(self, pint_setup):
        toa_data, params, pint_delay, _, comp = pint_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    def test_jit_compatible(self, pint_setup):
        toa_data, params, _, _, comp = pint_setup
        delay = jnp.zeros(toa_data.n_toas)
        eager = comp(toa_data, params, delay)
        jitted = jax.jit(comp)(toa_data, params, delay)
        np.testing.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-14)

    def test_grad_finite(self, pint_setup):
        toa_data, params, _, _, comp = pint_setup
        delay = jnp.zeros(toa_data.n_toas)

        def loss(p):
            return comp(toa_data, p, delay).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))

    def test_bridge_builds_wavex(self, pint_setup):
        _, _, _, model, _ = pint_setup
        tm, _ = build_timing_model(model)
        assert any(isinstance(c, WaveX) for c in tm.delay_components)


# =========================================================================
# DMWaveX
# =========================================================================

class TestDMWaveXvsPINT:

    @pytest.fixture(scope="class")
    def pint_setup(self):
        from pint.models import get_model
        from pint.simulation import make_fake_toas_uniform

        par = _BASE_PAR + """\
DMWXEPOCH     55000
DMWXFREQ_0001 0.005
DMWXSIN_0001  0.1
DMWXCOS_0001  -0.05
DMWXFREQ_0002 0.01
DMWXSIN_0002  0.03
DMWXCOS_0002  0.07
"""
        model = get_model(StringIO(par))
        toas = make_fake_toas_uniform(
            54500, 55500, 40, model, freq=1400.0, add_noise=False,
        )
        toas.compute_TDBs()
        toas.compute_posvels()

        pint_delay = np.array(
            model.components["DMWaveX"].dmwavex_delay(toas).to("s").value,
            dtype=np.float64,
        )

        toa_data = pint_toas_to_jax(toas, model)
        params = pint_model_to_params(model).params

        comp = DMWaveX(
            n_components=2,
            dmwxfreq_names=("DMWXFREQ_0001", "DMWXFREQ_0002"),
            dmwxsin_names=("DMWXSIN_0001", "DMWXSIN_0002"),
            dmwxcos_names=("DMWXCOS_0001", "DMWXCOS_0002"),
            dmwxepoch_name="DMWXEPOCH",
        )
        return toa_data, params, pint_delay, model, comp

    def test_delay_matches_pint(self, pint_setup):
        toa_data, params, pint_delay, _, comp = pint_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    def test_jit_compatible(self, pint_setup):
        toa_data, params, _, _, comp = pint_setup
        delay = jnp.zeros(toa_data.n_toas)
        eager = comp(toa_data, params, delay)
        jitted = jax.jit(comp)(toa_data, params, delay)
        np.testing.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-14)

    def test_grad_finite(self, pint_setup):
        toa_data, params, _, _, comp = pint_setup
        delay = jnp.zeros(toa_data.n_toas)

        def loss(p):
            return comp(toa_data, p, delay).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))

    def test_bridge_builds_dmwavex(self, pint_setup):
        _, _, _, model, _ = pint_setup
        tm, _ = build_timing_model(model)
        assert any(isinstance(c, DMWaveX) for c in tm.delay_components)


# =========================================================================
# CMWaveX
# =========================================================================

class TestCMWaveXvsPINT:

    @pytest.fixture(scope="class")
    def pint_setup(self):
        from pint.models import get_model
        from pint.simulation import make_fake_toas_uniform

        par = _BASE_PAR + """\
TNCHROMIDX    4.0
CMWXEPOCH     55000
CMWXFREQ_0001 0.005
CMWXSIN_0001  0.01
CMWXCOS_0001  -0.005
CMWXFREQ_0002 0.01
CMWXSIN_0002  0.003
CMWXCOS_0002  0.007
"""
        model = get_model(StringIO(par))
        toas = make_fake_toas_uniform(
            54500, 55500, 40, model, freq=1400.0, add_noise=False,
        )
        toas.compute_TDBs()
        toas.compute_posvels()

        pint_delay = np.array(
            model.components["CMWaveX"].cmwavex_delay(toas).to("s").value,
            dtype=np.float64,
        )

        toa_data = pint_toas_to_jax(toas, model)
        params = pint_model_to_params(model).params

        comp = CMWaveX(
            n_components=2,
            cmwxfreq_names=("CMWXFREQ_0001", "CMWXFREQ_0002"),
            cmwxsin_names=("CMWXSIN_0001", "CMWXSIN_0002"),
            cmwxcos_names=("CMWXCOS_0001", "CMWXCOS_0002"),
            cmwxepoch_name="CMWXEPOCH",
            tnchromidx_name="TNCHROMIDX",
        )
        return toa_data, params, pint_delay, model, comp

    def test_delay_matches_pint(self, pint_setup):
        toa_data, params, pint_delay, _, comp = pint_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    def test_jit_compatible(self, pint_setup):
        toa_data, params, _, _, comp = pint_setup
        delay = jnp.zeros(toa_data.n_toas)
        eager = comp(toa_data, params, delay)
        jitted = jax.jit(comp)(toa_data, params, delay)
        np.testing.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-14)

    def test_grad_finite(self, pint_setup):
        toa_data, params, _, _, comp = pint_setup
        delay = jnp.zeros(toa_data.n_toas)

        def loss(p):
            return comp(toa_data, p, delay).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))

    def test_bridge_builds_cmwavex(self, pint_setup):
        _, _, _, model, _ = pint_setup
        tm, _ = build_timing_model(model)
        assert any(isinstance(c, CMWaveX) for c in tm.delay_components)
