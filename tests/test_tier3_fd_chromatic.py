"""Tests for FrequencyDependent (FD), ChromaticCM, and ChromaticCMX against PINT."""

from __future__ import annotations

from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model
from jaxpint.delay.frequency_dependent import FrequencyDependent
from jaxpint.delay.chromatic_cm import ChromaticCM
from jaxpint.delay.chromatic_cmx import ChromaticCMX

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


def _make_multifreq_toas(model, n=20):
    """Create TOAs at two frequencies and merge."""
    import astropy.units as u
    from pint.simulation import make_fake_toas_uniform
    from pint.toa import merge_TOAs

    t1 = make_fake_toas_uniform(54500, 55500, n, model, freq=1400 * u.MHz, add_noise=False)
    t2 = make_fake_toas_uniform(54500, 55500, n, model, freq=2000 * u.MHz, add_noise=False)
    toas = merge_TOAs([t1, t2])
    toas.compute_TDBs()
    toas.compute_posvels()
    return toas


# =========================================================================
# FrequencyDependent (FD)
# =========================================================================

class TestFrequencyDependentvsPINT:

    @pytest.fixture(scope="class")
    def pint_setup(self):
        from pint.models import get_model

        par = _BASE_PAR + """\
FD1           1e-5
FD2           -2e-6
FD3           5e-7
"""
        model = get_model(StringIO(par))
        toas = _make_multifreq_toas(model)

        pint_delay = np.array(
            model.components["FD"].FD_delay(toas).to("s").value,
            dtype=np.float64,
        )

        toa_data = pint_toas_to_jax(toas, model)
        params = pint_model_to_params(model).params

        comp = FrequencyDependent(fd_param_names=("FD1", "FD2", "FD3"))
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

        def loss(p):
            return comp(toa_data, p, jnp.zeros(toa_data.n_toas)).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))

    def test_bridge_builds_fd(self, pint_setup):
        _, _, _, model, _ = pint_setup
        tm, _ = build_timing_model(model)
        assert any(isinstance(c, FrequencyDependent) for c in tm.delay_components)


# =========================================================================
# ChromaticCM
# =========================================================================

class TestChromaticCMvsPINT:

    @pytest.fixture(scope="class")
    def pint_setup(self):
        from pint.models import get_model

        par = _BASE_PAR + """\
CM            0.5
CM1           0.01
CMEPOCH       55000
TNCHROMIDX    4.0
"""
        model = get_model(StringIO(par))
        toas = _make_multifreq_toas(model)

        pint_delay = np.array(
            model.components["ChromaticCM"].chromatic_type_delay(toas).to("s").value,
            dtype=np.float64,
        )

        toa_data = pint_toas_to_jax(toas, model)
        params = pint_model_to_params(model).params

        comp = ChromaticCM(
            cm_param_names=("CM", "CM1"),
            cmepoch_name="CMEPOCH",
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

        def loss(p):
            return comp(toa_data, p, jnp.zeros(toa_data.n_toas)).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))

    def test_bridge_builds_chromatic_cm(self, pint_setup):
        _, _, _, model, _ = pint_setup
        tm, _ = build_timing_model(model)
        assert any(isinstance(c, ChromaticCM) for c in tm.delay_components)


# =========================================================================
# ChromaticCMX
# =========================================================================

class TestChromaticCMXvsPINT:

    @pytest.fixture(scope="class")
    def pint_setup(self):
        from pint.models import get_model

        par = _BASE_PAR + """\
TNCHROMIDX    4.0
CMX_0001      0.3
CMXR1_0001    54500
CMXR2_0001    55000
CMX_0002      -0.1
CMXR1_0002    55000
CMXR2_0002    55500
"""
        model = get_model(StringIO(par))
        toas = _make_multifreq_toas(model)

        pint_delay = np.array(
            model.components["ChromaticCMX"].chromatic_type_delay(toas).to("s").value,
            dtype=np.float64,
        )

        toa_data = pint_toas_to_jax(toas, model)
        params = pint_model_to_params(model).params

        comp = ChromaticCMX(
            n_bins=2,
            cmx_names=("CMX_0001", "CMX_0002"),
            cmxr1_names=("CMXR1_0001", "CMXR1_0002"),
            cmxr2_names=("CMXR2_0001", "CMXR2_0002"),
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

        def loss(p):
            return comp(toa_data, p, jnp.zeros(toa_data.n_toas)).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))

    def test_bridge_builds_chromatic_cmx(self, pint_setup):
        _, _, _, model, _ = pint_setup
        tm, _ = build_timing_model(model)
        assert any(isinstance(c, ChromaticCMX) for c in tm.delay_components)
