"""Tests for jaxpint.dispersion_dmx: DispersionDMX delay component."""

import jax
import jax.numpy as jnp
import pytest


from jaxpint.constants import DMCONST
from jaxpint.delay.dispersion_dmx import DispersionDMX
from tests.helpers import make_gbt_toa_data, make_dmx_params


def _make_dmx_component(n_bins):
    """Build a DispersionDMX with auto-named bins."""
    idx = tuple(f"{i + 1:04d}" for i in range(n_bins))
    return DispersionDMX(
        n_bins=n_bins,
        dmx_names=tuple(f"DMX_{i}" for i in idx),
        dmxr1_names=tuple(f"DMXR1_{i}" for i in idx),
        dmxr2_names=tuple(f"DMXR2_{i}" for i in idx),
    )


def _single_bin_component():
    return _make_dmx_component(1)


def _two_bin_component():
    return _make_dmx_component(2)


class TestConstruction:
    def test_single_bin(self):
        c = _single_bin_component()
        assert c.n_bins == 1
        assert c.dmx_names == ("DMX_0001",)

    def test_multiple_bins(self):
        c = _make_dmx_component(3)
        assert c.n_bins == 3

    def test_zero_bins_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            DispersionDMX(
                n_bins=0, dmx_names=(), dmxr1_names=(), dmxr2_names=()
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="does not match n_bins"):
            DispersionDMX(
                n_bins=2,
                dmx_names=("DMX_0001",),  # only 1
                dmxr1_names=("DMXR1_0001", "DMXR1_0002"),
                dmxr2_names=("DMXR2_0001", "DMXR2_0002"),
            )


class TestPytree:
    def test_zero_dynamic_leaves(self):
        c = _single_bin_component()
        leaves, _ = jax.tree.flatten(c)
        assert len(leaves) == 0

    def test_pytree_roundtrip(self):
        c = _two_bin_component()
        leaves, treedef = jax.tree.flatten(c)
        c2 = jax.tree.unflatten(treedef, leaves)
        assert c2.n_bins == c.n_bins
        assert c2.dmx_names == c.dmx_names


class TestDelay:
    @pytest.mark.parametrize("toa_mjd, bins, dmx_values, expected_idx", [
        pytest.param(
            59000.0, [(58900.0, 59100.0)], [0.5], 0,
            id="single_bin_inside",
        ),
        pytest.param(
            59000.0, [(58900.0, 58950.0)], [0.5], None,
            id="single_bin_toa_after_bin",
        ),
        pytest.param(
            58000.0, [(59000.0, 59100.0)], [1.0], None,
            id="single_bin_toa_before_bin",
        ),
        pytest.param(
            59000.0, [(59000.0, 59100.0)], [0.3], 0,
            id="boundary_start_inclusive",
        ),
        pytest.param(
            59100.0, [(59000.0, 59100.0)], [0.3], 0,
            id="boundary_end_inclusive",
        ),
    ])
    def test_dmx_bin_membership(self, toa_mjd, bins, dmx_values, expected_idx):
        """A TOA inside bin i picks up DMX_i * DMCONST / freq^2; outside all bins gives 0.
        """
        comp = _make_dmx_component(len(bins))
        freq = 1400.0
        params = make_dmx_params(
            dmx_values, [b[0] for b in bins], [b[1] for b in bins],
        )
        toa_data = make_gbt_toa_data(
            n_toas=1, tdb_int=toa_mjd, tdb_frac=0.0, freq=freq,
        )
        result = comp(toa_data, params, jnp.zeros(1))
        if expected_idx is None:
            assert jnp.isclose(result[0], 0.0, atol=1e-30)
        else:
            expected = dmx_values[expected_idx] * DMCONST / freq ** 2
            assert jnp.isclose(result[0], expected, rtol=1e-12)

    def test_frequency_dependence(self, freq_lo=800.0, freq_hi=1400.0):
        """Delay scales as 1/freq^2.
        """
        comp = _single_bin_component()
        dmx_val = 0.5
        params = make_dmx_params([dmx_val], [58900.0], [59100.0])

        toa_lo = make_gbt_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.0,
                                   freq=freq_lo)
        toa_hi = make_gbt_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.0,
                                   freq=freq_hi)
        d_lo = comp(toa_lo, params, jnp.zeros(1))
        d_hi = comp(toa_hi, params, jnp.zeros(1))

        ratio = d_lo[0] / d_hi[0]
        expected_ratio = (freq_hi / freq_lo) ** 2
        assert jnp.isclose(ratio, expected_ratio, rtol=1e-12)

    def test_multiple_toas_vectorized(self):
        """Correct bin assignment across an array of TOAs."""
        comp = _two_bin_component()
        dmx1, dmx2 = 0.5, -0.3
        freq = 1400.0
        params = make_dmx_params(
            [dmx1, dmx2],
            [58900.0, 59100.0],
            [59000.0, 59200.0],
        )
        # 4 TOAs: bin1, gap, bin2, bin2
        toa_data = make_gbt_toa_data(
            t_mjd=[58950.0, 59050.0, 59150.0, 59180.0],
            freq=freq,
        )
        result = comp(toa_data, params, jnp.zeros(4))

        assert jnp.isclose(result[0], dmx1 * DMCONST / freq**2, rtol=1e-12)
        assert jnp.isclose(result[1], 0.0, atol=1e-30)  # in gap
        assert jnp.isclose(result[2], dmx2 * DMCONST / freq**2, rtol=1e-12)
        assert jnp.isclose(result[3], dmx2 * DMCONST / freq**2, rtol=1e-12)

    def test_acc_delay_ignored(self):
        """Accumulated delay does not affect DMX dispersion."""
        comp = _single_bin_component()
        params = make_dmx_params([0.5], [58900.0], [59100.0])
        toa_data = make_gbt_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.0,
                                  freq=1400.0)

        r_no_delay = comp(toa_data, params, jnp.zeros(1))
        r_with_delay = comp(toa_data, params, jnp.array([0.5]))
        assert jnp.isclose(r_no_delay[0], r_with_delay[0])


class TestJIT:
    def test_jit_call(self):
        comp = _single_bin_component()
        params = make_dmx_params([0.5], [58900.0], [59100.0])
        toa_data = make_gbt_toa_data(n_toas=3, tdb_int=59000.0, tdb_frac=0.0,
                                  freq=1400.0)

        jitted = jax.jit(comp)
        result = jitted(toa_data, params, jnp.zeros(3))
        assert result.shape == (3,)

    def test_jit_same_trace_different_params(self):
        comp = _single_bin_component()
        params = make_dmx_params([0.5], [58900.0], [59100.0])
        toa_data = make_gbt_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.0,
                                  freq=1400.0)

        jitted = jax.jit(comp)
        r1 = jitted(toa_data, params, jnp.zeros(1))  # first call compiles one variant
        # Same structure, different value -> cache hit, must not recompile.
        r2 = jitted(toa_data, params.with_value("DMX_0001", 1.0), jnp.zeros(1))

        assert jitted._cache_size() == 1, (
            f"recompiled on same-structure inputs: {jitted._cache_size()} variants"
        )
        assert not jnp.array_equal(r1, r2)


class TestGrad:
    """Gradient correctness for DMX bin membership.
    """

    def test_grad_two_bins(self):
        """Gradients are exact and independent for non-overlapping bins."""
        comp = _two_bin_component()
        freq = 1400.0
        params = make_dmx_params(
            [0.5, -0.3],
            [58900.0, 59100.0],
            [59000.0, 59200.0],
        )
        # TOA only in bin 1
        toa_data = make_gbt_toa_data(n_toas=1, tdb_int=58950.0, tdb_frac=0.0,
                                  freq=freq)

        def loss(p):
            return comp(toa_data, p, jnp.zeros(1)).sum()

        grads = jax.grad(loss)(params)
        dmx1_idx = params.param_index("DMX_0001")
        dmx2_idx = params.param_index("DMX_0002")
        assert jnp.isclose(grads.values[dmx1_idx], DMCONST / freq**2, rtol=1e-10)
        assert jnp.isclose(grads.values[dmx2_idx], 0.0, atol=1e-30)


# ===========================================================================
# Cross-validation against PINT (oracle tests)
# ===========================================================================

class TestPINTOracle:
    """Compare JaxPINT DispersionDMX delay against PINT's implementation."""

    @pytest.fixture
    def pint_setup(self):
        from io import StringIO
        import numpy as np
        from pint.models import get_model
        from pint.simulation import make_fake_toas_uniform
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params

        par = """
PSR           J1234+5678
RAJ           12:34:56.789
DECJ          +56:07:08.12
F0            100.0
F1            -1e-15
PEPOCH        55000
DM            15.0
DMEPOCH       55000
DMX_0001      0.5
DMXR1_0001    54500
DMXR2_0001    54750
DMX_0002      -0.3
DMXR1_0002    54750
DMXR2_0002    55000
DMX_0003      0.1
DMXR1_0003    55000
DMXR2_0003    55250
DMX_0004      -0.2
DMXR1_0004    55250
DMXR2_0004    55500
EPHEM         DE421
CLK           UTC(NIST)
UNITS         TDB
CORRECT_TROPOSPHERE  N
PLANET_SHAPIRO       N
"""
        model = get_model(StringIO(par))
        toas = make_fake_toas_uniform(
            startMJD=54500, endMJD=55500,
            ntoas=40, model=model, freq=1400.0,
            add_noise=False,
        )
        toas.compute_TDBs()
        toas.compute_posvels()

        # Compute PINT's DMX dispersion delay
        dmx_comp = model.components["DispersionDMX"]
        pint_delay = np.array(
            dmx_comp.dispersion_type_delay(toas).to("s").value,
            dtype=np.float64,
        )

        toa_data = pint_toas_to_jax(toas, model)
        params = pint_model_to_params(model).params

        return toa_data, params, pint_delay, model

    @pytest.mark.slow
    def test_matches_pint(self, pint_setup):
        """JaxPINT DMX delay matches PINT within float64 tolerance."""
        toa_data, params, pint_delay, model = pint_setup

        comp = _make_dmx_component(4)
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        assert jnp.allclose(
            jax_delay, jnp.asarray(pint_delay), rtol=1e-10, atol=1e-15,
        )
