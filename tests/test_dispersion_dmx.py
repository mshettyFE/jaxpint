"""Tests for jaxpint.dispersion_dmx: DispersionDMX delay component."""

import jax
import jax.numpy as jnp
import pytest


from jaxpint.constants import DMCONST
from jaxpint.delay.dispersion_dmx import DispersionDMX
from tests.helpers import make_toa_data as _make_toa_data_base, make_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_toa_data(n_toas=5, t_mjd=None, tdb_int=59000.0, tdb_frac=None,
                   freq=1400.0):
    return _make_toa_data_base(n_toas, t_mjd=t_mjd, tdb_int=tdb_int,
                               tdb_frac=tdb_frac, freq=freq,
                               obs_names=("GBT",), planet_positions=None)


def _make_dmx_params(dmx_values, dmxr1_mjds, dmxr2_mjds, frozen_dmx=None):
    """Build a ParameterVector with DMX bins.

    Parameters
    ----------
    dmx_values : list of float
        DM values for each bin.
    dmxr1_mjds, dmxr2_mjds : list of float
        Bin boundary MJDs.
    frozen_dmx : list of bool, optional
        Whether each DMX param is frozen (default: all False).
    """
    n = len(dmx_values)
    names = []
    values = []
    units = []
    components = []
    epoch_int_values = {}
    frozen_mask = []

    for i in range(n):
        idx = f"{i + 1:04d}"
        # DMX value
        names.append(f"DMX_{idx}")
        values.append(dmx_values[i])
        units.append("pc cm^-3")
        components.append("DispersionDMX")
        frozen_mask.append(False if frozen_dmx is None else frozen_dmx[i])

        # DMXR1 (bin start, epoch param)
        names.append(f"DMXR1_{idx}")
        r1_int = float(int(dmxr1_mjds[i]))
        r1_frac = dmxr1_mjds[i] - r1_int
        values.append(r1_frac)
        units.append("day")
        components.append("DispersionDMX")
        epoch_int_values[f"DMXR1_{idx}"] = r1_int
        frozen_mask.append(True)

        # DMXR2 (bin end, epoch param)
        names.append(f"DMXR2_{idx}")
        r2_int = float(int(dmxr2_mjds[i]))
        r2_frac = dmxr2_mjds[i] - r2_int
        values.append(r2_frac)
        units.append("day")
        components.append("DispersionDMX")
        epoch_int_values[f"DMXR2_{idx}"] = r2_int
        frozen_mask.append(True)

    return make_params(
        names, values,
        units=tuple(units),
        components=tuple(components),
        epoch_int_values=epoch_int_values,
        frozen_mask=tuple(frozen_mask),
    )


def _single_bin_component():
    return DispersionDMX(
        n_bins=1,
        dmx_names=("DMX_0001",),
        dmxr1_names=("DMXR1_0001",),
        dmxr2_names=("DMXR2_0001",),
    )


def _two_bin_component():
    return DispersionDMX(
        n_bins=2,
        dmx_names=("DMX_0001", "DMX_0002"),
        dmxr1_names=("DMXR1_0001", "DMXR1_0002"),
        dmxr2_names=("DMXR2_0001", "DMXR2_0002"),
    )


# ===========================================================================
# Construction tests
# ===========================================================================

class TestConstruction:
    def test_single_bin(self):
        c = _single_bin_component()
        assert c.n_bins == 1
        assert c.dmx_names == ("DMX_0001",)

    def test_multiple_bins(self):
        c = DispersionDMX(
            n_bins=3,
            dmx_names=("DMX_0001", "DMX_0002", "DMX_0003"),
            dmxr1_names=("DMXR1_0001", "DMXR1_0002", "DMXR1_0003"),
            dmxr2_names=("DMXR2_0001", "DMXR2_0002", "DMXR2_0003"),
        )
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


# ===========================================================================
# Pytree tests
# ===========================================================================

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


# ===========================================================================
# Delay computation tests
# ===========================================================================

class TestDelay:
    def test_single_bin_toa_inside(self):
        """TOA inside the bin gets the DMX delay."""
        comp = _single_bin_component()
        dmx_val = 0.5
        freq = 1400.0
        params = _make_dmx_params([dmx_val], [58900.0], [59100.0])
        toa_data = _make_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.0,
                                  freq=freq)
        result = comp(toa_data, params, jnp.zeros(1))
        expected = dmx_val * DMCONST / freq**2
        assert jnp.isclose(result[0], expected, rtol=1e-12)

    def test_single_bin_toa_outside(self):
        """TOA outside the bin gets zero delay."""
        comp = _single_bin_component()
        params = _make_dmx_params([0.5], [58900.0], [58950.0])
        toa_data = _make_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.0,
                                  freq=1400.0)
        result = comp(toa_data, params, jnp.zeros(1))
        assert jnp.isclose(result[0], 0.0, atol=1e-30)

    def test_boundary_inclusion(self):
        """TOAs at exact bin boundaries are included."""
        comp = _single_bin_component()
        dmx_val = 0.3
        freq = 1400.0
        r1, r2 = 59000.0, 59100.0
        params = _make_dmx_params([dmx_val], [r1], [r2])
        expected = dmx_val * DMCONST / freq**2

        # TOA at start boundary
        toa_start = _make_toa_data(n_toas=1, tdb_int=r1, tdb_frac=0.0,
                                   freq=freq)
        assert jnp.isclose(comp(toa_start, params, jnp.zeros(1))[0],
                           expected, rtol=1e-12)

        # TOA at end boundary
        toa_end = _make_toa_data(n_toas=1, tdb_int=r2, tdb_frac=0.0,
                                 freq=freq)
        assert jnp.isclose(comp(toa_end, params, jnp.zeros(1))[0],
                           expected, rtol=1e-12)

    def test_two_bins_nonoverlapping(self):
        """Two non-overlapping bins assign correct DMX to each TOA."""
        comp = _two_bin_component()
        dmx1, dmx2 = 0.5, -0.3
        freq = 1400.0
        params = _make_dmx_params(
            [dmx1, dmx2],
            [58900.0, 59100.0],
            [59000.0, 59200.0],
        )
        # TOA in bin 1
        toa1 = _make_toa_data(n_toas=1, tdb_int=58950.0, tdb_frac=0.0,
                              freq=freq)
        r1 = comp(toa1, params, jnp.zeros(1))
        assert jnp.isclose(r1[0], dmx1 * DMCONST / freq**2, rtol=1e-12)

        # TOA in bin 2
        toa2 = _make_toa_data(n_toas=1, tdb_int=59150.0, tdb_frac=0.0,
                              freq=freq)
        r2 = comp(toa2, params, jnp.zeros(1))
        assert jnp.isclose(r2[0], dmx2 * DMCONST / freq**2, rtol=1e-12)

    def test_frequency_dependence(self):
        """Delay scales as 1/freq^2."""
        comp = _single_bin_component()
        dmx_val = 0.5
        params = _make_dmx_params([dmx_val], [58900.0], [59100.0])

        freq_lo, freq_hi = 800.0, 1400.0
        toa_lo = _make_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.0,
                                freq=freq_lo)
        toa_hi = _make_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.0,
                                freq=freq_hi)
        d_lo = comp(toa_lo, params, jnp.zeros(1))
        d_hi = comp(toa_hi, params, jnp.zeros(1))

        ratio = d_lo[0] / d_hi[0]
        expected_ratio = (freq_hi / freq_lo) ** 2
        assert jnp.isclose(ratio, expected_ratio, rtol=1e-12)

    def test_multiple_toas_vectorized(self):
        """Correct bin assignment across array of TOAs."""
        comp = _two_bin_component()
        dmx1, dmx2 = 0.5, -0.3
        freq = 1400.0
        params = _make_dmx_params(
            [dmx1, dmx2],
            [58900.0, 59100.0],
            [59000.0, 59200.0],
        )
        # 4 TOAs: bin1, gap, bin2, bin2
        toa_data = _make_toa_data(
            t_mjd=[58950.0, 59050.0, 59150.0, 59180.0],
            freq=freq,
        )
        result = comp(toa_data, params, jnp.zeros(4))

        assert jnp.isclose(result[0], dmx1 * DMCONST / freq**2, rtol=1e-12)
        assert jnp.isclose(result[1], 0.0, atol=1e-30)  # in gap
        assert jnp.isclose(result[2], dmx2 * DMCONST / freq**2, rtol=1e-12)
        assert jnp.isclose(result[3], dmx2 * DMCONST / freq**2, rtol=1e-12)

    def test_toa_in_no_bin(self):
        """TOA outside all bins gets zero delay."""
        comp = _single_bin_component()
        params = _make_dmx_params([1.0], [59000.0], [59100.0])
        toa_data = _make_toa_data(n_toas=1, tdb_int=58000.0, tdb_frac=0.0,
                                  freq=1400.0)
        result = comp(toa_data, params, jnp.zeros(1))
        assert jnp.isclose(result[0], 0.0, atol=1e-30)

    def test_acc_delay_ignored(self):
        """Accumulated delay does not affect DMX dispersion."""
        comp = _single_bin_component()
        params = _make_dmx_params([0.5], [58900.0], [59100.0])
        toa_data = _make_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.0,
                                  freq=1400.0)

        r_no_delay = comp(toa_data, params, jnp.zeros(1))
        r_with_delay = comp(toa_data, params, jnp.array([0.5]))
        assert jnp.isclose(r_no_delay[0], r_with_delay[0])


# ===========================================================================
# JIT tests
# ===========================================================================

class TestJIT:
    def test_jit_call(self):
        comp = _single_bin_component()
        params = _make_dmx_params([0.5], [58900.0], [59100.0])
        toa_data = _make_toa_data(n_toas=3, tdb_int=59000.0, tdb_frac=0.0,
                                  freq=1400.0)

        jitted = jax.jit(comp)
        result = jitted(toa_data, params, jnp.zeros(3))
        assert result.shape == (3,)

    def test_jit_same_trace_different_params(self):
        comp = _single_bin_component()
        params1 = _make_dmx_params([0.5], [58900.0], [59100.0])
        params2 = _make_dmx_params([1.0], [58900.0], [59100.0])
        toa_data = _make_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.0,
                                  freq=1400.0)

        jitted = jax.jit(comp)
        r1 = jitted(toa_data, params1, jnp.zeros(1))
        r2 = jitted(toa_data, params2, jnp.zeros(1))
        assert not jnp.array_equal(r1, r2)


# ===========================================================================
# Gradient tests
# ===========================================================================

class TestGrad:
    def test_grad_wrt_dmx_in_bin(self):
        """d(delay)/d(DMX_i) = DMCONST/freq^2 for TOA in bin i."""
        comp = _single_bin_component()
        freq = 1400.0
        params = _make_dmx_params([0.5], [58900.0], [59100.0])
        toa_data = _make_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.0,
                                  freq=freq)

        def loss(p):
            return comp(toa_data, p, jnp.zeros(1)).sum()

        grads = jax.grad(loss)(params)
        dmx_idx = params.param_index("DMX_0001")
        expected = DMCONST / freq**2
        assert jnp.isclose(grads.values[dmx_idx], expected, rtol=1e-10)

    def test_grad_wrt_dmx_out_of_bin(self):
        """d(delay)/d(DMX_i) = 0 for TOA outside bin i."""
        comp = _single_bin_component()
        params = _make_dmx_params([0.5], [59000.0], [59100.0])
        toa_data = _make_toa_data(n_toas=1, tdb_int=58000.0, tdb_frac=0.0,
                                  freq=1400.0)

        def loss(p):
            return comp(toa_data, p, jnp.zeros(1)).sum()

        grads = jax.grad(loss)(params)
        dmx_idx = params.param_index("DMX_0001")
        assert jnp.isclose(grads.values[dmx_idx], 0.0, atol=1e-30)

    def test_grad_two_bins(self):
        """Gradients are independent for non-overlapping bins."""
        comp = _two_bin_component()
        freq = 1400.0
        params = _make_dmx_params(
            [0.5, -0.3],
            [58900.0, 59100.0],
            [59000.0, 59200.0],
        )
        # TOA only in bin 1
        toa_data = _make_toa_data(n_toas=1, tdb_int=58950.0, tdb_frac=0.0,
                                  freq=freq)

        def loss(p):
            return comp(toa_data, p, jnp.zeros(1)).sum()

        grads = jax.grad(loss)(params)
        dmx1_idx = params.param_index("DMX_0001")
        dmx2_idx = params.param_index("DMX_0002")
        assert jnp.isclose(grads.values[dmx1_idx], DMCONST / freq**2, rtol=1e-10)
        assert jnp.isclose(grads.values[dmx2_idx], 0.0, atol=1e-30)

    def test_grad_finite(self):
        comp = _two_bin_component()
        params = _make_dmx_params([0.5, -0.3], [58900.0, 59100.0],
                                  [59000.0, 59200.0])
        toa_data = _make_toa_data(n_toas=3, tdb_int=58950.0, tdb_frac=0.0,
                                  freq=1400.0)

        def loss(p):
            return comp(toa_data, p, jnp.zeros(3)).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))


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
        params = pint_model_to_params(model)

        return toa_data, params, pint_delay, model

    def test_matches_pint(self, pint_setup):
        """JaxPINT DMX delay matches PINT within float64 tolerance."""
        toa_data, params, pint_delay, model = pint_setup

        comp = DispersionDMX(
            n_bins=4,
            dmx_names=("DMX_0001", "DMX_0002", "DMX_0003", "DMX_0004"),
            dmxr1_names=("DMXR1_0001", "DMXR1_0002", "DMXR1_0003", "DMXR1_0004"),
            dmxr2_names=("DMXR2_0001", "DMXR2_0002", "DMXR2_0003", "DMXR2_0004"),
        )
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        assert jnp.allclose(
            jax_delay, jnp.asarray(pint_delay), rtol=1e-10, atol=1e-15,
        )
