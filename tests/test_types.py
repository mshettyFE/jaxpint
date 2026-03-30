"""Tests for jaxpint.types: PhaseResult, TOAData, ParameterVector."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jaxpint.types import PhaseResult, TOAData, ParameterVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_toa_data(n_toas=10, with_planets=False, with_dm=False):
    """Create a minimal TOAData for testing."""
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 10)

    planet_positions = None
    if with_planets:
        planet_positions = {
            "jupiter": jax.random.normal(keys[7], (n_toas, 3)),
            "saturn": jax.random.normal(keys[8], (n_toas, 3)),
        }

    dm_values = jax.random.normal(keys[9], (n_toas,)) if with_dm else None
    dm_errors = jnp.abs(jax.random.normal(keys[9], (n_toas,))) if with_dm else None

    return TOAData(
        mjd_int=jnp.full(n_toas, 59000.0),
        mjd_frac=jax.random.uniform(keys[0], (n_toas,)),
        tdb_int=jnp.full(n_toas, 59000.0),
        tdb_frac=jax.random.uniform(keys[1], (n_toas,)),
        error=jnp.abs(jax.random.normal(keys[2], (n_toas,))) * 1e-6,
        freq=jnp.full(n_toas, 1400.0),
        delta_pulse_number=jnp.zeros(n_toas),
        ssb_obs_pos=jax.random.normal(keys[3], (n_toas, 3)),
        ssb_obs_vel=jax.random.normal(keys[4], (n_toas, 3)),
        obs_sun_pos=jax.random.normal(keys[5], (n_toas, 3)),
        obs_indices=jnp.zeros(n_toas, dtype=jnp.int32),
        flag_masks={
            "EFAC1": jnp.ones(n_toas, dtype=jnp.bool_),
            "JUMP1": jax.random.bernoulli(keys[6], shape=(n_toas,)),
        },
        planet_positions=planet_positions,
        dm_values=dm_values,
        dm_errors=dm_errors,
        n_toas=n_toas,
        obs_names=("GBT",),
    )


def _make_param_vector():
    """Create a minimal ParameterVector for testing."""
    names = ("F0", "F1", "DM", "PEPOCH", "RAJ", "DECJ")
    values = jnp.array([200.0, -1e-15, 15.0, 0.5, 1.5, 0.8])
    return ParameterVector(
        values=values,
        frozen_mask=(False, False, False, True, False, False),
        names=names,
        units=("Hz", "Hz/s", "pc cm^-3", "day", "rad", "rad"),
        components=("Spindown", "Spindown", "Dispersion", "Spindown", "Astrometry", "Astrometry"),
        _name_to_index={n: i for i, n in enumerate(names)},
        bounds=((None, None),) * 6,
        epoch_int_values={"PEPOCH": 59000.0},
    )


# ===========================================================================
# PhaseResult tests
# ===========================================================================

class TestPhaseResult:
    def test_normalize_invariant(self):
        """Frac must always be in [-0.5, 0.5) after create()."""
        p = PhaseResult.create(jnp.array([0.0, 5.0, -3.0]), jnp.array([0.7, -0.8, 1.3]))
        assert jnp.all(p.frac >= -0.5)
        assert jnp.all(p.frac < 0.5)

    def test_normalize_preserves_total(self):
        """int + frac should equal the original total phase."""
        int_in = jnp.array([3.0, -2.0, 0.0])
        frac_in = jnp.array([0.7, -1.3, 0.49999])
        p = PhaseResult.create(int_in, frac_in)
        expected = int_in + frac_in
        assert jnp.allclose(p.quantity, expected, atol=1e-12)

    def test_add_sub_roundtrip(self):
        """(a + b) - b should approximately equal a."""
        a = PhaseResult.create(jnp.array([10.0]), jnp.array([0.3]))
        b = PhaseResult.create(jnp.array([5.0]), jnp.array([-0.2]))
        result = (a + b) - b
        assert jnp.allclose(result.quantity, a.quantity, atol=1e-12)

    def test_mul_scalar(self):
        p = PhaseResult.create(jnp.array([1.0]), jnp.array([0.25]))
        doubled = p * 2.0
        assert jnp.allclose(doubled.quantity, jnp.array([2.5]), atol=1e-12)

    def test_rmul(self):
        p = PhaseResult.create(jnp.array([1.0]), jnp.array([0.25]))
        doubled = 2.0 * p
        assert jnp.allclose(doubled.quantity, jnp.array([2.5]), atol=1e-12)

    def test_negative_double(self):
        """Double negation should be identity."""
        p = PhaseResult.create(jnp.array([3.0]), jnp.array([0.4]))
        pp = -(-p)
        assert jnp.allclose(pp.int, p.int)
        assert jnp.allclose(pp.frac, p.frac)

    def test_quantity(self):
        p = PhaseResult.create(jnp.array([5.0]), jnp.array([0.3]))
        assert jnp.allclose(p.quantity, jnp.array([5.3]), atol=1e-12)

    def test_jit_compatible(self):
        a = PhaseResult.create(jnp.array([1.0]), jnp.array([0.2]))
        b = PhaseResult.create(jnp.array([2.0]), jnp.array([0.3]))

        @jax.jit
        def add_phases(x, y):
            return x + y

        result = add_phases(a, b)
        assert jnp.allclose(result.quantity, jnp.array([3.5]), atol=1e-12)

    def test_grad_through_phase(self):
        """Gradient should flow through PhaseResult arithmetic."""

        @jax.grad
        def loss(x):
            p = PhaseResult.create(jnp.array([0.0]), x)
            return jnp.sum(p.quantity ** 2)

        g = loss(jnp.array([0.3]))
        assert jnp.allclose(g, 2.0 * 0.3, atol=1e-12)


# ===========================================================================
# TOAData tests
# ===========================================================================

class TestTOAData:
    def test_pytree_roundtrip(self):
        td = _make_toa_data()
        leaves, treedef = jax.tree.flatten(td)
        td2 = jax.tree.unflatten(treedef, leaves)
        assert jnp.array_equal(td.mjd_int, td2.mjd_int)
        assert jnp.array_equal(td.freq, td2.freq)

    def test_jit_passthrough(self):
        td = _make_toa_data()

        @jax.jit
        def get_freqs(toa_data):
            return toa_data.freq

        result = get_freqs(td)
        assert jnp.array_equal(result, td.freq)

    def test_leaf_count(self):
        td = _make_toa_data()
        leaves = jax.tree.leaves(td)
        # 11 core arrays + 2 flag mask arrays + 0 planets + 0 dm = 13
        assert len(leaves) == 13

    def test_optional_planets_none(self):
        td = _make_toa_data(with_planets=False)
        assert td.planet_positions is None
        # Should survive JIT
        @jax.jit
        def identity(x):
            return x.freq
        assert jnp.array_equal(identity(td), td.freq)

    def test_optional_planets_present(self):
        td = _make_toa_data(with_planets=True)
        assert "jupiter" in td.planet_positions
        assert td.planet_positions["jupiter"].shape == (10, 3)
        leaves = jax.tree.leaves(td)
        # 11 core + 2 flags + 2 planets + 0 dm = 15
        assert len(leaves) == 15

    def test_wideband(self):
        td = _make_toa_data(with_dm=True)
        assert td.dm_values is not None
        assert td.dm_errors is not None
        assert td.dm_values.shape == (10,)

    def test_flag_masks_shape(self):
        td = _make_toa_data()
        for name, mask in td.flag_masks.items():
            assert mask.shape == (td.n_toas,)
            assert mask.dtype == jnp.bool_

    def test_static_fields(self):
        td = _make_toa_data()
        assert td.n_toas == 10
        assert td.obs_names == ("GBT",)


# ===========================================================================
# ParameterVector tests
# ===========================================================================

class TestParameterVector:
    def test_pytree_roundtrip(self):
        pv = _make_param_vector()
        leaves, treedef = jax.tree.flatten(pv)
        pv2 = jax.tree.unflatten(treedef, leaves)
        assert jnp.array_equal(pv.values, pv2.values)
        assert pv.names == pv2.names

    def test_param_index(self):
        pv = _make_param_vector()
        assert pv.param_index("F0") == 0
        assert pv.param_index("DM") == 2
        assert pv.param_index("PEPOCH") == 3

    def test_param_value(self):
        pv = _make_param_vector()
        assert jnp.isclose(pv.param_value("F0"), 200.0)
        assert jnp.isclose(pv.param_value("DM"), 15.0)

    def test_epoch_value(self):
        pv = _make_param_vector()
        int_day, frac_day = pv.epoch_value("PEPOCH")
        assert int_day == 59000.0
        assert jnp.isclose(frac_day, 0.5)

    def test_free_values_extraction(self):
        pv = _make_param_vector()
        # PEPOCH (index 3) is frozen, rest are free
        free = pv.free_values()
        assert free.shape == (5,)
        assert jnp.isclose(free[0], 200.0)  # F0
        assert jnp.isclose(free[1], -1e-15)  # F1
        assert jnp.isclose(free[2], 15.0)  # DM

    def test_free_names(self):
        pv = _make_param_vector()
        assert pv.free_names() == ("F0", "F1", "DM", "RAJ", "DECJ")

    def test_with_free_values_roundtrip(self):
        pv = _make_param_vector()
        free = pv.free_values()
        pv2 = pv.with_free_values(free)
        assert jnp.allclose(pv.values, pv2.values)

    def test_with_free_values_update(self):
        pv = _make_param_vector()
        new_free = jnp.array([201.0, -2e-15, 16.0, 1.6, 0.9])
        pv2 = pv.with_free_values(new_free)
        # Free params updated
        assert jnp.isclose(pv2.param_value("F0"), 201.0)
        assert jnp.isclose(pv2.param_value("DM"), 16.0)
        # Frozen param unchanged
        assert jnp.isclose(pv2.param_value("PEPOCH"), 0.5)

    def test_with_value_immutability(self):
        pv = _make_param_vector()
        pv2 = pv.with_value("F0", 999.0)
        # Original unchanged
        assert jnp.isclose(pv.param_value("F0"), 200.0)
        # New has update
        assert jnp.isclose(pv2.param_value("F0"), 999.0)

    def test_component_mask(self):
        pv = _make_param_vector()
        spin_mask = pv.component_mask("Spindown")
        expected = jnp.array([True, True, False, True, False, False])
        assert jnp.array_equal(spin_mask, expected)

    def test_n_params_and_n_free(self):
        pv = _make_param_vector()
        assert pv.n_params == 6
        assert pv.n_free == 5

    def test_grad_through_param_value(self):
        """jax.grad should flow through param_value()."""
        pv = _make_param_vector()

        @jax.grad
        def loss(params):
            f0 = params.param_value("F0")
            return f0 ** 2

        grad_pv = loss(pv)
        # Gradient w.r.t. F0 should be 2 * F0 = 400.0
        assert jnp.isclose(grad_pv.values[0], 400.0)
        # Other gradients should be zero
        assert jnp.allclose(grad_pv.values[1:], 0.0)

    def test_grad_through_loss(self):
        """Full quadratic loss with multiple params."""
        pv = _make_param_vector()

        @jax.grad
        def loss(params):
            f0 = params.param_value("F0")
            dm = params.param_value("DM")
            return f0 ** 2 + 3.0 * dm ** 2

        grad_pv = loss(pv)
        assert jnp.isclose(grad_pv.values[0], 2.0 * 200.0)  # dL/dF0
        assert jnp.isclose(grad_pv.values[2], 6.0 * 15.0)  # dL/dDM

    def test_epoch_grad_flows_through_frac(self):
        """Gradient of dt = toa_frac - epoch_frac should be -1 w.r.t. epoch_frac."""
        pv = _make_param_vector()
        toa_frac = jnp.array(0.7)

        @jax.grad
        def dt_loss(params):
            _, epoch_frac = params.epoch_value("PEPOCH")
            dt = toa_frac - epoch_frac
            return dt

        grad_pv = dt_loss(pv)
        pepoch_idx = pv.param_index("PEPOCH")
        assert jnp.isclose(grad_pv.values[pepoch_idx], -1.0)

    def test_jit_with_param_vector(self):
        pv = _make_param_vector()

        @jax.jit
        def get_f0(params):
            return params.param_value("F0")

        assert jnp.isclose(get_f0(pv), 200.0)


# ===========================================================================
# ParameterVector validation tests
# ===========================================================================

class TestParameterVectorValidation:
    """Tests for __check_init__ length validation."""

    def _defaults(self):
        """Return valid kwargs for ParameterVector construction."""
        names = ("F0", "F1", "DM", "PEPOCH", "RAJ", "DECJ")
        return dict(
            values=jnp.array([200.0, -1e-15, 15.0, 0.5, 1.5, 0.8]),
            frozen_mask=(False, False, False, True, False, False),
            names=names,
            units=("Hz", "Hz/s", "pc cm^-3", "day", "rad", "rad"),
            components=("Spindown", "Spindown", "Dispersion", "Spindown", "Astrometry", "Astrometry"),
            _name_to_index={n: i for i, n in enumerate(names)},
            bounds=((None, None),) * 6,
            epoch_int_values={"PEPOCH": 59000.0},
        )

    def test_valid_construction(self):
        ParameterVector(**self._defaults())

    def test_frozen_mask_length_mismatch(self):
        kw = self._defaults()
        kw["frozen_mask"] = (False, False, False)
        with pytest.raises(ValueError, match="len\\(frozen_mask\\)"):
            ParameterVector(**kw)

    def test_units_length_mismatch(self):
        kw = self._defaults()
        kw["units"] = ("Hz",)
        with pytest.raises(ValueError, match="len\\(units\\)"):
            ParameterVector(**kw)

    def test_components_length_mismatch(self):
        kw = self._defaults()
        kw["components"] = ("Spindown",) * 3
        with pytest.raises(ValueError, match="len\\(components\\)"):
            ParameterVector(**kw)

    def test_bounds_length_mismatch(self):
        kw = self._defaults()
        kw["bounds"] = ((None, None),) * 2
        with pytest.raises(ValueError, match="len\\(bounds\\)"):
            ParameterVector(**kw)

    def test_values_length_mismatch(self):
        kw = self._defaults()
        kw["values"] = jnp.array([1.0, 2.0])
        with pytest.raises(ValueError, match="values\\.shape\\[0\\]"):
            ParameterVector(**kw)

    def test_name_to_index_key_mismatch(self):
        kw = self._defaults()
        kw["_name_to_index"] = {"F0": 0, "BOGUS": 1}
        with pytest.raises(ValueError, match="_name_to_index keys"):
            ParameterVector(**kw)

    def test_name_to_index_out_of_range(self):
        kw = self._defaults()
        kw["_name_to_index"] = {n: i for i, n in enumerate(kw["names"])}
        kw["_name_to_index"]["DECJ"] = 99
        with pytest.raises(ValueError, match="out of range"):
            ParameterVector(**kw)

    def test_epoch_int_values_unknown_key(self):
        kw = self._defaults()
        kw["epoch_int_values"] = {"PEPOCH": 59000.0, "BOGUS": 0.0}
        with pytest.raises(ValueError, match="not in names"):
            ParameterVector(**kw)


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    def test_jit_function_with_all_types(self):
        """A JIT function taking (ParameterVector, TOAData) -> PhaseResult."""
        pv = _make_param_vector()
        td = _make_toa_data()

        @jax.jit
        def mock_phase(params, toa_data):
            f0 = params.param_value("F0")
            dt = toa_data.tdb_frac  # simplified: just use fractional day
            phase_val = f0 * dt
            phase_int = jnp.floor(phase_val)
            return PhaseResult.create(phase_int, phase_val - phase_int)

        result = mock_phase(pv, td)
        assert result.int.shape == (10,)
        assert result.frac.shape == (10,)
        assert jnp.all(result.frac >= -0.5)
        assert jnp.all(result.frac < 0.5)

    def test_jacobian_mock_residuals(self):
        """jax.jacobian of a mock residual function produces correct shape."""
        pv = _make_param_vector()
        td = _make_toa_data(n_toas=5)

        def mock_residuals(params, toa_data):
            f0 = params.param_value("F0")
            dm = params.param_value("DM")
            return f0 * toa_data.tdb_frac + dm * toa_data.freq

        # Jacobian w.r.t. params.values: shape (n_toas, n_params)
        jac_fn = jax.jacobian(lambda p: mock_residuals(p, td))
        grad_pv = jac_fn(pv)
        # grad_pv.values has shape (n_toas, n_params)
        assert grad_pv.values.shape == (5, 6)

    def test_precision_dt_subtraction(self):
        """Verify int/frac split preserves nanosecond precision over decades.

        The key insight: with the split, the fractional subtraction
        (toa_frac - epoch_frac) involves numbers in [0,1) so there's no
        catastrophic cancellation. Without the split, subtracting two
        ~60000-day MJDs directly loses ~5 digits.
        """
        # Two MJDs ~30 years apart
        toa_int = jnp.array(59000.0)
        toa_frac = jnp.array(0.123456789012345)
        epoch_int = 48000.0
        epoch_frac = jnp.array(0.987654321098765)

        # High-precision dt via int/frac split
        # Integer part is exact; fractional part has no cancellation
        dt_int = toa_int - epoch_int  # exactly 11000.0
        dt_frac = toa_frac - epoch_frac  # full precision, both in [0,1)
        dt_split = dt_int + dt_frac

        # Direct float64 subtraction (simulating no split)
        toa_full = toa_int + toa_frac  # 59000.123... loses precision in representation
        epoch_full = jnp.array(epoch_int) + epoch_frac
        dt_direct = toa_full - epoch_full

        # Both should give the same large-scale answer
        assert jnp.isclose(dt_split, dt_direct, atol=1e-8)

        # But the split preserves more precision in the fractional part.
        # We can verify this by checking that the fractional subtraction
        # is exact to float64 eps (~1e-16), while the direct subtraction
        # of ~60000-valued numbers can only be accurate to ~60000 * eps ~ 1e-11.
        #
        # Test: the fractional difference should be representable to full precision.
        frac_diff = float(toa_frac - epoch_frac)
        # This is a small number (~-0.864) computed from two [0,1) numbers:
        # no cancellation, full 16 digits of precision.
        expected_frac = 0.123456789012345 - 0.987654321098765
        assert abs(frac_diff - expected_frac) < 2e-16, (
            f"Fractional difference lost precision: err={abs(frac_diff - expected_frac)}"
        )

        # And the split dt preserves this: dt_split = 11000.0 + frac_diff (exact addition)
        # So dt_split has full precision of frac_diff, i.e. ~1e-16 relative to frac_diff,
        # which is ~1e-16 days ~ 0.0086 nanoseconds. Well under 1ns.
        one_ns_in_days = 1e-9 / 86400.0
        expected_dt = 11000.0 + expected_frac
        err_split = abs(float(dt_split) - expected_dt)
        assert err_split < one_ns_in_days, f"Split error {err_split} exceeds 1ns"
