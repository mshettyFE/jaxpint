"""Tests for jaxpint.jump: PhaseJump component."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from jaxpint.phase.jump import PhaseJump
from jaxpint.types.dual_float import DualFloat
from tests.helpers import make_toa_data, make_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jump_params(
    f0=200.0,
    jump_values=(1e-6, 2e-6),
    jump_names=("JUMP1", "JUMP2"),
    pepoch_int=59000.0,
):
    """Build a ParameterVector with F0, PEPOCH, and JUMP parameters."""
    names = ["F0"] + list(jump_names) + ["PEPOCH"]
    values = [f0] + list(jump_values) + [0.0]
    units = ["Hz"] + ["s"] * len(jump_names) + ["day"]
    components = ["Spindown"] + ["PhaseJump"] * len(jump_names) + ["Spindown"]
    return make_params(
        names, values,
        units=tuple(units),
        components=tuple(components),
        epoch_int_values={"PEPOCH": pepoch_int},
    )


def _make_toa_with_masks(n_toas=6, masks=None):
    """Build TOAData with specified flag_masks."""
    return make_toa_data(n_toas, flag_masks=masks or {})


class TestConstruction:
    def test_basic(self):
        j = PhaseJump(jump_param_names=("JUMP1",))
        assert j.jump_param_names == ("JUMP1",)
        assert j.f0_name == "F0"

    def test_multiple_jumps(self):
        j = PhaseJump(jump_param_names=("JUMP1", "JUMP2", "JUMP3"))
        assert len(j.jump_param_names) == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            PhaseJump(jump_param_names=())

    def test_zero_dynamic_leaves(self):
        j = PhaseJump(jump_param_names=("JUMP1", "JUMP2"))
        leaves, _ = jax.tree.flatten(j)
        assert len(leaves) == 0


class TestPhaseJump:
    def test_single_jump_masked_toas(self):
        """Jump applies only to masked TOAs: phase = JUMP * F0."""
        f0 = 200.0
        jump_val = 1e-6  # 1 microsecond
        mask = np.array([True, True, False, False, False, False])

        jump = PhaseJump(jump_param_names=("JUMP1",))
        params = _make_jump_params(f0=f0, jump_values=(jump_val,), jump_names=("JUMP1",))
        toa_data = _make_toa_with_masks(n_toas=6, masks={"JUMP1": mask})
        delay = jnp.zeros(6)

        result = jump(toa_data, params, delay)
        assert isinstance(result, DualFloat)

        expected = jnp.where(jnp.array(mask), jump_val * f0, 0.0)
        assert jnp.allclose(result.total, expected, atol=1e-15)

    def test_two_jumps_non_overlapping(self):
        """Two non-overlapping jumps each apply to their own TOA subset."""
        f0 = 200.0
        j1_val, j2_val = 1e-6, 3e-6
        mask1 = np.array([True, True, False, False, False, False])
        mask2 = np.array([False, False, False, True, True, False])

        jump = PhaseJump(jump_param_names=("JUMP1", "JUMP2"))
        params = _make_jump_params(f0=f0, jump_values=(j1_val, j2_val))
        toa_data = _make_toa_with_masks(
            n_toas=6, masks={"JUMP1": mask1, "JUMP2": mask2}
        )
        delay = jnp.zeros(6)

        result = jump(toa_data, params, delay)

        expected = jnp.zeros(6)
        expected = jnp.where(jnp.array(mask1), expected + j1_val * f0, expected)
        expected = jnp.where(jnp.array(mask2), expected + j2_val * f0, expected)
        assert jnp.allclose(result.total, expected, atol=1e-15)

    def test_overlapping_jumps(self):
        """A TOA affected by two jumps gets the sum of both."""
        f0 = 100.0
        j1_val, j2_val = 1e-6, 2e-6
        # TOA index 1 is in both masks
        mask1 = np.array([True, True, False, False])
        mask2 = np.array([False, True, True, False])

        jump = PhaseJump(jump_param_names=("JUMP1", "JUMP2"))
        params = _make_jump_params(
            f0=f0, jump_values=(j1_val, j2_val),
            jump_names=("JUMP1", "JUMP2"),
        )
        toa_data = _make_toa_with_masks(
            n_toas=4, masks={"JUMP1": mask1, "JUMP2": mask2}
        )
        delay = jnp.zeros(4)

        result = jump(toa_data, params, delay)

        # TOA 0: j1 only, TOA 1: j1+j2, TOA 2: j2 only, TOA 3: none
        expected = jnp.array([
            j1_val * f0,
            (j1_val + j2_val) * f0,
            j2_val * f0,
            0.0,
        ])
        assert jnp.allclose(result.total, expected, atol=1e-15)

    def test_no_mask_gives_zero(self):
        """If flag_masks is empty (e.g. TZR TOA), phase is zero."""
        jump = PhaseJump(jump_param_names=("JUMP1",))
        params = _make_jump_params(f0=200.0, jump_values=(1e-6,), jump_names=("JUMP1",))
        toa_data = _make_toa_with_masks(n_toas=3, masks={})
        delay = jnp.zeros(3)

        result = jump(toa_data, params, delay)
        assert jnp.allclose(result.total, 0.0, atol=1e-15)

    def test_all_false_mask(self):
        """A mask of all False gives zero phase."""
        jump = PhaseJump(jump_param_names=("JUMP1",))
        params = _make_jump_params(f0=200.0, jump_values=(1e-6,), jump_names=("JUMP1",))
        mask = np.zeros(4, dtype=bool)
        toa_data = _make_toa_with_masks(n_toas=4, masks={"JUMP1": mask})
        delay = jnp.zeros(4)

        result = jump(toa_data, params, delay)
        assert jnp.allclose(result.total, 0.0, atol=1e-15)

    def test_all_true_mask(self):
        """A mask of all True applies the jump to every TOA."""
        f0 = 150.0
        jump_val = 5e-7
        jump = PhaseJump(jump_param_names=("JUMP1",))
        params = _make_jump_params(f0=f0, jump_values=(jump_val,), jump_names=("JUMP1",))
        mask = np.ones(4, dtype=bool)
        toa_data = _make_toa_with_masks(n_toas=4, masks={"JUMP1": mask})
        delay = jnp.zeros(4)

        result = jump(toa_data, params, delay)
        expected = jnp.full(4, jump_val * f0)
        assert jnp.allclose(result.total, expected, atol=1e-15)


class TestJIT:
    def test_jit_call(self):
        """PhaseJump works under jax.jit."""
        jump = PhaseJump(jump_param_names=("JUMP1",))
        params = _make_jump_params(f0=200.0, jump_values=(1e-6,), jump_names=("JUMP1",))
        mask = np.array([True, False, True, False])
        toa_data = _make_toa_with_masks(n_toas=4, masks={"JUMP1": mask})
        delay = jnp.zeros(4)

        jitted = jax.jit(jump)
        result = jitted(toa_data, params, delay)
        assert isinstance(result, DualFloat)
        assert result.int.shape == (4,)


class TestGrad:
    def test_grad_wrt_jump(self):
        """d(phase)/d(JUMP) = F0 for masked TOAs, 0 for unmasked."""
        f0 = 200.0
        jump = PhaseJump(jump_param_names=("JUMP1",))
        mask = np.array([True, True, False, False])
        params = _make_jump_params(f0=f0, jump_values=(1e-6,), jump_names=("JUMP1",))
        toa_data = _make_toa_with_masks(n_toas=4, masks={"JUMP1": mask})
        delay = jnp.zeros(4)

        def loss(p):
            return jump(toa_data, p, delay).total.sum()

        grads = jax.grad(loss)(params)
        jump_idx = params.param_index("JUMP1")
        # d(sum(phase))/d(JUMP1) = F0 * n_masked_toas
        expected = f0 * np.sum(mask)
        assert jnp.isclose(grads.values[jump_idx], expected, rtol=1e-12)

    def test_jacobian_per_toa(self):
        """Per-TOA Jacobian: d(phase_i)/d(JUMP) = F0 if masked, else 0."""
        f0 = 200.0
        jump = PhaseJump(jump_param_names=("JUMP1",))
        mask = np.array([True, False, True, False])
        params = _make_jump_params(f0=f0, jump_values=(1e-6,), jump_names=("JUMP1",))
        toa_data = _make_toa_with_masks(n_toas=4, masks={"JUMP1": mask})
        delay = jnp.zeros(4)

        def phase_fn(p):
            return jump(toa_data, p, delay).total

        jac = jax.jacobian(phase_fn)(params)
        jump_idx = params.param_index("JUMP1")
        # Column of Jacobian for JUMP1
        d_phase_d_jump = jac.values[:, jump_idx]
        expected = jnp.where(jnp.array(mask), f0, 0.0)
        assert jnp.allclose(d_phase_d_jump, expected, atol=1e-12)

    def test_grad_finite(self):
        """All gradients are finite."""
        jump = PhaseJump(jump_param_names=("JUMP1", "JUMP2"))
        mask1 = np.array([True, True, False])
        mask2 = np.array([False, True, True])
        params = _make_jump_params(f0=200.0, jump_values=(1e-6, 2e-6))
        toa_data = _make_toa_with_masks(
            n_toas=3, masks={"JUMP1": mask1, "JUMP2": mask2}
        )
        delay = jnp.zeros(3)

        def loss(p):
            return jump(toa_data, p, delay).total.sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))


# ===========================================================================
# Integration tests: JaxPINT vs PINT
# ===========================================================================


@pytest.fixture
def b1855_9yv1():
    """B1855+09 NANOGrav 9yr v1 (1 JUMP on -fe L-wide)."""
    pytest.importorskip("pint")
    import pint.models as models
    import pint.toa as toa
    from pint.config import examplefile

    model = models.get_model(examplefile("B1855+09_NANOGrav_9yv1.gls.par"))
    toas = toa.get_TOAs(examplefile("B1855+09_NANOGrav_9yv1.tim"), ephem="DE421")
    return model, toas


@pytest.fixture
def b1855_dfg12():
    """B1855+09 dfg+12 (21 JUMPs on -chanid flags)."""
    pytest.importorskip("pint")
    import pint.models as models
    import pint.toa as toa
    from pint.config import examplefile

    model = models.get_model(examplefile("B1855+09_NANOGrav_dfg+12_TAI.par"))
    toas = toa.get_TOAs(examplefile("B1855+09_NANOGrav_dfg+12.tim"), ephem="DE421")
    return model, toas


class TestJumpIntegration:
    """Compare JaxPINT PhaseJump against PINT on real pulsar data."""

    # -- mask consistency ------------------------------------------------

    @pytest.mark.slow
    def test_mask_consistency(self, b1855_9yv1):
        """Bridge flag_masks match PINT's select_toa_mask for a single JUMP."""
        model, toas = b1855_9yv1
        from jaxpint.bridge import pint_toas_to_jax

        toa_data = pint_toas_to_jax(toas, model=model)

        # PINT mask: indices -> boolean
        pint_idx = model.JUMP1.select_toa_mask(toas)
        expected = np.zeros(toas.ntoas, dtype=bool)
        expected[pint_idx] = True

        jax_mask = np.asarray(toa_data.flag_masks["JUMP1"])
        np.testing.assert_array_equal(jax_mask, expected)

        # Mask is non-trivial
        assert np.any(expected)
        assert not np.all(expected)

    @pytest.mark.slow
    def test_mask_consistency_multi_jump(self, b1855_dfg12):
        """Bridge flag_masks match PINT for all 21 JUMPs."""
        model, toas = b1855_dfg12
        from jaxpint.bridge import pint_toas_to_jax, build_timing_model

        toa_data = pint_toas_to_jax(toas, model=model)
        jax_model, _ = build_timing_model(model)

        # Find the PhaseJump component
        jump_comp = [c for c in jax_model.phase_components
                     if isinstance(c, PhaseJump)]
        assert len(jump_comp) == 1
        jump_names = jump_comp[0].jump_param_names

        assert len(jump_names) == 21

        for jname in jump_names:
            pint_par = getattr(model, jname)
            pint_idx = pint_par.select_toa_mask(toas)
            expected = np.zeros(toas.ntoas, dtype=bool)
            expected[pint_idx] = True

            jax_mask = np.asarray(toa_data.flag_masks[jname])
            np.testing.assert_array_equal(jax_mask, expected, err_msg=jname)
            assert np.any(expected), f"{jname} mask is all-False"

    # -- phase values ----------------------------------------------------

    @pytest.mark.slow
    def test_phase_values_match(self, b1855_9yv1):
        """JaxPINT jump phase matches PINT for 1-JUMP data."""
        model, toas = b1855_9yv1
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model

        toa_data = pint_toas_to_jax(toas, model=model)
        params = pint_model_to_params(model).params
        jax_model, _ = build_timing_model(model)

        jump_comp = [c for c in jax_model.phase_components
                     if isinstance(c, PhaseJump)][0]

        # PINT phase
        pint_jump_comp = model.components["PhaseJump"]
        pint_jump_comp.setup()
        delay = model.delay(toas)
        pint_phase = pint_jump_comp.jump_phase(toas, delay).value

        # JaxPINT phase
        jax_result = jump_comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        jax_phase = np.asarray(jax_result.total)

        np.testing.assert_allclose(jax_phase, pint_phase, rtol=1e-12, atol=1e-15)

    @pytest.mark.slow
    def test_phase_values_match_multi_jump(self, b1855_dfg12):
        """JaxPINT jump phase matches PINT for 21-JUMP data."""
        model, toas = b1855_dfg12
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model

        toa_data = pint_toas_to_jax(toas, model=model)
        params = pint_model_to_params(model).params
        jax_model, _ = build_timing_model(model)

        jump_comp = [c for c in jax_model.phase_components
                     if isinstance(c, PhaseJump)][0]

        # PINT phase
        pint_jump_comp = model.components["PhaseJump"]
        pint_jump_comp.setup()
        delay = model.delay(toas)
        pint_phase = pint_jump_comp.jump_phase(toas, delay).value

        # JaxPINT phase
        jax_result = jump_comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        jax_phase = np.asarray(jax_result.total)

        np.testing.assert_allclose(jax_phase, pint_phase, rtol=1e-12, atol=1e-15)

        # Verify non-trivial: at least some TOAs have non-zero jump phase
        assert np.any(np.abs(pint_phase) > 0)

    # -- derivatives -----------------------------------------------------

    @pytest.mark.slow
    def test_derivative_match(self, b1855_9yv1):
        """JAX autodiff d(phase)/d(JUMP) matches PINT's analytical derivative."""
        model, toas = b1855_9yv1
        import astropy.units as u
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model

        toa_data = pint_toas_to_jax(toas, model=model)
        params = pint_model_to_params(model).params
        jax_model, _ = build_timing_model(model)

        jump_comp = [c for c in jax_model.phase_components
                     if isinstance(c, PhaseJump)][0]

        # PINT derivative
        pint_jump_comp = model.components["PhaseJump"]
        pint_jump_comp.setup()
        delay = model.delay(toas)
        pint_deriv = pint_jump_comp.d_phase_d_jump(
            toas, "JUMP1", delay
        ).to(1 / u.second).value

        # JaxPINT derivative via Jacobian
        def phase_fn(p):
            return jump_comp(toa_data, p, jnp.zeros(toa_data.n_toas)).total

        jac = jax.jacobian(phase_fn)(params)
        jump_idx = params.param_index("JUMP1")
        jax_deriv = np.asarray(jac.values[:, jump_idx])

        np.testing.assert_allclose(jax_deriv, pint_deriv, rtol=1e-12)

    @pytest.mark.slow
    def test_derivative_match_multi_jump(self, b1855_dfg12):
        """JAX autodiff derivatives match PINT for all 21 JUMPs."""
        model, toas = b1855_dfg12
        import astropy.units as u
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model

        toa_data = pint_toas_to_jax(toas, model=model)
        params = pint_model_to_params(model).params
        jax_model, _ = build_timing_model(model)

        jump_comp = [c for c in jax_model.phase_components
                     if isinstance(c, PhaseJump)][0]

        # PINT derivatives
        pint_jump_comp = model.components["PhaseJump"]
        pint_jump_comp.setup()
        delay = model.delay(toas)

        # JaxPINT Jacobian (computed once)
        def phase_fn(p):
            return jump_comp(toa_data, p, jnp.zeros(toa_data.n_toas)).total

        jac = jax.jacobian(phase_fn)(params)

        for jname in jump_comp.jump_param_names:
            pint_deriv = pint_jump_comp.d_phase_d_jump(
                toas, jname, delay
            ).to(1 / u.second).value

            jump_idx = params.param_index(jname)
            jax_deriv = np.asarray(jac.values[:, jump_idx])

            np.testing.assert_allclose(
                jax_deriv, pint_deriv, rtol=1e-12, err_msg=jname
            )
