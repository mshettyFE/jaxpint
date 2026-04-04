"""Tests for jaxpint.jump: PhaseJump component."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jaxpint.jump import PhaseJump
from jaxpint.phase_result import PhaseResult
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


# ===========================================================================
# Construction tests
# ===========================================================================

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


# ===========================================================================
# Phase computation tests
# ===========================================================================

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
        assert isinstance(result, PhaseResult)

        expected = jnp.where(jnp.array(mask), jump_val * f0, 0.0)
        assert jnp.allclose(result.quantity, expected, atol=1e-15)

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
        assert jnp.allclose(result.quantity, expected, atol=1e-15)

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
        assert jnp.allclose(result.quantity, expected, atol=1e-15)

    def test_no_mask_gives_zero(self):
        """If flag_masks is empty (e.g. TZR TOA), phase is zero."""
        jump = PhaseJump(jump_param_names=("JUMP1",))
        params = _make_jump_params(f0=200.0, jump_values=(1e-6,), jump_names=("JUMP1",))
        toa_data = _make_toa_with_masks(n_toas=3, masks={})
        delay = jnp.zeros(3)

        result = jump(toa_data, params, delay)
        assert jnp.allclose(result.quantity, 0.0, atol=1e-15)

    def test_all_false_mask(self):
        """A mask of all False gives zero phase."""
        jump = PhaseJump(jump_param_names=("JUMP1",))
        params = _make_jump_params(f0=200.0, jump_values=(1e-6,), jump_names=("JUMP1",))
        mask = np.zeros(4, dtype=bool)
        toa_data = _make_toa_with_masks(n_toas=4, masks={"JUMP1": mask})
        delay = jnp.zeros(4)

        result = jump(toa_data, params, delay)
        assert jnp.allclose(result.quantity, 0.0, atol=1e-15)

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
        assert jnp.allclose(result.quantity, expected, atol=1e-15)


# ===========================================================================
# JIT tests
# ===========================================================================

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
        assert isinstance(result, PhaseResult)
        assert result.int.shape == (4,)


# ===========================================================================
# Gradient tests
# ===========================================================================

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
            return jump(toa_data, p, delay).quantity.sum()

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
            return jump(toa_data, p, delay).quantity

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
            return jump(toa_data, p, delay).quantity.sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))
