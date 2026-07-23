"""Tests for FDJump delay component — unit tests (no PINT oracle).

FDJump uses maskParameters with system-dependent flags, which are
difficult to set up in synthetic PINT models. These tests verify
the JaxPINT component directly using hand-crafted TOAData.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.delay.fdjump import FDJump
from tests.helpers import make_toa_data, make_params


class TestFDJumpUnit:
    """Unit tests for FDJump component."""

    @pytest.fixture
    def setup(self):
        # Two systems: "SYS1" at 1400 MHz, "SYS2" at 2000 MHz
        n = 10
        mask1 = np.array([True] * 5 + [False] * 5)
        mask2 = np.array([False] * 5 + [True] * 5)
        freqs = np.array([1400.0] * 5 + [2000.0] * 5)

        toa_data = make_toa_data(
            n_toas=n,
            freq=freqs,
            flag_masks={"FD1JUMP1": mask1, "FD1JUMP2": mask2},
        )

        params = make_params(
            names=["FD1JUMP1", "FD1JUMP2"],
            values=[5e-6, 3e-6],
            units=("s", "s"),
        )

        comp = FDJump(
            fdjump_param_names=("FD1JUMP1", "FD1JUMP2"),
            fdjump_fd_indices=(1, 1),
            use_log=True,
        )
        return toa_data, params, comp

    def test_applies_per_system(self, setup):
        """FDJump applies different delays to different systems."""
        toa_data, params, comp = setup
        delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        # SYS1 at 1400 MHz: 5e-6 * log(1400/1000)
        expected_sys1 = 5e-6 * np.log(1400.0 / 1000.0)
        # SYS2 at 2000 MHz: 3e-6 * log(2000/1000)
        expected_sys2 = 3e-6 * np.log(2000.0 / 1000.0)

        np.testing.assert_allclose(np.array(delay[:5]), expected_sys1, rtol=1e-12)
        np.testing.assert_allclose(np.array(delay[5:]), expected_sys2, rtol=1e-12)

    def test_linear_mode(self, setup):
        """FDJump with use_log=False uses linear frequency."""
        toa_data, params, _ = setup
        comp_linear = FDJump(
            fdjump_param_names=("FD1JUMP1", "FD1JUMP2"),
            fdjump_fd_indices=(1, 1),
            use_log=False,
        )
        delay = comp_linear(toa_data, params, jnp.zeros(toa_data.n_toas))

        expected_sys1 = 5e-6 * (1400.0 / 1000.0)
        expected_sys2 = 3e-6 * (2000.0 / 1000.0)

        np.testing.assert_allclose(np.array(delay[:5]), expected_sys1, rtol=1e-12)
        np.testing.assert_allclose(np.array(delay[5:]), expected_sys2, rtol=1e-12)

    def test_jit_compatible(self, setup):
        toa_data, params, comp = setup
        delay = jnp.zeros(toa_data.n_toas)
        eager = comp(toa_data, params, delay)
        jitted = jax.jit(comp)(toa_data, params, delay)
        np.testing.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-14)

    def test_grad_finite(self, setup):
        toa_data, params, comp = setup

        def loss(p):
            return comp(toa_data, p, jnp.zeros(toa_data.n_toas)).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))
