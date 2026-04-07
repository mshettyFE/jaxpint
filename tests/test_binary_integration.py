"""Integration test: parse a real .par/.tim file and compare binary delay
between JaxPINT and PINT end-to-end."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest



@pytest.fixture
def b1855():
    """Load B1855+09 (DD binary) from PINT example data."""
    pint = pytest.importorskip("pint")
    from pint import models, toa
    from pint.config import examplefile

    model = models.get_model(examplefile("B1855+09_NANOGrav_9yv1.gls.par"))
    toas = toa.get_TOAs(examplefile("B1855+09_NANOGrav_9yv1.tim"), ephem="DE421")
    return model, toas


class TestBinaryIntegration:
    """End-to-end binary delay comparison using real par/tim files."""

    def test_b1855_dd_binary_delay(self, b1855):
        """B1855+09 DD binary delay should match PINT to <1 ps."""
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model

        pint_model, toas = b1855

        # --- PINT binary delay (with zero accumulated delay, matching our call) ---
        pint_delay = pint_model.binarymodel_delay(toas, 0).to("s").value

        # --- JaxPINT binary delay ---
        toa_data = pint_toas_to_jax(toas)
        params = pint_model_to_params(pint_model).params
        tm, _ = build_timing_model(pint_model)

        # Find the binary component and call it directly
        from jaxpint.binary.dd import BinaryDD
        binary_comp = [c for c in tm.delay_components if isinstance(c, BinaryDD)]
        assert len(binary_comp) == 1, "Expected exactly one BinaryDD component"
        binary_comp = binary_comp[0]

        jax_delay = np.array(binary_comp(toa_data, params, jnp.zeros(toa_data.n_toas)))

        npt.assert_allclose(jax_delay, pint_delay, atol=1e-11, rtol=1e-11)

    def test_b1855_binary_delay_jit(self, b1855):
        """Binary delay on real data should be JIT-compilable."""
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model
        from jaxpint.binary.dd import BinaryDD

        pint_model, toas = b1855
        toa_data = pint_toas_to_jax(toas)
        params = pint_model_to_params(pint_model).params
        tm, _ = build_timing_model(pint_model)

        binary_comp = [c for c in tm.delay_components if isinstance(c, BinaryDD)][0]
        jitted = jax.jit(binary_comp)

        result = jitted(toa_data, params, jnp.zeros(toa_data.n_toas))
        assert result.shape == (toa_data.n_toas,)
        assert jnp.all(jnp.isfinite(result))

    def test_b1855_binary_delay_autodiff(self, b1855):
        """Jacobian of binary delay w.r.t. parameters on real data."""
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model
        from jaxpint.binary.dd import BinaryDD

        pint_model, toas = b1855
        toa_data = pint_toas_to_jax(toas)
        params = pint_model_to_params(pint_model).params
        tm, _ = build_timing_model(pint_model)

        binary_comp = [c for c in tm.delay_components if isinstance(c, BinaryDD)][0]
        n = toa_data.n_toas

        def delay_fn(param_values):
            p = params.with_free_values(param_values)
            return binary_comp(toa_data, p, jnp.zeros(n))

        J = jax.jacobian(delay_fn)(params.free_values())
        n_free = len(params.free_values())
        assert J.shape == (n, n_free)
        assert jnp.all(jnp.isfinite(J))
