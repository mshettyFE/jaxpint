"""Tests for BinaryELL1 delay model against PINT."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest


from tests.helpers import make_binary_toa_data, make_binary_params


@pytest.fixture
def ell1_params():
    """Typical ELL1 binary parameters."""
    return {
        "PB": 1.2,           # days
        "TASC": 54000.0,     # MJD
        "A1": 1.5,           # light-seconds
        "EPS1": 0.01,        # e*sin(omega)
        "EPS2": 0.02,        # e*cos(omega)
        "M2": 0.25,          # solar masses
        "SINI": 0.85,
        "PBDOT": 1e-13,
    }


class TestBinaryELL1vsPINT:
    """ELL1 variants beyond the standard parametrized suite in ``test_binary_common.py``."""

    @pytest.mark.slow
    def test_ell1_no_shapiro(self, ell1_params):
        """ELL1 without Shapiro delay."""
        pytest.importorskip("pint")
        import astropy.units as u
        from pint.models.stand_alone_psr_binaries.ELL1_model import ELL1model

        from jaxpint.binary.ell1 import BinaryELL1

        bm = ELL1model()
        pint_params = {
            "PB": ell1_params["PB"] * u.day,
            "TASC": np.longdouble(ell1_params["TASC"]) * u.day,
            "A1": ell1_params["A1"] * u.lightsecond,
            "EPS1": ell1_params["EPS1"] * u.Unit(""),
            "EPS2": ell1_params["EPS2"] * u.Unit(""),
        }
        t = np.linspace(54001.0, 54100.0, 200) * u.day
        bm.update_input(barycentric_toa=t, **pint_params)
        pint_delay = bm.ELL1delay().to(u.second).value

        ell1 = BinaryELL1(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2", shapiro_mode="none",
        )

        tasc_int = np.floor(ell1_params["TASC"])
        tasc_frac = ell1_params["TASC"] - tasc_int

        param_names = ("PB", "TASC", "A1", "EPS1", "EPS2")
        param_values = [ell1_params["PB"], tasc_frac, ell1_params["A1"],
                        ell1_params["EPS1"], ell1_params["EPS2"]]
        params = make_binary_params(param_names, param_values, "BinaryELL1",
                              epoch_int_values={"TASC": tasc_int})

        toa_data = make_binary_toa_data(np.linspace(54001.0, 54100.0, 200))
        jax_delay = np.array(ell1(toa_data, params, jnp.zeros(200)))

        npt.assert_allclose(jax_delay, pint_delay, atol=1e-12, rtol=1e-12)

