"""Tests for BinaryELL1 delay model against PINT."""

import jax
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
        params = make_binary_params(param_names, param_values,
                              epoch_int_values={"TASC": tasc_int})

        toa_data = make_binary_toa_data(np.linspace(54001.0, 54100.0, 200))
        jax_delay = np.array(ell1(toa_data, params, jnp.zeros(200)))

        npt.assert_allclose(jax_delay, pint_delay, atol=1e-12, rtol=1e-12)



class TestXPBDOT:
    """XPBDOT must actually enter the orbital phase (PINT OrbitPB:
    orbits = tt0/PB - 0.5*(PBDOT+XPBDOT)*(tt0/PB)^2, while
    pbprime = PB + PBDOT*tt0 only).  The component formerly declared and
    populated ``xpbdot_name`` but never read it, so the parameter silently
    had zero effect."""

    def _delay(self, pbdot, xpbdot):
        from jaxpint.binary.ell1 import BinaryELL1

        ell1 = BinaryELL1(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2",
            pbdot_name="PBDOT", xpbdot_name="XPBDOT",
            shapiro_mode="none",
        )
        t = np.linspace(54001.0, 55800.0, 120)
        params = make_binary_params(
            ("PB", "TASC", "A1", "EPS1", "EPS2", "PBDOT", "XPBDOT"),
            (10.0, 0.0, 50.0, 1e-5, -2e-5, pbdot, xpbdot),
            epoch_int_values={"TASC": 54000.0},
        )
        toa_data = make_binary_toa_data(t)
        return np.array(ell1(toa_data, params, jnp.zeros(len(t))))

    def test_xpbdot_has_effect(self):
        base = self._delay(pbdot=0.0, xpbdot=0.0)
        with_x = self._delay(pbdot=0.0, xpbdot=1e-10)
        assert np.max(np.abs(with_x - base)) > 1e-9

    def test_xpbdot_adds_to_pbdot_in_phase(self):
        """PBDOT and XPBDOT enter the phase only through their sum; the sole
        difference between (pbdot=x, 0) and (0, xpbdot=x) is the pbprime
        (nhat) factor in the inverse-timing correction, which is orders of
        magnitude below the leading effect."""
        x = 1e-10
        via_pbdot = self._delay(pbdot=x, xpbdot=0.0)
        via_xpbdot = self._delay(pbdot=0.0, xpbdot=x)
        effect = np.max(np.abs(via_pbdot - self._delay(0.0, 0.0)))
        diff = np.max(np.abs(via_pbdot - via_xpbdot))
        assert diff < 1e-6 * effect


class TestH3StigmaGuard:
    """get_sini_m2's h3stigma/h3h4 modes divide by STIGMA^3 (and H3 for
    h3h4).  STIGMA (or H3) free at its initial 0 -- or a fitter stepping
    through 0 -- formerly produced an inf forward value and nan gradients;
    the guard returns (sini, m2) = (0, 0) so the objective stays finite."""

    def _component_and_params(self, mode, **overrides):
        from jaxpint.binary.ell1 import BinaryELL1

        names = ["PB", "TASC", "A1", "EPS1", "EPS2", "H3", "STIGMA", "H4"]
        values = {
            "PB": 10.0, "TASC": 0.0, "A1": 50.0,
            "EPS1": 1e-5, "EPS2": -2e-5,
            "H3": 1e-7, "STIGMA": 0.3, "H4": 3e-8,
        }
        values.update(overrides)
        kwargs = dict(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2",
            h3_name="H3", shapiro_mode=mode,
        )
        if mode == "h3stigma":
            kwargs["stigma_name"] = "STIGMA"
        else:
            kwargs["h4_name"] = "H4"
        ell1 = BinaryELL1(**kwargs)
        params = make_binary_params(
            tuple(names), tuple(values[n] for n in names),
            epoch_int_values={"TASC": 54000.0},
        )
        return ell1, params

    @pytest.mark.parametrize(
        "mode,zero_param",
        [("h3stigma", "STIGMA"), ("h3h4", "H3")],
    )
    def test_zero_orthometric_param_finite(self, mode, zero_param):
        ell1, params = self._component_and_params(mode, **{zero_param: 0.0})
        t = np.linspace(54001.0, 54100.0, 50)
        toa_data = make_binary_toa_data(t)

        delay = ell1(toa_data, params, jnp.zeros(len(t)))
        assert jnp.all(jnp.isfinite(delay))

        def loss(p):
            return ell1(toa_data, p, jnp.zeros(len(t))).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values)), (
            f"non-finite gradients: "
            f"{[n for n, g in zip(params.names, grads.values) if not jnp.isfinite(g)]}"
        )

    def test_nonzero_stigma_unchanged(self):
        """The guard must not perturb the regular h3stigma path."""
        ell1, params = self._component_and_params("h3stigma")
        t = np.linspace(54001.0, 54100.0, 50)
        toa_data = make_binary_toa_data(t)
        delay = ell1(toa_data, params, jnp.zeros(len(t)))
        assert jnp.all(jnp.isfinite(delay))
        # Shapiro contributes: differs from the no-shapiro configuration.
        from jaxpint.binary.ell1 import BinaryELL1

        no_shap = BinaryELL1(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2", shapiro_mode="none",
        )
        base = no_shap(toa_data, params, jnp.zeros(len(t)))
        assert float(jnp.max(jnp.abs(delay - base))) > 0.0
