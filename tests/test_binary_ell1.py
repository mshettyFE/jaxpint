"""Tests for BinaryELL1 delay model against PINT."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)


def _make_params(param_names, param_values, epoch_int_values=None):
    from jaxpint.types import ParameterVector
    if epoch_int_values is None:
        epoch_int_values = {}
    return ParameterVector(
        values=jnp.array(param_values),
        names=param_names,
        frozen_mask=tuple(False for _ in param_names),
        units=tuple("" for _ in param_names),
        components=tuple("BinaryELL1" for _ in param_names),
        bounds=tuple((None, None) for _ in param_names),
        epoch_int_values=epoch_int_values,
        _name_to_index={n: i for i, n in enumerate(param_names)},
    )


def _make_toa_data(t_mjd):
    from jaxpint.types import TOAData
    t_np = np.asarray(t_mjd)
    tdb_int = jnp.array(np.floor(t_np))
    tdb_frac = jnp.array(t_np - np.floor(t_np))
    n = len(t_np)
    return TOAData(
        mjd_int=tdb_int, mjd_frac=tdb_frac,
        tdb_int=tdb_int, tdb_frac=tdb_frac,
        error=jnp.ones(n) * 1e-6, freq=jnp.ones(n) * 1400.0,
        delta_pulse_number=jnp.zeros(n),
        ssb_obs_pos=jnp.zeros((n, 3)), ssb_obs_vel=jnp.zeros((n, 3)),
        obs_sun_pos=jnp.zeros((n, 3)),
        obs_indices=jnp.zeros(n, dtype=jnp.int32),
        flag_masks={}, planet_positions={},
        dm_values=None, dm_errors=None,
        tropo_alt=None, tropo_alt_valid=None,
        obs_geodetic_lat=None, obs_height_km=None,
        tzr_tdb_int=jnp.array(54000.0), tzr_tdb_frac=jnp.array(0.5),
        tzr_freq=jnp.array(jnp.inf), tzr_ssb_obs_pos=jnp.zeros(3),
        n_toas=n, obs_names=("fake",),
    )


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
    """Compare JaxPINT BinaryELL1 against PINT's standalone ELL1model."""

    def test_ell1_delay_matches_pint(self, ell1_params):
        """ELL1 delay should match PINT to float64 precision."""
        pytest.importorskip("pint")
        import astropy.units as u
        from pint.models.stand_alone_psr_binaries.ELL1_model import ELL1model
        from pint import Tsun

        from jaxpint.binary.ell1 import BinaryELL1

        # --- PINT ELL1 model ---
        bm = ELL1model()
        pint_params = {
            "PB": ell1_params["PB"] * u.day,
            "TASC": np.longdouble(ell1_params["TASC"]) * u.day,
            "A1": ell1_params["A1"] * u.lightsecond,
            "EPS1": ell1_params["EPS1"] * u.Unit(""),
            "EPS2": ell1_params["EPS2"] * u.Unit(""),
            "M2": ell1_params["M2"] * u.M_sun,
            "SINI": ell1_params["SINI"] * u.Unit(""),
            "PBDOT": ell1_params["PBDOT"] * u.Unit(""),
        }
        t = np.linspace(54000.5, 54300.0, 500) * u.day
        bm.update_input(barycentric_toa=t, **pint_params)
        pint_delay = bm.ELL1delay().to(u.second).value

        # --- JaxPINT BinaryELL1 ---
        ell1 = BinaryELL1(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2",
            m2_name="M2", sini_name="SINI", pbdot_name="PBDOT",
        )

        tasc_int = np.floor(ell1_params["TASC"])
        tasc_frac = ell1_params["TASC"] - tasc_int

        param_names = ("PB", "TASC", "A1", "EPS1", "EPS2", "M2", "SINI", "PBDOT")
        param_values = [ell1_params["PB"], tasc_frac, ell1_params["A1"],
                        ell1_params["EPS1"], ell1_params["EPS2"],
                        ell1_params["M2"], ell1_params["SINI"], ell1_params["PBDOT"]]
        params = _make_params(param_names, param_values,
                              epoch_int_values={"TASC": tasc_int})

        toa_data = _make_toa_data(np.linspace(54000.5, 54300.0, 500))
        jax_delay = np.array(ell1(toa_data, params, jnp.zeros(500)))

        npt.assert_allclose(jax_delay, pint_delay, atol=1e-12, rtol=1e-12)

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
        params = _make_params(param_names, param_values,
                              epoch_int_values={"TASC": tasc_int})

        toa_data = _make_toa_data(np.linspace(54001.0, 54100.0, 200))
        jax_delay = np.array(ell1(toa_data, params, jnp.zeros(200)))

        npt.assert_allclose(jax_delay, pint_delay, atol=1e-12, rtol=1e-12)

    def test_ell1_jit(self, ell1_params):
        """BinaryELL1 should be JIT-compilable."""
        from jaxpint.binary.ell1 import BinaryELL1

        ell1 = BinaryELL1(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2", shapiro_mode="none",
        )

        tasc_int = np.floor(ell1_params["TASC"])
        tasc_frac = ell1_params["TASC"] - tasc_int

        params = _make_params(
            ("PB", "TASC", "A1", "EPS1", "EPS2"),
            [ell1_params["PB"], tasc_frac, ell1_params["A1"],
             ell1_params["EPS1"], ell1_params["EPS2"]],
            epoch_int_values={"TASC": tasc_int},
        )

        n = 10
        toa_data = _make_toa_data(np.linspace(54100.1, 54100.9, n))
        jitted = jax.jit(ell1)
        result = jitted(toa_data, params, jnp.zeros(n))
        assert result.shape == (n,)
        assert jnp.all(jnp.isfinite(result))

    def test_ell1_autodiff(self, ell1_params):
        """BinaryELL1 should be differentiable via JAX autodiff."""
        from jaxpint.binary.ell1 import BinaryELL1

        ell1 = BinaryELL1(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2",
            m2_name="M2", sini_name="SINI",
        )

        tasc_int = np.floor(ell1_params["TASC"])
        tasc_frac = ell1_params["TASC"] - tasc_int

        param_names = ("PB", "TASC", "A1", "EPS1", "EPS2", "M2", "SINI")
        param_values = [ell1_params["PB"], tasc_frac, ell1_params["A1"],
                        ell1_params["EPS1"], ell1_params["EPS2"],
                        ell1_params["M2"], ell1_params["SINI"]]
        params = _make_params(param_names, param_values,
                              epoch_int_values={"TASC": tasc_int})

        n = 10
        toa_data = _make_toa_data(np.linspace(54100.1, 54100.9, n))

        def delay_fn(param_values):
            p = params.with_free_values(param_values)
            return ell1(toa_data, p, jnp.zeros(n))

        J = jax.jacobian(delay_fn)(params.free_values())
        assert J.shape == (n, len(param_names))
        assert jnp.all(jnp.isfinite(J))
        a1_col = list(param_names).index("A1")
        assert jnp.any(jnp.abs(J[:, a1_col]) > 0)
