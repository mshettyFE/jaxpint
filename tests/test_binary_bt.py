"""Tests for BinaryBT delay model against PINT."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)


def _make_params(param_names, param_values, epoch_int_values=None):
    """Helper to build a ParameterVector for tests."""
    from jaxpint.types import ParameterVector

    if epoch_int_values is None:
        epoch_int_values = {}
    return ParameterVector(
        values=jnp.array(param_values),
        names=param_names,
        frozen_mask=tuple(False for _ in param_names),
        units=tuple("" for _ in param_names),
        components=tuple("BinaryBT" for _ in param_names),
        bounds=tuple((None, None) for _ in param_names),
        epoch_int_values=epoch_int_values,
        _name_to_index={n: i for i, n in enumerate(param_names)},
    )


def _make_toa_data(t_mjd):
    """Helper to build minimal TOAData from MJD array."""
    from jaxpint.types import TOAData

    t_np = np.asarray(t_mjd)
    tdb_int = jnp.array(np.floor(t_np))
    tdb_frac = jnp.array(t_np - np.floor(t_np))
    n = len(t_np)
    return TOAData(
        mjd_int=tdb_int,
        mjd_frac=tdb_frac,
        tdb_int=tdb_int,
        tdb_frac=tdb_frac,
        error=jnp.ones(n) * 1e-6,
        freq=jnp.ones(n) * 1400.0,
        delta_pulse_number=jnp.zeros(n),
        ssb_obs_pos=jnp.zeros((n, 3)),
        ssb_obs_vel=jnp.zeros((n, 3)),
        obs_sun_pos=jnp.zeros((n, 3)),
        obs_indices=jnp.zeros(n, dtype=jnp.int32),
        flag_masks={},
        planet_positions={},
        dm_values=None,
        dm_errors=None,
        tzr_tdb_int=jnp.array(54000.0),
        tzr_tdb_frac=jnp.array(0.5),
        tzr_freq=jnp.array(jnp.inf),
        tzr_ssb_obs_pos=jnp.zeros(3),
        n_toas=n,
        obs_names=("fake",),
    )


_DEG_YR_TO_RAD_S = np.pi / 180.0 / (365.25 * 86400.0)


@pytest.fixture
def bt_params():
    """Typical BT binary parameters in JaxPINT conventions (post-bridge)."""
    return {
        "PB": 1.5,                       # days
        "T0": 54000.0,                   # MJD
        "A1": 2.0,                       # light-seconds
        "ECC": 0.3,
        "OM_deg": 45.0,                  # degrees (for PINT)
        "OM": 45.0 * np.pi / 180.0,     # radians (post-bridge)
        "OMDOT_deg_yr": 0.5,            # deg/yr (for PINT)
        "OMDOT": 0.5 * _DEG_YR_TO_RAD_S,  # rad/s (post-bridge)
        "GAMMA": 0.001,                  # seconds
        "PBDOT": 1e-12,                  # s/s
    }


class TestBinaryBTvsPINT:
    """Compare JaxPINT BinaryBT against PINT's standalone BTmodel."""

    def test_bt_delay_matches_pint(self, bt_params):
        """BT delay should match PINT to float64 precision."""
        pytest.importorskip("pint")
        import astropy.units as u
        from pint.models.stand_alone_psr_binaries.BT_model import BTmodel

        from jaxpint.binary.bt import BinaryBT

        # --- Set up PINT BT model ---
        bm = BTmodel()
        pint_params = {
            "PB": bt_params["PB"] * u.day,
            "T0": np.longdouble(bt_params["T0"]) * u.day,
            "A1": bt_params["A1"] * u.lightsecond,
            "ECC": bt_params["ECC"] * u.Unit(""),
            "OM": bt_params["OM_deg"] * u.deg,
            "OMDOT": bt_params["OMDOT_deg_yr"] * u.deg / u.year,
            "GAMMA": bt_params["GAMMA"] * u.second,
            "PBDOT": bt_params["PBDOT"] * u.Unit(""),
        }
        t = np.linspace(54000.5, 54500.0, 500) * u.day
        bm.update_input(barycentric_toa=t, **pint_params)
        pint_delay = bm.BTdelay().to(u.second).value

        # --- Set up JaxPINT BinaryBT ---
        bt = BinaryBT(
            pb_name="PB", t0_name="T0", a1_name="A1", ecc_name="ECC",
            om_name="OM", omdot_name="OMDOT", gamma_name="GAMMA", pbdot_name="PBDOT",
        )

        t0_int = np.floor(bt_params["T0"])
        t0_frac = bt_params["T0"] - t0_int

        param_names = ("PB", "T0", "A1", "ECC", "OM", "OMDOT", "GAMMA", "PBDOT")
        param_values = [bt_params["PB"], t0_frac, bt_params["A1"], bt_params["ECC"],
                        bt_params["OM"], bt_params["OMDOT"], bt_params["GAMMA"], bt_params["PBDOT"]]
        params = _make_params(param_names, param_values, epoch_int_values={"T0": t0_int})

        toa_data = _make_toa_data(np.linspace(54000.5, 54500.0, 500))
        jax_delay = np.array(bt(toa_data, params, jnp.zeros(500)))

        npt.assert_allclose(jax_delay, pint_delay, atol=1e-12, rtol=1e-12)

    def test_bt_jit(self, bt_params):
        """BinaryBT should be JIT-compilable."""
        from jaxpint.binary.bt import BinaryBT

        bt = BinaryBT(pb_name="PB", t0_name="T0", a1_name="A1",
                       ecc_name="ECC", om_name="OM")

        t0_int = np.floor(bt_params["T0"])
        t0_frac = bt_params["T0"] - t0_int

        params = _make_params(
            ("PB", "T0", "A1", "ECC", "OM"),
            [bt_params["PB"], t0_frac, bt_params["A1"], bt_params["ECC"], bt_params["OM"]],
            epoch_int_values={"T0": t0_int},
        )
        toa_data = _make_toa_data(np.linspace(54100.0, 54100.9, 10))

        jitted = jax.jit(bt)
        result = jitted(toa_data, params, jnp.zeros(10))
        assert result.shape == (10,)
        assert jnp.all(jnp.isfinite(result))

    def test_bt_autodiff(self, bt_params):
        """BinaryBT should be differentiable via JAX autodiff."""
        from jaxpint.binary.bt import BinaryBT

        bt = BinaryBT(pb_name="PB", t0_name="T0", a1_name="A1",
                       ecc_name="ECC", om_name="OM", gamma_name="GAMMA")

        t0_int = np.floor(bt_params["T0"])
        t0_frac = bt_params["T0"] - t0_int

        param_names = ("PB", "T0", "A1", "ECC", "OM", "GAMMA")
        params = _make_params(
            param_names,
            [bt_params["PB"], t0_frac, bt_params["A1"], bt_params["ECC"],
             bt_params["OM"], bt_params["GAMMA"]],
            epoch_int_values={"T0": t0_int},
        )
        n = 10
        toa_data = _make_toa_data(np.linspace(54100.1, 54100.9, n))

        def delay_fn(param_values):
            p = params.with_free_values(param_values)
            return bt(toa_data, p, jnp.zeros(n))

        J = jax.jacobian(delay_fn)(params.free_values())
        assert J.shape == (n, len(param_names))
        assert jnp.all(jnp.isfinite(J))
        # A1 column should be nonzero (delay scales linearly with A1)
        a1_col = list(param_names).index("A1")
        assert jnp.any(jnp.abs(J[:, a1_col]) > 0)
