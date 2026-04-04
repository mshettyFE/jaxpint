"""Tests for BinaryDD delay model and variants against PINT."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from tests.helpers import make_toa_data as _make_toa_data_base, make_params

_DEG_YR_TO_RAD_S = np.pi / 180.0 / (365.25 * 86400.0)


def _make_params(param_names, param_values, epoch_int_values=None):
    return make_params(param_names, param_values, components="BinaryDD",
                       epoch_int_values=epoch_int_values or {})


def _make_toa_data(t_mjd):
    return _make_toa_data_base(
        t_mjd=t_mjd,
        tzr_tdb_int=jnp.array(54000.0), tzr_tdb_frac=jnp.array(0.5),
        tzr_freq=jnp.array(jnp.inf), tzr_ssb_obs_pos=jnp.zeros(3),
    )


@pytest.fixture
def dd_params():
    """Typical DD binary parameters in JaxPINT conventions (post-bridge)."""
    return {
        "PB": 0.7,           # days
        "T0": 54000.0,       # MJD
        "A1": 3.0,           # light-seconds
        "ECC": 0.2,
        "OM_deg": 120.0,
        "OM": 120.0 * np.pi / 180.0,          # radians
        "OMDOT_deg_yr": 1.0,
        "OMDOT": 1.0 * _DEG_YR_TO_RAD_S,     # rad/s
        "GAMMA": 0.004,      # seconds
        "PBDOT": 5e-13,
        "M2": 0.3,           # solar masses
        "SINI": 0.9,
        "A0": 1e-5,          # seconds
        "B0": 2e-5,          # seconds
        "DR": 1e-4,
        "DTH": 1e-4,
    }


class TestBinaryDDvsPINT:
    """Compare JaxPINT BinaryDD against PINT's standalone DDmodel."""

    def test_dd_delay_matches_pint(self, dd_params):
        """DD delay should match PINT to float64 precision."""
        pytest.importorskip("pint")
        import astropy.units as u
        from pint.models.stand_alone_psr_binaries.DD_model import DDmodel

        from jaxpint.binary.dd import BinaryDD

        # --- PINT DD model ---
        bm = DDmodel()
        pint_params = {
            "PB": dd_params["PB"] * u.day,
            "T0": np.longdouble(dd_params["T0"]) * u.day,
            "A1": dd_params["A1"] * u.lightsecond,
            "ECC": dd_params["ECC"] * u.Unit(""),
            "OM": dd_params["OM_deg"] * u.deg,
            "OMDOT": dd_params["OMDOT_deg_yr"] * u.deg / u.year,
            "GAMMA": dd_params["GAMMA"] * u.second,
            "PBDOT": dd_params["PBDOT"] * u.Unit(""),
            "M2": dd_params["M2"] * u.M_sun,
            "SINI": dd_params["SINI"] * u.Unit(""),
            "A0": dd_params["A0"] * u.second,
            "B0": dd_params["B0"] * u.second,
            "DR": dd_params["DR"] * u.Unit(""),
            "DTH": dd_params["DTH"] * u.Unit(""),
        }
        t = np.linspace(54000.5, 54200.0, 500) * u.day
        bm.update_input(barycentric_toa=t, **pint_params)
        pint_delay = bm.DDdelay().to(u.second).value

        # --- JaxPINT BinaryDD ---
        dd = BinaryDD(
            pb_name="PB", t0_name="T0", a1_name="A1", ecc_name="ECC", om_name="OM",
            omdot_name="OMDOT", gamma_name="GAMMA", pbdot_name="PBDOT",
            m2_name="M2", sini_name="SINI", a0_name="A0", b0_name="B0",
            dr_name="DR", dth_name="DTH",
        )

        om_rad = dd_params["OM"]
        t0_int = np.floor(dd_params["T0"])
        t0_frac = dd_params["T0"] - t0_int

        param_names = ("PB", "T0", "A1", "ECC", "OM", "OMDOT", "GAMMA", "PBDOT",
                        "M2", "SINI", "A0", "B0", "DR", "DTH")
        param_values = [dd_params["PB"], t0_frac, dd_params["A1"], dd_params["ECC"],
                        om_rad, dd_params["OMDOT"], dd_params["GAMMA"], dd_params["PBDOT"],
                        dd_params["M2"], dd_params["SINI"], dd_params["A0"], dd_params["B0"],
                        dd_params["DR"], dd_params["DTH"]]
        params = _make_params(param_names, param_values, epoch_int_values={"T0": t0_int})

        toa_data = _make_toa_data(np.linspace(54000.5, 54200.0, 500))
        jax_delay = np.array(dd(toa_data, params, jnp.zeros(500)))

        npt.assert_allclose(jax_delay, pint_delay, atol=1e-12, rtol=1e-12)

    def test_dd_no_shapiro(self, dd_params):
        """DD without Shapiro (M2=SINI=0) should match inverse + aberration only."""
        pytest.importorskip("pint")
        import astropy.units as u
        from pint.models.stand_alone_psr_binaries.DD_model import DDmodel

        from jaxpint.binary.dd import BinaryDD

        bm = DDmodel()
        pint_params = {
            "PB": dd_params["PB"] * u.day,
            "T0": np.longdouble(dd_params["T0"]) * u.day,
            "A1": dd_params["A1"] * u.lightsecond,
            "ECC": dd_params["ECC"] * u.Unit(""),
            "OM": dd_params["OM_deg"] * u.deg,
            "OMDOT": dd_params["OMDOT_deg_yr"] * u.deg / u.year,
            "GAMMA": dd_params["GAMMA"] * u.second,
            "PBDOT": dd_params["PBDOT"] * u.Unit(""),
        }
        t = np.linspace(54001.0, 54100.0, 200) * u.day
        bm.update_input(barycentric_toa=t, **pint_params)
        pint_delay = bm.DDdelay().to(u.second).value

        dd = BinaryDD(
            pb_name="PB", t0_name="T0", a1_name="A1", ecc_name="ECC", om_name="OM",
            omdot_name="OMDOT", gamma_name="GAMMA", pbdot_name="PBDOT",
        )

        om_rad = dd_params["OM"]
        t0_int = np.floor(dd_params["T0"])
        t0_frac = dd_params["T0"] - t0_int

        param_names = ("PB", "T0", "A1", "ECC", "OM", "OMDOT", "GAMMA", "PBDOT")
        param_values = [dd_params["PB"], t0_frac, dd_params["A1"], dd_params["ECC"],
                        om_rad, dd_params["OMDOT"], dd_params["GAMMA"], dd_params["PBDOT"]]
        params = _make_params(param_names, param_values, epoch_int_values={"T0": t0_int})

        toa_data = _make_toa_data(np.linspace(54001.0, 54100.0, 200))
        jax_delay = np.array(dd(toa_data, params, jnp.zeros(200)))

        npt.assert_allclose(jax_delay, pint_delay, atol=1e-12, rtol=1e-12)

    def test_dds_delay_matches_pint(self, dd_params):
        """DDS delay should match PINT to float64 precision."""
        pytest.importorskip("pint")
        import astropy.units as u
        from pint.models.stand_alone_psr_binaries.DDS_model import DDSmodel

        from jaxpint.binary.dds import BinaryDDS

        sini = dd_params["SINI"]
        shapmax = -np.log(1 - sini)

        bm = DDSmodel()
        pint_params = {
            "PB": dd_params["PB"] * u.day,
            "T0": np.longdouble(dd_params["T0"]) * u.day,
            "A1": dd_params["A1"] * u.lightsecond,
            "ECC": dd_params["ECC"] * u.Unit(""),
            "OM": dd_params["OM_deg"] * u.deg,
            "OMDOT": dd_params["OMDOT_deg_yr"] * u.deg / u.year,
            "GAMMA": dd_params["GAMMA"] * u.second,
            "PBDOT": dd_params["PBDOT"] * u.Unit(""),
            "M2": dd_params["M2"] * u.M_sun,
            "SHAPMAX": shapmax,
        }
        t = np.linspace(54000.5, 54200.0, 300) * u.day
        bm.update_input(barycentric_toa=t, **pint_params)
        pint_delay = bm.DDdelay().to(u.second).value

        dds = BinaryDDS(
            pb_name="PB", t0_name="T0", a1_name="A1", ecc_name="ECC", om_name="OM",
            omdot_name="OMDOT", gamma_name="GAMMA", pbdot_name="PBDOT",
            m2_name="M2", shapmax_name="SHAPMAX",
        )

        om_rad = dd_params["OM"]
        t0_int = np.floor(dd_params["T0"])
        t0_frac = dd_params["T0"] - t0_int

        param_names = ("PB", "T0", "A1", "ECC", "OM", "OMDOT", "GAMMA", "PBDOT",
                        "M2", "SHAPMAX")
        param_values = [dd_params["PB"], t0_frac, dd_params["A1"], dd_params["ECC"],
                        om_rad, dd_params["OMDOT"], dd_params["GAMMA"], dd_params["PBDOT"],
                        dd_params["M2"], shapmax]
        params = _make_params(param_names, param_values, epoch_int_values={"T0": t0_int})

        toa_data = _make_toa_data(np.linspace(54000.5, 54200.0, 300))
        jax_delay = np.array(dds(toa_data, params, jnp.zeros(300)))

        npt.assert_allclose(jax_delay, pint_delay, atol=1e-12, rtol=1e-12)

    def test_dd_jit(self, dd_params):
        """BinaryDD should be JIT-compilable."""
        from jaxpint.binary.dd import BinaryDD

        dd = BinaryDD(
            pb_name="PB", t0_name="T0", a1_name="A1", ecc_name="ECC", om_name="OM",
            m2_name="M2", sini_name="SINI",
        )

        om_rad = dd_params["OM"]
        t0_int = np.floor(dd_params["T0"])
        t0_frac = dd_params["T0"] - t0_int

        param_names = ("PB", "T0", "A1", "ECC", "OM", "M2", "SINI")
        param_values = [dd_params["PB"], t0_frac, dd_params["A1"], dd_params["ECC"],
                        om_rad, dd_params["M2"], dd_params["SINI"]]
        params = _make_params(param_names, param_values, epoch_int_values={"T0": t0_int})

        n = 10
        toa_data = _make_toa_data(np.linspace(54100.1, 54100.9, n))

        jitted = jax.jit(dd)
        result = jitted(toa_data, params, jnp.zeros(n))
        assert result.shape == (n,)
        assert jnp.all(jnp.isfinite(result))

    def test_dd_autodiff(self, dd_params):
        """BinaryDD should be differentiable via JAX autodiff."""
        from jaxpint.binary.dd import BinaryDD

        dd = BinaryDD(
            pb_name="PB", t0_name="T0", a1_name="A1", ecc_name="ECC", om_name="OM",
            gamma_name="GAMMA", m2_name="M2", sini_name="SINI",
        )

        om_rad = dd_params["OM"]
        t0_int = np.floor(dd_params["T0"])
        t0_frac = dd_params["T0"] - t0_int

        param_names = ("PB", "T0", "A1", "ECC", "OM", "GAMMA", "M2", "SINI")
        param_values = [dd_params["PB"], t0_frac, dd_params["A1"], dd_params["ECC"],
                        om_rad, dd_params["GAMMA"], dd_params["M2"], dd_params["SINI"]]
        params = _make_params(param_names, param_values, epoch_int_values={"T0": t0_int})

        n = 10
        toa_data = _make_toa_data(np.linspace(54100.1, 54100.9, n))

        def delay_fn(param_values):
            p = params.with_free_values(param_values)
            return dd(toa_data, p, jnp.zeros(n))

        J = jax.jacobian(delay_fn)(params.free_values())
        assert J.shape == (n, len(param_names))
        assert jnp.all(jnp.isfinite(J))
        # A1 column should be nonzero
        a1_col = list(param_names).index("A1")
        assert jnp.any(jnp.abs(J[:, a1_col]) > 0)
