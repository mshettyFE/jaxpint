"""Tests for BinaryDDGR delay model against PINT."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest


from tests.helpers import make_toa_data as _make_toa_data_base, make_params


def _make_params_ddgr(param_names, param_values, epoch_int_values=None):
    return make_params(param_names, param_values, components="BinaryDDGR",
                       epoch_int_values=epoch_int_values or {})


def _make_toa_data(t_mjd):
    return _make_toa_data_base(
        t_mjd=t_mjd,
        tzr_tdb_int=jnp.array(54000.0), tzr_tdb_frac=jnp.array(0.5),
        tzr_freq=jnp.array(jnp.inf), tzr_ssb_obs_pos=jnp.zeros(3),
    )


@pytest.fixture
def ddgr_params():
    """DDGR parameters: typical double neutron star system."""
    return {
        "PB": 0.5,            # days
        "T0": 54000.0,        # MJD
        "A1": 3.0,            # light-seconds
        "ECC": 0.2,
        "OM_deg": 120.0,
        "OM": 120.0 * np.pi / 180.0,
        "MTOT": 2.5,          # solar masses
        "M2": 1.1,            # solar masses
    }


class TestBinaryDDGRvsPINT:
    """Compare JaxPINT BinaryDDGR against PINT's standalone DDGRmodel."""

    def test_ddgr_pk_parameters(self, ddgr_params):
        """Verify GR-derived PK parameters match PINT."""
        pytest.importorskip("pint")
        import astropy.units as u
        from pint.models.stand_alone_psr_binaries.DDGR_model import DDGRmodel

        from jaxpint.binary.ddgr import (
            BinaryDDGR, _solve_relativistic_kepler, _GM_SUN_SI,
        )
        from jaxpint.constants import C_M_PER_S, SECS_PER_DAY, TSUN

        # --- PINT DDGR model ---
        bm = DDGRmodel()
        pint_params = {
            "PB": ddgr_params["PB"] * u.day,
            "T0": np.longdouble(ddgr_params["T0"]) * u.day,
            "A1": ddgr_params["A1"] * u.lightsecond,
            "ECC": ddgr_params["ECC"] * u.Unit(""),
            "OM": ddgr_params["OM_deg"] * u.deg,
            "MTOT": ddgr_params["MTOT"] * u.Msun,
            "M2": ddgr_params["M2"] * u.Msun,
        }
        t = np.linspace(54000.5, 54200.0, 10) * u.day
        bm.update_input(barycentric_toa=t, **pint_params)

        # Extract PINT PK values (SINI is an array over TOAs, take first)
        pint_sini = float(np.asarray(bm.SINI.decompose().value).flat[0])
        pint_gamma = float(np.asarray(bm.GAMMA.to(u.s).value).flat[0])
        pint_pbdot = float(np.asarray(bm.PBDOT.decompose().value).flat[0])
        pint_dr = float(np.asarray(bm.DR.decompose().value).flat[0])
        pint_dth = float(np.asarray(bm.DTH.decompose().value).flat[0])
        pint_k = float(np.asarray(bm.k.decompose().value).flat[0])

        # --- JaxPINT PK derivation ---
        mtot = ddgr_params["MTOT"]
        m2 = ddgr_params["M2"]
        m1 = mtot - m2
        pb_s = ddgr_params["PB"] * SECS_PER_DAY
        n = 2.0 * np.pi / pb_s
        gm_tot = mtot * _GM_SUN_SI
        c = C_M_PER_S
        c2 = c ** 2

        arr0, arr = _solve_relativistic_kepler(
            jnp.float64(mtot), jnp.float64(m1), jnp.float64(m2), jnp.float64(n)
        )
        arr0, arr = float(arr0), float(arr)

        ar = arr * m2 / mtot
        jax_sini = ddgr_params["A1"] * c / ar
        jax_gamma = ddgr_params["ECC"] * gm_tot * m2 * (m1 + 2.0 * m2) / (n * c2 * arr0 * mtot ** 2)
        jax_k = 3.0 * gm_tot / (c2 * arr0 * (1.0 - ddgr_params["ECC"] ** 2))

        gr_factor = _GM_SUN_SI / (c2 * mtot * arr)
        jax_dr = gr_factor * (3.0 * m1 ** 2 + 6.0 * m1 * m2 + 2.0 * m2 ** 2)
        jax_dth = gr_factor * (3.5 * m1 ** 2 + 6.0 * m1 * m2 + 2.0 * m2 ** 2)

        npt.assert_allclose(jax_sini, pint_sini, rtol=1e-8,
                            err_msg="SINI mismatch")
        npt.assert_allclose(jax_gamma, pint_gamma, rtol=1e-8,
                            err_msg="GAMMA mismatch")
        npt.assert_allclose(jax_k, pint_k, rtol=1e-8,
                            err_msg="k mismatch")
        npt.assert_allclose(jax_dr, pint_dr, rtol=1e-8,
                            err_msg="DR mismatch")
        npt.assert_allclose(jax_dth, pint_dth, rtol=1e-8,
                            err_msg="DTH mismatch")

    def test_ddgr_delay_matches_pint(self, ddgr_params):
        """Full DDGR delay should match PINT."""
        pytest.importorskip("pint")
        import astropy.units as u
        from pint.models.stand_alone_psr_binaries.DDGR_model import DDGRmodel

        from jaxpint.binary.ddgr import BinaryDDGR

        # --- PINT ---
        bm = DDGRmodel()
        pint_params = {
            "PB": ddgr_params["PB"] * u.day,
            "T0": np.longdouble(ddgr_params["T0"]) * u.day,
            "A1": ddgr_params["A1"] * u.lightsecond,
            "ECC": ddgr_params["ECC"] * u.Unit(""),
            "OM": ddgr_params["OM_deg"] * u.deg,
            "MTOT": ddgr_params["MTOT"] * u.Msun,
            "M2": ddgr_params["M2"] * u.Msun,
        }
        t = np.linspace(54000.5, 54200.0, 500) * u.day
        bm.update_input(barycentric_toa=t, **pint_params)
        pint_delay = bm.DDdelay().to(u.second).value

        # --- JaxPINT ---
        ddgr = BinaryDDGR(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            mtot_name="MTOT", m2_name="M2",
        )

        t0_int = np.floor(ddgr_params["T0"])
        t0_frac = ddgr_params["T0"] - t0_int

        param_names = ("PB", "T0", "A1", "ECC", "OM", "MTOT", "M2")
        param_values = [
            ddgr_params["PB"], t0_frac, ddgr_params["A1"],
            ddgr_params["ECC"], ddgr_params["OM"],
            ddgr_params["MTOT"], ddgr_params["M2"],
        ]
        params = _make_params_ddgr(
            param_names, param_values,
            epoch_int_values={"T0": t0_int},
        )
        toa_data = _make_toa_data(np.linspace(54000.5, 54200.0, 500))
        jax_delay = np.array(ddgr(toa_data, params, jnp.zeros(500)))

        npt.assert_allclose(jax_delay, pint_delay, atol=1e-11, rtol=1e-11)

    def test_ddgr_matches_dd_with_same_pk(self, ddgr_params):
        """DDGR delay should equal DD delay when given equivalent PK values."""
        pytest.importorskip("pint")
        import astropy.units as u
        from pint.models.stand_alone_psr_binaries.DDGR_model import DDGRmodel

        from jaxpint.binary.ddgr import BinaryDDGR
        from jaxpint.binary.dd import BinaryDD

        _DEG_YR_TO_RAD_S = np.pi / 180.0 / (365.25 * 86400.0)

        # Get PK values from PINT DDGR
        bm = DDGRmodel()
        pint_params = {
            "PB": ddgr_params["PB"] * u.day,
            "T0": np.longdouble(ddgr_params["T0"]) * u.day,
            "A1": ddgr_params["A1"] * u.lightsecond,
            "ECC": ddgr_params["ECC"] * u.Unit(""),
            "OM": ddgr_params["OM_deg"] * u.deg,
            "MTOT": ddgr_params["MTOT"] * u.Msun,
            "M2": ddgr_params["M2"] * u.Msun,
        }
        t = np.linspace(54000.5, 54100.0, 200) * u.day
        bm.update_input(barycentric_toa=t, **pint_params)

        pint_sini = float(np.asarray(bm.SINI.decompose().value).flat[0])
        pint_gamma = float(np.asarray(bm.GAMMA.to(u.s).value).flat[0])
        pint_pbdot = float(np.asarray(bm.PBDOT.decompose().value).flat[0])
        pint_dr = float(np.asarray(bm.DR.decompose().value).flat[0])
        pint_dth = float(np.asarray(bm.DTH.decompose().value).flat[0])
        pint_k = float(np.asarray(bm.k.decompose().value).flat[0])
        # OMDOT = k * n, convert to rad/s
        pb_s = ddgr_params["PB"] * 86400.0
        n_val = 2.0 * np.pi / pb_s
        pint_omdot_rad_s = float(pint_k * n_val)

        t0_int = np.floor(ddgr_params["T0"])
        t0_frac = ddgr_params["T0"] - t0_int
        toa_data = _make_toa_data(np.linspace(54000.5, 54100.0, 200))
        n_toas = 200

        # --- DDGR delay ---
        ddgr = BinaryDDGR(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            mtot_name="MTOT", m2_name="M2",
        )
        ddgr_param_names = ("PB", "T0", "A1", "ECC", "OM", "MTOT", "M2")
        ddgr_param_values = [
            ddgr_params["PB"], t0_frac, ddgr_params["A1"],
            ddgr_params["ECC"], ddgr_params["OM"],
            ddgr_params["MTOT"], ddgr_params["M2"],
        ]
        ddgr_p = _make_params_ddgr(
            ddgr_param_names, ddgr_param_values,
            epoch_int_values={"T0": t0_int},
        )
        delay_ddgr = np.array(ddgr(toa_data, ddgr_p, jnp.zeros(n_toas)))

        # --- DD delay with same PK values ---
        dd = BinaryDD(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            omdot_name="OMDOT", gamma_name="GAMMA",
            pbdot_name="PBDOT", dr_name="DR", dth_name="DTH",
            m2_name="M2", sini_name="SINI",
        )
        dd_param_names = (
            "PB", "T0", "A1", "ECC", "OM",
            "OMDOT", "GAMMA", "PBDOT", "DR", "DTH", "M2", "SINI",
        )
        dd_param_values = [
            ddgr_params["PB"], t0_frac, ddgr_params["A1"],
            ddgr_params["ECC"], ddgr_params["OM"],
            pint_omdot_rad_s, pint_gamma, pint_pbdot,
            pint_dr, pint_dth, ddgr_params["M2"], pint_sini,
        ]
        dd_p = make_params(
            dd_param_names, dd_param_values,
            components="BinaryDD",
            epoch_int_values={"T0": t0_int},
        )
        delay_dd = np.array(dd(toa_data, dd_p, jnp.zeros(n_toas)))

        npt.assert_allclose(delay_ddgr, delay_dd, atol=1e-11, rtol=1e-11)

    def test_ddgr_xomdot(self, ddgr_params):
        """XOMDOT should produce a different delay than without it."""
        from jaxpint.binary.ddgr import BinaryDDGR

        _DEG_YR_TO_RAD_S = np.pi / 180.0 / (365.25 * 86400.0)

        t0_int = np.floor(ddgr_params["T0"])
        t0_frac = ddgr_params["T0"] - t0_int
        toa_data = _make_toa_data(np.linspace(54000.5, 54200.0, 100))

        base_names = ("PB", "T0", "A1", "ECC", "OM", "MTOT", "M2")
        base_values = [
            ddgr_params["PB"], t0_frac, ddgr_params["A1"],
            ddgr_params["ECC"], ddgr_params["OM"],
            ddgr_params["MTOT"], ddgr_params["M2"],
        ]

        # Without XOMDOT
        ddgr_no = BinaryDDGR(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            mtot_name="MTOT", m2_name="M2",
        )
        params_no = _make_params_ddgr(base_names, base_values, {"T0": t0_int})
        d_no = np.array(ddgr_no(toa_data, params_no, jnp.zeros(100)))

        # With XOMDOT
        ddgr_x = BinaryDDGR(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            mtot_name="MTOT", m2_name="M2",
            xomdot_name="XOMDOT",
        )
        xomdot_rad_s = 0.5 * _DEG_YR_TO_RAD_S
        params_x = _make_params_ddgr(
            base_names + ("XOMDOT",), base_values + [xomdot_rad_s],
            {"T0": t0_int},
        )
        d_x = np.array(ddgr_x(toa_data, params_x, jnp.zeros(100)))

        assert not np.allclose(d_no, d_x), "XOMDOT should change the delay"

    def test_ddgr_jit(self, ddgr_params):
        """BinaryDDGR should be JIT-compilable."""
        from jaxpint.binary.ddgr import BinaryDDGR

        ddgr = BinaryDDGR(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            mtot_name="MTOT", m2_name="M2",
        )

        t0_int = np.floor(ddgr_params["T0"])
        t0_frac = ddgr_params["T0"] - t0_int

        param_names = ("PB", "T0", "A1", "ECC", "OM", "MTOT", "M2")
        param_values = [
            ddgr_params["PB"], t0_frac, ddgr_params["A1"],
            ddgr_params["ECC"], ddgr_params["OM"],
            ddgr_params["MTOT"], ddgr_params["M2"],
        ]
        params = _make_params_ddgr(param_names, param_values, {"T0": t0_int})
        n = 10
        toa_data = _make_toa_data(np.linspace(54100.1, 54100.9, n))

        jitted = jax.jit(ddgr)
        result = jitted(toa_data, params, jnp.zeros(n))
        assert result.shape == (n,)
        assert jnp.all(jnp.isfinite(result))

    def test_ddgr_autodiff(self, ddgr_params):
        """BinaryDDGR should be differentiable via JAX autodiff."""
        from jaxpint.binary.ddgr import BinaryDDGR

        ddgr = BinaryDDGR(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            mtot_name="MTOT", m2_name="M2",
        )

        t0_int = np.floor(ddgr_params["T0"])
        t0_frac = ddgr_params["T0"] - t0_int

        param_names = ("PB", "T0", "A1", "ECC", "OM", "MTOT", "M2")
        param_values = [
            ddgr_params["PB"], t0_frac, ddgr_params["A1"],
            ddgr_params["ECC"], ddgr_params["OM"],
            ddgr_params["MTOT"], ddgr_params["M2"],
        ]
        params = _make_params_ddgr(param_names, param_values, {"T0": t0_int})
        n = 10
        toa_data = _make_toa_data(np.linspace(54100.1, 54100.9, n))

        def delay_fn(param_values):
            p = params.with_free_values(param_values)
            return ddgr(toa_data, p, jnp.zeros(n))

        J = jax.jacobian(delay_fn)(params.free_values())
        assert J.shape == (n, len(param_names))
        assert jnp.all(jnp.isfinite(J))
        # MTOT and M2 columns should be nonzero
        mtot_col = list(param_names).index("MTOT")
        m2_col = list(param_names).index("M2")
        assert jnp.any(jnp.abs(J[:, mtot_col]) > 0)
        assert jnp.any(jnp.abs(J[:, m2_col]) > 0)
