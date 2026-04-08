"""Tests for BinaryBTPiecewise delay model."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest


from tests.helpers import make_binary_toa_data, make_params


@pytest.fixture
def bt_params():
    """Base BT parameters."""
    return {
        "PB": 0.2,          # days
        "T0": 55000.0,      # MJD
        "A1": 0.343,        # light-seconds
        "ECC": 0.01,
        "OM_deg": 60.0,
        "OM": 60.0 * np.pi / 180.0,  # radians
    }


class TestBinaryBTPiecewise:

    def test_no_pieces_matches_bt(self, bt_params):
        """With no pieces, BTPiecewise should exactly equal BT."""
        from jaxpint.binary.bt import BinaryBT
        from jaxpint.binary.bt_piecewise import BinaryBTPiecewise

        t0_int = np.floor(bt_params["T0"])
        t0_frac = bt_params["T0"] - t0_int

        param_names = ("PB", "T0", "A1", "ECC", "OM")
        param_values = [bt_params["PB"], t0_frac, bt_params["A1"],
                        bt_params["ECC"], bt_params["OM"]]

        p = make_params(param_names, param_values, components="BinaryBT",
                        epoch_int_values={"T0": t0_int})

        t_mjd = np.linspace(55000.5, 55200.0, 100)
        toa_data = make_binary_toa_data(t_mjd, tzr_tdb_int=55000.0)
        n = len(t_mjd)

        bt = BinaryBT(pb_name="PB", t0_name="T0", a1_name="A1",
                       ecc_name="ECC", om_name="OM")
        bt_pw = BinaryBTPiecewise(pb_name="PB", t0_name="T0", a1_name="A1",
                                   ecc_name="ECC", om_name="OM", n_pieces=0)

        d_bt = np.array(bt(toa_data, p, jnp.zeros(n)))
        d_pw = np.array(bt_pw(toa_data, p, jnp.zeros(n)))

        npt.assert_allclose(d_pw, d_bt, atol=1e-15)

    def test_one_a1_piece(self, bt_params):
        """Single A1X piece: in-piece TOAs use A1X, others use global A1."""
        from jaxpint.binary.bt_piecewise import BinaryBTPiecewise

        t0_int = np.floor(bt_params["T0"])
        t0_frac = bt_params["T0"] - t0_int

        a1_global = bt_params["A1"]
        a1x = a1_global + 0.001  # slightly different

        param_names = ("PB", "T0", "A1", "ECC", "OM", "A1X_0000", "XR1_0000", "XR2_0000")
        param_values = [
            bt_params["PB"], t0_frac, a1_global, bt_params["ECC"], bt_params["OM"],
            a1x, 55000.0, 55100.0,
        ]
        p = make_params(param_names, param_values, components="BinaryBTPiecewise",
                        epoch_int_values={"T0": t0_int})

        bt_pw = BinaryBTPiecewise(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            n_pieces=1,
            t0x_names=(), a1x_names=("A1X_0000",),
            xr1_names=("XR1_0000",), xr2_names=("XR2_0000",),
        )

        # TOAs: half in piece, half outside
        t_in = np.linspace(55010.0, 55090.0, 10)
        t_out = np.linspace(55110.0, 55190.0, 10)
        t_all = np.concatenate([t_in, t_out])
        toa_data = make_binary_toa_data(t_all, tzr_tdb_int=55000.0)
        d = np.array(bt_pw(toa_data, p, jnp.zeros(20)))

        # With global A1 everywhere
        from jaxpint.binary.bt import BinaryBT
        bt_global = BinaryBT(pb_name="PB", t0_name="T0", a1_name="A1",
                              ecc_name="ECC", om_name="OM")
        p_global = make_params(
            ("PB", "T0", "A1", "ECC", "OM"),
            [bt_params["PB"], t0_frac, a1_global, bt_params["ECC"], bt_params["OM"]],
            components="BinaryBT", epoch_int_values={"T0": t0_int},
        )
        d_global = np.array(bt_global(toa_data, p_global, jnp.zeros(20)))

        # Out-of-piece TOAs should match global
        npt.assert_allclose(d[10:], d_global[10:], atol=1e-15)
        # In-piece TOAs should differ (different A1)
        assert not np.allclose(d[:10], d_global[:10]), "In-piece delays should differ"

    def test_one_t0_piece(self, bt_params):
        """Single T0X piece changes the delay for in-piece TOAs."""
        from jaxpint.binary.bt_piecewise import BinaryBTPiecewise

        t0_int = np.floor(bt_params["T0"])
        t0_frac = bt_params["T0"] - t0_int
        t0x = bt_params["T0"] + 0.00001  # slightly shifted
        t0x_int = np.floor(t0x)
        t0x_frac = t0x - t0x_int

        param_names = ("PB", "T0", "A1", "ECC", "OM", "T0X_0000", "XR1_0000", "XR2_0000")
        param_values = [
            bt_params["PB"], t0_frac, bt_params["A1"], bt_params["ECC"], bt_params["OM"],
            t0x_frac, 55000.0, 55100.0,
        ]
        p = make_params(param_names, param_values, components="BinaryBTPiecewise",
                        epoch_int_values={"T0": t0_int, "T0X_0000": t0x_int})

        bt_pw = BinaryBTPiecewise(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            n_pieces=1,
            t0x_names=("T0X_0000",), a1x_names=(),
            xr1_names=("XR1_0000",), xr2_names=("XR2_0000",),
        )

        t_in = np.linspace(55010.0, 55090.0, 10)
        t_out = np.linspace(55110.0, 55190.0, 10)
        t_all = np.concatenate([t_in, t_out])
        toa_data = make_binary_toa_data(t_all, tzr_tdb_int=55000.0)
        d = np.array(bt_pw(toa_data, p, jnp.zeros(20)))

        # With global T0 only
        from jaxpint.binary.bt import BinaryBT
        bt_global = BinaryBT(pb_name="PB", t0_name="T0", a1_name="A1",
                              ecc_name="ECC", om_name="OM")
        p_global = make_params(
            ("PB", "T0", "A1", "ECC", "OM"),
            [bt_params["PB"], t0_frac, bt_params["A1"], bt_params["ECC"], bt_params["OM"]],
            components="BinaryBT", epoch_int_values={"T0": t0_int},
        )
        d_global = np.array(bt_global(toa_data, p_global, jnp.zeros(20)))

        # Out-of-piece should match global
        npt.assert_allclose(d[10:], d_global[10:], atol=1e-15)
        # In-piece should differ
        assert not np.allclose(d[:10], d_global[:10]), "In-piece delays should differ"

    def test_two_pieces(self, bt_params):
        """Two non-overlapping pieces with different A1X values."""
        from jaxpint.binary.bt_piecewise import BinaryBTPiecewise

        t0_int = np.floor(bt_params["T0"])
        t0_frac = bt_params["T0"] - t0_int

        a1_global = bt_params["A1"]
        a1x_0 = a1_global + 0.001
        a1x_1 = a1_global - 0.001

        param_names = (
            "PB", "T0", "A1", "ECC", "OM",
            "A1X_0000", "XR1_0000", "XR2_0000",
            "A1X_0001", "XR1_0001", "XR2_0001",
        )
        param_values = [
            bt_params["PB"], t0_frac, a1_global, bt_params["ECC"], bt_params["OM"],
            a1x_0, 55000.0, 55100.0,
            a1x_1, 55100.0, 55200.0,
        ]
        p = make_params(param_names, param_values, components="BinaryBTPiecewise",
                        epoch_int_values={"T0": t0_int})

        bt_pw = BinaryBTPiecewise(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            n_pieces=2,
            t0x_names=(), a1x_names=("A1X_0000", "A1X_0001"),
            xr1_names=("XR1_0000", "XR1_0001"),
            xr2_names=("XR2_0000", "XR2_0001"),
        )

        # 10 TOAs in each piece
        t_piece0 = np.linspace(55010.0, 55090.0, 10)
        t_piece1 = np.linspace(55110.0, 55190.0, 10)
        t_all = np.concatenate([t_piece0, t_piece1])
        toa_data = make_binary_toa_data(t_all, tzr_tdb_int=55000.0)
        d = np.array(bt_pw(toa_data, p, jnp.zeros(20)))

        # Delays in piece 0 and piece 1 should differ (different A1X)
        assert not np.allclose(d[:10], d[10:]), "Pieces should have different delays"

    @pytest.mark.slow
    def test_piecewise_matches_pint_regular_bt(self, bt_params):
        """Each piece should match PINT's regular BT with piece parameters.

        We verify that in-piece TOAs match PINT's regular BT model run with
        T0=T0X and A1=A1X, and out-of-piece TOAs match PINT's BT with
        global T0/A1.  This validates correctness per-piece.
        """
        pytest.importorskip("pint")
        import astropy.units as u
        from pint.models.stand_alone_psr_binaries.BT_model import BTmodel

        from jaxpint.binary.bt_piecewise import BinaryBTPiecewise

        a1_global = bt_params["A1"]
        a1x = a1_global + 0.001
        t0x = bt_params["T0"] + 0.00001

        t0_int = np.floor(bt_params["T0"])
        t0_frac = bt_params["T0"] - t0_int
        t0x_int = np.floor(t0x)
        t0x_frac = t0x - t0x_int

        t_in = np.linspace(55010.0, 55090.0, 10)
        t_out = np.linspace(55110.0, 55190.0, 10)
        t_all = np.concatenate([t_in, t_out])

        # --- JaxPINT Piecewise ---
        bt_pw = BinaryBTPiecewise(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            n_pieces=1,
            t0x_names=("T0X_0000",), a1x_names=("A1X_0000",),
            xr1_names=("XR1_0000",), xr2_names=("XR2_0000",),
        )
        param_names = ("PB", "T0", "A1", "ECC", "OM",
                        "T0X_0000", "A1X_0000", "XR1_0000", "XR2_0000")
        param_values = [
            bt_params["PB"], t0_frac, a1_global, bt_params["ECC"], bt_params["OM"],
            t0x_frac, a1x, 55000.0, 55100.0,
        ]
        p = make_params(param_names, param_values, components="BinaryBTPiecewise",
                        epoch_int_values={"T0": t0_int, "T0X_0000": t0x_int})
        toa_data = make_binary_toa_data(t_all, tzr_tdb_int=55000.0)
        jax_delay = np.array(bt_pw(toa_data, p, jnp.zeros(20)))

        # --- PINT reference for in-piece TOAs (T0=T0X, A1=A1X) ---
        bm_in = BTmodel()
        bm_in.update_input(
            barycentric_toa=t_in * u.day,
            PB=bt_params["PB"] * u.day,
            T0=np.longdouble(t0x) * u.day,
            A1=a1x * u.lightsecond,
            ECC=bt_params["ECC"] * u.Unit(""),
            OM=bt_params["OM_deg"] * u.deg,
        )
        pint_in = bm_in.BTdelay().to(u.second).value

        # --- PINT reference for out-of-piece TOAs (global T0, A1) ---
        bm_out = BTmodel()
        bm_out.update_input(
            barycentric_toa=t_out * u.day,
            PB=bt_params["PB"] * u.day,
            T0=np.longdouble(bt_params["T0"]) * u.day,
            A1=a1_global * u.lightsecond,
            ECC=bt_params["ECC"] * u.Unit(""),
            OM=bt_params["OM_deg"] * u.deg,
        )
        pint_out = bm_out.BTdelay().to(u.second).value

        npt.assert_allclose(jax_delay[:10], pint_in, atol=1e-12, rtol=1e-12,
                            err_msg="In-piece delays should match PINT BT with T0X/A1X")
        npt.assert_allclose(jax_delay[10:], pint_out, atol=1e-12, rtol=1e-12,
                            err_msg="Out-of-piece delays should match PINT BT with global T0/A1")

    def test_piecewise_jit(self, bt_params):
        """BinaryBTPiecewise should be JIT-compilable."""
        from jaxpint.binary.bt_piecewise import BinaryBTPiecewise

        t0_int = np.floor(bt_params["T0"])
        t0_frac = bt_params["T0"] - t0_int

        bt_pw = BinaryBTPiecewise(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            n_pieces=1,
            t0x_names=(), a1x_names=("A1X_0000",),
            xr1_names=("XR1_0000",), xr2_names=("XR2_0000",),
        )

        param_names = ("PB", "T0", "A1", "ECC", "OM",
                        "A1X_0000", "XR1_0000", "XR2_0000")
        param_values = [
            bt_params["PB"], t0_frac, bt_params["A1"], bt_params["ECC"], bt_params["OM"],
            bt_params["A1"] + 0.001, 55000.0, 55100.0,
        ]
        p = make_params(param_names, param_values, components="BinaryBTPiecewise",
                        epoch_int_values={"T0": t0_int})

        n = 10
        toa_data = make_binary_toa_data(np.linspace(55010.0, 55090.0, n), tzr_tdb_int=55000.0)

        jitted = jax.jit(bt_pw)
        result = jitted(toa_data, p, jnp.zeros(n))
        assert result.shape == (n,)
        assert jnp.all(jnp.isfinite(result))

    def test_piecewise_autodiff(self, bt_params):
        """Jacobian w.r.t. A1X should be nonzero only for in-piece TOAs."""
        from jaxpint.binary.bt_piecewise import BinaryBTPiecewise

        t0_int = np.floor(bt_params["T0"])
        t0_frac = bt_params["T0"] - t0_int

        bt_pw = BinaryBTPiecewise(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            n_pieces=1,
            t0x_names=(), a1x_names=("A1X_0000",),
            xr1_names=("XR1_0000",), xr2_names=("XR2_0000",),
        )

        param_names = ("PB", "T0", "A1", "ECC", "OM",
                        "A1X_0000", "XR1_0000", "XR2_0000")
        param_values = [
            bt_params["PB"], t0_frac, bt_params["A1"], bt_params["ECC"], bt_params["OM"],
            bt_params["A1"] + 0.001, 55000.0, 55100.0,
        ]
        p = make_params(param_names, param_values, components="BinaryBTPiecewise",
                        epoch_int_values={"T0": t0_int})

        # 5 in-piece, 5 out-of-piece
        t_in = np.linspace(55010.0, 55090.0, 5)
        t_out = np.linspace(55110.0, 55190.0, 5)
        t_all = np.concatenate([t_in, t_out])
        toa_data = make_binary_toa_data(t_all, tzr_tdb_int=55000.0)
        n = 10

        def delay_fn(param_values):
            pp = p.with_free_values(param_values)
            return bt_pw(toa_data, pp, jnp.zeros(n))

        J = jax.jacobian(delay_fn)(p.free_values())
        assert J.shape == (n, len(param_names))
        assert jnp.all(jnp.isfinite(J))

        # A1X_0000 column: nonzero for in-piece (first 5), zero for out-of-piece (last 5)
        a1x_col = list(param_names).index("A1X_0000")
        assert jnp.any(jnp.abs(J[:5, a1x_col]) > 0), "A1X should affect in-piece TOAs"
        npt.assert_allclose(J[5:, a1x_col], 0.0, atol=1e-15,
                            err_msg="A1X should not affect out-of-piece TOAs")
