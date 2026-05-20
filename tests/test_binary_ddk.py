"""Tests for BinaryDDK delay model against PINT."""

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest


from tests.helpers import make_toa_data as _make_toa_data_base, make_params

_DEG_YR_TO_RAD_S = np.pi / 180.0 / (365.25 * 86400.0)


@pytest.fixture
def ddk_params():
    """J1713+0747-like DDK parameters."""
    return {
        "PB": 67.825,             # days
        "T0": 54187.0,            # MJD
        "A1": 32.342,             # light-seconds
        "ECC": 0.0000749,
        "OM_deg": 176.2,
        "OM": 176.2 * np.pi / 180.0,
        "KIN_deg": 72.0,
        "KIN": 72.0 * np.pi / 180.0,
        "KOM_deg": 89.0,
        "KOM": 89.0 * np.pi / 180.0,
        "PX": 0.8,               # mas
        "M2": 0.29,              # solar masses
        "GAMMA": 0.0,
        "RAJ_deg": 258.48,       # degrees (17h 13m ~= 258.48 deg)
        "RAJ": 258.48 * np.pi / 180.0,
        "DECJ_deg": 7.79,        # degrees
        "DECJ": 7.79 * np.pi / 180.0,
        "PMRA": 4.917,           # mas/yr
        "PMDEC": -3.937,         # mas/yr
    }


class TestBinaryDDKvsPINT:
    """Compare JaxPINT BinaryDDK against PINT's standalone DDKmodel."""

    def _setup_pint_ddk(self, ddk_params, t_mjd, k96=True):
        """Set up PINT DDK model and return delay + obs_pos."""
        import astropy.units as u
        from pint.models.stand_alone_psr_binaries.DDK_model import DDKmodel

        bm = DDKmodel()
        pint_params = {
            "PB": ddk_params["PB"] * u.day,
            "T0": np.longdouble(ddk_params["T0"]) * u.day,
            "A1": ddk_params["A1"] * u.lightsecond,
            "ECC": ddk_params["ECC"] * u.Unit(""),
            "OM": ddk_params["OM_deg"] * u.deg,
            "KIN": ddk_params["KIN_deg"] * u.deg,
            "KOM": ddk_params["KOM_deg"] * u.deg,
            "PX": ddk_params["PX"] * u.mas,
            "M2": ddk_params["M2"] * u.M_sun,
            "PMLONG_DDK": ddk_params["PMRA"] * u.mas / u.yr,
            "PMLAT_DDK": ddk_params["PMDEC"] * u.mas / u.yr,
            "K96": k96,
        }
        t = t_mjd * u.day

        # Generate realistic SSB obs_pos using Earth orbit
        # Simple circular approximation: Earth at 1 AU
        phase = 2 * np.pi * (t_mjd - 51544.5) / 365.25  # J2000 epoch
        AU_km = 149597870.7
        obs_pos = np.column_stack([
            AU_km * np.cos(phase),
            AU_km * np.sin(phase),
            np.zeros_like(phase),
        ]) * u.km

        # Pulsar direction unit vector
        ra_rad = ddk_params["RAJ"]
        dec_rad = ddk_params["DECJ"]
        psr_dir = np.array([
            np.cos(ra_rad) * np.cos(dec_rad),
            np.sin(ra_rad) * np.cos(dec_rad),
            np.sin(dec_rad),
        ])
        psr_pos = np.tile(psr_dir, (len(t_mjd), 1))

        bm.update_input(barycentric_toa=t, obs_pos=obs_pos, psr_pos=psr_pos, **pint_params)
        pint_delay = bm.DDdelay().to(u.second).value

        return pint_delay, obs_pos.to(u.km).value

    def _make_jax_ddk(self, ddk_params, t_mjd, obs_pos_km, k96=True):
        """Set up JaxPINT DDK model and return delay."""
        from jaxpint.binary.ddk import BinaryDDK

        ddk = BinaryDDK(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            m2_name="M2",
            kin_name="KIN", kom_name="KOM", px_name="PX",
            raj_name="RAJ", decj_name="DECJ",
            pmra_name="PMRA", pmdec_name="PMDEC",
            posepoch_name="POSEPOCH",
            k96=k96,
        )

        t0_int = np.floor(ddk_params["T0"])
        t0_frac = ddk_params["T0"] - t0_int
        posepoch = ddk_params["T0"]
        posepoch_int = np.floor(posepoch)
        posepoch_frac = posepoch - posepoch_int

        param_names = (
            "PB", "T0", "A1", "ECC", "OM", "M2",
            "KIN", "KOM", "PX",
            "RAJ", "DECJ", "PMRA", "PMDEC", "POSEPOCH",
        )
        param_values = [
            ddk_params["PB"], t0_frac, ddk_params["A1"],
            ddk_params["ECC"], ddk_params["OM"], ddk_params["M2"],
            ddk_params["KIN"], ddk_params["KOM"], ddk_params["PX"],
            ddk_params["RAJ"], ddk_params["DECJ"],
            ddk_params["PMRA"], ddk_params["PMDEC"], posepoch_frac,
        ]
        params = make_params(
            param_names, param_values,
            components="BinaryDDK",
            epoch_int_values={"T0": t0_int, "POSEPOCH": posepoch_int},
        )

        n_toas = len(t_mjd)
        toa_data = _make_toa_data_base(
            t_mjd=t_mjd,
            tzr_tdb_int=jnp.array(54000.0), tzr_tdb_frac=jnp.array(0.5),
            tzr_freq=jnp.array(jnp.inf), tzr_ssb_obs_pos=jnp.zeros(3),
        )
        # Replace ssb_obs_pos with realistic values
        import equinox as eqx
        toa_data = eqx.tree_at(
            lambda t: t.ssb_obs_pos,
            toa_data,
            jnp.array(obs_pos_km),
        )

        jax_delay = np.array(ddk(toa_data, params, jnp.zeros(n_toas)))
        return jax_delay

    @pytest.mark.slow
    def test_ddk_no_k96(self, ddk_params):
        """DDK with K96=False (parallax only)."""
        pytest.importorskip("pint")

        t_mjd = np.linspace(54200.0, 54600.0, 100)
        pint_k96, obs_pos = self._setup_pint_ddk(ddk_params, t_mjd, k96=True)
        pint_nok96, _ = self._setup_pint_ddk(ddk_params, t_mjd, k96=False)
        jax_nok96 = self._make_jax_ddk(ddk_params, t_mjd, obs_pos, k96=False)

        npt.assert_allclose(jax_nok96, pint_nok96, atol=1e-12, rtol=1e-12)
        # K96 proper motion corrections are small but nonzero
        assert np.max(np.abs(pint_k96 - pint_nok96)) > 1e-9, "K96 should change the delay"

    @pytest.mark.slow
    def test_ddk_large_px_matches_dd(self, ddk_params):
        """With large PX (far distance), DDK should reduce to DD with SINI=sin(KIN)."""
        from jaxpint.binary.ddk import BinaryDDK
        from jaxpint.binary.dd import BinaryDD

        t0_int = np.floor(ddk_params["T0"])
        t0_frac = ddk_params["T0"] - t0_int

        n = 50
        t_mjd = np.linspace(54200.0, 54400.0, n)
        toa_data = _make_toa_data_base(
            t_mjd=t_mjd,
            tzr_tdb_int=jnp.array(54000.0), tzr_tdb_frac=jnp.array(0.5),
            tzr_freq=jnp.array(jnp.inf), tzr_ssb_obs_pos=jnp.zeros(3),
        )
        # Zero obs_pos → zero parallax corrections
        # (alternative: very large PX → very close → small corrections)

        # DDK with zero obs_pos and K96=False
        ddk = BinaryDDK(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM", m2_name="M2",
            kin_name="KIN", kom_name="KOM", px_name="PX",
            raj_name="RAJ", decj_name="DECJ",
            k96=False,
        )
        ddk_param_names = ("PB", "T0", "A1", "ECC", "OM", "M2",
                           "KIN", "KOM", "PX", "RAJ", "DECJ")
        ddk_param_values = [
            ddk_params["PB"], t0_frac, ddk_params["A1"],
            ddk_params["ECC"], ddk_params["OM"], ddk_params["M2"],
            ddk_params["KIN"], ddk_params["KOM"], ddk_params["PX"],
            ddk_params["RAJ"], ddk_params["DECJ"],
        ]
        ddk_p = make_params(ddk_param_names, ddk_param_values, components="BinaryDDK",
                            epoch_int_values={"T0": t0_int})
        d_ddk = np.array(ddk(toa_data, ddk_p, jnp.zeros(n)))

        # DD with SINI = sin(KIN)
        sini = np.sin(ddk_params["KIN"])
        dd = BinaryDD(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            m2_name="M2", sini_name="SINI",
        )
        dd_param_names = ("PB", "T0", "A1", "ECC", "OM", "M2", "SINI")
        dd_param_values = [
            ddk_params["PB"], t0_frac, ddk_params["A1"],
            ddk_params["ECC"], ddk_params["OM"], ddk_params["M2"], sini,
        ]
        dd_p = make_params(dd_param_names, dd_param_values, components="BinaryDD",
                           epoch_int_values={"T0": t0_int})
        d_dd = np.array(dd(toa_data, dd_p, jnp.zeros(n)))

        npt.assert_allclose(d_ddk, d_dd, atol=1e-12, rtol=1e-12)
