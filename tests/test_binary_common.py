"""Parametrized binary delay tests shared across BT, DD, DDGR, DDK, ELL1.

The PINT-equivalence, JIT-smoke, and autodiff tests all follow the same
shape; per-model specs encapsulate the differences (model class, parameter
dict, PINT setup, TOA grid, tolerance, sensitivity parameter).

Model-specific tests stay in their per-model files (e.g.
``test_binary_dd.py::test_dd_no_shapiro``, ``test_binary_ddgr.py::test_ddgr_xomdot``).
The piecewise variant and the real-file integration suite are also untouched
because their assertions are structurally different.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import functools
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from tests.helpers import (
    ddk_earth_obs_pos_km,
    make_binary_params,
    make_binary_toa_data,
    make_params,
    make_toa_data as _make_toa_data_base,
)


_DEG_YR_TO_RAD_S = np.pi / 180.0 / (365.25 * 86400.0)


@dataclass(frozen=True)
class BinarySpec:
    name: str
    # () -> (jax_model, params, toa_data, n_toas, pint_delay_array, atol)
    make_full: Callable[[], tuple]
    # () -> (jax_model, params, toa_data, n_toas, sensitivity_param_name)
    make_min: Callable[[], tuple]


# ---------------------------------------------------------------------------
# BT
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)  # 3 tests x this spec reuse one PINT build (read-only)
def _bt_make_full() -> tuple:
    import astropy.units as u
    from pint.models.stand_alone_psr_binaries.BT_model import BTmodel

    from jaxpint.binary.bt import BinaryBT

    pb, t0, a1, ecc = 1.5, 54000.0, 2.0, 0.3
    om_deg = 45.0
    omdot_deg_yr = 0.5
    gamma = 0.001
    pbdot = 1e-12

    bm = BTmodel()
    pint_params = {
        "PB": pb * u.day,
        "T0": np.longdouble(t0) * u.day,
        "A1": a1 * u.lightsecond,
        "ECC": ecc * u.Unit(""),
        "OM": om_deg * u.deg,
        "OMDOT": omdot_deg_yr * u.deg / u.year,
        "GAMMA": gamma * u.second,
        "PBDOT": pbdot * u.Unit(""),
    }
    t_mjd = np.linspace(54000.5, 54500.0, 500)
    bm.update_input(barycentric_toa=t_mjd * u.day, **pint_params)
    pint_delay = bm.BTdelay().to(u.second).value

    bt = BinaryBT(
        pb_name="PB", t0_name="T0", a1_name="A1", ecc_name="ECC",
        om_name="OM", omdot_name="OMDOT", gamma_name="GAMMA", pbdot_name="PBDOT",
    )
    t0_int = np.floor(t0)
    t0_frac = t0 - t0_int
    om_rad = om_deg * np.pi / 180.0
    omdot_rad_s = omdot_deg_yr * _DEG_YR_TO_RAD_S
    params = make_binary_params(
        ("PB", "T0", "A1", "ECC", "OM", "OMDOT", "GAMMA", "PBDOT"),
        [pb, t0_frac, a1, ecc, om_rad, omdot_rad_s, gamma, pbdot],
        epoch_int_values={"T0": t0_int},
    )
    toa_data = make_binary_toa_data(t_mjd)
    return bt, params, toa_data, len(t_mjd), pint_delay, 1e-12


def _bt_make_min() -> tuple:
    from jaxpint.binary.bt import BinaryBT

    pb, t0, a1, ecc = 1.5, 54000.0, 2.0, 0.3
    om_rad = 45.0 * np.pi / 180.0
    t0_int = np.floor(t0)
    t0_frac = t0 - t0_int

    bt = BinaryBT(pb_name="PB", t0_name="T0", a1_name="A1",
                  ecc_name="ECC", om_name="OM")
    params = make_binary_params(
        ("PB", "T0", "A1", "ECC", "OM"),
        [pb, t0_frac, a1, ecc, om_rad],
        epoch_int_values={"T0": t0_int},
    )
    n = 10
    toa_data = make_binary_toa_data(np.linspace(54100.0, 54100.9, n))
    return bt, params, toa_data, n, "A1"


# ---------------------------------------------------------------------------
# DD
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)  # 3 tests x this spec reuse one PINT build (read-only)
def _dd_make_full() -> tuple:
    import astropy.units as u
    from pint.models.stand_alone_psr_binaries.DD_model import DDmodel

    from jaxpint.binary.dd import BinaryDD

    pb, t0, a1, ecc = 0.7, 54000.0, 3.0, 0.2
    om_deg = 120.0
    omdot_deg_yr = 1.0
    gamma, pbdot = 0.004, 5e-13
    m2, sini = 0.3, 0.9
    a0, b0 = 1e-5, 2e-5
    dr, dth = 1e-4, 1e-4

    bm = DDmodel()
    pint_params = {
        "PB": pb * u.day,
        "T0": np.longdouble(t0) * u.day,
        "A1": a1 * u.lightsecond,
        "ECC": ecc * u.Unit(""),
        "OM": om_deg * u.deg,
        "OMDOT": omdot_deg_yr * u.deg / u.year,
        "GAMMA": gamma * u.second,
        "PBDOT": pbdot * u.Unit(""),
        "M2": m2 * u.M_sun,
        "SINI": sini * u.Unit(""),
        "A0": a0 * u.second,
        "B0": b0 * u.second,
        "DR": dr * u.Unit(""),
        "DTH": dth * u.Unit(""),
    }
    t_mjd = np.linspace(54000.5, 54200.0, 500)
    bm.update_input(barycentric_toa=t_mjd * u.day, **pint_params)
    pint_delay = bm.DDdelay().to(u.second).value

    dd = BinaryDD(
        pb_name="PB", t0_name="T0", a1_name="A1", ecc_name="ECC", om_name="OM",
        omdot_name="OMDOT", gamma_name="GAMMA", pbdot_name="PBDOT",
        m2_name="M2", sini_name="SINI", a0_name="A0", b0_name="B0",
        dr_name="DR", dth_name="DTH",
    )
    t0_int = np.floor(t0)
    t0_frac = t0 - t0_int
    om_rad = om_deg * np.pi / 180.0
    omdot_rad_s = omdot_deg_yr * _DEG_YR_TO_RAD_S
    params = make_binary_params(
        ("PB", "T0", "A1", "ECC", "OM", "OMDOT", "GAMMA", "PBDOT",
         "M2", "SINI", "A0", "B0", "DR", "DTH"),
        [pb, t0_frac, a1, ecc, om_rad, omdot_rad_s, gamma, pbdot,
         m2, sini, a0, b0, dr, dth],
        epoch_int_values={"T0": t0_int},
    )
    toa_data = make_binary_toa_data(t_mjd)
    return dd, params, toa_data, len(t_mjd), pint_delay, 1e-12


def _dd_make_min() -> tuple:
    from jaxpint.binary.dd import BinaryDD

    pb, t0, a1, ecc = 0.7, 54000.0, 3.0, 0.2
    om_rad = 120.0 * np.pi / 180.0
    m2, sini = 0.3, 0.9
    t0_int = np.floor(t0)
    t0_frac = t0 - t0_int

    dd = BinaryDD(
        pb_name="PB", t0_name="T0", a1_name="A1", ecc_name="ECC", om_name="OM",
        m2_name="M2", sini_name="SINI",
    )
    params = make_binary_params(
        ("PB", "T0", "A1", "ECC", "OM", "M2", "SINI"),
        [pb, t0_frac, a1, ecc, om_rad, m2, sini],
        epoch_int_values={"T0": t0_int},
    )
    n = 10
    toa_data = make_binary_toa_data(np.linspace(54100.1, 54100.9, n))
    return dd, params, toa_data, n, "A1"


# ---------------------------------------------------------------------------
# DDGR
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)  # 3 tests x this spec reuse one PINT build (read-only)
def _ddgr_make_full() -> tuple:
    import astropy.units as u
    from pint.models.stand_alone_psr_binaries.DDGR_model import DDGRmodel

    from jaxpint.binary.ddgr import BinaryDDGR

    pb, t0, a1, ecc = 0.5, 54000.0, 3.0, 0.2
    om_deg = 120.0
    mtot, m2 = 2.5, 1.1

    bm = DDGRmodel()
    pint_params = {
        "PB": pb * u.day,
        "T0": np.longdouble(t0) * u.day,
        "A1": a1 * u.lightsecond,
        "ECC": ecc * u.Unit(""),
        "OM": om_deg * u.deg,
        "MTOT": mtot * u.Msun,
        "M2": m2 * u.Msun,
    }
    t_mjd = np.linspace(54000.5, 54200.0, 500)
    bm.update_input(barycentric_toa=t_mjd * u.day, **pint_params)
    pint_delay = bm.DDdelay().to(u.second).value

    ddgr = BinaryDDGR(
        pb_name="PB", t0_name="T0", a1_name="A1",
        ecc_name="ECC", om_name="OM",
        mtot_name="MTOT", m2_name="M2",
    )
    t0_int = np.floor(t0)
    t0_frac = t0 - t0_int
    om_rad = om_deg * np.pi / 180.0
    params = make_binary_params(
        ("PB", "T0", "A1", "ECC", "OM", "MTOT", "M2"),
        [pb, t0_frac, a1, ecc, om_rad, mtot, m2],
        epoch_int_values={"T0": t0_int},
    )
    toa_data = make_binary_toa_data(t_mjd)
    return ddgr, params, toa_data, len(t_mjd), pint_delay, 1e-11


def _ddgr_make_min() -> tuple:
    from jaxpint.binary.ddgr import BinaryDDGR

    pb, t0, a1, ecc = 0.5, 54000.0, 3.0, 0.2
    om_rad = 120.0 * np.pi / 180.0
    mtot, m2 = 2.5, 1.1
    t0_int = np.floor(t0)
    t0_frac = t0 - t0_int

    ddgr = BinaryDDGR(
        pb_name="PB", t0_name="T0", a1_name="A1",
        ecc_name="ECC", om_name="OM",
        mtot_name="MTOT", m2_name="M2",
    )
    params = make_binary_params(
        ("PB", "T0", "A1", "ECC", "OM", "MTOT", "M2"),
        [pb, t0_frac, a1, ecc, om_rad, mtot, m2],
        epoch_int_values={"T0": t0_int},
    )
    n = 10
    toa_data = make_binary_toa_data(np.linspace(54100.1, 54100.9, n))
    return ddgr, params, toa_data, n, "MTOT"


# ---------------------------------------------------------------------------
# DDK
# ---------------------------------------------------------------------------


def _ddk_toa_data_with_obs_pos(t_mjd: np.ndarray):
    toa_data = _make_toa_data_base(
        t_mjd=t_mjd,
        tzr_tdb_int=jnp.array(54000.0), tzr_tdb_frac=jnp.array(0.5),
        tzr_freq=jnp.array(jnp.inf), tzr_ssb_obs_pos=jnp.zeros(3),
    )
    return eqx.tree_at(
        lambda t: t.ssb_obs_pos, toa_data,
        jnp.array(ddk_earth_obs_pos_km(t_mjd)),
    )


@functools.lru_cache(maxsize=None)  # 3 tests x this spec reuse one PINT build (read-only)
def _ddk_make_full() -> tuple:
    import astropy.units as u
    from pint.models.stand_alone_psr_binaries.DDK_model import DDKmodel

    from jaxpint.binary.ddk import BinaryDDK

    pb, t0, a1 = 67.825, 54187.0, 32.342
    ecc = 0.0000749
    om_deg, kin_deg, kom_deg = 176.2, 72.0, 89.0
    px, m2 = 0.8, 0.29
    ra_deg, dec_deg = 258.48, 7.79
    pmra, pmdec = 4.917, -3.937

    t_mjd = np.linspace(54200.0, 54600.0, 200)

    bm = DDKmodel()
    pint_params = {
        "PB": pb * u.day,
        "T0": np.longdouble(t0) * u.day,
        "A1": a1 * u.lightsecond,
        "ECC": ecc * u.Unit(""),
        "OM": om_deg * u.deg,
        "KIN": kin_deg * u.deg,
        "KOM": kom_deg * u.deg,
        "PX": px * u.mas,
        "M2": m2 * u.M_sun,
        "PMLONG_DDK": pmra * u.mas / u.yr,
        "PMLAT_DDK": pmdec * u.mas / u.yr,
        "K96": True,
    }
    obs_pos_km = ddk_earth_obs_pos_km(t_mjd)
    obs_pos_q = obs_pos_km * u.km
    ra_rad = ra_deg * np.pi / 180.0
    dec_rad = dec_deg * np.pi / 180.0
    psr_dir = np.array([
        np.cos(ra_rad) * np.cos(dec_rad),
        np.sin(ra_rad) * np.cos(dec_rad),
        np.sin(dec_rad),
    ])
    psr_pos = np.tile(psr_dir, (len(t_mjd), 1))
    bm.update_input(
        barycentric_toa=t_mjd * u.day,
        obs_pos=obs_pos_q, psr_pos=psr_pos,
        **pint_params,
    )
    pint_delay = bm.DDdelay().to(u.second).value

    ddk = BinaryDDK(
        pb_name="PB", t0_name="T0", a1_name="A1",
        ecc_name="ECC", om_name="OM",
        m2_name="M2",
        kin_name="KIN", kom_name="KOM", px_name="PX",
        raj_name="RAJ", decj_name="DECJ",
        pmra_name="PMRA", pmdec_name="PMDEC",
        posepoch_name="POSEPOCH",
        k96=True,
    )

    t0_int = np.floor(t0)
    t0_frac = t0 - t0_int
    posepoch_int = np.floor(t0)
    posepoch_frac = t0 - posepoch_int
    om_rad = om_deg * np.pi / 180.0
    kin_rad = kin_deg * np.pi / 180.0
    kom_rad = kom_deg * np.pi / 180.0
    ra_rad_p = ra_deg * np.pi / 180.0
    dec_rad_p = dec_deg * np.pi / 180.0

    params = make_params(
        ("PB", "T0", "A1", "ECC", "OM", "M2",
         "KIN", "KOM", "PX",
         "RAJ", "DECJ", "PMRA", "PMDEC", "POSEPOCH"),
        [pb, t0_frac, a1, ecc, om_rad, m2,
         kin_rad, kom_rad, px,
         ra_rad_p, dec_rad_p, pmra, pmdec, posepoch_frac],
        epoch_int_values={"T0": t0_int, "POSEPOCH": posepoch_int},
    )
    toa_data = _make_toa_data_base(
        t_mjd=t_mjd,
        tzr_tdb_int=jnp.array(54000.0), tzr_tdb_frac=jnp.array(0.5),
        tzr_freq=jnp.array(jnp.inf), tzr_ssb_obs_pos=jnp.zeros(3),
    )
    toa_data = eqx.tree_at(lambda t: t.ssb_obs_pos, toa_data, jnp.array(obs_pos_km))
    return ddk, params, toa_data, len(t_mjd), pint_delay, 1e-12


def _ddk_make_min() -> tuple:
    from jaxpint.binary.ddk import BinaryDDK

    pb, t0, a1 = 67.825, 54187.0, 32.342
    ecc = 0.0000749
    om_rad = 176.2 * np.pi / 180.0
    kin_rad = 72.0 * np.pi / 180.0
    kom_rad = 89.0 * np.pi / 180.0
    px = 0.8
    ra_rad = 258.48 * np.pi / 180.0
    dec_rad = 7.79 * np.pi / 180.0
    t0_int = np.floor(t0)
    t0_frac = t0 - t0_int

    ddk = BinaryDDK(
        pb_name="PB", t0_name="T0", a1_name="A1",
        ecc_name="ECC", om_name="OM",
        kin_name="KIN", kom_name="KOM", px_name="PX",
        raj_name="RAJ", decj_name="DECJ",
        k96=False,
    )
    params = make_params(
        ("PB", "T0", "A1", "ECC", "OM", "KIN", "KOM", "PX", "RAJ", "DECJ"),
        [pb, t0_frac, a1, ecc, om_rad, kin_rad, kom_rad, px, ra_rad, dec_rad],
        epoch_int_values={"T0": t0_int},
    )
    n = 10
    t_mjd = np.linspace(54200.0, 54400.0, n)
    toa_data = _ddk_toa_data_with_obs_pos(t_mjd)
    return ddk, params, toa_data, n, "KIN"


# ---------------------------------------------------------------------------
# ELL1
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)  # 3 tests x this spec reuse one PINT build (read-only)
def _ell1_make_full() -> tuple:
    import astropy.units as u
    from pint.models.stand_alone_psr_binaries.ELL1_model import ELL1model

    from jaxpint.binary.ell1 import BinaryELL1

    pb, tasc, a1 = 1.2, 54000.0, 1.5
    eps1, eps2 = 0.01, 0.02
    m2, sini = 0.25, 0.85
    pbdot = 1e-13

    bm = ELL1model()
    pint_params = {
        "PB": pb * u.day,
        "TASC": np.longdouble(tasc) * u.day,
        "A1": a1 * u.lightsecond,
        "EPS1": eps1 * u.Unit(""),
        "EPS2": eps2 * u.Unit(""),
        "M2": m2 * u.M_sun,
        "SINI": sini * u.Unit(""),
        "PBDOT": pbdot * u.Unit(""),
    }
    t_mjd = np.linspace(54000.5, 54300.0, 500)
    bm.update_input(barycentric_toa=t_mjd * u.day, **pint_params)
    pint_delay = bm.ELL1delay().to(u.second).value

    ell1 = BinaryELL1(
        pb_name="PB", tasc_name="TASC", a1_name="A1",
        eps1_name="EPS1", eps2_name="EPS2",
        m2_name="M2", sini_name="SINI", pbdot_name="PBDOT",
    )
    tasc_int = np.floor(tasc)
    tasc_frac = tasc - tasc_int
    params = make_binary_params(
        ("PB", "TASC", "A1", "EPS1", "EPS2", "M2", "SINI", "PBDOT"),
        [pb, tasc_frac, a1, eps1, eps2, m2, sini, pbdot],
        epoch_int_values={"TASC": tasc_int},
    )
    toa_data = make_binary_toa_data(t_mjd)
    return ell1, params, toa_data, len(t_mjd), pint_delay, 1e-12


def _ell1_make_min() -> tuple:
    from jaxpint.binary.ell1 import BinaryELL1

    pb, tasc, a1 = 1.2, 54000.0, 1.5
    eps1, eps2 = 0.01, 0.02
    m2, sini = 0.25, 0.85
    tasc_int = np.floor(tasc)
    tasc_frac = tasc - tasc_int

    ell1 = BinaryELL1(
        pb_name="PB", tasc_name="TASC", a1_name="A1",
        eps1_name="EPS1", eps2_name="EPS2",
        m2_name="M2", sini_name="SINI",
    )
    params = make_binary_params(
        ("PB", "TASC", "A1", "EPS1", "EPS2", "M2", "SINI"),
        [pb, tasc_frac, a1, eps1, eps2, m2, sini],
        epoch_int_values={"TASC": tasc_int},
    )
    n = 10
    toa_data = make_binary_toa_data(np.linspace(54100.1, 54100.9, n))
    return ell1, params, toa_data, n, "A1"


BINARY_SPECS = [
    BinarySpec("bt", _bt_make_full, _bt_make_min),
    BinarySpec("dd", _dd_make_full, _dd_make_min),
    BinarySpec("ddgr", _ddgr_make_full, _ddgr_make_min),
    BinarySpec("ddk", _ddk_make_full, _ddk_make_min),
    BinarySpec("ell1", _ell1_make_full, _ell1_make_min),
]


@pytest.fixture(params=BINARY_SPECS, ids=[s.name for s in BINARY_SPECS])
def binary_spec(request) -> BinarySpec:
    return request.param


class TestBinaryCommon:
    """Tests shared by all PL binary delay models."""

    @pytest.mark.slow
    def test_delay_matches_pint(self, binary_spec):
        pytest.importorskip("pint")
        jax_model, params, toa_data, n, pint_delay, tol = binary_spec.make_full()
        jax_delay = np.array(jax_model(toa_data, params, jnp.zeros(n)))
        npt.assert_allclose(jax_delay, pint_delay, atol=tol, rtol=tol)

    def test_jit(self, binary_spec):
        jax_model, params, toa_data, n, _ = binary_spec.make_min()
        jitted = jax.jit(jax_model)
        result = jitted(toa_data, params, jnp.zeros(n))
        assert result.shape == (n,)
        assert jnp.all(jnp.isfinite(result))

    def test_autodiff(self, binary_spec):
        jax_model, params, toa_data, n, sens_param = binary_spec.make_min()

        def delay_fn(values):
            p = params.with_free_values(values)
            return jax_model(toa_data, p, jnp.zeros(n))

        J = jax.jacobian(delay_fn)(params.free_values())
        n_params = len(params.names)
        assert J.shape == (n, n_params)
        assert jnp.all(jnp.isfinite(J))
        col = list(params.names).index(sens_param)
        assert jnp.any(jnp.abs(J[:, col]) > 0)
