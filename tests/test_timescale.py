"""PINT-free unit tests for the native time/geometry stage.

Covers TT(BIPM)->TDB (`jaxpint.clock.timescale`), barycentric posvels
(`jaxpint.clock.posvels`), and the barycentric-freq Doppler helper
(`jaxpint.utils`). Network (ephemeris) is required only for the
posvel tests, which are marked slow.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxpint.clock.observatory import resolve_observatory
from jaxpint.clock.timescale import to_tdb


def test_itrf_xyz_present_in_metadata():
    for name in ("gbt", "arecibo", "chime"):
        xyz = resolve_observatory(name).itrf_xyz
        assert xyz is not None and len(xyz) == 3
    assert resolve_observatory("@").itrf_xyz is None  # barycenter


def test_to_tdb_matches_astropy_pulsar_mjd():
    """to_tdb equals astropy's pulsar_mjd .tdb (the reference path) -- incl a
    leap-second day (MJD 57753 = 2016-12-31)."""
    import pint.pulsar_mjd  # noqa: F401  -- registers the astropy "pulsar_mjd" format
    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    gbt = resolve_observatory("gbt").itrf_xyz
    mjd_int = np.array([57000.0, 57753.0, 55000.0])
    mjd_frac = np.array([0.3, 0.6, 0.123456])
    xyz = np.array([gbt, gbt, gbt])

    ti, tf = to_tdb(mjd_int, mjd_frac, xyz)
    ours = ti.astype(np.longdouble) + tf.astype(np.longdouble)

    loc = EarthLocation.from_geocentric(gbt[0] * u.m, gbt[1] * u.m, gbt[2] * u.m)
    ref_t = Time(mjd_int, mjd_frac, format="pulsar_mjd", scale="utc", location=loc).tdb
    ref = np.array(
        [np.longdouble(x.jd1 - 2400000.5) + np.longdouble(x.jd2) for x in ref_t]
    )
    assert np.max(np.abs(ours - ref)) * 86400.0 < 1e-3  # sub-ms; in practice 0


# --- barycentric freq helper -------------------------------------------------


def test_doppler_shift_basic():
    from jaxpint.utils import doppler_shift_freq
    from jaxpint.constants import C_KM_PER_S

    freq = np.array([1400.0])
    # velocity purely along L_hat: shift = -v/c
    l_hat = np.array([[1.0, 0.0, 0.0]])
    vel = np.array([[30.0, 0.0, 0.0]])  # km/s
    out = np.asarray(doppler_shift_freq(freq, vel, l_hat))
    assert out[0] == pytest.approx(1400.0 * (1 - 30.0 / C_KM_PER_S))
    # zero velocity -> topocentric unchanged
    out0 = np.asarray(doppler_shift_freq(freq, np.zeros((1, 3)), l_hat))
    assert out0[0] == pytest.approx(1400.0)


def test_precompute_staleness_below_ns():
    """Precomputing barycentric freq with build-time astrometry is safe.

    The Doppler-shifted freq is baked into ``TOAData.freq`` at build time (see
    jaxpint/utils.py) and is not refreshed if astrometry is refit.
    Real ``.par`` astrometry uncertainties are sub-arcsec, so take 1" as a
    generous worst-case fit step, push the Doppler factor at the max observatory
    speed (~30 km/s), and confirm the resulting frequency-dependent-delay error
    stays well below ns-level timing precision.
    """
    from jaxpint.utils import doppler_shift_freq

    f_topo = np.array([1400.0])           # MHz
    v_obs = np.array([[30.0, 0.0, 0.0]])  # km/s (max observatory speed scale)

    def lhat(ra, dec):
        return np.array(
            [[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]]
        )

    ra, dec = np.radians(120.0), np.radians(-25.0)
    dtheta = np.radians(1.0 / 3600.0)  # 1 arcsec: worst-case fit step
    f0 = float(np.asarray(doppler_shift_freq(f_topo, v_obs, lhat(ra, dec)))[0])
    f1 = float(
        np.asarray(doppler_shift_freq(f_topo, v_obs, lhat(ra + dtheta, dec)))[0]
    )

    df_over_f = abs(f1 - f0) / f0
    # dispersion delay ~ f^-2  =>  d(delay) = 2 (df/f) * delay; DM=30 @ 1400 MHz
    disp_delay = 4.148808e3 * 30.0 / 1400.0**2  # seconds
    d_delay = 2.0 * df_over_f * disp_delay
    assert d_delay < 1e-9, f"precompute staleness {d_delay:.2e} s exceeds 1 ns"
    assert d_delay < 1e-10  # actually ~5e-11 s; tight bound trips a real regression


# --- posvels (needs ephemeris; slow) ----------------------------------------


@pytest.mark.slow
def test_compute_posvels_vs_pint():
    import pint.pulsar_mjd  # noqa: F401  registers formats
    from astropy.time import Time
    from pint.observatory import get_observatory

    from jaxpint.clock.posvels import compute_posvels

    gbt = resolve_observatory("gbt").itrf_xyz
    mjd_int = np.array([57000.0, 58000.0])
    mjd_frac = np.array([0.3, 0.6])
    ti, tf = to_tdb(mjd_int, mjd_frac, np.array([gbt, gbt]))
    out = compute_posvels(ti, tf, gbt, ephem="DE440", planets=True)

    site = get_observatory("gbt")
    ssb = site.posvel(Time(ti, tf, format="mjd", scale="tdb"), "DE440")
    assert np.max(np.abs(out["ssb_obs_pos"] - ssb.pos.T.to_value("km"))) < 1e-3  # km
    assert np.max(np.abs(out["ssb_obs_vel"] - ssb.vel.T.to_value("km/s"))) < 1e-9
    assert set(out["planet_positions"]) == {
        "obs_jupiter_pos", "obs_saturn_pos", "obs_venus_pos",
        "obs_uranus_pos", "obs_neptune_pos", "obs_earth_pos",
    }


@pytest.mark.slow
def test_compute_posvels_barycenter_zero_obs_term():
    from jaxpint.clock.posvels import compute_posvels

    ti = np.array([57000.0])
    tf = np.array([0.3])
    bary = compute_posvels(ti, tf, None, ephem="DE440")
    geo = compute_posvels(ti, tf, (0.0, 0.0, 0.0), ephem="DE440")
    # The barycentre observatory IS the SSB, so its SSB-relative posvel is zero --
    # distinct from the geocentre (itrf_xyz == 0), whose ssb_obs == earth posvel.
    assert bary["ssb_obs_pos"].shape == (1, 3)
    assert np.allclose(bary["ssb_obs_pos"], 0.0)
    assert np.allclose(bary["ssb_obs_vel"], 0.0)
    assert not np.allclose(bary["ssb_obs_pos"], geo["ssb_obs_pos"])
