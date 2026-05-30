"""End-to-end native `.tim` -> TOAData parity vs PINT.

PINT is pointed at our pinned clock snapshot ($PINT_CLOCK_OVERRIDE) and pinned to
the same ref ($JAXPINT_CLOCK_REF), and both sides use the same ephem / BIPM, so
inputs are identical. All marked slow (need PINT + ephemeris download).
"""

from __future__ import annotations

import numpy as np
import pytest

EPHEM = "DE440"
BIPM = "BIPM2023"

_CORPUS = [
    "B1855+09_NANOGrav_dfg+12.tim",      # ao
    "J0740+6620.FCP+21.wb.tim",          # gbt + chime, wideband
    "J1614-2230_NANOGrav_12yv3.wb.tim",  # gbt, wideband
]


@pytest.fixture
def _pinned_clock(monkeypatch):
    from jaxpint.clock import SEED_CLOCK_REF, clock_dir, ensure_fresh

    monkeypatch.setenv("JAXPINT_CLOCK_REF", SEED_CLOCK_REF)
    ensure_fresh(force=True)
    monkeypatch.setenv("PINT_CLOCK_OVERRIDE", str(clock_dir()))


def _load_pint(timname, model):
    import pint.toa as pt
    from pint.config import examplefile

    path = examplefile(timname)
    toas = pt.get_TOAs(
        path, model=model, ephem=EPHEM, include_bipm=True, bipm_version=BIPM,
        planets=False,
    )
    return path, toas


@pytest.mark.slow
@pytest.mark.parametrize("timname", _CORPUS)
def test_topocentric_core_parity(timname, _pinned_clock):
    """`.par`-free: native core vs PINT get_TOAs(model=None)."""
    from jaxpint.loaders.native import topocentric_core

    try:
        path, toas = _load_pint(timname, model=None)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load {timname}: {exc}")

    core = topocentric_core(
        path, ephem=EPHEM, include_bipm=True, bipm_version=BIPM, planets=False
    )
    assert core.n_toas == len(toas)

    p_tdb = np.asarray(toas.table["tdbld"], dtype=np.float64)
    p_pos = np.asarray(toas.table["ssb_obs_pos"].to("km").value)
    p_vel = np.asarray(toas.table["ssb_obs_vel"].to("km/s").value)
    p_sun = np.asarray(toas.table["obs_sun_pos"].to("km").value)
    p_freq = toas.get_freqs().to("MHz").value
    p_err = toas.get_errors().to("s").value

    o_mjd = core.mjd_int + core.mjd_frac
    po = np.argsort(np.asarray(toas.get_mjds().value))
    jo = np.argsort(o_mjd)

    assert np.max(np.abs((core.tdb_int + core.tdb_frac)[jo] - p_tdb[po])) * 86400 < 1e-6
    assert np.max(np.abs(core.ssb_obs_pos[jo] - p_pos[po])) < 1e-3   # km (~m)
    assert np.max(np.abs(core.ssb_obs_vel[jo] - p_vel[po])) < 1e-6
    assert np.max(np.abs(core.obs_sun_pos[jo] - p_sun[po])) < 1e-3
    assert np.allclose(core.freq_mhz[jo], p_freq[po], rtol=0, atol=1e-9)
    assert np.allclose(core.error_s[jo], p_err[po], rtol=1e-12, atol=0)


@pytest.mark.slow
@pytest.mark.parametrize("timname", _CORPUS)
def test_full_toadata_parity(timname, _pinned_clock):
    """Full native TOAData (incl. barycentric freq + wideband DM) vs PINT bridge."""
    import pint.models as pm
    from pint.config import examplefile

    from jaxpint.bridge import pint_toas_to_jax
    from jaxpint.loaders.native import native_toas_to_jax
    import jaxpint.par as par

    try:
        # find the matching .par in PINT examples
        par_path = examplefile(_par_for(timname))
        model = pm.get_model(par_path)
        path, toas = _load_pint(timname, model=model)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load {timname}: {exc}")

    ref = pint_toas_to_jax(toas, model=model)
    par_result = par.get_model(par_path)
    nat = native_toas_to_jax(
        path, par_result, ephem=EPHEM, include_bipm=True, bipm_version=BIPM,
        planets=False,
    )

    o_mjd = np.asarray(nat.mjd_int) + np.asarray(nat.mjd_frac)
    r_mjd = np.asarray(ref.mjd_int) + np.asarray(ref.mjd_frac)
    jo, ro = np.argsort(o_mjd), np.argsort(r_mjd)

    # barycentric freq parity (the helper applied once vs PINT). rtol 1e-8:
    # the Doppler factor is a float dot product (v.Lhat/c), so it agrees to
    # ~1e-10 relative -- far below timing relevance -- but not bit-exact.
    assert np.allclose(np.asarray(nat.freq)[jo], np.asarray(ref.freq)[ro],
                       rtol=1e-8, atol=0), "barycentric freq"
    # tdb + geometry
    assert np.max(np.abs((np.asarray(nat.tdb_int)+np.asarray(nat.tdb_frac))[jo]
                         - (np.asarray(ref.tdb_int)+np.asarray(ref.tdb_frac))[ro])) * 86400 < 1e-6
    assert np.max(np.abs(np.asarray(nat.ssb_obs_pos)[jo] - np.asarray(ref.ssb_obs_pos)[ro])) < 1e-3

    # wideband DM
    if ref.dm_values is not None:
        assert nat.dm_values is not None
        assert np.allclose(np.asarray(nat.dm_values)[jo], np.asarray(ref.dm_values)[ro],
                           rtol=1e-9, atol=0)
        assert np.allclose(np.asarray(nat.dm_errors)[jo], np.asarray(ref.dm_errors)[ro],
                           rtol=1e-9, atol=0)


def _par_for(timname):
    return {
        "B1855+09_NANOGrav_dfg+12.tim": "B1855+09_NANOGrav_dfg+12_TAI.par",
        "J0740+6620.FCP+21.wb.tim": "J0740+6620.FCP+21.wb.DMX3.0.par",
        "J1614-2230_NANOGrav_12yv3.wb.tim": "J1614-2230_NANOGrav_12yv3.wb.gls.par",
    }[timname]
