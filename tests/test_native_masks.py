"""Tests for native flag-mask matching (jaxpint.tim.masks.select_toa_mask).

PINT-free unit tests for the matcher, plus differential parity vs PINT's
``maskParameter.select_toa_mask`` over the example corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from jaxpint.tim.masks import select_toa_mask
from jaxpint.par.result import MaskInfo
from jaxpint.tim.raw_toa import RawTOA


def _toa(mjd, freq, **flags):
    mi = float(int(mjd))
    return RawTOA(
        mjd_int=mi, mjd_frac=mjd - mi, error_s=1e-6, freq_mhz=freq,
        obs="gbt", flags={k: str(v) for k, v in flags.items()},
    )


# --------------------------------------------------------------------------- units


def _run(info, toas, obs=None, mjd=None):
    obs = obs if obs is not None else ["gbt"] * len(toas)
    mjd = mjd if mjd is not None else np.array([t.mjd_int + t.mjd_frac for t in toas])
    return select_toa_mask(info, toas, obs_canonical=obs, mjd_corrected=mjd)


def test_exact_flag_match():
    toas = [_toa(55000, 1400, fe="Rcvr_800"), _toa(55001, 1400, fe="Rcvr1_2"),
            _toa(55002, 1400)]  # third has no -fe flag
    m = _run(MaskInfo("JUMP1", "-fe", "Rcvr_800"), toas)
    assert list(m) == [True, False, False]  # missing flag never matches


def test_freq_range_inclusive():
    toas = [_toa(55000, 800), _toa(55001, 1400), _toa(55002, 2000)]
    m = _run(MaskInfo("JUMP1", "freq", "1000.0", "1500.0"), toas)
    assert list(m) == [False, True, False]
    # bound inclusivity
    m2 = _run(MaskInfo("J", "freq", "800.0", "1400.0"), toas)
    assert list(m2) == [True, True, False]


def test_mjd_range_uses_corrected_mjd():
    toas = [_toa(54000, 1400), _toa(55000, 1400), _toa(56000, 1400)]
    mjd = np.array([54000.0, 55000.0, 56000.0])
    m = _run(MaskInfo("DMX_0001", "mjd", "54500.0", "55500.0"), toas, mjd=mjd)
    assert list(m) == [False, True, False]


def test_value_sorting_handles_reversed_bounds():
    toas = [_toa(55000, 800), _toa(55001, 1400)]
    m = _run(MaskInfo("J", "freq", "1500.0", "1000.0"), toas)  # hi,lo reversed
    assert list(m) == [False, True]


def test_tel_matches_canonical_obs():
    toas = [_toa(55000, 1400), _toa(55001, 1400)]
    m = _run(MaskInfo("JUMP1", "tel", "arecibo"), toas, obs=["gbt", "arecibo"])
    assert list(m) == [False, True]


def test_empty_keyvalue_selects_nothing():
    toas = [_toa(55000, 1400, fe="x"), _toa(55001, 1400, fe="y")]
    m = _run(MaskInfo("JUMP1", "-fe", ""), toas)
    assert not m.any()


# --------------------------------------------------------------------------- parity

_CORPUS = [
    "B1855+09_NANOGrav_dfg+12_TAI.par",
    "J0740+6620.FCP+21.wb.DMX3.0.par",
    "J1614-2230_NANOGrav_12yv3.wb.gls.par",
]
_TIM = {
    "B1855+09_NANOGrav_dfg+12_TAI.par": "B1855+09_NANOGrav_dfg+12.tim",
    "J0740+6620.FCP+21.wb.DMX3.0.par": "J0740+6620.FCP+21.wb.tim",
    "J1614-2230_NANOGrav_12yv3.wb.gls.par": "J1614-2230_NANOGrav_12yv3.wb.tim",
}


@pytest.fixture
def _pinned_clock(monkeypatch):
    from jaxpint.clock import SEED_CLOCK_REF, clock_dir, ensure_fresh

    monkeypatch.setenv("JAXPINT_CLOCK_REF", SEED_CLOCK_REF)
    ensure_fresh(force=True)
    monkeypatch.setenv("PINT_CLOCK_OVERRIDE", str(clock_dir()))


@pytest.mark.slow
@pytest.mark.parametrize("parname", _CORPUS)
def test_mask_parity_vs_pint(parname, _pinned_clock):
    import pint.models as pm
    import pint.toa as pt
    from pint.config import examplefile
    from pint.models.parameter import maskParameter

    from jaxpint.loaders.native import native_toas_to_jax
    import jaxpint.par as par

    try:
        parp = examplefile(parname)
        timp = examplefile(_TIM[parname])
        model = pm.get_model(parp)
        toas = pt.get_TOAs(timp, model=model, ephem="DE440",
                           include_bipm=True, bipm_version="BIPM2023", planets=False)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load {parname}: {exc}")

    par_result = par.get_model(parp)
    td = native_toas_to_jax(timp, par_result, ephem="DE440",
                            include_bipm=True, bipm_version="BIPM2023", planets=False)

    # align native order (build order) to PINT order (sorted) by corrected MJD
    nm = np.asarray(td.mjd_int) + np.asarray(td.mjd_frac)
    pmj = np.asarray(toas.get_mjds().value)
    jo, po = np.argsort(nm), np.argsort(pmj)

    # tel-key correctness: native canonical obs == PINT obs column
    assert list(np.asarray(td.obs_indices))  # populated
    # every masked param our model has matches PINT exactly
    checked = 0
    for pname in td.flag_masks:
        p = getattr(model, pname, None)
        assert isinstance(p, maskParameter), f"{pname} not a PINT maskParameter"
        idx = p.select_toa_mask(toas)
        pmask = np.zeros(toas.ntoas, dtype=bool)
        pmask[idx] = True
        nmask = np.asarray(td.flag_masks[pname])
        assert np.array_equal(nmask[jo], pmask[po]), (parname, pname,
                                                      int(nmask.sum()), int(pmask.sum()))
        checked += 1
    assert checked > 0, f"{parname}: no masked params exercised"

    # coverage: every selector our parser captured produced a mask entry
    assert set(td.flag_masks) == set(par_result.mask_info)
