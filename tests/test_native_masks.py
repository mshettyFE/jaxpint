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

    # tel-key correctness: the native per-TOA observatory partition must match
    # PINT's obs column.  Compare the *grouping* (which TOAs share an obs),
    # robust to any observatory name-canonicalization differences.
    native_obs = np.asarray(td.obs_indices)[jo]
    pint_obs = np.asarray(toas.table["obs"])[po]
    assert len(native_obs) == toas.ntoas
    obs_map = {}
    for n_obs, p_obs in zip(native_obs, pint_obs):
        assert obs_map.setdefault(n_obs, p_obs) == p_obs, \
            "native obs grouping disagrees with PINT"
    assert len(set(obs_map.values())) == len(obs_map), \
        "native obs not injective to PINT obs"
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


# ------------------------------------------------------------- flag-key case parity


def test_mixed_case_flag_key_matches_case_insensitively():
    """Flag-key matching is case-insensitive -- and this agrees with PINT.
    """
    toas = [_toa(55000, 1400, f="L-wide"), _toa(55001, 1400, f="S-wide")]
    assert list(_run(MaskInfo("EFAC1", "-F", "L-wide"), toas)) == [True, False]
    assert list(_run(MaskInfo("EFAC1", "-f", "L-wide"), toas)) == [True, False]


@pytest.mark.slow
def test_mixed_case_flag_key_mask_matches_pint(tmp_path):
    """Differential: JaxPINT's and PINT's masks agree for a mixed-case flag key.
    """
    import io

    import astropy.units as u
    import pint.models as pm
    import pint.toa as pt
    from pint.simulation import make_fake_toas_uniform

    import jaxpint.par as jpar
    from jaxpint.loaders.native import native_toas_to_jax

    par_txt = (
        "PSR J0000+0000\nRAJ 00:00:00\nDECJ 00:00:00\nPEPOCH 55000\n"
        "F0 100\nDM 15\n"
        "EFAC -F L-wide 1.7\n"          # UPPERCASE key ...
        "TZRMJD 55000\nTZRFRQ 1400\nTZRSITE @\n"
        "EPHEM DE421\nCLOCK TT(BIPM2019)\nUNITS TDB\n"
    )
    parp = tmp_path / "mixedcase.par"
    timp = tmp_path / "mixedcase.tim"
    parp.write_text(par_txt)

    model = pm.get_model(io.StringIO(par_txt))
    chunks = []
    for j, be in enumerate(["L-wide", "S-wide"]):        # ... lowercase '-f' flags
        t = make_fake_toas_uniform(54000 + 10 * j, 55000 + 10 * j, 8, model=model,
                                   obs="gbt", freq=1400.0 + 100.0 * j,
                                   error=1.0 * u.us)
        for k in range(t.ntoas):
            t.table["flags"][k]["f"] = be
        chunks.append(t)
    merged = pt.merge_TOAs(chunks)
    merged.write_TOA_file(str(timp), format="tempo2")

    toas = pt.get_TOAs(str(timp), model=model, ephem="DE421", planets=False)
    par_result = jpar.get_model(str(parp))
    td = native_toas_to_jax(str(timp), par_result, ephem="DE421", planets=False)

    # PINT's index array -> boolean; align both to MJD order (native build order
    # differs from PINT's sorted order).
    pint_mask = np.zeros(toas.ntoas, dtype=bool)
    pint_mask[getattr(model, "EFAC1").select_toa_mask(toas)] = True
    jax_mask = np.asarray(td.flag_masks["EFAC1"])

    jo = np.argsort(np.asarray(td.mjd_int) + np.asarray(td.mjd_frac))
    po = np.argsort(np.asarray(toas.get_mjds().value))

    assert jax_mask.sum() > 0, "mixed-case key selected nothing in JaxPINT"
    assert np.array_equal(jax_mask[jo], pint_mask[po]), (
        f"mixed-case flag-key masks disagree: "
        f"JaxPINT selected {int(jax_mask.sum())}, PINT selected {int(pint_mask.sum())}"
    )
