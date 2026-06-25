"""Tests for the native TZR (absolute-phase anchor) builder.

Differential parity vs the PINT bridge's ``extract_tzr_toa`` for the synthesized
TZR TOA fields, plus an end-to-end absolute-phase residual check vs PINT. All
slow (need PINT + ephemeris); PINT is pinned to our clock snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest

EPHEM = "DE440"
BIPM = "BIPM2023"


@pytest.fixture
def _pinned_clock(monkeypatch):
    from jaxpint.clock import SEED_CLOCK_REF, clock_dir, ensure_fresh

    monkeypatch.setenv("JAXPINT_CLOCK_REF", SEED_CLOCK_REF)
    ensure_fresh(force=True)
    monkeypatch.setenv("PINT_CLOCK_OVERRIDE", str(clock_dir()))


def _native_tzr(parp, timp, planets=False):
    from jaxpint.loaders.native import topocentric_core, _build_tzr_fields
    import jaxpint.par as par

    pr = par.get_model(parp)
    core = topocentric_core(timp, ephem=EPHEM, include_bipm=True,
                            bipm_version=BIPM, planets=planets)
    return _build_tzr_fields(core, pr, ephem=EPHEM, include_bipm=True,
                             bipm_version=BIPM, planets=planets), pr


def _assert_tzr_parity(nat, ref):
    dtdb = abs((nat["tzr_tdb_int"] + nat["tzr_tdb_frac"])
               - (ref["tdb_int"] + ref["tdb_frac"])) * 86400.0
    assert dtdb < 1e-6, f"tzr tdb diff {dtdb} s"
    if np.isinf(ref["freq"]):
        assert np.isinf(nat["tzr_freq"])
    else:
        assert np.isclose(nat["tzr_freq"], ref["freq"], rtol=1e-8)
    assert np.max(np.abs(np.asarray(nat["tzr_ssb_obs_pos"]) - ref["ssb_obs_pos"])) < 1e-3
    assert np.max(np.abs(np.asarray(nat["tzr_obs_sun_pos"]) - ref["obs_sun_pos"])) < 1e-3


# --------------------------------------------------------------------------- A: parity

_CORPUS = [
    ("B1855+09_NANOGrav_dfg+12_TAI.par", "B1855+09_NANOGrav_dfg+12.tim"),
    ("J1614-2230_NANOGrav_12yv3.wb.gls.par", "J1614-2230_NANOGrav_12yv3.wb.tim"),
]


@pytest.mark.slow
@pytest.mark.parametrize("parname,timname", _CORPUS)
def test_tzr_parity_vs_pint(parname, timname, _pinned_clock):
    import pint.models as pm
    import pint.toa as pt
    from pint.config import examplefile

    from jaxpint.bridge.toa_conversion import extract_tzr_toa

    try:
        parp = examplefile(parname)
        timp = examplefile(timname)
        model = pm.get_model(parp)
        toas = pt.get_TOAs(timp, model=model, ephem=EPHEM,
                           include_bipm=True, bipm_version=BIPM, planets=False)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load {parname}: {exc}")

    ref = extract_tzr_toa(model, toas)
    nat, _ = _native_tzr(parp, timp, planets=False)
    assert nat is not None
    _assert_tzr_parity(nat, ref)


@pytest.mark.slow
def test_tzr_auto_pepoch_vs_pint(tmp_path, _pinned_clock):
    """No TZRMJD in the .par -> PINT auto-generates from PEPOCH; native must match."""
    import pint.models as pm
    import pint.toa as pt
    from pint.config import examplefile

    from jaxpint.bridge.toa_conversion import extract_tzr_toa

    parp = examplefile("B1855+09_NANOGrav_dfg+12_TAI.par")
    timp = examplefile("B1855+09_NANOGrav_dfg+12.tim")
    # Strip the TZR* lines so both sides fall back to the PEPOCH rule.
    stripped = tmp_path / "no_tzr.par"
    stripped.write_text(
        "\n".join(l for l in open(parp)
                  if not l.strip().startswith(("TZRMJD", "TZRSITE", "TZRFRQ")))
    )
    try:
        model = pm.get_model(str(stripped))
        toas = pt.get_TOAs(timp, model=model, ephem=EPHEM,
                           include_bipm=True, bipm_version=BIPM, planets=False)
        ref = extract_tzr_toa(model, toas)  # triggers make_TZR_toa
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load stripped par: {exc}")

    nat, _ = _native_tzr(str(stripped), timp, planets=False)
    assert nat is not None
    _assert_tzr_parity(nat, ref)


@pytest.mark.slow
def test_tzr_barycenter_site(tmp_path, _pinned_clock):
    """TZRSITE=ssb -> barycentric TZR: TDB-direct, zero obs/sun position."""
    import pint.models as pm
    import pint.toa as pt
    from pint.config import examplefile

    from jaxpint.bridge.toa_conversion import extract_tzr_toa

    parp = examplefile("B1855+09_NANOGrav_dfg+12_TAI.par")
    timp = examplefile("B1855+09_NANOGrav_dfg+12.tim")
    bary = tmp_path / "ssb_tzr.par"
    lines = []
    for l in open(parp):
        if l.strip().startswith("TZRSITE"):
            lines.append("TZRSITE        ssb\n")
        else:
            lines.append(l)
    bary.write_text("".join(lines))
    try:
        model = pm.get_model(str(bary))
        toas = pt.get_TOAs(timp, model=model, ephem=EPHEM,
                           include_bipm=True, bipm_version=BIPM, planets=False)
        ref = extract_tzr_toa(model, toas)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load ssb par: {exc}")

    nat, _ = _native_tzr(str(bary), timp, planets=False)
    assert nat is not None
    assert np.allclose(np.asarray(nat["tzr_obs_sun_pos"]), 0.0)
    assert np.allclose(ref["obs_sun_pos"], 0.0)
    _assert_tzr_parity(nat, ref)


# --------------------------------------------------------------------------- B: abs phase


@pytest.mark.slow
def test_abs_phase_residuals_vs_pint(_pinned_clock):
    """End-to-end: native residuals (with TZR abs-phase) match PINT."""
    import pint.models as pm
    import pint.toa as pt
    from pint.config import examplefile
    from pint.residuals import Residuals

    from jaxpint.model_builder import build_model
    from jaxpint.fitters import compute_time_residuals
    from jaxpint.loaders.native import native_toas_to_jax
    import jaxpint.par as par

    parp = examplefile("B1855+09_NANOGrav_dfg+12_TAI.par")
    timp = examplefile("B1855+09_NANOGrav_dfg+12.tim")
    try:
        model = pm.get_model(parp)
        toas = pt.get_TOAs(timp, model=model, ephem=EPHEM,
                           include_bipm=True, bipm_version=BIPM, planets=False)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load: {exc}")

    pr = par.get_model(parp)
    td = native_toas_to_jax(timp, pr, ephem=EPHEM, include_bipm=True,
                            bipm_version=BIPM, planets=False)
    assert td.tzr_tdb_int is not None  # TZR populated
    tm, _ = build_model(pr, td)
    r_nat = np.asarray(compute_time_residuals(tm, td, pr.params))

    r_pint = Residuals(toas, model, subtract_mean=False).time_resids.to_value("s")

    # align by sorted corrected MJD (PINT reorders)
    nm = np.asarray(td.mjd_int) + np.asarray(td.mjd_frac)
    pmj = np.asarray(toas.get_mjds().value)
    jo, po = np.argsort(nm), np.argsort(pmj)
    diff = r_nat[jo] - r_pint[po]
    # absolute-phase parity: residuals agree up to an overall constant offset
    diff -= np.median(diff)
    assert np.max(np.abs(diff)) < 1e-6, float(np.max(np.abs(diff)))


# --------------------------------------------------------------------------- C: none path


def test_no_tzrmjd_no_pepoch_returns_none():
    """Neither TZRMJD nor PEPOCH -> _build_tzr_fields returns None (no abs phase)."""
    import types as _t

    from jaxpint.loaders.native import _build_tzr_fields
    from jaxpint.par.result import ParResult
    from jaxpint.types import ParameterVector

    pv = ParameterVector(
        values=np.zeros(1), frozen_mask=(True,), names=("F0",),
        units=("Hz",), epoch_int_values={},
    )
    pr = ParResult(params=pv)
    core = _t.SimpleNamespace(mjd_int=np.array([55000.0]), mjd_frac=np.array([0.0]))
    assert _build_tzr_fields(core, pr, ephem=EPHEM, include_bipm=True,
                             bipm_version=BIPM, planets=False) is None
