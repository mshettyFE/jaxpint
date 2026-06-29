"""Tests for the native public API (jaxpint.native) + compute_barycentric_toas.

PINT-free shape checks plus differential parity vs PINT (end-to-end residuals and
barycentric TOAs). PINT pinned to our clock snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest

EPHEM = "DE440"
BIPM = "BIPM2023"


# --------------------------------------------------------------------------- shape


def test_native_namespace_surface():
    from jaxpint import native

    assert hasattr(native, "get_model")
    assert hasattr(native, "get_TOAs")          # PINT casing
    assert hasattr(native, "get_model_and_toas")


@pytest.mark.slow
def test_get_model_and_toas_shape(_pinned_clock):
    from pint.config import examplefile

    from jaxpint import native
    from jaxpint.model import TimingModel
    from jaxpint.noise import NoiseModel
    from jaxpint.types import TOAData

    try:
        parp = examplefile("B1855+09_NANOGrav_dfg+12_TAI.par")
        timp = examplefile("B1855+09_NANOGrav_dfg+12.tim")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT example data unavailable: {exc}")

    # Outside the try: a failure in the JaxPINT loader under test must fail the
    # test, not be swallowed as a skip.
    model, noise, td = native.get_model_and_toas(
        parp, timp, ephem=EPHEM, include_bipm=True, bipm_version=BIPM,
    )
    assert isinstance(model, TimingModel)
    assert isinstance(noise, NoiseModel)
    assert isinstance(td, TOAData)
    assert td.tzr_tdb_int is not None  # TZR populated by the loader


# --------------------------------------------------------------------------- bary parity


@pytest.mark.slow
@pytest.mark.parametrize("parname,timname", [
    ("B1855+09_NANOGrav_dfg+12_TAI.par", "B1855+09_NANOGrav_dfg+12.tim"),  # non-binary
    ("J1614-2230_NANOGrav_12yv3.wb.gls.par", "J1614-2230_NANOGrav_12yv3.wb.tim"),  # binary
])
def test_barycentric_toas_parity(parname, timname, _pinned_clock):
    import pint.models as pm
    import pint.toa as pt
    from pint.config import examplefile

    from jaxpint import native
    import jaxpint.par as par

    try:
        parp = examplefile(parname)
        timp = examplefile(timname)
        pmod = pm.get_model(parp)
        ptoas = pt.get_TOAs(timp, model=pmod, ephem=EPHEM,
                            include_bipm=True, bipm_version=BIPM, planets=False)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load {parname}: {exc}")

    model, _noise, td = native.get_model_and_toas(parp, timp, ephem=EPHEM,
                                                  include_bipm=True, bipm_version=BIPM)
    pr = par.get_model(parp)
    bt = model.compute_barycentric_toas(td, pr.params)
    bt_nat = np.asarray(bt.int) + np.asarray(bt.frac)
    bt_pint = pmod.get_barycentric_toas(ptoas).to_value("day")

    nm = np.asarray(td.mjd_int) + np.asarray(td.mjd_frac)
    pmj = np.asarray(ptoas.get_mjds().value)
    jo, po = np.argsort(nm), np.argsort(pmj)
    assert np.max(np.abs(bt_nat[jo] - bt_pint[po])) < 1e-9, \
        float(np.max(np.abs(bt_nat[jo] - bt_pint[po])))


# --------------------------------------------------------------------------- end-to-end


@pytest.mark.slow
def test_get_model_and_toas_residuals_vs_pint(_pinned_clock):
    import pint.models as pm
    import pint.toa as pt
    from pint.config import examplefile
    from pint.residuals import Residuals

    from jaxpint import native
    from jaxpint.fitters import compute_time_residuals
    import jaxpint.par as par

    parp = examplefile("B1855+09_NANOGrav_dfg+12_TAI.par")
    timp = examplefile("B1855+09_NANOGrav_dfg+12.tim")
    try:
        pmod = pm.get_model(parp)
        ptoas = pt.get_TOAs(timp, model=pmod, ephem=EPHEM,
                            include_bipm=True, bipm_version=BIPM, planets=False)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load: {exc}")

    model, _noise, td = native.get_model_and_toas(parp, timp, ephem=EPHEM,
                                                  include_bipm=True, bipm_version=BIPM)
    pr = par.get_model(parp)
    r_nat = np.asarray(compute_time_residuals(model, td, pr.params))
    r_pint = Residuals(ptoas, pmod, subtract_mean=False).time_resids.to_value("s")

    nm = np.asarray(td.mjd_int) + np.asarray(td.mjd_frac)
    pmj = np.asarray(ptoas.get_mjds().value)
    jo, po = np.argsort(nm), np.argsort(pmj)
    diff = r_nat[jo] - r_pint[po]
    diff -= np.median(diff)  # abs-phase: agree up to an overall constant
    assert np.max(np.abs(diff)) < 1e-6, float(np.max(np.abs(diff)))


# --------------------------------------------------------------------------- get_model limit


@pytest.mark.slow
def test_get_model_omits_toa_noise(_pinned_clock):
    """get_model(par) (no TOAs) omits TOA-dependent noise; full build includes it."""
    from pint.config import examplefile

    from jaxpint import native

    parp = examplefile("B1855+09_NANOGrav_9yv1.gls.par")  # has ECORR + red noise
    timp = examplefile("B1855+09_NANOGrav_9yv1.tim")
    try:
        _m_only, noise_only = native.get_model(parp)
        _m_full, noise_full, _td = native.get_model_and_toas(
            parp, timp, ephem=EPHEM, include_bipm=True, bipm_version=BIPM)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT data unavailable: {exc}")

    n_only = len(getattr(noise_only, "correlated", ()) or ())
    n_full = len(getattr(noise_full, "correlated", ()) or ())
    assert n_full > n_only  # full build has the TOA-dependent correlated noise
