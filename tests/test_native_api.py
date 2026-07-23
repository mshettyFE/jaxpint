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

@pytest.mark.slow
def test_residuals_vs_pint_with_padd(tmp_path, _pinned_clock):
    """End-to-end residual parity vs PINT for a fractional ``-padd`` offset.

    Unlike an integer PHASE command (inert under nearest-pulse tracking), a
    *fractional* ``-padd`` shifts the phase residual — and must match PINT, which
    folds ``-padd`` into ``delta_pulse_number`` and adds it to the model phase
    before wrapping to the nearest pulse.  This guards the ``compute_phase_residuals``
    fix that adds ``delta_pulse_number`` to the *fractional* phase rather than the
    (discarded) integer slot; with the old code the ``0.25``-turn shift vanished.
    """
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
        # Add a fractional -padd to the tail half of the TOAs.
        lines = open(timp).read().splitlines()
        data = [
            i for i, ln in enumerate(lines)
            if len(ln.split()) >= 4 and ln.split()[0] not in ("FORMAT", "MODE", "C")
        ]
        mod = list(lines)
        for i in data[len(data) // 2:]:
            mod[i] += " -padd 0.25"
        mtim = tmp_path / "b1855_padd.tim"
        mtim.write_text("\n".join(mod) + "\n")

        pmod = pm.get_model(parp)
        ptoas = pt.get_TOAs(str(mtim), model=pmod, ephem=EPHEM,
                            include_bipm=True, bipm_version=BIPM, planets=False)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load: {exc}")

    # JaxPINT loader outside the try: a failure here is a real bug, not a skip.
    model, _noise, td = native.get_model_and_toas(parp, str(mtim), ephem=EPHEM,
                                                  include_bipm=True, bipm_version=BIPM)
    pr = par.get_model(parp)

    # A genuinely *fractional* offset is present (else the fix goes untested).
    dpn = np.asarray(td.delta_pulse_number)
    assert np.count_nonzero(dpn) > 0 and np.any(dpn % 1.0 != 0.0)

    r_nat = np.asarray(compute_time_residuals(model, td, pr.params))
    r_pint = Residuals(ptoas, pmod, subtract_mean=False,
                       track_mode="nearest").time_resids.to_value("s")
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
    # native.get_model / get_model_and_toas ARE the code under test -- call them
    # directly. A failure here is a real bug and must surface, not be swallowed
    # as a "PINT data unavailable" skip (the example-file lookups above are the
    # only data-availability step, and they are outside any skip guard).
    _m_only, noise_only = native.get_model(parp)
    _m_full, noise_full, _td = native.get_model_and_toas(
        parp, timp, ephem=EPHEM, include_bipm=True, bipm_version=BIPM)

    n_only = len(getattr(noise_only, "correlated", ()) or ())
    n_full = len(getattr(noise_full, "correlated", ()) or ())
    assert n_full > n_only  # full build has the TOA-dependent correlated noise
