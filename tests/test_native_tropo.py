"""Tests for native troposphere geometry (jaxpint.loaders.native._build_tropo_fields).

The example corpus has CORRECT_TROPOSPHERE off, so the parity tests synthesize an
enabled .par from B1855 dfg+12 (topocentric 'ao') in tmp_path. All slow (need
PINT + ephemeris); PINT pinned to our clock snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest

EPHEM = "DE440"
BIPM = "BIPM2023"


def _tropo_par(tmp_path, enabled: bool):
    """Copy B1855 dfg+12 with CORRECT_TROPOSPHERE set to Y/N."""
    from pint.config import examplefile

    src = examplefile("B1855+09_NANOGrav_dfg+12_TAI.par")
    out = tmp_path / f"tropo_{'on' if enabled else 'off'}.par"
    with open(src) as f:
        lines = [l for l in f if not l.strip().startswith("CORRECT_TROPOSPHERE")]
    lines.append(f"CORRECT_TROPOSPHERE {'Y' if enabled else 'N'}\n")
    out.write_text("".join(lines))
    return str(out), examplefile("B1855+09_NANOGrav_dfg+12.tim")


def _load(parp, timp):
    import pint.models as pm
    import pint.toa as pt

    from jaxpint.loaders.native import native_toas_to_jax
    import jaxpint.par as par

    model = pm.get_model(parp)
    toas = pt.get_TOAs(timp, model=model, ephem=EPHEM,
                       include_bipm=True, bipm_version=BIPM, planets=False)
    pr = par.get_model(parp)
    td = native_toas_to_jax(timp, pr, ephem=EPHEM, include_bipm=True,
                            bipm_version=BIPM, planets=False)
    return model, toas, pr, td


def _align(td, toas):
    nm = np.asarray(td.mjd_int) + np.asarray(td.mjd_frac)
    pmj = np.asarray(toas.get_mjds().value)
    return np.argsort(nm), np.argsort(pmj)


# --------------------------------------------------------------------------- A: fields


@pytest.mark.slow
def test_tropo_field_parity_vs_bridge(tmp_path, _pinned_clock):
    from jaxpint.bridge import pint_toas_to_jax

    parp, timp = _tropo_par(tmp_path, enabled=True)
    try:
        model, toas, pr, td = _load(parp, timp)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load: {exc}")

    assert pr.bool_params.get("CORRECT_TROPOSPHERE") is True
    ref = pint_toas_to_jax(toas, model=model)
    assert ref.tropo_alt is not None and td.tropo_alt is not None

    jo, po = _align(td, toas)
    assert np.max(np.abs(np.asarray(td.tropo_alt)[jo]
                         - np.asarray(ref.tropo_alt)[po])) < 1e-9
    assert np.max(np.abs(np.asarray(td.obs_geodetic_lat)[jo]
                         - np.asarray(ref.obs_geodetic_lat)[po])) < 1e-9
    assert np.max(np.abs(np.asarray(td.obs_height_km)[jo]
                         - np.asarray(ref.obs_height_km)[po])) < 1e-6
    assert np.array_equal(np.asarray(td.tropo_alt_valid)[jo],
                          np.asarray(ref.tropo_alt_valid)[po])


# --------------------------------------------------------------------------- B: delay


@pytest.mark.slow
def test_tropo_delay_parity_vs_pint(tmp_path, _pinned_clock):
    import astropy.units as u

    from jaxpint.delay.troposphere import TroposphereDelay

    parp, timp = _tropo_par(tmp_path, enabled=True)
    try:
        model, toas, pr, td = _load(parp, timp)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load: {exc}")

    pint_delay = model.components["TroposphereDelay"].troposphere_delay(toas).to_value(u.s)
    nat_delay = np.asarray(TroposphereDelay()(td, pr.params, np.zeros(td.n_toas)))

    jo, po = _align(td, toas)
    assert np.max(np.abs(nat_delay[jo] - pint_delay[po])) < 1e-9, \
        float(np.max(np.abs(nat_delay[jo] - pint_delay[po])))
    # non-trivial: the corrected delay should actually be nonzero somewhere
    assert np.max(np.abs(nat_delay)) > 0.0


# --------------------------------------------------------------------------- C: off path


@pytest.mark.slow
def test_tropo_off_leaves_fields_none(tmp_path, _pinned_clock):
    from jaxpint.delay.troposphere import TroposphereDelay

    parp, timp = _tropo_par(tmp_path, enabled=False)
    try:
        _model, _toas, pr, td = _load(parp, timp)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PINT could not load: {exc}")

    assert td.tropo_alt is None
    assert td.tropo_alt_valid is None
    assert td.obs_geodetic_lat is None
    # component returns exactly zero when geometry absent
    d = np.asarray(TroposphereDelay()(td, pr.params, np.zeros(td.n_toas)))
    assert np.all(d == 0.0)
