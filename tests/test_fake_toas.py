"""Tests for fake-TOA generation from scratch (no ``.tim`` file).

Covers :func:`jaxpint.simulation.make_toa_data_from_mjds`,
:func:`make_uniform_toa_data` and :func:`make_fake_toas_uniform` -- the
counterparts of PINT's ``make_fake_toas_fromMJDs`` / ``make_fake_toas_uniform``
"""

from __future__ import annotations

import pathlib

import jax
import numpy as np
import pytest

import jaxpint.par as jpar
from jaxpint import build_model, native
from jaxpint.fitters import compute_time_residuals
from jaxpint.simulation import (
    make_fake_toas_uniform,
    make_toa_data_from_mjds,
    make_uniform_toa_data,
)

_DATA = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"


@pytest.fixture(scope="module")
def ngc6440e():
    return jpar.get_model(str(_DATA / "NGC6440E.par"))


# ---------------------------------------------------------------------------
# Epoch scaffolding
# ---------------------------------------------------------------------------


def test_uniform_grid_endpoints_and_count(ngc6440e):
    td = make_uniform_toa_data(53000.0, 54000.0, 11, ngc6440e, obs="gbt")
    mjd = np.asarray(td.mjd_int) + np.asarray(td.mjd_frac)
    assert td.n_toas == 11
    # linspace convention, both ends included -- matches PINT.
    assert mjd[0] == pytest.approx(53000.0, abs=1e-9)
    assert mjd[-1] == pytest.approx(54000.0, abs=1e-9)


def test_from_mjds_preserves_epochs_and_units(ngc6440e):
    mjds = np.array([53001.25, 53500.5, 53999.75])
    td = make_toa_data_from_mjds(
        mjds, ngc6440e, obs="gbt", freq_mhz=820.0, error_us=2.5
    )
    mjd = np.asarray(td.mjd_int) + np.asarray(td.mjd_frac)
    np.testing.assert_allclose(mjd, mjds, atol=1e-9)
    assert np.all(np.asarray(td.freq) > 0)  # barycentric-corrected, still ~820
    np.testing.assert_allclose(np.asarray(td.error), 2.5e-6, rtol=1e-12)


def test_frequency_array_is_cycled(ngc6440e):
    """PINT's multi-frequency convention: an array of frequencies alternates
    across the grid. Cycled at the raw level -- checked against the parsed
    records via a dispersion-free site to keep frequencies unmodified."""
    td = make_toa_data_from_mjds(
        np.linspace(53000, 53010, 6), ngc6440e, obs="@", freq_mhz=[430.0, 1400.0]
    )
    np.testing.assert_array_equal(
        np.asarray(td.freq), [430.0, 1400.0, 430.0, 1400.0, 430.0, 1400.0]
    )


def test_zero_frequency_means_infinite(ngc6440e):
    """freq_mhz=0 -> inf, the .tim parser's rule, applied to synthetic TOAs too."""
    td = make_toa_data_from_mjds([53000.5], ngc6440e, obs="@", freq_mhz=0.0)
    assert np.isinf(np.asarray(td.freq)).all()


def test_bad_grid_arguments_raise(ngc6440e):
    with pytest.raises(ValueError, match="n_toas"):
        make_uniform_toa_data(53000.0, 54000.0, 0, ngc6440e)
    with pytest.raises(ValueError, match="1-D"):
        make_toa_data_from_mjds(np.zeros((2, 2)), ngc6440e)


# ---------------------------------------------------------------------------
# Model realization (zero_residuals through the generator)
# ---------------------------------------------------------------------------


def test_fake_toas_realize_the_model(ngc6440e):
    """The generated TOAs encode the timing model to < 1 ns."""
    td = make_fake_toas_uniform(53000.0, 54000.0, 50, ngc6440e, obs="gbt")
    tm, _ = build_model(ngc6440e, td)
    r = np.asarray(compute_time_residuals(tm, td, ngc6440e.params))
    assert np.abs(r).max() < 1e-9


def test_barycentric_fake_toas(ngc6440e):
    """obs="@" at infinite frequency: the from-scratch barycentric path.

    Pins the TDB-native-site handling end to end: TDB equals the recorded MJD
    exactly (no UTC->TDB conversion for a site that records TDB), the
    SSB-relative posvels are zero, and the model is still realized to < 1 ns.
    """
    td = make_fake_toas_uniform(53000.0, 54000.0, 50, ngc6440e, obs="@", freq_mhz=0.0)
    tdb = np.asarray(td.tdb_int) + np.asarray(td.tdb_frac)
    mjd = np.asarray(td.mjd_int) + np.asarray(td.mjd_frac)
    np.testing.assert_array_equal(tdb, mjd)
    assert not np.asarray(td.ssb_obs_pos).any()
    assert not np.asarray(td.ssb_obs_vel).any()

    tm, _ = build_model(ngc6440e, td)
    r = np.asarray(compute_time_residuals(tm, td, ngc6440e.params))
    assert np.abs(r).max() < 1e-9


def test_add_noise_scales_with_errors(ngc6440e):
    """add_noise perturbs by N(0, error): rms ~ error_us, and reproducible."""
    kw = dict(obs="gbt", error_us=1.0, add_noise=True, key=jax.random.PRNGKey(7))
    td = make_fake_toas_uniform(53000.0, 54000.0, 400, ngc6440e, **kw)
    tm, _ = build_model(ngc6440e, td)
    r = np.asarray(compute_time_residuals(tm, td, ngc6440e.params))
    # 400 draws: rms within ~5 sigma of 1 us (sigma_rms ~ 1/sqrt(2n) ~ 3.5%)
    assert 0.8e-6 < r.std() < 1.2e-6
    # Same key -> identical realization; the generator is deterministic.
    td2 = make_fake_toas_uniform(53000.0, 54000.0, 400, ngc6440e, **kw)
    np.testing.assert_array_equal(np.asarray(td.mjd_frac), np.asarray(td2.mjd_frac))


def test_noise_requires_a_key(ngc6440e):
    with pytest.raises(ValueError, match="PRNG key"):
        make_fake_toas_uniform(53000.0, 54000.0, 5, ngc6440e, add_noise=True)


# ---------------------------------------------------------------------------
# Barycentre fix: the previously-unloadable files
# ---------------------------------------------------------------------------


def test_barycentred_tim_files_load():
    """The three vendored files with "@" TOAs, all unloadable before the fix.

    Kept with the generator tests because the generator's obs="@" support and
    these loads are the same code path (core_from_raw_toas' timescale
    partition).
    """
    for name, n in (("parkes.toa", 8), ("piecewise.tim", 6), ("slug.tim", 6)):
        td = native.get_TOAs(str(_DATA / name))
        assert td.n_toas == n, name
        tdb = np.asarray(td.tdb_int) + np.asarray(td.tdb_frac)
        mjd = np.asarray(td.mjd_int) + np.asarray(td.mjd_frac)
        np.testing.assert_array_equal(tdb, mjd, err_msg=name)


def test_parkes_toa_tdb_matches_pint_reference():
    """PINT's tdbld for parkes.toa[0], pinned as a literal.

    58852.7590686063892 is what PINT computes for this file (it special-cases
    BarycenterObs the same way). Float64 int+frac carries ~2e-12 day here.
    """
    td = native.get_TOAs(str(_DATA / "parkes.toa"))
    tdb0 = float(np.asarray(td.tdb_int)[0]) + float(np.asarray(td.tdb_frac)[0])
    assert tdb0 == pytest.approx(58852.7590686063892, abs=5e-12)


def test_utc_site_without_coordinates_rejected(ngc6440e):
    """stl_geo records UTC but has no ITRF coordinates: must raise clearly.

    The discriminator for the TDB bypass is the *timescale*, not "has no
    coordinates" -- keying on coordinates would silently treat this spacecraft
    placeholder as the barycentre and misconvert its TOAs. Before the fix this
    was a cryptic NaN crash inside astropy; now it names the site.
    """
    with pytest.raises(ValueError, match="stl_geo.*no ITRF"):
        make_toa_data_from_mjds([53000.5], ngc6440e, obs="stl_geo")


# ---------------------------------------------------------------------------
# Cross-implementation
# ---------------------------------------------------------------------------


def test_pint_generated_fake_toas_agree():
    """TOAs faked by PINT evaluate to ~zero residuals under JaxPINT's model.

    The generators themselves cannot be compared array-by-array (each
    implementation zeroes against its own model), so the meaningful statement
    is cross-evaluation: PINT zeroes to 1 ns against its model, and JaxPINT's
    model agrees about those same TOAs to a few ns. Bridged, not parsed -- the
    fake TOAs never touch disk.
    """
    pytest.importorskip("pint")
    import astropy.units as u
    import pint.models
    import pint.simulation
    from jaxpint.bridge import (
        build_timing_model,
        pint_model_to_params,
        pint_toas_to_jax,
    )

    m = pint.models.get_model(str(_DATA / "NGC6440E.par"))
    t = pint.simulation.make_fake_toas_uniform(
        53000, 54000, 50, m, obs="GBT", freq=1400 * u.MHz, error=1 * u.us
    )
    toa_data = pint_toas_to_jax(t, model=m)
    params = pint_model_to_params(m).params
    tm, _noise = build_timing_model(m)
    r = np.asarray(compute_time_residuals(tm, toa_data, params))
    assert np.abs(r).max() < 1e-8  # measured 1.9e-9


def test_grid_builders_accept_no_par_but_the_realizer_does_not():
    """par_result=None: scaffolding works with defaults, realizing raises clearly.

    The grid builders mirror native_toas_to_jax's optional par (DE440, packaged
    clock, topocentric freq, no basis). make_fake_toas_uniform cannot work
    without a par -- zeroing needs a model -- and must say so, not surface an
    AttributeError from inside build_model.
    """
    td = make_toa_data_from_mjds([53000.5, 53100.5], None, obs="gbt")
    assert td.n_toas == 2
    assert td.clock_realization == "TT(BIPM2023)"  # packaged default
    with pytest.raises(ValueError, match="needs a ParResult"):
        make_fake_toas_uniform(53000.0, 53100.0, 5, None)
