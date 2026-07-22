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


# ---------------------------------------------------------------------------
# setup_synthetic_pta: the notebook cascade point, now native
# ---------------------------------------------------------------------------


def _random_par_strings(n):
    from jaxpint.notebook_utils import generate_random_par

    rng = np.random.default_rng(1234)
    return [generate_random_par(i, rng, start_mjd=57000.0) for i in range(n)]


def test_setup_synthetic_pta_is_pint_free():
    """Par strings in, fully-built native PTA out -- no PINT anywhere.

    This is the function all 16 example notebooks route through; it used to
    generate via pint.simulation and the bridge. The assertion that matters is
    the last one: every pulsar's TOAs realize its own timing model to < 1 ns,
    which is the property the notebooks rely on when they inject signals on
    top of "quiet" TOAs.
    """
    from jaxpint.notebook_utils import setup_synthetic_pta

    pars = _random_par_strings(3)
    synth = setup_synthetic_pta(
        pars, start_mjd=57000.0, end_mjd=58000.0, n_toas=40,
        toa_error_s=1e-7, freq_mhz=1400.0,
    )
    assert len(synth.toa_data_list) == len(synth.timing_models) == 3
    for td, tm, params in zip(
        synth.toa_data_list, synth.timing_models, synth.pulsar_params_list
    ):
        assert td.n_toas == 40
        np.testing.assert_allclose(np.asarray(td.error), 1e-7, rtol=1e-12)
        r = np.asarray(compute_time_residuals(tm, td, params))
        assert np.abs(r).max() < 1e-9


def test_setup_synthetic_pta_custom_mjds():
    """mjds_per_pulsar mode: each pulsar gets exactly its own epochs."""
    from jaxpint.notebook_utils import setup_synthetic_pta

    pars = _random_par_strings(2)
    grids = [np.linspace(57000, 57500, 10), np.linspace(57100, 57900, 17)]
    synth = setup_synthetic_pta(
        pars, mjds_per_pulsar=grids, toa_error_s=1e-7, freq_mhz=1400.0
    )
    for td, grid in zip(synth.toa_data_list, grids):
        assert td.n_toas == len(grid)
        mjd = np.asarray(td.mjd_int) + np.asarray(td.mjd_frac)
        # Zeroing shifts timestamps by (sub-second) residuals, not by epochs.
        np.testing.assert_allclose(mjd, grid, atol=1.0)


def test_setup_synthetic_pta_rejects_mismatched_mjds():
    from jaxpint.notebook_utils import setup_synthetic_pta

    with pytest.raises(ValueError, match="mjds_per_pulsar"):
        setup_synthetic_pta(
            _random_par_strings(2),
            mjds_per_pulsar=[np.linspace(57000, 57500, 5)],
            toa_error_s=1e-7,
            freq_mhz=1400.0,
        )
    with pytest.raises(ValueError, match="Uniform mode"):
        setup_synthetic_pta(_random_par_strings(1), toa_error_s=1e-7, freq_mhz=1400.0)


def test_setup_synthetic_pta_accepts_pint_models():
    """Backward compatibility: the notebooks pass parsed PINT models.

    Round-trips through the PINT model's own as_parfile() into the native
    parser, so generation stays native while old callers keep working. The
    residual check is the proof the round-trip preserved the model.
    """
    pytest.importorskip("pint")
    from io import StringIO

    import pint.models as pm

    from jaxpint.notebook_utils import setup_synthetic_pta

    par = _random_par_strings(1)[0]
    model = pm.get_model(StringIO(par))
    synth = setup_synthetic_pta(
        [model], start_mjd=57000.0, end_mjd=57500.0, n_toas=20,
        toa_error_s=1e-7, freq_mhz=1400.0,
    )
    td, tm, params = (
        synth.toa_data_list[0], synth.timing_models[0], synth.pulsar_params_list[0]
    )
    r = np.asarray(compute_time_residuals(tm, td, params))
    assert np.abs(r).max() < 1e-9


def test_as_par_result_rejects_junk():
    from jaxpint.notebook_utils import _as_par_result

    with pytest.raises(TypeError, match="cannot interpret"):
        _as_par_result(42)


def test_get_model_reads_file_like(ngc6440e):
    """get_model(StringIO(text)) == get_model(path), PINT's convention."""
    from io import StringIO

    from_text = jpar.get_model(StringIO((_DATA / "NGC6440E.par").read_text()))
    np.testing.assert_array_equal(
        np.asarray(from_text.params.values), np.asarray(ngc6440e.params.values)
    )
    assert from_text.params.names == ngc6440e.params.names


# ---------------------------------------------------------------------------
# pulsar_positions_from_models / build_cw_injectors: native sky positions
# ---------------------------------------------------------------------------


def test_pulsar_positions_native_inputs():
    """Par strings in, unit vectors out -- no PINT objects anywhere."""
    from jaxpint.notebook_utils import pulsar_positions_from_models

    pars = _random_par_strings(4)
    pos = np.asarray(pulsar_positions_from_models(pars))
    assert pos.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), 1.0, rtol=1e-12)


def test_pulsar_positions_match_pint_model_input():
    """The same par through a PINT model gives the same vectors to 1e-12.

    This is the parity claim for the native rewrite: RAJ/DECJ read from the
    ParameterVector (radians) must equal PINT's quantity conversion. The
    tolerance absorbs the as_parfile round-trip's decimal formatting.
    """
    pytest.importorskip("pint")
    from io import StringIO

    import pint.models as pm

    from jaxpint.notebook_utils import pulsar_positions_from_models

    pars = _random_par_strings(3)
    native = np.asarray(pulsar_positions_from_models(pars))
    via_pint = np.asarray(
        pulsar_positions_from_models([pm.get_model(StringIO(p)) for p in pars])
    )
    np.testing.assert_allclose(native, via_pint, atol=1e-12)


def test_build_cw_injectors_native():
    """The injector builder accepts par strings end to end."""
    from jaxpint.notebook_utils import build_cw_injectors

    rng = np.random.default_rng(5)
    injectors, positions = build_cw_injectors(
        _random_par_strings(3), n_sources=2, rng=rng
    )
    assert len(injectors) == 2
    assert np.asarray(positions).shape == (3, 3)


def test_pulsar_positions_rejects_ecliptic():
    """Ecliptic-only astrometry raises instead of returning garbage."""
    from jaxpint.notebook_utils import pulsar_positions_from_models

    par = (
        "PSR J0000+0000\nPEPOCH 54000\nF0 100.0 1\nDM 15.0\n"
        "ELONG 120.0\nELAT 30.0\n"
    )
    with pytest.raises(ValueError, match="RAJ/DECJ"):
        pulsar_positions_from_models([par])
