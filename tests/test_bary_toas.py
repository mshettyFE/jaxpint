"""Tests for TOAData.basis_seconds (the GP basis time coordinate).

``basis_seconds`` is the time coordinate GP Fourier bases and ECORR
quantization are evaluated at.  Every producer chooses it explicitly — there
is deliberately no fallback:

- bridge / native loader: barycentered TOAs (enterprise/discovery
  convention; enterprise's ``PintPulsar._toas``);
- synthetic zero-geometry data (tests/helpers.py): TDB, which equals
  barycentric time exactly there;
- unset + GP component build: hard error (``require_basis_seconds``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from tests.helpers import make_toa_data


def test_synthetic_helper_sets_basis_to_tdb():
    """Zero-geometry synthetic data: basis time is TDB (== barycentric)."""
    toa_data = make_toa_data(n_toas=5)
    assert toa_data.basis_seconds is not None
    assert toa_data.basis_coord == "tdb"
    npt.assert_array_equal(
        np.asarray(toa_data.basis_seconds), np.asarray(toa_data.tdb_seconds)
    )

def test_basis_seconds_and_coord_must_travel_together():
    """__check_init__ invariant: the times are meaningless without their
    coordinate label, and a dangling label is a stale declaration."""
    import dataclasses

    toa_data = make_toa_data(n_toas=5)
    with pytest.raises(ValueError, match="set together"):
        dataclasses.replace(toa_data, basis_coord=None)
    with pytest.raises(ValueError, match="set together"):
        dataclasses.replace(toa_data, basis_seconds=None)
    with pytest.raises(ValueError, match="Unknown basis_coord"):
        dataclasses.replace(toa_data, basis_coord="utc")


def test_pta_config_rejects_mixed_basis_coords():
    """PTAConfig refuses configs whose pulsars declare different frames.

    The correlated kernels compare basis phases between pulsars, so mixed
    coordinates silently corrupt the cross-terms — this is the
    config-construction-time guard for the failure mode that motivated the
    explicit-choice design.
    """
    import dataclasses

    from jaxpint.pta.likelihood import PTAConfig
    from tests.helpers import make_simple_pulsar

    td0, tm0, nm0, _ = make_simple_pulsar(n_toas=10, f0=200.0, f1=-1e-15, seed=1)
    td1, tm1, nm1, _ = make_simple_pulsar(n_toas=10, f0=310.0, f1=-2e-15, seed=2)
    td1 = dataclasses.replace(td1, basis_coord="barycentric")

    with pytest.raises(ValueError, match="Mixed GP basis time coordinates"):
        PTAConfig(
            toa_data_list=(td0, td1),
            timing_models=(tm0, tm1),
            noise_models=(nm0, nm1),
            signal_injectors=(),
            correlated_injectors=(),
        )


def test_unset_basis_seconds_raises():
    """No producer choice -> require_basis_seconds refuses to guess."""
    import dataclasses

    toa_data = dataclasses.replace(
        make_toa_data(n_toas=5), basis_seconds=None, basis_coord=None
    )
    assert toa_data.basis_seconds is None
    with pytest.raises(ValueError, match="basis_seconds is not set"):
        toa_data.require_basis_seconds()


def test_unset_basis_seconds_fails_gp_build(tmp_path):
    """Building a GP noise component without the coordinate is a hard error,
    not a silent convention pick (the failure mode that motivated this)."""
    import dataclasses

    from jaxpint.par import get_model as parse_par
    from jaxpint.model_builder import build_model

    par_path = tmp_path / "red.par"
    par_path.write_text(
        """\
PSR J0000+0000
PEPOCH 55000
F0 100 1
TNREDAMP -13
TNREDGAM 3.5
TNREDC 10
UNITS TDB
"""
    )
    par = parse_par(str(par_path))
    toa_data = dataclasses.replace(
        make_toa_data(n_toas=8), basis_seconds=None, basis_coord=None
    )
    with pytest.raises(ValueError, match="basis_seconds is not set"):
        build_model(par, toa_data)


def test_with_basis_seconds_roundtrip():
    """with_basis_seconds sets the field and leaves everything else alone."""
    import dataclasses

    toa_data = dataclasses.replace(
        make_toa_data(n_toas=5), basis_seconds=None, basis_coord=None
    )
    basis = np.arange(5, dtype=np.float64) * 86400.0
    updated = toa_data.with_basis_seconds(basis, "tdb")

    assert updated.basis_seconds is not None
    assert updated.basis_coord == "tdb"
    npt.assert_array_equal(np.asarray(updated.basis_seconds), basis)
    assert updated.basis_seconds.dtype == jnp.float64
    # untouched leaves and the original object
    assert toa_data.basis_seconds is None
    npt.assert_array_equal(
        np.asarray(updated.tdb_seconds), np.asarray(toa_data.tdb_seconds)
    )
    assert updated.n_toas == toa_data.n_toas


def _example_par_tim() -> tuple[Path, Path]:
    from pint.config import examplefile

    return (
        Path(examplefile("B1855+09_NANOGrav_9yv1.gls.par")),
        Path(examplefile("B1855+09_NANOGrav_9yv1.tim")),
    )


@pytest.mark.slow
def test_bridge_matches_pint(_pinned_clock):
    """Bridge basis_seconds == enterprise's own computation, bit for bit.

    Enterprise's PintPulsar does
    ``np.array(model.get_barycentric_toas(toas).value, dtype=float64) * 86400``;
    the bridge must reproduce that exactly (same call, same float64
    truncation point) so the bases inherit enterprise's numbers.
    """
    from pint.models import get_model_and_toas

    from jaxpint.bridge import pint_toas_to_jax

    par, tim = _example_par_tim()
    model, toas = get_model_and_toas(str(par), str(tim))
    toa_data = pint_toas_to_jax(toas, model=model)

    assert toa_data.basis_seconds is not None
    assert toa_data.basis_coord == "barycentric"
    expected = (
        np.array(model.get_barycentric_toas(toas).value, dtype=np.float64) * 86400.0
    )
    npt.assert_array_equal(np.asarray(toa_data.basis_seconds), expected)

    # Sanity on the geometry: differs from TDB by solar-system/dispersion
    # delays (sub-Roemer-scale, nonzero).
    gap = np.abs(np.asarray(toa_data.basis_seconds) - np.asarray(toa_data.tdb_seconds))
    assert 0.0 < gap.max() < 600.0


@pytest.mark.slow
def test_bridge_noise_basis_built_at_bary_times():
    """Bridge-built PLRedNoise evaluates its Fourier basis at basis_seconds.

    Wiring test for the enterprise/discovery basis-time convention: the
    stored basis must equal build_fourier_basis at the barycentered times
    (and, for a real observatory, therefore differ from a TDB-built basis
    by the differential Roemer delay).
    """
    import io

    import astropy.units as u
    import pint.models as models
    from pint.simulation import make_fake_toas_uniform

    from jaxpint.bridge import build_timing_model, pint_toas_to_jax
    from jaxpint.noise.red_noise import PLRedNoise
    from jaxpint.utils import build_fourier_basis

    par = """\
PSR           J0000+0000
RAJ           05:00:00   1
DECJ          +20:00:00  1
PEPOCH        55000
F0            100        1
F1            -1e-15     1
DM            15         1
TNREDAMP      -13
TNREDGAM      3.5
TNREDC        10
TZRMJD        55000
TZRFRQ        1400
TZRSITE       @
EPHEM         DE421
CLOCK         TT(BIPM2019)
UNITS         TDB
"""
    m = models.get_model(io.StringIO(par))
    toas = make_fake_toas_uniform(
        54500, 55500, 100, model=m, obs="gbt", freq=1400.0,
        error=1.0 * u.us, add_noise=False,
    )
    toas.compute_TDBs()

    toa_data = pint_toas_to_jax(toas, model=m)
    _tm, noise_model = build_timing_model(m, toas)
    red = next(c for c in noise_model.correlated if isinstance(c, PLRedNoise))

    bary = np.asarray(toa_data.basis_seconds)
    tspan = float(bary.max() - bary.min())
    F_expected, _, _ = build_fourier_basis(bary, 10, tspan)
    npt.assert_array_equal(np.asarray(red.fourier_basis), np.asarray(F_expected))

    tdb = np.asarray(toa_data.tdb_seconds)
    F_tdb, _, _ = build_fourier_basis(tdb, 10, float(tdb.max() - tdb.min()))
    assert np.abs(np.asarray(red.fourier_basis) - np.asarray(F_tdb)).max() > 1e-6, (
        "bridge basis matches the TDB-built basis — the barycentric "
        "convention switch has been undone"
    )


@pytest.mark.slow
def test_native_loader_matches_pint(tmp_path, _pinned_clock):
    """Native loader's basis_seconds matches PINT's get_barycentric_toas,
    and the loader's precomputed noise bases are built at those times.

    The second assertion guards the build ordering: the times must be
    attached BEFORE the noise model is built, or the intrinsic-noise bases
    silently end up on a different time coordinate than the injectors
    (the bug that motivated the no-fallback design).
    """
    from pint.models import get_model_and_toas

    from jaxpint import load_nanograv_pta
    from jaxpint.noise.red_noise import PLRedNoise
    from jaxpint.utils import build_fourier_basis

    par, tim = _example_par_tim()
    (tmp_path / "par").mkdir()
    (tmp_path / "tim").mkdir()
    shutil.copy2(par, tmp_path / "par" / par.name)
    shutil.copy2(tim, tmp_path / "tim" / tim.name)

    # Ephemeris/BIPM pinned identically on both sides: the loader's defaults
    # (DE440/BIPM2019) deliberately override the par's EPHEM DE421, and the
    # DE421->DE440 SSB relocation alone shifts the Roemer delay by ~60 us.
    psrs = load_nanograv_pta(
        tmp_path, planets=False, ephem="DE421", bipm_version="BIPM2019"
    )
    toa_data = psrs.toa_data_list[0]
    assert toa_data.basis_seconds is not None

    model, toas = get_model_and_toas(
        str(par), str(tim), ephem="DE421", include_bipm=True, bipm_version="BIPM2019"
    )
    expected = (
        np.array(model.get_barycentric_toas(toas).value, dtype=np.float64) * 86400.0
    )
    # PINT TOAs keep file order; the native loader also preserves it, so no
    # sorting is needed.  Achieved agreement: max ~9.5e-7 s, which is the
    # pre-existing native-vs-PINT *TDB* parity for these TOAs (the raw tdbld
    # columns differ by the same amount); the delay part agrees far below
    # that.  The assert message records the achieved level.
    diff = np.abs(np.asarray(toa_data.basis_seconds) - expected)
    npt.assert_allclose(
        np.asarray(toa_data.basis_seconds),
        expected,
        atol=1e-6,
        rtol=0,
        err_msg=f"native basis TOAs vs PINT: max |diff| = {diff.max():.3e} s",
    )

    # Build-ordering guard: the precomputed red-noise basis must be evaluated
    # at basis_seconds, not at TDB.
    red = next(
        c for c in psrs.noise_models[0].correlated if isinstance(c, PLRedNoise)
    )
    basis_t = np.asarray(toa_data.basis_seconds)
    F_expected, _, _ = build_fourier_basis(
        basis_t, red.freqs.shape[0], float(basis_t.max() - basis_t.min())
    )
    npt.assert_array_equal(np.asarray(red.fourier_basis), np.asarray(F_expected))
