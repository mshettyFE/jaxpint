"""Tests for TOAData.bary_seconds (barycentered TOAs, enterprise convention).

``bary_seconds`` is the time coordinate enterprise/discovery evaluate their GP
Fourier bases at (``PintPulsar._toas``)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from tests.helpers import make_toa_data


def test_default_is_none():
    """No model at conversion time -> no barycentered TOAs."""
    toa_data = make_toa_data(n_toas=5)
    assert toa_data.bary_seconds is None


def test_with_bary_seconds_roundtrip():
    """with_bary_seconds sets the field and leaves everything else alone."""
    toa_data = make_toa_data(n_toas=5)
    bary = np.arange(5, dtype=np.float64) * 86400.0
    updated = toa_data.with_bary_seconds(bary)

    assert updated.bary_seconds is not None
    npt.assert_array_equal(np.asarray(updated.bary_seconds), bary)
    assert updated.bary_seconds.dtype == jnp.float64
    # untouched leaves and the original object
    assert toa_data.bary_seconds is None
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
    """Bridge field == enterprise's own computation, bit for bit.

    Enterprise's PintPulsar does
    ``np.array(model.get_barycentric_toas(toas).value, dtype=float64) * 86400``;
    the bridge must reproduce that exactly (same call, same float64
    truncation point) so a future basis switch inherits enterprise's numbers.
    """
    from pint.models import get_model_and_toas

    from jaxpint.bridge import pint_toas_to_jax

    par, tim = _example_par_tim()
    model, toas = get_model_and_toas(str(par), str(tim))
    toa_data = pint_toas_to_jax(toas, model=model)

    assert toa_data.bary_seconds is not None
    expected = (
        np.array(model.get_barycentric_toas(toas).value, dtype=np.float64) * 86400.0
    )
    npt.assert_array_equal(np.asarray(toa_data.bary_seconds), expected)

    # Sanity on the geometry: differs from TDB by solar-system/dispersion
    # delays (sub-Roemer-scale, nonzero).
    gap = np.abs(np.asarray(toa_data.bary_seconds) - np.asarray(toa_data.tdb_seconds))
    assert 0.0 < gap.max() < 600.0


@pytest.mark.slow
def test_native_loader_matches_pint(tmp_path, _pinned_clock):
    """Native loader's bary_seconds matches PINT's get_barycentric_toas.
    """
    from pint.models import get_model_and_toas

    from jaxpint import load_nanograv_pta

    par, tim = _example_par_tim()
    (tmp_path / "par").mkdir()
    (tmp_path / "tim").mkdir()
    shutil.copy2(par, tmp_path / "par" / par.name)
    shutil.copy2(tim, tmp_path / "tim" / tim.name)

    psrs = load_nanograv_pta(
        tmp_path, planets=False, ephem="DE421", bipm_version="BIPM2019"
    )
    toa_data = psrs.toa_data_list[0]
    assert toa_data.bary_seconds is not None

    model, toas = get_model_and_toas(
        str(par), str(tim), ephem="DE421", include_bipm=True, bipm_version="BIPM2019"
    )
    expected = (
        np.array(model.get_barycentric_toas(toas).value, dtype=np.float64) * 86400.0
    )
    diff = np.abs(np.asarray(toa_data.bary_seconds) - expected)
    npt.assert_allclose(
        np.asarray(toa_data.bary_seconds),
        expected,
        atol=1e-6,
        rtol=0,
        err_msg=f"native bary TOAs vs PINT: max |diff| = {diff.max():.3e} s",
    )
