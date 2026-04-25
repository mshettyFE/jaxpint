"""Tests for ``jaxpint.loaders.nanograv.load_nanograv_pta``.

Uses PINT's bundled ``B1855+09_NANOGrav_9yv1`` par/tim pair as a stand-in for a
single-pulsar "PTA". Stages the pair into a tmpdir under both Zenodo layouts
(``par/`` + ``tim/`` siblings, and ``<PSR>/<files>``) to exercise the
discovery branch.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pint.config import examplefile

from jaxpint import NanogravPTA, load_nanograv_pta
from jaxpint.pta.likelihood import PTAConfig


def _example_par_tim() -> tuple[Path, Path]:
    return (
        Path(examplefile("B1855+09_NANOGrav_9yv1.gls.par")),
        Path(examplefile("B1855+09_NANOGrav_9yv1.tim")),
    )


def _stage_par_tim_layout(root: Path) -> None:
    """``<root>/par/<stem>.par`` + ``<root>/tim/<stem>.tim``."""
    par_src, tim_src = _example_par_tim()
    (root / "par").mkdir(parents=True)
    (root / "tim").mkdir(parents=True)
    shutil.copy2(par_src, root / "par" / par_src.name)
    shutil.copy2(tim_src, root / "tim" / tim_src.name)


def _stage_per_pulsar_layout(root: Path) -> None:
    """``<root>/<PSR>/<stem>.par`` + sibling ``.tim``."""
    par_src, tim_src = _example_par_tim()
    psr_dir = root / "B1855+09"
    psr_dir.mkdir(parents=True)
    shutil.copy2(par_src, psr_dir / par_src.name)
    shutil.copy2(tim_src, psr_dir / tim_src.name)


@pytest.mark.parametrize(
    "stager", [_stage_par_tim_layout, _stage_per_pulsar_layout]
)
def test_load_nanograv_pta_layouts(tmp_path, stager):
    stager(tmp_path)

    psrs = load_nanograv_pta(tmp_path, planets=False)

    assert isinstance(psrs, NanogravPTA)
    assert psrs.pulsar_names == ("B1855+09",)
    assert (
        len(psrs.toa_data_list)
        == len(psrs.pulsar_params_list)
        == len(psrs.timing_models)
        == len(psrs.noise_models)
        == 1
    )
    # Real TOAs (not synthetic) → at least a few hundred entries.
    assert psrs.toa_data_list[0].mjd_int.shape[0] > 100

    # Result must drop straight into PTAConfig without further massaging.
    cfg = PTAConfig(
        toa_data_list=psrs.toa_data_list,
        timing_models=psrs.timing_models,
        noise_models=psrs.noise_models,
        signal_injectors=(),
    )
    assert cfg.n_pulsars == 1


def test_load_nanograv_pta_pulsar_names_and_exclude(tmp_path):
    _stage_per_pulsar_layout(tmp_path)

    # Explicit selection of a known pulsar works.
    psrs = load_nanograv_pta(
        tmp_path, pulsar_names=["B1855+09"], planets=False
    )
    assert psrs.pulsar_names == ("B1855+09",)

    # Unknown pulsar → KeyError.
    with pytest.raises(KeyError):
        load_nanograv_pta(tmp_path, pulsar_names=["J9999+9999"], planets=False)

    # Excluding the only pulsar empties the set → ValueError.
    with pytest.raises(ValueError):
        load_nanograv_pta(tmp_path, exclude=["B1855+09"], planets=False)


def test_load_nanograv_pta_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_nanograv_pta(tmp_path / "does-not-exist")


def test_load_nanograv_pta_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_nanograv_pta(tmp_path)
