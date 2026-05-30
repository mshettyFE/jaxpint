"""Load the NANOGrav narrowband PTA dataset from disk into JaxPINT.

The NANOGrav 15-Year Data Set (Zenodo DOI ``10.5281/zenodo.8423265``) ships
``.par`` and ``.tim`` files for 68 millisecond pulsars in either of two on-disk
layouts:

- sibling ``par/`` and ``tim/`` directories (``<root>/par/<stem>.par`` +
  ``<root>/tim/<stem>.tim``), or
- one directory per pulsar (``<root>/<PSR>/<stem>.par`` + sibling ``.tim``).

The user downloads + extracts the tarball once and points
:func:`load_nanograv_pta` at the resulting tree. The loader parses each pair
**natively** (no PINT) via JaxPINT's own ``.par`` / ``.tim`` reader, returning
the same tuple-of-tuples shape that :class:`jaxpint.pta.likelihood.PTAConfig`
expects.

The native ``.tim`` reader handles the TEMPO2 line format, which is what the
NANOGrav narrowband releases ship; legacy fixed-column formats are not
supported (see :doc:`/guides/loading_data`).

This first pass supports narrowband only — wideband ingestion uses different
fitting plumbing and will land separately.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, Sequence

from jaxpint.loaders.native import native_toas_to_jax
from jaxpint.model import TimingModel
from jaxpint.model_builder import build_model
from jaxpint.noise import NoiseModel
from jaxpint.par import get_model as parse_par
from jaxpint.types import ParameterVector, TOAData

log = logging.getLogger(__name__)


class NanogravPTA(NamedTuple):
    """Output of :func:`load_nanograv_pta`.

    Tuple fields after ``pulsar_names`` mirror
    ``jaxpint.notebook_utils.SyntheticPTA`` exactly, so the result drops
    straight into :class:`~jaxpint.pta.PTAConfig`::

        psrs = load_nanograv_pta("/data/NG15yr/narrowband")
        cfg = PTAConfig(
            toa_data_list=psrs.toa_data_list,
            timing_models=psrs.timing_models,
            noise_models=psrs.noise_models,
            signal_injectors=(...),
        )
    """

    pulsar_names: tuple[str, ...]
    toa_data_list: tuple[TOAData, ...]
    pulsar_params_list: tuple[ParameterVector, ...]
    timing_models: tuple[TimingModel, ...]
    noise_models: tuple[NoiseModel, ...]


def _find_matching_tim(par_path: Path, psr_name: str) -> Path | None:
    """Locate the ``.tim`` that goes with ``par_path``.

    NANOGrav releases use several conventions: ``X.gls.par`` paired with
    ``X.tim`` (9yr), ``X_PINT_*.nb.par`` paired with same-stem ``.nb.tim``
    (15yr), and Zenodo's split ``par/`` / ``tim/`` layout. We try, in order:

    1. Same directory, identical stem.
    2. Sibling ``tim/`` directory with identical stem (Zenodo split layout).
    3. Any ``.tim`` whose filename starts with the pulsar name in either the
       same directory or the sibling ``tim/`` — accepted only when exactly
       one candidate matches, to avoid silently pairing the wrong file.
    """
    same_stem = par_path.with_suffix(".tim")
    if same_stem.exists():
        return same_stem

    if par_path.parent.name == "par":
        sibling_tim_dir = par_path.parent.parent / "tim"
        sibling_same_stem = sibling_tim_dir / (par_path.stem + ".tim")
        if sibling_same_stem.exists():
            return sibling_same_stem
        candidates = sorted(sibling_tim_dir.glob(f"{psr_name}*.tim"))
        if len(candidates) == 1:
            return candidates[0]

    candidates = sorted(par_path.parent.glob(f"{psr_name}*.tim"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _pair_par_tim(data_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Discover ``{pulsar_name: (par_path, tim_path)}`` pairs under ``data_dir``.

    Pulsar name is the prefix of the ``.par`` filename stem up to the first
    underscore, which captures the standard NANOGrav naming convention
    (``J0030+0451_PINT_20230327.nb.par`` → ``J0030+0451``) and also the simpler
    ``B1855+09.par`` form. ``.tim`` discovery is delegated to
    :func:`_find_matching_tim`.
    """
    pairs: dict[str, tuple[Path, Path]] = {}
    for par_path in sorted(data_dir.rglob("*.par")):
        psr_name = par_path.stem.split("_", 1)[0]
        tim_path = _find_matching_tim(par_path, psr_name)
        if tim_path is None:
            log.warning("No matching .tim for %s — skipping", par_path)
            continue
        if psr_name in pairs:
            log.warning(
                "Duplicate pulsar %s; keeping %s, ignoring %s",
                psr_name,
                pairs[psr_name][0],
                par_path,
            )
            continue
        pairs[psr_name] = (par_path, tim_path)
    return pairs


def load_nanograv_pta(
    data_dir: str | Path,
    *,
    pulsar_names: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    ephem: str = "DE440",
    bipm_version: str = "BIPM2019",
    planets: bool = True,
) -> NanogravPTA:
    """Load a NANOGrav narrowband PTA dataset into JaxPINT.

    Parameters
    ----------
    data_dir
        Path to the extracted ``narrowband/`` directory of the Zenodo archive
        (or any directory tree of paired ``.par``/``.tim`` files in the layouts
        described in the module docstring).
    pulsar_names
        If given, load exactly these pulsars in this order. ``KeyError`` is
        raised if any name is not discovered under ``data_dir``. ``None`` (the
        default) loads every discovered pulsar in sorted order.
    exclude
        Pulsar names to drop after discovery / selection.
    ephem
        Solar System ephemeris passed to the native TOA loader
        (:func:`jaxpint.native.get_TOAs`). Defaults to ``DE440`` to match the
        15yr release's reference analysis.
    bipm_version
        BIPM clock realisation. The loader applies BIPM (``include_bipm=True``)
        with this version.
    planets
        Whether to compute SSB-to-planet position vectors. These are consumed by
        the ``PLANET_SHAPIRO`` delay component (Shapiro delay through the gas
        giants); the default of ``True`` is safe for any model, and pulsars
        without ``PLANET_SHAPIRO`` simply ignore them.

    Returns
    -------
    NanogravPTA
        Per-pulsar tuples ready to feed into
        :class:`jaxpint.pta.likelihood.PTAConfig`.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a directory")

    pairs = _pair_par_tim(root)
    if not pairs:
        raise FileNotFoundError(f"No par/tim pairs found under {root}")

    if pulsar_names is None:
        names = list(pairs.keys())
    else:
        missing = [n for n in pulsar_names if n not in pairs]
        if missing:
            raise KeyError(f"Pulsars not found in {root}: {missing}")
        names = list(pulsar_names)

    excl = set(exclude)
    names = [n for n in names if n not in excl]
    if not names:
        raise ValueError(f"No pulsars left after applying exclude={list(exclude)!r}")

    out_names: list[str] = []
    toa_data_list: list[TOAData] = []
    params_list: list[ParameterVector] = []
    timing_models: list[TimingModel] = []
    noise_models: list[NoiseModel] = []

    for name in names:
        par_path, tim_path = pairs[name]
        log.info("Loading %s from %s", name, par_path)

        par_result = parse_par(str(par_path))
        toa_data = native_toas_to_jax(
            str(tim_path),
            par_result,
            ephem=ephem,
            include_bipm=True,
            bipm_version=bipm_version,
            planets=planets,
        )
        tm, nm = build_model(par_result, toa_data)

        out_names.append(name)
        toa_data_list.append(toa_data)
        params_list.append(par_result.params)
        timing_models.append(tm)
        noise_models.append(nm)

    return NanogravPTA(
        pulsar_names=tuple(out_names),
        toa_data_list=tuple(toa_data_list),
        pulsar_params_list=tuple(params_list),
        timing_models=tuple(timing_models),
        noise_models=tuple(noise_models),
    )
