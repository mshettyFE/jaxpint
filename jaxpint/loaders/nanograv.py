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

import gc
import logging
import warnings
from pathlib import Path
from typing import Callable, Collection, Iterator, NamedTuple, Sequence, TypeVar

from jaxpint.loaders.native import native_toas_to_jax
from jaxpint.model import TimingModel
from jaxpint.model_builder import build_model
from jaxpint.noise import NoiseModel
from jaxpint.par import get_model as parse_par
from jaxpint.types import ParameterVector, TOAData

log = logging.getLogger(__name__)

T = TypeVar("T")


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


class PulsarRecord(NamedTuple):
    """One pulsar's fully-built inputs, as yielded by :func:`iter_nanograv_pta`."""

    name: str
    toa_data: TOAData
    params: ParameterVector
    timing_model: TimingModel
    noise_model: NoiseModel


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


def _resolve_pairs(
    data_dir: str | Path,
    pulsar_names: Sequence[str] | None,
    exclude: Collection[str],
) -> list[tuple[str, Path, Path]]:
    """Discover + select + order the ``(name, par, tim)`` work list.

    Shared front half of :func:`load_nanograv_pta` and
    :func:`iter_nanograv_pta`, so both resolve identically.
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

    return [(n, *pairs[n]) for n in names]


class MixedClockRealization(UserWarning):
    """Pulsars in one PTA were built against different clock realizations."""


def _check_uniform_clock(records: Sequence[PulsarRecord]) -> None:
    """Warn if the array's pulsars disagree on the clock realization.

    Clock errors are common-mode across pulsars, so they project onto the
    monopole and leak into correlated-signal searches.  An array whose pulsars
    sit on different realizations carries a spurious inter-pulsar offset that
    looks exactly like the signal a GWB search measures -- ~27 us between
    TT(TAI) and TT(BIPM), tens of ns between BIPM realizations.

    This warns rather than raises: a mixed array is a red flag, not
    categorically invalid, and forcing uniformity is one kwarg away.
    """
    seen: dict[str, list[str]] = {}
    for r in records:
        key = r.toa_data.clock_realization
        if key is not None:
            seen.setdefault(key, []).append(r.name)
    if len(seen) > 1:
        detail = "; ".join(
            f"{clk}: {len(names)} pulsar(s) e.g. {', '.join(sorted(names)[:3])}"
            for clk, names in sorted(seen.items())
        )
        warnings.warn(
            "PTA pulsars were built against different clock realizations "
            f"({detail}). Clock errors are common-mode and leak into "
            "correlated-signal searches. Pass an explicit bipm_version=... "
            "to force one realization across the array.",
            MixedClockRealization,
            stacklevel=3,
        )


def _load_one(
    name: str,
    par_path: Path,
    tim_path: Path,
    *,
    ephem: str | None,
    bipm_version: str | None,
    planets: bool,
) -> PulsarRecord:
    """Parse + build one pulsar (the shared back half of both loaders).

    ``ephem``/``bipm_version`` of ``None`` mean "derive from the par file";
    passing a value forces it, overriding every par.  Previously the clock was
    forced unconditionally (``include_bipm=True``, ``bipm_version="BIPM2019"``),
    which silently overrode any pulsar whose ``CLK`` disagreed -- including the
    68 ``TT(TAI)`` pars in NANOGrav 15yr's ``narrowband/alternate/tempo2/``,
    where applying BIPM anyway is a ~27 us error.
    """
    log.info("Loading %s from %s", name, par_path)
    par_result = parse_par(str(par_path))
    toa_data = native_toas_to_jax(
        str(tim_path),
        par_result,
        ephem=ephem,
        include_bipm=None,
        bipm_version=bipm_version,
        planets=planets,
    )
    tm, nm = build_model(par_result, toa_data)
    return PulsarRecord(
        name=name,
        toa_data=toa_data,
        params=par_result.params,
        timing_model=tm,
        noise_model=nm,
    )


def load_nanograv_pta(
    data_dir: str | Path,
    *,
    pulsar_names: Sequence[str] | None = None,
    exclude: Collection[str] = (),
    ephem: str | None = None,
    bipm_version: str | None = None,
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
        (:func:`jaxpint.native.get_TOAs`). ``None`` (the default) takes it from
        each par's ``EPHEM``; pass a value to force one ephemeris across the
        array. The 15yr release's pars all specify ``DE440``, so the default
        reproduces the reference analysis without overriding the files.
    bipm_version
        BIPM clock realisation. ``None`` (the default) derives it from each
        par's ``CLK`` line -- including ``TT(TAI)``/``UNCORR``, which disable
        the BIPM term entirely. Pass a value to force one realization across the
        array. A mixed array warns (:class:`MixedClockRealization`), since clock
        errors are common-mode and leak into correlated-signal searches.
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

    Notes
    -----
    This materializes (and retains references to) **every** pulsar at once. For
    one-pulsar-at-a-time workflows on memory-constrained machines, prefer
    :func:`iter_nanograv_pta`
    """
    records = [
        _load_one(
            name, parp, timp, ephem=ephem, bipm_version=bipm_version, planets=planets
        )
        for name, parp, timp in _resolve_pairs(data_dir, pulsar_names, exclude)
    ]
    _check_uniform_clock(records)
    return NanogravPTA(
        pulsar_names=tuple(r.name for r in records),
        toa_data_list=tuple(r.toa_data for r in records),
        pulsar_params_list=tuple(r.params for r in records),
        timing_models=tuple(r.timing_model for r in records),
        noise_models=tuple(r.noise_model for r in records),
    )


def iter_nanograv_pta(
    data_dir: str | Path,
    *,
    pulsar_names: Sequence[str] | None = None,
    exclude: Collection[str] = (),
    ephem: str | None = None,
    bipm_version: str | None = None,
    planets: bool = True,
) -> Iterator[PulsarRecord]:
    """Stream a NANOGrav PTA dataset one pulsar at a time, loading lazily.

    The iterable-style counterpart of :func:`load_nanograv_pta`.
    Each pulsar is loaded on demand, and the generator
    **retains no reference to yielded records**, so a consumer that drops each
    record after use keeps peak memory at ~one pulsar regardless of array
    size. This is the intended idiom for full-array sweeps on
    memory-constrained machines::

        for rec in iter_nanograv_pta(data_dir):
            slab = extract_something(rec)   # heavy, per-pulsar
            slabs.append(slab)              # tiny
            del rec                         # last reference -> buffers freed
            jax.clear_caches()              # per-shape kernels never reused

    Parameters
    ----------
    data_dir, pulsar_names, exclude, ephem, bipm_version, planets
        As in :func:`load_nanograv_pta`.

    Yields
    ------
    PulsarRecord
        ``(name, toa_data, params, timing_model, noise_model)`` per pulsar,
        in selection order.
    """
    for name, parp, timp in _resolve_pairs(data_dir, pulsar_names, exclude):
        yield _load_one(
            name,
            parp,
            timp,
            ephem=ephem,
            bipm_version=bipm_version,
            planets=planets,
        )


def map_pulsars(
    fn: Callable[[PulsarRecord], T],
    data_dir: str | Path,
    *,
    clear_caches: bool = True,
    pulsar_names: Sequence[str] | None = None,
    exclude: Collection[str] = (),
    ephem: str | None = None,
    bipm_version: str | None = None,
    planets: bool = True,
) -> Iterator[T]:
    """Apply ``fn`` to each pulsar with build → use → purge memory hygiene.

    The streaming combinator over :func:`iter_nanograv_pta`: loads one pulsar,
    calls ``fn`` on it, then releases the pulsar's data (and, by default, the
    XLA compilation cache — each pulsar's kernels are uniquely shaped and never
    reused, so that cache otherwise only grows) before loading the next.  Peak
    memory is one pulsar's build plus whatever ``fn`` returns, however many
    pulsars the dataset holds.  This packages the per-pulsar purge idiom the
    example drivers use, so consumers reduce to::

        slabs = list(map_pulsars(extract_blocks, data_dir, exclude=DROP))

    ``fn`` must return a *reduced* result (scalars, small arrays, tuples
    thereof): a result that references the record's own arrays (or the record
    itself) keeps that pulsar's buffers alive and defeats the purge.  Results
    are blocked on (``jax.block_until_ready``) before the purge, so async
    dispatches reading the record's buffers finish first and the memory
    profile stays deterministic.

    Parameters
    ----------
    fn
        ``PulsarRecord -> result``.  The heavy per-pulsar work goes here.
    data_dir, pulsar_names, exclude, ephem, bipm_version, planets
        As in :func:`load_nanograv_pta`.
    clear_caches
        Clear the JAX compilation cache after each pulsar (default).  The
        trade-off: unbounded cache growth across differently-shaped pulsars is
        eliminated, at the cost of recompiling any *shared*-shape helper
        kernels each iteration.  Set ``False`` when iterating few pulsars or
        same-shaped (padded/synthetic) data.

    Yields
    ------
    T
        The return value of ``fn`` for each pulsar, in selection order.
    """
    import jax

    for record in iter_nanograv_pta(
        data_dir,
        pulsar_names=pulsar_names,
        exclude=exclude,
        ephem=ephem,
        bipm_version=bipm_version,
        planets=planets,
    ):
        result = fn(record)
        jax.block_until_ready(result)
        del record
        if clear_caches:
            jax.clear_caches()
        gc.collect()
        yield result
