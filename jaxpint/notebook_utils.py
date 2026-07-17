"""Shared helpers for the ``examples/`` notebooks and scripts.

This module is **not** part of the core JaxPINT API — it exists solely to
factor out scaffolding (random-pulsar generation, synthetic-TOA setup,
CW injection, likelihood-grid sweeps, delta-log-L plots, and the
NANOGrav-dataset load/marginalize/HEALPix boilerplate shared by the
``examples/*.py`` sky-map scripts) that would otherwise be copy-pasted
across the examples.  It is deliberately kept out of ``jaxpint.__init__``
so that the top-level namespace reflects the library, not the demos.

Typical usage in a notebook::

    from io import StringIO
    import numpy as np
    import pint.models as pm

    from jaxpint.notebook_utils import (
        generate_random_par,
        setup_synthetic_pta,
        build_cw_injectors,
        inject_and_build_config,
        sweep_1d_logL,
        plot_1d_delta_logL,
    )

    rng = np.random.default_rng(127)
    par_strings = [generate_random_par(i, rng, start_mjd=57000.0) for i in range(10)]
    pint_models = [pm.get_model(StringIO(p)) for p in par_strings]

    synthetic = setup_synthetic_pta(
        pint_models, start_mjd=57000.0, end_mjd=60000.0,
        n_toas=200, toa_error_s=1e-8, freq_mhz=1400.0,
    )
    injectors, _ = build_cw_injectors(pint_models, n_sources=1, rng=rng)
    gp, cfg = inject_and_build_config(synthetic, injectors)
    # ... now define an `eval_fn` and call `sweep_1d_logL(eval_fn, grid)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple, Optional

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh

# PINT-backed helpers are imported lazily inside the functions that use them so
# this demo module does not hard-require PINT at import time.
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.pta.likelihood import PTAConfig, SignalInjector
from jaxpint.types import GlobalParams
from jaxpint.pta.signals.cw import CWInjector
from jaxpint.simulation import apply_delay_to_toas
from jaxpint.types import ParameterVector, TOAData

if TYPE_CHECKING:
    # NanogravPTA is structurally identical to SyntheticPTA (see its docstring);
    # the injector helpers accept either.
    from jaxpint.loaders.nanograv import NanogravPTA


# ---------------------------------------------------------------------------
# Random .par string generation
# ---------------------------------------------------------------------------


_SIMPLE_NOISE_LINES = "EFAC tel gbt 1.0\n"

_REALISTIC_NOISE_TEMPLATE = (
    "EFAC tel gbt 1.0\n"
    "EQUAD tel gbt 0.1\n"
    "ECORR tel gbt {ecorr_us:.4f}\n"
    "TNRedAmp {tnredamp:.6f}\n"
    "TNRedGam {tnredgam:.6f}\n"
    "TNRedC {tnredc}\n"
)


def generate_random_par(
    idx: int,
    rng: np.random.Generator,
    *,
    start_mjd: float,
    noise: str = "simple",
    include_dm: bool = False,
    free_params: bool = False,
    extra_params: Optional[dict[str, str]] = None,
) -> str:
    """Return a PINT-parsable ``.par`` string for a randomly drawn pulsar.

    Parameters
    ----------
    idx : int
        Integer suffix appended to the pulsar name (``J<RA><DEC>_<idx>``).
    rng : numpy.random.Generator
        RNG used to draw sky position, spin parameters, distance, and
        (optionally) DM and red-noise parameters.
    start_mjd : float
        Reference epoch (``PEPOCH``).
    noise : {"simple", "realistic"}
        ``"simple"`` (default) emits only ``EFAC tel gbt 1.0``.
        ``"realistic"`` adds ``EQUAD``, ``ECORR``, and power-law red noise
        (``TNRedAmp``/``TNRedGam``/``TNRedC``) with randomized amplitudes.
    include_dm : bool
        If True, include a random ``DM`` line drawn from U(10, 50) pc/cm^3.
    free_params : bool
        If True, append PINT's fit flag (`` 1``) after each value line for
        RAJ, DECJ, F0, F1, DM (if present), PX. Use this when downstream
        code needs those parameters to be free (e.g. WLS fitting, NUTS).
    extra_params : dict[str, str] | None
        Additional ``.par`` lines to append verbatim as ``"{key} {value}\\n"``.
        Use this to add DMX, EQUAD-per-backend overrides, etc. without
        bloating the signature.

    Returns
    -------
    str
        A ``.par`` string consumable by ``pint.models.get_model(StringIO(...))``.

    Notes
    -----
    ``PX`` is stored as parallax in mas (types.py convention). A distance is
    drawn from U(0.5, 3.0) kpc and inverted to mas before being written into
    the par string; ``CWInjector`` converts back to kpc internally following
    Ellis+2012.
    """
    if noise not in ("simple", "realistic"):
        raise ValueError(f"noise must be 'simple' or 'realistic', got {noise!r}")

    ra_hours = rng.uniform(0, 24)
    dec_deg = np.degrees(np.arcsin(rng.uniform(-1, 1)))

    ra_h = int(ra_hours)
    ra_m = int((ra_hours - ra_h) * 60)
    ra_s = (ra_hours - ra_h - ra_m / 60) * 3600

    dec_sign = "+" if dec_deg >= 0 else "-"
    dec_abs = abs(dec_deg)
    dec_d = int(dec_abs)
    dec_m = int((dec_abs - dec_d) * 60)
    dec_s = (dec_abs - dec_d - dec_m / 60) * 3600

    f0 = rng.uniform(100, 500)
    f1 = -(10 ** rng.uniform(-16, -14))
    distance_kpc = rng.uniform(0.5, 3.0)
    px_mas = 1.0 / distance_kpc

    fit = "  1" if free_params else ""
    lines = [
        f"PSR           J{ra_h:02d}{ra_m:02d}{dec_sign}{dec_d:02d}{dec_m:02d}_{idx:02d}",
        f"RAJ           {ra_h:02d}:{ra_m:02d}:{ra_s:08.5f}{fit}",
        f"DECJ          {dec_sign}{dec_d:02d}:{dec_m:02d}:{dec_s:07.4f}{fit}",
        f"F0            {f0:.10f}{fit}",
        f"F1            {f1:.6e}{fit}",
        f"PEPOCH        {start_mjd:.1f}",
    ]
    if include_dm:
        dm = rng.uniform(10, 50)
        lines.append(f"DM            {dm:.4f}{fit}")
    lines.extend(
        [
            f"PX            {px_mas:.6f}{fit}",
            "EPHEM         DE440",
            "CLK           TT(BIPM2019)",
            "UNITS         TDB",
        ]
    )

    par = "\n".join(lines) + "\n"

    if noise == "simple":
        par += _SIMPLE_NOISE_LINES
    else:
        par += _REALISTIC_NOISE_TEMPLATE.format(
            ecorr_us=rng.uniform(0.01, 1.0),
            tnredamp=rng.uniform(-15, -12),
            tnredgam=rng.uniform(1.5, 5.0),
            tnredc=14,
        )

    if extra_params:
        for key, value in extra_params.items():
            par += f"{key} {value}\n"

    return par


# ---------------------------------------------------------------------------
# Synthetic PTA setup (TOAs + JaxPINT conversion)
# ---------------------------------------------------------------------------


class SyntheticPTA(NamedTuple):
    """Output of :func:`setup_synthetic_pta`.

    Fields are tuples (not lists) so the result drops straight into
    :class:`jaxpint.pta.likelihood.PTAConfig`.
    """

    toa_data_list: tuple[TOAData, ...]
    pulsar_params_list: tuple[ParameterVector, ...]
    timing_models: tuple[TimingModel, ...]
    noise_models: tuple[NoiseModel, ...]


def setup_synthetic_pta(
    pint_models: list,
    *,
    start_mjd: Optional[float] = None,
    end_mjd: Optional[float] = None,
    n_toas: Optional[int] = None,
    toa_error_s: float,
    freq_mhz: float,
    obs: str = "GBT",
    mjds_per_pulsar: Optional[list[np.ndarray]] = None,
) -> SyntheticPTA:
    """Generate fake TOAs for a list of PINT models and convert to JaxPINT.

    Wraps the ``make_fake_toas_uniform`` / ``make_fake_toas_fromMJDs`` →
    ``pint_toas_to_jax`` → ``pint_model_to_params`` → ``build_timing_model``
    pipeline that appears verbatim in most example notebooks.

    Two modes are supported:

    1. **Uniform cadence (default)**: pass ``start_mjd``, ``end_mjd``, and
       ``n_toas``. All pulsars share the same TOA span and count.
    2. **Custom MJDs**: pass ``mjds_per_pulsar`` — a list of ``np.ndarray``
       of TOA MJDs, one entry per pulsar. Each pulsar gets exactly those
       TOAs. ``start_mjd``/``end_mjd``/``n_toas`` are ignored.

    Parameters
    ----------
    pint_models : list
        Parsed PINT timing models (one per pulsar).
    start_mjd, end_mjd : float, optional
        Uniform-cadence TOA span. Required unless ``mjds_per_pulsar`` is given.
    n_toas : int, optional
        Number of TOAs per pulsar. Required unless ``mjds_per_pulsar`` is given.
    toa_error_s : float
        TOA uncertainty in seconds.
    freq_mhz : float
        Observing frequency in MHz.
    obs : str
        PINT observatory code (default ``"GBT"``).
    mjds_per_pulsar : list of arrays, optional
        If given, each element is the MJDs for that pulsar; routed through
        ``pint.simulation.make_fake_toas_fromMJDs``. Length must match
        ``len(pint_models)``.

    Returns
    -------
    SyntheticPTA
        Named tuple of per-pulsar tuples ready to feed into
        :func:`inject_and_build_config`.
    """
    import pint.simulation as psim

    from jaxpint.bridge import build_timing_model, pint_toas_to_jax
    from jaxpint.bridge.model_conversion import pint_model_to_params

    if mjds_per_pulsar is not None:
        if len(mjds_per_pulsar) != len(pint_models):
            raise ValueError(
                f"mjds_per_pulsar has length {len(mjds_per_pulsar)}, "
                f"expected {len(pint_models)} (one per pulsar)."
            )
    else:
        if start_mjd is None or end_mjd is None or n_toas is None:
            raise ValueError(
                "Uniform mode requires start_mjd, end_mjd, and n_toas. "
                "Alternatively pass mjds_per_pulsar for custom cadence."
            )

    toa_data_list: list[TOAData] = []
    pulsar_params_list: list[ParameterVector] = []
    timing_models: list[TimingModel] = []
    noise_models: list[NoiseModel] = []

    for p, model in enumerate(pint_models):
        if mjds_per_pulsar is not None:
            toas = psim.make_fake_toas_fromMJDs(
                mjds_per_pulsar[p],
                model,
                obs=obs,
                error=toa_error_s * u.s,
                freq=freq_mhz * u.MHz,
            )
        else:
            assert start_mjd is not None and end_mjd is not None and n_toas is not None
            toas = psim.make_fake_toas_uniform(
                start_mjd,
                end_mjd,
                n_toas,
                model,
                obs=obs,
                error=toa_error_s * u.s,
                freq=freq_mhz * u.MHz,
            )
        toa_data = pint_toas_to_jax(toas, model)
        par_result = pint_model_to_params(model)
        tm, nm = build_timing_model(model, toas)

        toa_data_list.append(toa_data)
        pulsar_params_list.append(par_result.params)
        timing_models.append(tm)
        noise_models.append(nm)

    return SyntheticPTA(
        toa_data_list=tuple(toa_data_list),
        pulsar_params_list=tuple(pulsar_params_list),
        timing_models=tuple(timing_models),
        noise_models=tuple(noise_models),
    )


# ---------------------------------------------------------------------------
# CW injection + PTAConfig assembly
# ---------------------------------------------------------------------------


def pulsar_positions_from_models(pint_models: list) -> Float[Array, "n_psr 3"]:
    """Compute ICRS unit vectors from each PINT model's ``RAJ``/``DECJ``."""
    positions = []
    for model in pint_models:
        ra_rad = model.RAJ.quantity.to(u.rad).value
        dec_rad = model.DECJ.quantity.to(u.rad).value
        positions.append(
            np.array(
                [
                    np.cos(dec_rad) * np.cos(ra_rad),
                    np.cos(dec_rad) * np.sin(ra_rad),
                    np.sin(dec_rad),
                ]
            )
        )
    return jnp.array(np.array(positions))


def build_cw_injectors(
    pint_models: list,
    n_sources: int,
    *,
    rng: np.random.Generator,
    log10_h: float = -14.0,
    prefix_fmt: str = "cw{m}_",
) -> tuple[tuple[CWInjector, ...], Float[Array, "n_psr 3"]]:
    """Construct ``n_sources`` :class:`CWInjector`s with randomized sky/frequency.

    Parameters
    ----------
    pint_models : list
        PINT models, used only to compute pulsar unit vectors.
    n_sources : int
        Number of CW sources to build.
    rng : numpy.random.Generator
        RNG for ``cos_gwtheta``, ``gwphi``, and ``log10_fgw``.
    log10_h : float
        Fixed ``log10_h`` assigned to every source.
    prefix_fmt : str
        Format string used to build each injector's parameter-name prefix.
        Must contain ``{m}``.

    Returns
    -------
    injectors : tuple of CWInjector
    positions : (n_psr, 3) jax array
        Returned so callers can reuse the positions (e.g. to build another
        injector with a different prefix) without recomputing them.
    """
    positions = pulsar_positions_from_models(pint_models)

    injectors = tuple(
        CWInjector(
            positions,
            prefix=prefix_fmt.format(m=m),
            initial_values={
                "log10_h": float(log10_h),
                "cos_gwtheta": float(rng.uniform(-1, 1)),
                "gwphi": float(rng.uniform(0, 2 * np.pi)),
                "log10_fgw": float(rng.uniform(-9, -7)),
            },
        )
        for m in range(n_sources)
    )
    return injectors, positions


def inject_and_build_config(
    synthetic: SyntheticPTA | NanogravPTA,
    injectors: tuple[SignalInjector, ...],
) -> tuple[GlobalParams, PTAConfig]:
    """Register injector params, apply their delays to TOAs, and build a ``PTAConfig``.

    Parameters
    ----------
    synthetic : SyntheticPTA
        Output of :func:`setup_synthetic_pta`.
    injectors : tuple of SignalInjector
        Any mix of deterministic (CW) and stochastic (GWB, correlated GWB)
        injectors. Injectors whose ``delay()`` returns ``None`` contribute
        no timing delay (covariance-only signals pass through unchanged).

    Returns
    -------
    global_params : GlobalParams
        Populated with every injector's parameters, in registration order.
    config : PTAConfig
        Ready to pass to :func:`jaxpint.pta.pta_logL`.
    """
    gp = GlobalParams.empty()
    for inj in injectors:
        gp = inj.register_params(gp)

    injected_toa_data: list[TOAData] = []
    for p, td in enumerate(synthetic.toa_data_list):
        total_delay = jnp.zeros(td.n_toas)
        for inj in injectors:
            d = inj.delay(p, td, synthetic.pulsar_params_list[p], gp)
            if d is not None:
                total_delay = total_delay + d
        injected_toa_data.append(apply_delay_to_toas(td, total_delay))

    config = PTAConfig(
        toa_data_list=tuple(injected_toa_data),
        timing_models=synthetic.timing_models,
        noise_models=synthetic.noise_models,
        signal_injectors=tuple(injectors),
    )
    return gp, config


# ---------------------------------------------------------------------------
# Likelihood-grid sweep helpers
# ---------------------------------------------------------------------------


def sweep_1d_logL(
    eval_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    grid: np.ndarray,
    *,
    jit_eval_fn: bool = True,
) -> np.ndarray:
    """Evaluate ``eval_fn`` on a 1D grid with JIT + vmap + warmup.

    The caller supplies ``eval_fn(x) -> scalar`` as a closure over the PTA
    state (global params, per-pulsar params, config, which param to vary).
    This helper just removes the JIT/vmap/warmup/``block_until_ready``
    boilerplate. NOTE: ensure that large constant blocks of data are passed as
    inputs to eval_fn and not by closure; This prevents the memory footprint from
    blowing up!

    Parameters
    ----------
    eval_fn : callable
        Scalar-in, scalar-out function suitable for ``jax.vmap``.
    grid : (N,) array
        Parameter values at which to evaluate ``eval_fn``.
    jit_eval_fn : bool, default True
        When ``True`` (default), wrap ``eval_fn`` in ``jax.jit(jax.vmap(...))``
        for the standard fast path.  Set to ``False`` when ``eval_fn``
        already handles its own JIT internally and must not be wrapped
        in an outer JIT.  In that case the grid is iterated in plain
        Python and each cell is a single ``eval_fn(x)`` call.

    Returns
    -------
    (N,) numpy array of log-likelihood values.
    """
    grid_jax = jnp.asarray(grid)

    if not jit_eval_fn:
        return np.asarray([float(eval_fn(x)) for x in grid_jax])

    eval_vmap = jax.jit(jax.vmap(eval_fn))
    _ = eval_vmap(grid_jax[:2]).block_until_ready()
    return np.asarray(eval_vmap(grid_jax))


def sweep_2d_logL(
    eval_fn: Callable[..., Float[Array, ""]],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    extra_args: tuple = (),
    jit_eval_fn: bool = True,
) -> np.ndarray:
    """Evaluate ``eval_fn(x, y, *extra_args)`` on a 2D grid, returned with shape ``(n_y, n_x)``.

    Parameters
    ----------
    eval_fn : callable
        Takes two scalar JAX arguments plus zero or more extras and
        returns a scalar.
    grid_x, grid_y : arrays
        Grid values along each axis.
    extra_args : tuple, optional
        Additional positional arguments threaded through to ``eval_fn``
        after the two scalar grid coordinates, broadcast (not mapped) by
        the inner/outer ``vmap``. Use this to pass containers like
        ``PTAConfig`` that hold large JAX arrays — passing them here
        means equinox flattens their dynamic fields into traced JIT
        leaves rather than letting closure capture bake them into the
        HLO as compile-time constants.
    jit_eval_fn : bool, default True
        When ``True`` (default), wrap ``eval_fn`` in
        ``jax.jit(jax.vmap(jax.vmap(...)))`` for the standard fast path.
        Set to ``False`` when ``eval_fn`` already handles its own JIT
        internally and must not be wrapped in an outer JIT.  In that
        case the grid is iterated cell-by-cell in plain Python.

    Returns
    -------
    (n_y, n_x) numpy array.
    """
    grid_x_jax = jnp.asarray(grid_x)
    grid_y_jax = jnp.asarray(grid_y)
    n_x = grid_x_jax.shape[0]
    n_y = grid_y_jax.shape[0]
    extra_in_axes = (None,) * len(extra_args)

    if not jit_eval_fn:
        out = np.empty((n_y, n_x), dtype=np.float64)
        for j in range(n_y):
            y = grid_y_jax[j]
            for i in range(n_x):
                out[j, i] = float(eval_fn(grid_x_jax[i], y, *extra_args))
        return out

    eval_grid = jax.jit(
        jax.vmap(
            jax.vmap(eval_fn, in_axes=(0, None) + extra_in_axes),
            in_axes=(None, 0) + extra_in_axes,
        )
    )
    _ = eval_grid(grid_x_jax[:2], grid_y_jax[:2], *extra_args).block_until_ready()
    return np.asarray(eval_grid(grid_x_jax, grid_y_jax, *extra_args))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_1d_delta_logL(
    ax: Axes,
    grid: np.ndarray,
    logL: np.ndarray,
    *,
    true_value: Optional[float] = None,
    label: Optional[str] = None,
    xlabel: str = "",
    clip_min: Optional[float] = None,
    linewidth: float = 1.2,
) -> None:
    """Plot ``logL - logL.max()`` on ``ax`` with an optional true-value marker.

    Parameters
    ----------
    ax : matplotlib Axes
        Target axes. Layout (figure, title, legend) is the caller's job.
    grid, logL : arrays of equal length
        Parameter grid and corresponding log-likelihood values.
    true_value : float or None
        If given, a black dashed vertical line is drawn at this x value.
    label : str or None
        Line label passed to ``ax.plot`` (no legend is drawn here).
    xlabel : str
        Axis label applied only if non-empty.
    clip_min : float or None
        If given, delta values are clipped from below to this floor
        (useful when the tails span many decades under ``symlog``).
    linewidth : float
        Line width for the delta-logL curve.
    """
    delta = np.asarray(logL) - np.asarray(logL).max()
    if clip_min is not None:
        delta = np.clip(delta, clip_min, 0.0)
    ax.plot(grid, delta, linewidth=linewidth, label=label)
    if true_value is not None:
        ax.axvline(
            true_value,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"True = {true_value:.4g}",
        )
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(r"$\Delta$ Log-likelihood", fontsize=13)


def plot_2d_delta_logL(
    ax: Axes,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    logL_2d: np.ndarray,
    *,
    true_xy: Optional[tuple[float, float]] = None,
    clip_min: float = -500.0,
    cmap: str = "viridis",
) -> QuadMesh:
    """pcolormesh of ``logL_2d - logL_2d.max()`` with optional truth marker.

    Parameters
    ----------
    ax : matplotlib Axes
    grid_x, grid_y : arrays
        Axes grids. ``logL_2d`` is expected to have shape ``(len(grid_y), len(grid_x))``
        (the convention produced by :func:`sweep_2d_logL`).
    logL_2d : (n_y, n_x) array
        Raw log-likelihood values.
    true_xy : (x, y) or None
        If given, a red star marker is placed at ``(x, y)``.
    clip_min : float
        Lower bound applied to delta-logL before plotting. Set to
        ``-np.inf`` for no clipping.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    QuadMesh
        The pcolormesh artist. The caller owns colorbar placement:
        e.g. ``fig.colorbar(mesh, ax=ax, label=...)``.
    """
    delta = np.asarray(logL_2d) - np.asarray(logL_2d).max()
    delta = np.clip(delta, clip_min, 0.0)
    mesh = ax.pcolormesh(
        np.asarray(grid_x),
        np.asarray(grid_y),
        delta,
        shading="auto",
        cmap=cmap,
    )
    if true_xy is not None:
        ax.plot(
            true_xy[0],
            true_xy[1],
            "r*",
            markersize=15,
            label=f"True = ({true_xy[0]:.3g}, {true_xy[1]:.3g})",
        )
    return mesh


# ---------------------------------------------------------------------------
# Irregular observing cadence
# ---------------------------------------------------------------------------


def random_obs_window(
    rng: np.random.Generator,
    *,
    global_start_mjd: float,
    global_end_mjd: float,
    min_span_days: float,
) -> tuple[float, float]:
    """Draw a random per-pulsar (start, end) MJD window inside a global window.

    The window is at least ``min_span_days`` long and never extends past
    ``global_end_mjd``.
    """
    span = global_end_mjd - global_start_mjd
    start = rng.uniform(global_start_mjd, global_end_mjd - min_span_days)
    end = rng.uniform(start + min_span_days, min(start + span, global_end_mjd))
    return start, end


def generate_irregular_mjds(
    rng: np.random.Generator,
    *,
    start_mjd: float,
    end_mjd: float,
    n_approx: int,
) -> np.ndarray:
    """Generate non-uniformly spaced TOA MJDs via a Poisson-like process.

    Gaps are drawn from an exponential distribution whose mean gives roughly
    ``n_approx`` TOAs over ``[start_mjd, end_mjd]``.  Feed the result to
    :func:`setup_synthetic_pta` via ``mjds_per_pulsar``.
    """
    avg_gap = (end_mjd - start_mjd) / n_approx
    mjds = [start_mjd]
    while mjds[-1] < end_mjd:
        mjds.append(mjds[-1] + rng.exponential(avg_gap))
    out = np.array(mjds[:-1])
    return out[out < end_mjd]


# ---------------------------------------------------------------------------
# Bundled real-data example pulsar (PINT's NGC6440E)
# ---------------------------------------------------------------------------


class ExamplePulsar(NamedTuple):
    """Output of :func:`load_example_pulsar` — one real pulsar, bridged to JaxPINT."""

    toa_data: TOAData
    params: ParameterVector
    timing_model: TimingModel
    noise_model: NoiseModel
    pint_model: object
    pint_toas: object


def load_example_pulsar(
    name: str = "NGC6440E", *, ephem: str = "DE421"
) -> ExamplePulsar:
    """Load one of PINT's bundled example datasets and bridge it to JaxPINT.

    Wraps the ``examplefile`` → ``get_model``/``get_TOAs`` →
    ``pint_toas_to_jax``/``pint_model_to_params``/``build_timing_model``
    preamble shared by the single-pulsar example notebooks.

    Parameters
    ----------
    name : str
        Basename of the bundled PINT example (``.par``/``.tim`` pair).
    ephem : str
        Solar-system ephemeris passed to ``pint.toa.get_TOAs``.
    """
    import pint.models as pm
    import pint.toa as pt
    from pint.config import examplefile

    from jaxpint.bridge import (
        build_timing_model,
        pint_model_to_params,
        pint_toas_to_jax,
    )

    pint_model = pm.get_model(examplefile(f"{name}.par"))
    pint_toas = pt.get_TOAs(examplefile(f"{name}.tim"), ephem=ephem)

    toa_data = pint_toas_to_jax(pint_toas, model=pint_model)
    params = pint_model_to_params(pint_model).params
    timing_model, noise_model = build_timing_model(pint_model, pint_toas)
    return ExamplePulsar(
        toa_data=toa_data,
        params=params,
        timing_model=timing_model,
        noise_model=noise_model,
        pint_model=pint_model,
        pint_toas=pint_toas,
    )


# ---------------------------------------------------------------------------
# NANOGrav-dataset helpers shared by the examples/*.py sky-map scripts
# ---------------------------------------------------------------------------

# Six pulsars appear as a combined file plus per-telescope (ao/gbt) splits; the
# combined .par/.tim already holds every telescope's TOAs (and VLA-only TOAs
# that are in no split), so keep only the combined one and drop the splits to
# avoid double-counting (worst for J1713/J1909, the most sensitive pulsars).
DROP_PULSARS = frozenset(
    {
        "B1937+21ao",
        "B1937+21gbt",
        "J1600-3053gbt",
        "J1643-1224gbt",
        "J1713+0747ao",
        "J1713+0747gbt",
        "J1903+0327ao",
        "J1909-3744gbt",
    }
)

# Small, well-timed default subset for smoke tests. All four pulsars have
# measured PX in the ocarina par files, so the subset works in both Earth-term
# and pulsar-term modes.
SMOKE_SUBSET = ["J1909-3744", "J1713+0747", "J0613-0200", "J1744-1134"]

# Linear timing-model params to marginalize analytically (improper priors).
# The dominant low-frequency degeneracies; all linear in the residuals.
# Pulsar distance (parallax PX) is deliberately EXCLUDED — held fixed at the
# par-file value. This is a *design invariant* required by pulsar-term modes:
# pegging PX is what gives the pulsar-term sinusoid a coherent matched-filter
# contribution. Marginalizing PX would re-scramble the pulsar-term phase
# (Delta_p ~ 10^4 rad at 27 nHz; fractional PX errors of ~1e-5 wrap a full
# cycle) and collapse the result back to the Earth-term limit.
MARG_PARAMS = {
    "F0",
    "F1",
    "RAJ",
    "DECJ",
    "ELONG",
    "ELAT",
    "PMRA",
    "PMDEC",
    "PMELONG",
    "PMELAT",
}


def log_flush(msg: str) -> None:
    """Flushed progress line (keeps SLURM .out files live under block buffering)."""
    print(msg, flush=True)


def import_healpy():
    """Import healpy with a clear hint if the optional extra isn't installed."""
    try:
        import healpy as hp
    except ImportError as e:  # pragma: no cover - import-time guard
        raise ImportError(
            "This example needs healpy (HEALPix sky grid + Mollweide plots). "
            "Install the optional extra:  uv pip install 'jaxpint[skymap]'  "
            "(or: pip install healpy matplotlib)."
        ) from e
    return hp


class LoadedPTA(NamedTuple):
    """Output of :func:`load_filtered_pta`.

    Field names/order match :class:`SyntheticPTA` (plus ``names``/``positions``)
    so the result drops straight into :func:`inject_and_build_config` and
    :class:`jaxpint.pta.likelihood.PTAConfig`.
    """

    toa_data_list: tuple[TOAData, ...]
    pulsar_params_list: tuple[ParameterVector, ...]
    timing_models: tuple[TimingModel, ...]
    noise_models: tuple[NoiseModel, ...]
    names: tuple[str, ...]
    positions: np.ndarray  # (n_psr, 3) ICRS unit vectors


def load_filtered_pta(
    data_dir,
    *,
    pulsar_names: Optional[list[str]] = None,
    drop: frozenset[str] = DROP_PULSARS,
) -> LoadedPTA:
    """Load a NANOGrav-style par/tim dataset, drop duplicates, compute positions.

    Wraps the ``load_nanograv_pta`` → drop :data:`DROP_PULSARS` → build
    names/TOA/params/model tuples → ICRS unit vectors pipeline shared by the
    ``examples/*.py`` sky-map scripts.

    Parameters
    ----------
    data_dir : path-like
        Dataset directory understood by :func:`jaxpint.load_nanograv_pta`.
    pulsar_names : list[str] or None
        Subset to load (``None`` loads all pulsars in ``data_dir``).
    drop : frozenset[str]
        Pulsar names to discard after loading (default :data:`DROP_PULSARS`,
        the per-telescope split duplicates).
    """
    from pathlib import Path

    from jaxpint import load_nanograv_pta
    from jaxpint.utils import pulsar_unit_vector

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir {data_dir} not found.")
    psrs = load_nanograv_pta(data_dir, pulsar_names=pulsar_names)
    keep = [i for i, n in enumerate(psrs.pulsar_names) if n not in drop]
    pp_list = tuple(psrs.pulsar_params_list[i] for i in keep)
    positions = np.stack([np.asarray(pulsar_unit_vector(pp)) for pp in pp_list])
    return LoadedPTA(
        toa_data_list=tuple(psrs.toa_data_list[i] for i in keep),
        pulsar_params_list=pp_list,
        timing_models=tuple(psrs.timing_models[i] for i in keep),
        noise_models=tuple(psrs.noise_models[i] for i in keep),
        names=tuple(psrs.pulsar_names[i] for i in keep),
        positions=positions,
    )


def marginalize_each_pulsar(
    pta: LoadedPTA,
    *,
    marg_params: set[str] = MARG_PARAMS,
    allow_nonlinear: bool = True,
    validate_linearity: bool = False,
) -> list:
    """Per-pulsar analytic timing-model marginalization (improper priors).

    Runs :func:`jaxpint.bayes.marginalize_single_pulsar` for every pulsar in
    ``pta``, marginalizing each pulsar's free parameters that appear in
    ``marg_params``.  Returns the list of ``marginalize_single_pulsar`` results
    (one ``(g, ..., skeleton)`` tuple per pulsar), in pulsar order.
    """
    from jaxpint.bayes import marginalize_single_pulsar

    out = []
    for td, tm, nm, pp in zip(
        pta.toa_data_list, pta.timing_models, pta.noise_models, pta.pulsar_params_list
    ):
        over = {n for n in pp.free_names() if n in marg_params}
        out.append(
            marginalize_single_pulsar(
                over=over,
                toa_data=td,
                timing_model=tm,
                noise_model=nm,
                fiducial_params=pp,
                allow_nonlinear=allow_nonlinear,
                validate_linearity=validate_linearity,
            )
        )
    return out


def marginalize_pta_timing(
    pta: LoadedPTA,
    config: PTAConfig,
    gp: GlobalParams,
    *,
    marg_params: set[str] = MARG_PARAMS,
    allow_nonlinear: bool = True,
    validate_linearity: bool = False,
):
    """PTA-level analytic timing-model marginalization (improper priors).

    Builds the fully-qualified ``{pulsar}_{param}`` name set from each pulsar's
    free parameters that appear in ``marg_params`` and calls
    :func:`jaxpint.bayes.marginalize_pta`.  Returns whatever ``marginalize_pta``
    returns (``(g, ..., reduced_pulsar_params)``).
    """
    from jaxpint.bayes import marginalize_pta

    over = set()
    for pn, pp in zip(pta.names, pta.pulsar_params_list):
        for n in pp.free_names():
            if n in marg_params:
                over.add(f"{pn}_{n}")
    return marginalize_pta(
        over=over,
        config=config,
        pulsar_names=pta.names,
        fiducial_pulsar_params=pta.pulsar_params_list,
        fiducial_global_params=gp,
        validate_linearity=validate_linearity,
        allow_nonlinear=allow_nonlinear,
    )


class HealpixGrid(NamedTuple):
    """Output of :func:`healpix_grid` (RING ordering, exactly equal-area).

    ``sky`` is the ``(npix, 2)`` array of ``(cos_gwtheta, gwphi)`` pairs the CW
    likelihood closures take; ``omhat`` is the ``(npix, 3)`` GW propagation
    direction (pointing from the source through the SSB).
    """

    nside: int
    npix: int
    theta: np.ndarray  # (npix,) colatitude
    phi: np.ndarray  # (npix,) longitude
    sky: Float[Array, "npix 2"]  # (cos_gwtheta, gwphi) per pixel
    omhat: np.ndarray  # (npix, 3) propagation direction


def healpix_grid(nside: int) -> HealpixGrid:
    """Build the standard HEALPix sky grid used by the sky-map example scripts."""
    hp = import_healpy()

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    sky = jnp.stack([jnp.cos(jnp.asarray(theta)), jnp.asarray(phi)], axis=1)
    sin_th = np.sin(theta)
    omhat = np.stack(
        [-sin_th * np.cos(phi), -sin_th * np.sin(phi), -np.cos(theta)], axis=1
    )
    return HealpixGrid(
        nside=int(nside), npix=int(npix), theta=theta, phi=phi, sky=sky, omhat=omhat
    )


def save_npz_results(path, results: dict) -> None:
    """Write a results dict to a compressed ``.npz`` (every key becomes an array)."""
    np.savez_compressed(path, **results)
    log_flush(f"Saved -> {path}")


def load_npz_results(path) -> dict:
    """Load an ``.npz`` written by :func:`save_npz_results`, unwrapping 0-d arrays."""
    data = np.load(path, allow_pickle=False)
    return {k: (v.item() if v.ndim == 0 else v) for k, v in data.items()}


def overlay_pulsars(
    positions: np.ndarray,
    anchor_mask: Optional[np.ndarray] = None,
    *,
    star_kwargs: Optional[dict] = None,
    dot_kwargs: Optional[dict] = None,
) -> None:
    """Overlay pulsars on the *current* healpy Mollweide projection.

    Parameters
    ----------
    positions : (n_psr, 3) array
        ICRS unit vectors (e.g. ``LoadedPTA.positions``).
    anchor_mask : (n_psr,) bool array or None
        Pulsars where the mask is True are drawn as stars, the rest as dots.
        ``None`` draws every pulsar as a star.
    star_kwargs, dot_kwargs : dict or None
        Overrides merged into the default ``hp.projscatter`` styles
        (red/black-edged stars; white/black-edged dots).
    """
    hp = import_healpy()

    pos = np.atleast_2d(np.asarray(positions))
    theta = np.arccos(np.clip(pos[:, 2], -1.0, 1.0))
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    mask = (
        np.ones(len(pos), dtype=bool)
        if anchor_mask is None
        else np.asarray(anchor_mask, dtype=bool)
    )

    star_style = dict(
        marker="*", s=120, color="red", edgecolors="black", linewidths=0.5, zorder=5
    )
    star_style.update(star_kwargs or {})
    dot_style = dict(marker="o", s=20, color="white", edgecolors="k", zorder=5)
    dot_style.update(dot_kwargs or {})

    if mask.any():
        hp.projscatter(theta[mask], phi[mask], **star_style)
    if (~mask).any():
        hp.projscatter(theta[~mask], phi[~mask], **dot_style)


__all__ = [
    "DROP_PULSARS",
    "ExamplePulsar",
    "HealpixGrid",
    "LoadedPTA",
    "MARG_PARAMS",
    "SMOKE_SUBSET",
    "SyntheticPTA",
    "build_cw_injectors",
    "generate_irregular_mjds",
    "generate_random_par",
    "healpix_grid",
    "import_healpy",
    "inject_and_build_config",
    "load_example_pulsar",
    "load_filtered_pta",
    "log_flush",
    "marginalize_each_pulsar",
    "marginalize_pta_timing",
    "overlay_pulsars",
    "plot_1d_delta_logL",
    "plot_2d_delta_logL",
    "pulsar_positions_from_models",
    "random_obs_window",
    "save_npz_results",
    "load_npz_results",
    "setup_synthetic_pta",
    "sweep_1d_logL",
    "sweep_2d_logL",
]
