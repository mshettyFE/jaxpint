"""Shared helpers for the ``examples/`` notebooks.

This module is **not** part of the core JaxPINT API — it exists solely to
factor out scaffolding (random-pulsar generation, synthetic-TOA setup,
CW injection, likelihood-grid sweeps, and delta-log-L plots) that would
otherwise be copy-pasted across the example notebooks.  It is deliberately
kept out of ``jaxpint.__init__`` so that the top-level namespace reflects
the library, not the demos.

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

from typing import Callable, NamedTuple, Optional

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import pint.simulation as psim
from jaxtyping import Array, Float
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh

from jaxpint.bridge.model_conversion import pint_model_to_params
from jaxpint.bridge import build_timing_model, pint_toas_to_jax
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.pta.likelihood import PTAConfig, SignalInjector
from jaxpint.pta.params import GlobalParams
from jaxpint.pta.signals.cw import CWInjector
from jaxpint.simulation import apply_delay_to_toas
from jaxpint.types import ParameterVector, TOAData


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
    lines.extend([
        f"PX            {px_mas:.6f}{fit}",
        "EPHEM         DE440",
        "CLK           TT(BIPM2019)",
        "UNITS         TDB",
    ])

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


def _pulsar_positions_from_models(pint_models: list) -> Float[Array, "n_psr 3"]:
    """Compute ICRS unit vectors from each model's ``RAJ``/``DECJ``."""
    positions = []
    for model in pint_models:
        ra_rad = model.RAJ.quantity.to(u.rad).value
        dec_rad = model.DECJ.quantity.to(u.rad).value
        positions.append(
            np.array([
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            ])
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
    positions = _pulsar_positions_from_models(pint_models)

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
    synthetic: SyntheticPTA,
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
        Ready to pass to :func:`jaxpint.pta.likelihood.pta_logL`.
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
) -> np.ndarray:
    """Evaluate ``eval_fn`` on a 1D grid with JIT + vmap + warmup.

    The caller supplies ``eval_fn(x) -> scalar`` as a closure over the PTA
    state (global params, per-pulsar params, config, which param to vary).
    This helper just removes the JIT/vmap/warmup/``block_until_ready``
    boilerplate.

    Parameters
    ----------
    eval_fn : callable
        Scalar-in, scalar-out function suitable for ``jax.vmap``.
    grid : (N,) array
        Parameter values at which to evaluate ``eval_fn``.

    Returns
    -------
    (N,) numpy array of log-likelihood values.
    """
    grid_jax = jnp.asarray(grid)
    eval_vmap = jax.jit(jax.vmap(eval_fn))
    _ = eval_vmap(grid_jax[:2]).block_until_ready()
    return np.asarray(eval_vmap(grid_jax))


def sweep_2d_logL(
    eval_fn: Callable[[Float[Array, ""], Float[Array, ""]], Float[Array, ""]],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    chunk_rows: Optional[int] = None,
) -> np.ndarray:
    """Evaluate ``eval_fn(x, y)`` on a 2D grid, returned with shape ``(n_y, n_x)``.

    Parameters
    ----------
    eval_fn : callable
        Takes two scalar JAX arguments and returns a scalar.
    grid_x, grid_y : arrays
        Grid values along each axis.
    chunk_rows : int or None
        When ``None``, evaluates the full grid with nested ``vmap`` inside a
        single JIT. When an integer, iterates over ``chunk_rows`` rows of
        ``grid_y`` at a time to bound peak memory. Set to ``1`` for the
        most memory-conservative sweep (equivalent to the inner-loop pattern
        used by the pulsar0-vs-pulsar1 contour notebook).

    Returns
    -------
    (n_y, n_x) numpy array.
    """
    grid_x_jax = jnp.asarray(grid_x)
    grid_y_jax = jnp.asarray(grid_y)
    n_x = grid_x_jax.shape[0]
    n_y = grid_y_jax.shape[0]

    if chunk_rows is None:
        eval_grid = jax.jit(
            jax.vmap(
                jax.vmap(eval_fn, in_axes=(0, None)),
                in_axes=(None, 0),
            )
        )
        _ = eval_grid(grid_x_jax[:2], grid_y_jax[:2]).block_until_ready()
        return np.asarray(eval_grid(grid_x_jax, grid_y_jax))

    if chunk_rows <= 0:
        raise ValueError(f"chunk_rows must be positive, got {chunk_rows}")

    if chunk_rows == 1:
        eval_row = jax.jit(jax.vmap(eval_fn, in_axes=(0, None)))
        _ = eval_row(grid_x_jax[:2], grid_y_jax[0]).block_until_ready()
        out = np.empty((n_y, n_x), dtype=np.float64)
        for j in range(n_y):
            out[j, :] = np.asarray(eval_row(grid_x_jax, grid_y_jax[j]).block_until_ready())
        return out

    eval_chunk = jax.jit(
        jax.vmap(
            jax.vmap(eval_fn, in_axes=(0, None)),
            in_axes=(None, 0),
        )
    )
    _ = eval_chunk(grid_x_jax[:2], grid_y_jax[:2]).block_until_ready()
    out = np.empty((n_y, n_x), dtype=np.float64)
    for start in range(0, n_y, chunk_rows):
        stop = min(start + chunk_rows, n_y)
        out[start:stop, :] = np.asarray(
            eval_chunk(grid_x_jax, grid_y_jax[start:stop]).block_until_ready()
        )
    return out


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
            true_xy[0], true_xy[1],
            "r*",
            markersize=15,
            label=f"True = ({true_xy[0]:.3g}, {true_xy[1]:.3g})",
        )
    return mesh


__all__ = [
    "SyntheticPTA",
    "build_cw_injectors",
    "generate_random_par",
    "inject_and_build_config",
    "plot_1d_delta_logL",
    "plot_2d_delta_logL",
    "setup_synthetic_pta",
    "sweep_1d_logL",
    "sweep_2d_logL",
]
