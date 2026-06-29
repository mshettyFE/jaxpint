"""Dependency-aware N-D scans of the PTA log-likelihood.

For multi-dimensional grid scans where each axis varies a single
parameter — a per-pulsar parameter (e.g. one pulsar's ``PX``) or a
global signal-injector parameter (e.g. ``cw_log10_h``) — most of the
per-pulsar log-likelihood contributions are unchanged across grid cells:
a scan that varies pulsar A's PX along x and pulsar B's PX along y
leaves every other pulsar's contribution constant at every cell. The
naive approach (call :func:`pta_logL` per cell) recomputes those
constants once per grid point.

:func:`scan_logL` exploits the per-pulsar decomposition of
:func:`pta_logL` to avoid that waste:

1. **Determine each pulsar's dependencies.** A
   :class:`PerPulsarScanAxis` for pulsar ``q`` affects only pulsar
   ``q``'s log-likelihood; a :class:`GlobalScanAxis` affects every
   pulsar (because every signal injector receives ``global_params``).
2. **Pre-compute constants.** Pulsars not affected by any axis have
   their :func:`single_pulsar_pta_logL` evaluated once at the base
   parameter values.
3. **Vmap variables only.** Pulsars affected by some subset of axes
   are nest-:func:`jax.vmap`-ed over only the axes that actually
   touch them. When none of those axes touch the noise covariance
   (e.g. a per-pulsar ``PX`` axis when only timing-domain params are
   varied), the Woodbury factor is precomputed once at the base
   parameter values via
   :func:`~jaxpint.pta.likelihood.precompute_single_pulsar_pta_factor`
   and reused inside the vmapped body, avoiding a per-cell Cholesky
   and the associated GPU memory blow-up.
4. **Sum with broadcasting.** Per-pulsar arrays are reshaped to the
   full grid shape (with size-1 dims at non-dependency positions) and
   summed.

For a 2D 400×400 distance scan over two pulsars in a 5-pulsar PTA, the
naive cost is 800,000 single-pulsar evaluations; the dependency-aware
cost is 803. The function is **pure-functional** (no Python-mutable
state), so :func:`jax.grad` and :func:`jax.vmap` compose with
:func:`scan_logL` end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.pta.likelihood import (
    PTAConfig,
    SignalInjector,
    precompute_single_pulsar_pta_factor,
    pta_logL,
    single_pulsar_pta_logL,
    single_pulsar_pta_logL_with_factor,
)
from jaxpint.types import GlobalParams
from jaxpint.types import ParameterVector


__all__ = [
    "PerPulsarScanAxis",
    "GlobalScanAxis",
    "ScanAxis",
    "scan_logL",
]


# ---------------------------------------------------------------------------
# Scan-axis specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PerPulsarScanAxis:
    """One scan axis varying a single per-pulsar parameter over a 1D grid.

    Affects only pulsar ``pulsar_idx``'s contribution to the PTA
    log-likelihood; other pulsars are constant along this axis.

    Parameters
    ----------
    pulsar_idx
        Index of the pulsar whose parameter varies along this axis.
        Must satisfy ``0 <= pulsar_idx < len(base_pulsar_params)``.
    param_name
        Name of the parameter (e.g. ``"PX"``, ``"F0"``). Must be present
        in ``base_pulsar_params[pulsar_idx].names``.
    values
        ``(n_values,)`` JAX array of parameter values to scan.
    """

    pulsar_idx: int
    param_name: str
    values: Float[Array, " n_values"]


@dataclass(frozen=True, eq=False)
class GlobalScanAxis:
    """One scan axis varying a single global parameter over a 1D grid.

    Global parameters live in :class:`GlobalParams` and feed into every
    signal injector's ``delay`` / ``covariance`` calls; consequently
    they affect *every* pulsar's contribution.

    Parameters
    ----------
    param_name
        Name of the global parameter (e.g. ``"cw_log10_h"``,
        ``"gw_log10_A"``). Must be present in
        ``base_global_params.names``.
    values
        ``(n_values,)`` JAX array of parameter values to scan.
    """

    param_name: str
    values: Float[Array, " n_values"]


ScanAxis = Union[PerPulsarScanAxis, GlobalScanAxis]


def _dep_axes_for_pulsar(p: int, axes: Sequence[ScanAxis]) -> tuple[int, ...]:
    """Return the indices (into ``axes``) of axes that affect pulsar ``p``.

    Rule: ``PerPulsarScanAxis(pulsar_idx=q)`` affects only pulsar ``q``;
    ``GlobalScanAxis`` affects every pulsar.

    Parameters
    ----------
    p
        Pulsar index.
    axes
        The full scan-axis sequence.

    Returns
    -------
    tuple of int
        Indices into ``axes`` (in input order) of the axes pulsar ``p``
        depends on; empty if ``p`` is constant across the whole scan.
    """
    out = []
    for i, ax in enumerate(axes):
        if isinstance(ax, GlobalScanAxis):
            out.append(i)
        elif isinstance(ax, PerPulsarScanAxis):
            if ax.pulsar_idx == p:
                out.append(i)
        else:
            raise TypeError(
                f"axes[{i}] has unexpected type {type(ax).__name__}; "
                f"expected PerPulsarScanAxis or GlobalScanAxis."
            )
    return tuple(out)


def _injectors_contribute_covariance(
    injectors: Sequence[SignalInjector],
) -> bool:
    """True if any injector overrides :meth:`SignalInjector.covariance`.

    Used to decide whether a :class:`GlobalScanAxis` could change the
    Woodbury covariance (and thus invalidate a precomputed factor).
    Conservative: if even one injector overrides ``covariance``, treat
    every global axis as covariance-touching.

    Parameters
    ----------
    injectors
        The PTA's per-pulsar signal injectors (``config.signal_injectors``).

    Returns
    -------
    bool
        ``True`` if at least one injector overrides ``covariance`` (i.e. can
        contribute a stochastic ``(U, Phi)`` block).
    """
    return any(
        type(inj).covariance is not SignalInjector.covariance for inj in injectors
    )


def _axes_touch_covariance(
    p: int,
    axes: Sequence[ScanAxis],
    dep_p: tuple[int, ...],
    config: PTAConfig,
) -> bool:
    """True iff any axis in ``dep_p`` could change pulsar ``p``'s Woodbury C.

    The Woodbury factor (``Σ`` Cholesky + ``log det C``) for pulsar ``p``
    is reusable across grid cells iff none of the axes that affect
    ``p`` perturb either:

    * the noise model's :meth:`covariance` (i.e. any param in
      ``noise_model_p.required_params()``), or
    * any stochastic signal injector's :meth:`covariance` for ``p``.

    The check is conservative on the second item: it only confirms
    "factor reusable" when no injector contributes covariance at all
    (every injector inherits the default ``covariance`` returning
    ``None``). Mixed setups with stochastic injectors fall back to the
    full path.

    Parameters
    ----------
    p
        Pulsar index.
    axes
        The full scan-axis sequence.
    dep_p
        Indices (into ``axes``) of the axes that affect pulsar ``p``, as
        returned by :func:`_dep_axes_for_pulsar`.
    config
        The PTA configuration (read for ``noise_models[p]`` and
        ``signal_injectors``).

    Returns
    -------
    bool
        ``True`` if pulsar ``p``'s Woodbury factor may change along some axis
        in ``dep_p`` (so it must be recomputed per cell); ``False`` if the
        factor can be precomputed once and reused.
    """
    nm_p = config.noise_models[p]
    noise_params: set[str] = set()
    for comp in nm_p.components:
        noise_params.update(comp.required_params())
    injectors_have_cov = _injectors_contribute_covariance(
        config.signal_injectors,
    )
    for axis_idx in dep_p:
        ax = axes[axis_idx]
        if isinstance(ax, GlobalScanAxis):
            if injectors_have_cov:
                return True
        else:  # PerPulsarScanAxis (only enters dep_p when pulsar_idx == p)
            if ax.param_name in noise_params:
                return True
            if injectors_have_cov:
                # An injector might read this per-pulsar param in its
                # covariance call; can't prove otherwise without running
                # it, so be conservative.
                return True
    return False


def _build_per_pulsar_array(
    p: int,
    base_global_params: GlobalParams,
    base_pulsar_params_p: ParameterVector,
    config: PTAConfig,
    axes: Sequence[ScanAxis],
    dep_p: tuple[int, ...],
    use_factor: bool,
    chunk_size: int | None = None,
) -> Float[Array, "..."]:
    """Compute pulsar ``p``'s contribution as an array shaped by ``dep_p``.

    Output shape (in ``'ij'`` order over ``dep_p``): ``(axes[dep_p[0]].
    values.size, axes[dep_p[1]].values.size, ..., axes[dep_p[-1]].
    values.size)``. Each element is the value of
    :func:`single_pulsar_pta_logL` at the corresponding grid point, with
    only the parameters varied by ``dep_p`` substituted.

    When ``use_factor`` is true, the noise-side Woodbury factor is
    precomputed once at the base parameter values and reused inside the
    vmapped body (see :func:`precompute_single_pulsar_pta_factor`).
    Caller is responsible for verifying the factor is valid — see
    :func:`_axes_touch_covariance`.

    Implements step (3) of :func:`scan_logL`: build the outer-product grid of
    the dependency axes with :func:`jax.numpy.meshgrid` (in ``'ij'`` order),
    flatten it to one row of ``K = len(dep_p)`` parameter values per cell,
    evaluate the per-cell log-likelihood at every row, and reshape back to the
    grid. The evaluation uses :func:`jax.vmap` (all cells at once) or
    :func:`jax.lax.map` with ``batch_size=chunk_size`` to cap memory.

    Parameters
    ----------
    p
        Pulsar index.
    base_global_params, base_pulsar_params_p
        Reference parameters; axis values are substituted onto copies of
        these per grid cell.
    config
        The PTA configuration.
    axes
        The full scan-axis sequence.
    dep_p
        Indices (into ``axes``) of the axes affecting pulsar ``p``; must be
        non-empty (the caller handles the constant case separately).
    use_factor
        If ``True``, precompute the noise-side Woodbury factor once and reuse
        it (valid only when :func:`_axes_touch_covariance` is ``False``).
    chunk_size
        If set, evaluate the flattened grid with :func:`jax.lax.map` in
        batches of this many cells (a compiled scan over batches) to cap peak
        memory; ``None`` vmaps all cells at once.

    Returns
    -------
    array
        ``len(dep_p)``-dimensional array of pulsar ``p``'s
        :func:`single_pulsar_pta_logL` over the grid of its dependency axes,
        in ``'ij'`` order (leading axis = ``dep_p[0]``).
    """
    assert dep_p, "use the constant fast path when dep_p is empty"

    def _substitute(cell):
        """Apply one grid cell onto pulsar ``p``'s base params.

        ``cell`` holds one value per dependency axis, in ``dep_p`` order.
        """
        gp = base_global_params
        pp_p = base_pulsar_params_p
        for axis_idx, val in zip(dep_p, cell):
            ax = axes[axis_idx]
            if isinstance(ax, PerPulsarScanAxis):
                pp_p = pp_p.with_value(ax.param_name, val)
            else:  # GlobalScanAxis
                gp = gp.with_value(ax.param_name, val)
        return gp, pp_p

    factor = (
        precompute_single_pulsar_pta_factor(
            p, base_global_params, base_pulsar_params_p, config
        )
        if use_factor
        else None
    )

    def f_cell(cell):
        """Log-likelihood at one grid cell."""
        gp, pp_p = _substitute(cell)
        if use_factor:
            assert factor is not None  # built whenever use_factor is True
            return single_pulsar_pta_logL_with_factor(p, gp, pp_p, factor, config)
        return single_pulsar_pta_logL(p, gp, pp_p, config)

    # Enumerate every grid cell as a row of K = len(dep_p) parameter values, in
    # 'ij' (outer-product) order; evaluate the log-likelihood at each, then
    # reshape back to the grid. jax.vmap evaluates all cells at once;
    # jax.lax.map with batch_size processes `chunk_size` cells per batch (a
    # compiled scan over batches) to cap peak memory.
    ax_values = tuple(axes[i].values for i in dep_p)
    grid_shape = tuple(v.shape[0] for v in ax_values)
    mesh = jnp.meshgrid(*ax_values, indexing="ij")
    cells = jnp.stack([m.ravel() for m in mesh], axis=-1)

    if chunk_size is None or chunk_size >= cells.shape[0]:
        flat = jax.vmap(f_cell)(cells)
    else:
        flat = jax.lax.map(f_cell, cells, batch_size=chunk_size)
    return flat.reshape(grid_shape)


def _inflate_to_full_ij_shape(
    arr: Float[Array, "..."],
    dep_p: tuple[int, ...],
    full_axis_sizes: tuple[int, ...],
) -> Float[Array, "..."]:
    """Reshape a per-pulsar array to the full ``'ij'`` grid shape.

    ``arr`` has dim ``j`` corresponding to ``axes[dep_p[j]]``. The
    output has shape ``full_axis_sizes`` with size-1 dims at every
    position not in ``dep_p`` (so it broadcasts when summed with arrays
    that depend on different axis subsets).

    Parameters
    ----------
    arr
        Per-pulsar array from :func:`_build_per_pulsar_array`, with one dim
        per entry of ``dep_p`` (in order).
    dep_p
        Indices (into the full axis list) of the axes ``arr`` varies over.
    full_axis_sizes
        Sizes of *all* scan axes, in input order.

    Returns
    -------
    array
        ``arr`` reshaped to ``len(full_axis_sizes)`` dims: its own sizes at
        the ``dep_p`` positions and size 1 elsewhere (broadcast-ready).
    """
    n_axes = len(full_axis_sizes)
    if not dep_p:
        return arr.reshape((1,) * n_axes)
    new_shape = []
    arr_dim = 0
    for axis in range(n_axes):
        if axis in dep_p:
            new_shape.append(arr.shape[arr_dim])
            arr_dim += 1
        else:
            new_shape.append(1)
    return arr.reshape(tuple(new_shape))


def scan_logL(
    base_global_params: GlobalParams,
    base_pulsar_params: tuple[ParameterVector, ...],
    config: PTAConfig,
    *,
    axes: Sequence[ScanAxis],
    indexing: str = "xy",
    chunk_size: int | None = None,
) -> Float[Array, "..."]:
    """Evaluate ``pta_logL`` on the outer-product grid of ``axes``.

    Pulsars whose parameters don't depend on any axis are evaluated once
    at base values and contribute as constants; pulsars that do depend
    on one or more axes are :func:`jax.vmap`-ed only over those axes.
    Per-pulsar contributions are summed with broadcasting.

    Output shape mirrors :func:`numpy.meshgrid`'s ``indexing`` parameter
    exactly:

    - ``indexing="xy"`` (default, numpy's default): for 2D
      ``axes=[ax_x, ax_y]`` → shape ``(n_y, n_x)``
      (matplotlib-friendly). For ≥3D, only the first two axes swap;
      ``[ax_x, ax_y, ax_z, ax_w]`` → ``(n_y, n_x, n_z, n_w)``.
    - ``indexing="ij"``: input order preserved end-to-end;
      ``[ax_0, ax_1, ax_2, ax_3]`` → ``(n_0, n_1, n_2, n_3)``.

    Parameters
    ----------
    base_global_params
        Reference global parameters; values along ``GlobalScanAxis`` axes
        are substituted via ``base_global_params.with_value(name, val)``.
    base_pulsar_params
        Reference per-pulsar parameter vectors; values along
        ``PerPulsarScanAxis`` axes are substituted via
        ``base_pulsar_params[pulsar_idx].with_value(name, val)``.
    config
        :class:`~jaxpint.pta.likelihood.PTAConfig`.
    axes
        Sequence of :class:`PerPulsarScanAxis` and/or
        :class:`GlobalScanAxis`. Empty sequence returns the scalar
        :func:`pta_logL` at the base values.
    indexing
        ``"xy"`` (default) or ``"ij"``. See above.
    chunk_size
        If set, each per-pulsar dependency grid is flattened to one cell per
        row and evaluated with :func:`jax.lax.map` in batches of this many
        cells (a compiled scan over batches), rather than a single
        :func:`jax.vmap` over the whole grid. Caps peak GPU memory roughly
        linearly in ``chunk_size``.

        Pick this when a single full :func:`jax.vmap` OOMs on the target
        device — typically because the timing-model phase computation has many
        ``(n_toas,)`` intermediates and a scanned pulsar has a long TOA list.
        Leave ``None`` for the unchunked default; values ``>=`` the total
        number of grid cells are equivalent to ``None``.

    Returns
    -------
    logL_grid : :class:`jax.Array`
        ``len(axes)``-dimensional log-likelihood grid. Pure-functional;
        :func:`jax.grad`, :func:`jax.vmap`, and :func:`jax.jit` compose
        with this function in the usual way.

    Raises
    ------
    ValueError
        If ``indexing`` is not ``"xy"`` or ``"ij"``.

    Examples
    --------
    A 2-D distance scan varying two pulsars' parallaxes (``PX``)
    independently — ``base_global_params``/``base_pulsar_params``/``config``
    come from the bridge or loader layer::

        from jaxpint.pta import PerPulsarScanAxis, scan_logL

        px0 = jnp.linspace(0.5, 2.0, 200)   # pulsar 0 parallax grid (x)
        px1 = jnp.linspace(0.3, 1.5, 200)   # pulsar 1 parallax grid (y)
        grid = scan_logL(
            base_global_params,
            base_pulsar_params,
            config,
            axes=[
                PerPulsarScanAxis(pulsar_idx=0, param_name="PX", values=px0),
                PerPulsarScanAxis(pulsar_idx=1, param_name="PX", values=px1),
            ],
        )
        # grid.shape == (200, 200)  (indexing="xy" → (n_y, n_x))

    A global axis (affects every pulsar) mixed with a per-pulsar one::

        from jaxpint.pta import GlobalScanAxis

        grid = scan_logL(
            base_global_params, base_pulsar_params, config,
            axes=[
                GlobalScanAxis(param_name="cw0_log10_h", values=h_grid),
                PerPulsarScanAxis(pulsar_idx=1, param_name="PX", values=px_grid),
            ],
        )
    """
    if indexing not in ("xy", "ij"):
        raise ValueError(f"indexing must be 'xy' or 'ij', got {indexing!r}")

    n_axes = len(axes)
    if n_axes == 0:
        return pta_logL(base_global_params, base_pulsar_params, config)

    n_pulsars = len(base_pulsar_params)

    # Validate every axis up front: a typo'd param_name otherwise surfaces as a
    # deep KeyError inside `with_value` during tracing, and an out-of-range
    # pulsar_idx is silently dropped (the axis matches no pulsar and varies
    # nothing). Fail here with a message that names the offending axis.
    for i, ax in enumerate(axes):
        if isinstance(ax, GlobalScanAxis):
            if ax.param_name not in base_global_params:
                raise ValueError(
                    f"axes[{i}]: global param {ax.param_name!r} is not in "
                    f"base_global_params (have {tuple(base_global_params.names)})."
                )
        elif isinstance(ax, PerPulsarScanAxis):
            if not (0 <= ax.pulsar_idx < n_pulsars):
                raise ValueError(
                    f"axes[{i}]: pulsar_idx {ax.pulsar_idx} is out of range "
                    f"[0, {n_pulsars})."
                )
            pp = base_pulsar_params[ax.pulsar_idx]
            if ax.param_name not in pp:
                raise ValueError(
                    f"axes[{i}]: param {ax.param_name!r} is not in pulsar "
                    f"{ax.pulsar_idx}'s ParameterVector (have {tuple(pp.names)})."
                )
        else:
            raise TypeError(
                f"axes[{i}] has unexpected type {type(ax).__name__}; "
                f"expected PerPulsarScanAxis or GlobalScanAxis."
            )

    full_axis_sizes = tuple(int(ax.values.shape[0]) for ax in axes)

    constants_total = jnp.float64(0.0)
    inflated_arrays: list = []

    for p in range(n_pulsars):
        dep_p = _dep_axes_for_pulsar(p, axes)
        if not dep_p:
            c_p = single_pulsar_pta_logL(
                p,
                base_global_params,
                base_pulsar_params[p],
                config,
            )
            constants_total = constants_total + c_p
        else:
            use_factor = not _axes_touch_covariance(p, axes, dep_p, config)
            arr = _build_per_pulsar_array(
                p,
                base_global_params,
                base_pulsar_params[p],
                config,
                axes,
                dep_p,
                use_factor=use_factor,
                chunk_size=chunk_size,
            )
            inflated = _inflate_to_full_ij_shape(arr, dep_p, full_axis_sizes)
            inflated_arrays.append(inflated)

    # Combine in 'ij' shape, then apply the 'xy' swap if requested.
    if inflated_arrays:
        result_ij = inflated_arrays[0]
        for arr in inflated_arrays[1:]:
            result_ij = result_ij + arr
        result_ij = result_ij + constants_total
    else:
        # Every pulsar is constant. Broadcast scalar to full grid.
        result_ij = jnp.broadcast_to(constants_total, full_axis_sizes)

    if indexing == "xy" and n_axes >= 2:
        return jnp.swapaxes(result_ij, 0, 1)
    return result_ij
