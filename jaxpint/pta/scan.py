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

from jaxpint.components import _collect_param_names
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
    """
    nm_p = config.noise_models[p]
    noise_params: set[str] = set()
    for comp in nm_p.components:
        # Some noise components (e.g. ScaleDmError) are eqx.Module but
        # don't inherit from NoiseComponent; fall back to the raw
        # name-collection helper for those.
        if hasattr(comp, "required_params"):
            noise_params.update(comp.required_params())
        else:
            noise_params.update(_collect_param_names(comp))
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

    Implements step (3) of :func:`scan_logL`: build a closure ``f_p`` of
    ``len(dep_p)`` scalar arguments, then nest :func:`jax.vmap` from the
    last argument inward so the resulting array's leading axis
    corresponds to ``dep_p[0]`` and trailing axis to ``dep_p[-1]``.
    """
    K = len(dep_p)
    assert K > 0, "use the constant fast path when dep_p is empty"

    if use_factor:
        factor = precompute_single_pulsar_pta_factor(
            p,
            base_global_params,
            base_pulsar_params_p,
            config,
        )

        def f_p(*ax_values):
            gp = base_global_params
            pp_p = base_pulsar_params_p
            for axis_idx, val in zip(dep_p, ax_values):
                ax = axes[axis_idx]
                if isinstance(ax, PerPulsarScanAxis):
                    pp_p = pp_p.with_value(ax.param_name, val)
                else:  # GlobalScanAxis
                    gp = gp.with_value(ax.param_name, val)
            return single_pulsar_pta_logL_with_factor(
                p,
                gp,
                pp_p,
                factor,
                config,
            )
    else:

        def f_p(*ax_values):
            gp = base_global_params
            pp_p = base_pulsar_params_p
            for axis_idx, val in zip(dep_p, ax_values):
                ax = axes[axis_idx]
                if isinstance(ax, PerPulsarScanAxis):
                    pp_p = pp_p.with_value(ax.param_name, val)
                else:  # GlobalScanAxis
                    gp = gp.with_value(ax.param_name, val)
            return single_pulsar_pta_logL(p, gp, pp_p, config)

    # Nested vmap: innermost over the last argument (dep_p[-1]),
    # outermost over the first (dep_p[0]). After nesting, the output's
    # leading axis is dep_p[0] and trailing is dep_p[-1].
    f_vmapped = f_p
    for k in range(K - 1, -1, -1):
        in_axes = tuple(0 if i == k else None for i in range(K))
        f_vmapped = jax.vmap(f_vmapped, in_axes=in_axes)

    ax_values = tuple(axes[i].values for i in dep_p)

    n_outer = int(axes[dep_p[0]].values.shape[0])
    if chunk_size is None or chunk_size >= n_outer:
        return f_vmapped(*ax_values)

    # Chunked path: split the outermost dependency axis into contiguous
    # chunks of length `chunk_size` (final chunk may be shorter), vmap
    # within each chunk, then concatenate. This caps the per-chunk
    # working set to chunk_size × (per-cell intermediate size), at the
    # cost of issuing ceil(n_outer / chunk_size) sequential dispatches.
    outer_values = ax_values[0]
    rest_values = ax_values[1:]
    chunks = []
    for start in range(0, n_outer, chunk_size):
        stop = min(start + chunk_size, n_outer)
        chunks.append(
            f_vmapped(outer_values[start:stop], *rest_values),
        )
    return jnp.concatenate(chunks, axis=0)


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


# ---------------------------------------------------------------------------
# Public: scan_logL
# ---------------------------------------------------------------------------


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
        If set, the outermost dependency axis (``dep_p[0]``) of each
        per-pulsar :func:`jax.vmap`'d body is split into contiguous
        chunks of this many cells. Within each chunk, all axes are
        :func:`jax.vmap`-ed; chunks are issued sequentially and
        concatenated. Caps per-dispatch GPU memory roughly linearly in
        ``chunk_size`` at the cost of ``ceil(n_outer / chunk_size)``
        sequential dispatches.

        Pick this when a single full-axis :func:`jax.vmap` OOMs on the
        target device — typically because the timing-model phase
        computation has many ``(n_toas,)`` intermediates and one of the
        scanned pulsars has a long TOA list. Leave ``None`` for the
        unchunked default; values larger than the outer axis are
        ignored.

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
    """
    if indexing not in ("xy", "ij"):
        raise ValueError(f"indexing must be 'xy' or 'ij', got {indexing!r}")

    n_axes = len(axes)
    if n_axes == 0:
        return pta_logL(base_global_params, base_pulsar_params, config)

    n_pulsars = len(base_pulsar_params)
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
