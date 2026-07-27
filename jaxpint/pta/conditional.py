r"""Conditional GP posteriors: coefficient distributions given the data.

Injection samples the prior, ``a ~ N(0, \Phi)``; conditioning inverts it.
Given observed residuals ``r = F a + n`` with ``n ~ N(0, C)``, the
coefficients are Gaussian, ``a | r ~ N(\hat{a}, \Sigma)``, with

.. math::

    P \equiv \Sigma^{-1} = \Phi^{-1} + F^T C^{-1} F,
    \qquad
    \hat a = \Sigma\, F^T C^{-1} r

— discovery's ``conditional`` / ``sample_conditional``.  Two levels:

- :func:`conditional_single_pulsar` — the joint posterior of **one
  pulsar's** GP coefficients (all of its ``NoiseModel``'s correlated
  components, plus any injector ``(U, \Phi)`` blocks passed as
  ``external_cov``), given the white-noise diagonal.
- :func:`conditional_gwb` — the posterior of the **correlated-signal**
  coefficients across the whole array, jointly coupled through the ORF
  prior ``\Gamma \otimes diag(S)``.  This is the inferred GWB realization: the
  posterior precision is exactly the ``Sigma_joint`` matrix the
  correlated :func:`~jaxpint.pta.pta_logL` already factors,
  and ``F_p^T C_p^{-1} r_p`` / ``F_p^T C_p^{-1} F_p`` are the same
  inner-tier blocks it consumes.

Uses: noise subtraction / whitening, time-domain GWB waveform
reconstruction (:func:`conditional_gwb_delays`), and posterior
predictive checks.  The mean of the per-pulsar conditional generalizes
the GLS fitter's BLUP ``noise_realizations`` with a covariance and draws.

https://scoste.fr/posts/schur/
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int

from jaxpint.likelihood import _residuals_and_woodbury
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.noise._basis_gp import _BasisGPNoise
from jaxpint.types import GlobalParams, ParameterVector, TOAData
from jaxpint.pta.likelihood import (
    PTAConfig,
    joint_correlated_blocks,
    n_basis_per_injector,
)

__all__ = [
    "ConditionalGP",
    "DelayBand",
    "conditional_single_pulsar",
    "conditional_noise_delays",
    "conditional_noise_delay_bands",
    "conditional_gwb",
    "conditional_gwb_delays",
    "conditional_gwb_delay_bands",
    "conditional_covariance",
    "sample_conditional",
]


class ConditionalGP(NamedTuple):
    r"""Gaussian posterior of GP coefficients, ``a | r ~ N(mean, \Sigma)``.

    Stored in precision form: ``precision_chol`` is the lower-triangular
    Cholesky factor ``L`` of the posterior precision ``P = \Sigma^{-1} = L L^{T}``.
    Materialize ``\Sigma`` with :func:`conditional_covariance`; draw with
    :func:`sample_conditional` (one triangular solve per draw, no dense
    inverse).

    Attributes
    ----------
    mean : (n_coeff,) array
        Posterior mean ``â`` of the coefficients.
    precision_chol : (n_coeff, n_coeff) array
        Lower Cholesky factor of the posterior precision.
    """

    mean: Float[Array, " n_coeff"]
    precision_chol: Float[Array, "n_coeff n_coeff"]


def conditional_covariance(cond: ConditionalGP) -> Float[Array, "n_coeff n_coeff"]:
    r"""Dense posterior covariance ``\Sigma =L^{-T} L^{-1}`` from the precision factor."""
    n = cond.mean.shape[0]
    L_inv = jax.scipy.linalg.solve_triangular(
        cond.precision_chol, jnp.eye(n), lower=True
    )
    return L_inv.T @ L_inv


def sample_conditional(
    key: jax.Array,
    cond: ConditionalGP,
    n_draws: Optional[int] = None,
) -> Float[Array, "... n_coeff"]:
    """Draw coefficient realizations from the conditional posterior.

    ``x = mean + L^{-T} z`` with ``z ~ N(0, I)`` and ``P = L L^{-T}``, so
    ``Cov(x) = L^{-T} L^{-1} = Σ`` exactly.

    Parameters
    ----------
    key
        PRNG key.
    cond
        The conditional posterior.
    n_draws
        If ``None`` (default) return one draw of shape ``(n_coeff,)``;
        otherwise ``(n_draws, n_coeff)``.
    """
    n = cond.mean.shape[0]
    shape = (n,) if n_draws is None else (n_draws, n)
    z = jax.random.normal(key, shape)
    x = jax.scipy.linalg.solve_triangular(
        cond.precision_chol.T,
        z[..., :, None] if n_draws is None else z.T,
        lower=False,
    )
    if n_draws is None:
        return cond.mean + x[:, 0]
    return cond.mean[None, :] + x.T


def _conditional_from_blocks(
    Phi_inv: Float[Array, "n_coeff n_coeff"],
    basis_overlap: Float[Array, "n_coeff n_coeff"],
    basis_proj_residual: Float[Array, " n_coeff"],
) -> ConditionalGP:
    r"""Assemble ``N(\hat{a}, \Sigma)`` from ``\Phi^{-1}``, ``F^{T}C^{-1}F`` and ``F^{T}C^{-1}r``."""
    P = Phi_inv + basis_overlap
    L = jnp.linalg.cholesky(P)
    mean = jax.scipy.linalg.cho_solve((L, True), basis_proj_residual)
    return ConditionalGP(mean=mean, precision_chol=L)


def conditional_single_pulsar(
    toa_data: TOAData,
    timing_model: TimingModel,
    noise_model: NoiseModel,
    params: ParameterVector,
    external_delay: Optional[Float[Array, " n_toas"]] = None,
    external_cov: Optional[
        tuple[Float[Array, "n_toas n_ext"], Float[Array, " n_ext"]]
    ] = None,
) -> ConditionalGP:
    r"""Posterior of one pulsar's GP coefficients given its residuals.

    Conditions on the coefficients of every correlated block in
    ``noise_model`` (red noise, DM/chromatic GPs, ECORR, …) plus any
    ``external_cov`` blocks, in the same stacked column order as
    :func:`~jaxpint.likelihood.single_pulsar_logL` uses — i.e. the
    ``U``,``\Phi`` returned by ``noise_model.covariance`` with
    ``external_cov`` concatenated last.  The time-domain realization of
    the posterior mean is ``U @ cond.mean``.

    Parameters
    ----------
    toa_data, timing_model, noise_model, params
        As for :func:`~jaxpint.likelihood.single_pulsar_logL`.
    external_delay : (n_toas,) array, optional
        Deterministic delay subtracted from the residuals before
        conditioning (e.g. a CW signal).
    external_cov : (U_ext, Phi_ext), optional
        Extra stochastic blocks (e.g. a CURN injector's contribution)
        appended to the noise model's basis.
    """
    # Kernel ECORR note: the returned blocks may be pre-whitened (Ndiag = 1);
    # everything below is coefficient-space (Uᵀ N⁻¹ U, Uᵀ N⁻¹ r), which is
    # invariant under consistent whitening — no correction needed.
    r, Ndiag, U, Phi, _whitener = _residuals_and_woodbury(
        toa_data, timing_model, noise_model, params, external_delay, external_cov
    )

    Ninv_U = U / Ndiag[:, None]
    return _conditional_from_blocks(jnp.diag(1.0 / Phi), U.T @ Ninv_U, Ninv_U.T @ r)


def _noise_component_layout(
    toa_data: TOAData,
    noise_model: NoiseModel,
    params: ParameterVector,
    times: Optional[Float[Array, " n_times"]],
    freq_mhz: Optional[Float[Array, " n_times"] | float] = None,
) -> list[tuple[str, Optional[Float[Array, "n_times n_basis"]], int]]:
    """Per-component ``(name, basis, width)`` in the conditional's column order.

    Walks ``noise_model.correlated`` in order — the same order
    ``NoiseModel.covariance`` stacks basis blocks, hence the layout of
    :func:`conditional_single_pulsar`'s coefficient vector.  ``times is
    None`` evaluates each component's on-grid basis (always possible);
    otherwise ``basis_at`` is asked, and components that cannot evaluate
    off-grid get ``basis = None`` with their ``width`` still counted, so
    the coefficient offsets of later components stay correct.

    Names are the component class names, suffixed ``_1``, ``_2``, … in
    ``correlated`` order when a class appears more than once.
    """
    base_names = [type(c).__name__ for c in noise_model.correlated]
    duplicated = {n for n in base_names if base_names.count(n) > 1}
    seen: dict[str, int] = {}
    layout: list[tuple[str, Optional[Float[Array, "n_times n_basis"]], int]] = []
    for comp, base in zip(noise_model.correlated, base_names):
        if base in duplicated:
            seen[base] = seen.get(base, 0) + 1
            name = f"{base}_{seen[base]}"
        else:
            name = base
        if times is None:
            basis = comp.covariance(toa_data, params)[1]
            width = int(basis.shape[1])
        elif isinstance(comp, _BasisGPNoise):
            width = comp.basis_width()
            basis = comp.basis_at(times, params, freq_mhz=freq_mhz)
        else:
            # Non-basis-GP correlated component (none in-tree): count its
            # width from the covariance triple; no off-grid capability.
            width = int(comp.covariance(toa_data, params)[1].shape[1])
            basis = None
        layout.append((name, basis, width))
    return layout


def _reject_ungridded(
    layout: list[tuple[str, Optional[Float[Array, "n_times n_basis"]], int]],
) -> None:
    """Raise for components that cannot evaluate at the requested times."""
    ungridded = [name for name, basis, _ in layout if basis is None]
    if ungridded:
        raise ValueError(
            f"{', '.join(ungridded)} cannot be evaluated at arbitrary times. "
            "Chromatic/DM components need the evaluation frequency: pass "
            "freq_mhz (e.g. freq_mhz=1400.0 for the standard reference). "
            "Epoch-indicator ECORR has no off-grid form: evaluate at the "
            "TOA epochs (times_seconds=None) or pass skip_ungridded=True "
            "to get explicit None entries for such components."
        )


def conditional_noise_delays(
    toa_data: TOAData,
    noise_model: NoiseModel,
    params: ParameterVector,
    coefficients: Float[Array, " n_coeff"],
    times_seconds: Optional[ArrayLike] = None,
    *,
    freq_mhz: Optional[ArrayLike] = None,
    skip_ungridded: bool = False,
) -> dict[str, Optional[Float[Array, " n_times"]]]:
    r"""Per-component time-domain noise realizations of a coefficient vector.

    The per-pulsar counterpart of :func:`conditional_gwb_delays`: maps a
    coefficient vector in :func:`conditional_single_pulsar`'s stacked
    layout — the posterior ``mean`` or a :func:`sample_conditional` draw
    — to one delay array per correlated component,
    ``delay_c = U_c\, a_c``.

    Parameters
    ----------
    toa_data, noise_model, params
        Exactly what :func:`conditional_single_pulsar` conditioned with —
        the layout (and any parameter-dependent basis scaling) must match.
    coefficients : (n_coeff,) array
        Stacked GP coefficients.  Trailing entries beyond the noise
        model's own blocks (an ``external_cov`` block passed to the
        conditional) are ignored here; fewer than the noise model needs
        is an error.
    times_seconds : optional
        ``None`` (default) evaluates each component at the TOA epochs.
        An array of TDB seconds evaluates on that grid instead (smooth
        curves) via each component's ``basis_at`` hook.
    freq_mhz : float or (n_times,) array, optional
        Radio frequency of the evaluation points in MHz, forwarded to
        every component's ``basis_at`` (achromatic components ignore
        it).  Chromatic/DM components need it off-grid: a scalar
        evaluates at one reference frequency (``1400.0`` is the
        plotting convention); an array gives each time its own
        observing frequency.
    skip_ungridded : bool
        Components that cannot evaluate off-grid (epoch-indicator ECORR;
        chromatic components without ``freq_mhz``) raise a
        ``ValueError`` by default when ``times_seconds`` is given.  Pass
        ``True`` to acknowledge the gap: such components map to explicit
        ``None`` entries instead (never silently dropped) while the
        capable ones are evaluated.

    Returns
    -------
    dict
        Component name (class name, ``_k``-suffixed on duplicates, in
        ``correlated`` order) → delay array, or ``None`` under
        ``skip_ungridded=True`` for components that cannot be evaluated
        at ``times_seconds``.  Kernel ECORR never appears: it has no
        coefficients to reconstruct (use the basis form when epoch
        offsets are wanted).
    """
    coeff = jnp.asarray(coefficients)
    times = None if times_seconds is None else jnp.asarray(times_seconds)
    freq = None if freq_mhz is None else jnp.asarray(freq_mhz)
    layout = _noise_component_layout(toa_data, noise_model, params, times, freq)
    if not skip_ungridded:
        _reject_ungridded(layout)
    total = sum(width for _, _, width in layout)
    if coeff.shape[0] < total:
        raise ValueError(
            f"coefficients has {coeff.shape[0]} entries; the noise model's "
            f"correlated blocks span {total}."
        )
    delays: dict[str, Optional[Float[Array, " n_times"]]] = {}
    offset = 0
    for name, basis, width in layout:
        delays[name] = None if basis is None else basis @ coeff[offset : offset + width]
        offset += width
    return delays


def conditional_noise_delay_bands(
    toa_data: TOAData,
    noise_model: NoiseModel,
    params: ParameterVector,
    cond: ConditionalGP,
    times_seconds: Optional[ArrayLike] = None,
    *,
    freq_mhz: Optional[ArrayLike] = None,
    skip_ungridded: bool = False,
) -> dict[str, Optional[DelayBand]]:
    r"""Per-component reconstruction bands: posterior mean ± 1\sigma delay.

    The per-pulsar counterpart of :func:`conditional_gwb_delay_bands`,
    from a :func:`conditional_single_pulsar` posterior: for each
    correlated component, the mean delay curve and its pointwise
    uncertainty ``std_c(t) = \sqrt{diag(U_c \Sigma_c U_c^T)}``, where
    ``\Sigma_c`` is the component's diagonal block of the joint
    coefficient covariance — i.e. the *marginal* posterior of that
    component's coefficients, coupling to the other components already
    integrated over.

    Parameters
    ----------
    toa_data, noise_model, params, times_seconds, freq_mhz, skip_ungridded
        As for :func:`conditional_noise_delays`.
    cond
        The posterior from :func:`conditional_single_pulsar` (built with
        the same ``toa_data`` / ``noise_model`` / ``params``).
    """
    times = None if times_seconds is None else jnp.asarray(times_seconds)
    freq = None if freq_mhz is None else jnp.asarray(freq_mhz)
    layout = _noise_component_layout(toa_data, noise_model, params, times, freq)
    if not skip_ungridded:
        _reject_ungridded(layout)
    total = sum(width for _, _, width in layout)
    if cond.mean.shape[0] < total:
        raise ValueError(
            f"cond has {cond.mean.shape[0]} coefficients; the noise model's "
            f"correlated blocks span {total}."
        )
    cov = conditional_covariance(cond)
    bands: dict[str, Optional[DelayBand]] = {}
    offset = 0
    for name, basis, width in layout:
        if basis is None:
            bands[name] = None
        else:
            sl = slice(offset, offset + width)
            mean_c = basis @ cond.mean[sl]
            var_c = jnp.einsum("tb,bc,tc->t", basis, cov[sl, sl], basis)
            bands[name] = DelayBand(mean=mean_c, std=jnp.sqrt(var_c))
        offset += width
    return bands


def conditional_gwb(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: PTAConfig,
) -> ConditionalGP:
    r"""Joint posterior of the correlated-signal coefficients across the array.

    The conditional counterpart of the correlated
    :func:`~jaxpint.pta.pta_logL` branch: the same per-pulsar
    inner-tier blocks and the same joint prior
    ``Phi_joint = blockdiag_k(\Gamma_k \otimes diag(S_k))`` assemble the posterior
    precision ``P = Phi_joint^{-1} + blockdiag_p(F_p^{T} C_p^{-1} F_p)`` and mean
    ``\hat{a} = P^{-1} · stack(F_p^{T} C_p^{-1} r_p)``, with ``C_p`` each pulsar's noise
    *excluding* the correlated signal.  Coefficients follow the
    likelihood's (k, p, b) layout: injector-major, then pulsar, then
    basis column — for a single correlated injector,
    ``mean.reshape(n_psr, n_basis)`` gives per-pulsar coefficient rows.

    Because the coupling through \Gamma is retained, the inferred realization
    in pulsar *a* is informed by the residuals of every other pulsar —
    this is the object behind GWB waveform-reconstruction plots.

    Parameters
    ----------
    global_params, pulsar_params, config
        As for :func:`~jaxpint.pta.pta_logL`;
        ``config.correlated_injectors`` must be non-empty.
    """
    if not config.correlated_injectors:
        raise ValueError(
            "conditional_gwb requires at least one correlated injector; "
            "for uncorrelated per-pulsar processes use "
            "conditional_single_pulsar (with the injector's (U, Phi) as "
            "external_cov)."
        )
    # Same joint blocks the correlated pta_logL branch assembles; the
    # conditional reads them as its precision (Phi_joint_inv + FᵀC⁻¹F) and RHS.
    blk = joint_correlated_blocks(global_params, pulsar_params, config)
    return _conditional_from_blocks(
        blk.Phi_joint_inv, blk.basis_overlap_joint, blk.basis_proj_residual_joint
    )


class DelayBand(NamedTuple):
    r"""Pointwise reconstruction band: posterior mean ± 1\sigma of the delay.

    Attributes
    ----------
    mean : (n_times,) array
        Posterior-mean delay at each evaluation time.
    std : (n_times,) array
        Pointwise 1\sigma posterior uncertainty of the delay,
        ``sqrt(diag(J \Sigma_p J^{T}))`` with ``J`` the pulsar's stacked basis.
    """

    mean: Float[Array, " n_times"]
    std: Float[Array, " n_times"]


# Evaluation times for a reconstruction, in seconds: either one shared grid
# for every pulsar, or one grid per pulsar (``len == n_pulsars``).  ``None``
# (the usual default) means "each pulsar's own TOA epochs".
TimesSpec = Union[ArrayLike, Sequence[ArrayLike]]


def _resolve_times(
    config: PTAConfig,
    times_seconds: Optional[TimesSpec],
) -> list[Optional[Float[Array, " n_times"]]]:
    """Per-pulsar evaluation times; ``None`` entries mean "the TOA epochs".

    Returns one entry per pulsar, in ``config`` order.  Entry lengths may
    differ (each pulsar has its own TOA epochs), so ``n_times`` is read
    per-element and does not bind across the list.
    """
    if times_seconds is None:
        return [None] * config.n_pulsars
    if isinstance(times_seconds, (tuple, list)):
        if len(times_seconds) != config.n_pulsars:
            raise ValueError(
                f"times_seconds has {len(times_seconds)} entries, expected "
                f"{config.n_pulsars} (one per pulsar), or a single shared array."
            )
        return [jnp.asarray(t) for t in times_seconds]
    shared = jnp.asarray(times_seconds)
    return [shared] * config.n_pulsars


def _pulsar_bases_and_indices(
    config: PTAConfig,
    times: list[Optional[Float[Array, " n_times"]]],
) -> tuple[
    list[Float[Array, "n_times n_basis_total"]],
    list[Int[Array, " n_basis_total"]],
]:
    """Per-pulsar stacked basis ``J_p`` and coefficient indices ``idx_p``.

    ``J_p`` horizontally stacks every correlated injector's basis at
    pulsar ``p``'s evaluation times; ``idx_p`` gathers that pulsar's
    coefficient positions out of the flat (k, p, b) layout, in the same
    column order, so ``J_p @ coefficients[idx_p]`` is the pulsar's total
    correlated-signal delay.

    Parameters
    ----------
    config
        As for :func:`conditional_gwb`.
    times
        One entry per pulsar, as returned by :func:`_resolve_times`; a
        ``None`` entry evaluates that pulsar at its TOA epochs.

    Returns
    -------
    J : list of (n_times, n_basis_total) arrays
        One per pulsar; ``n_basis_total = sum_k n_basis_k``.  Row counts
        may differ between pulsars (see :func:`_resolve_times`).
    idx : list of (n_basis_total,) integer arrays
        One per pulsar; positions into the flat (k, p, b) coefficient
        vector, column-aligned with the matching ``J``.
    """
    n_psr = config.n_pulsars
    n_basis_per_k = n_basis_per_injector(
        config.correlated_injectors, config.toa_data_list[0]
    )
    bases: list[list[Float[Array, "n_times n_basis_k"]]] = [[] for _ in range(n_psr)]
    indices: list[list[Int[Array, " n_basis_k"]]] = [[] for _ in range(n_psr)]
    offset = 0
    for k, cinj in enumerate(config.correlated_injectors):
        nb = n_basis_per_k[k]
        for p in range(n_psr):
            if times[p] is None:
                F_kp = cinj.get_fourier_basis(config.toa_data_list[p])
            else:
                basis_at = getattr(cinj, "basis_at", None)
                if basis_at is None:
                    raise NotImplementedError(
                        f"{type(cinj).__name__} does not implement "
                        "basis_at(times_seconds); evaluation at "
                        "non-TOA times needs it."
                    )
                F_kp = basis_at(times[p])
            bases[p].append(F_kp)
            indices[p].append(offset + p * nb + jnp.arange(nb))
        offset += n_psr * nb
    J = [jnp.concatenate(b, axis=1) for b in bases]
    idx = [jnp.concatenate(i) for i in indices]
    return J, idx


def conditional_gwb_delays(
    config: PTAConfig,
    coefficients: Float[Array, " n_coeff"],
    times_seconds: Optional[TimesSpec] = None,
) -> tuple[Float[Array, " n_times"], ...]:
    r"""Per-pulsar time-domain realization of correlated-signal coefficients.

    Maps a coefficient vector in :func:`conditional_gwb`'s (k, p, b)
    layout — the posterior ``mean`` or a :func:`sample_conditional`
    draw — to one delay array per pulsar,
    ``delay_p = \Sigma_k F_{k,p} a_{k,p}``.  This is the reconstructed
    waveform to overplot on (or subtract from) each pulsar's residuals.

    Parameters
    ----------
    config, coefficients
        As for :func:`conditional_gwb`.
    times_seconds : optional
        Evaluation times in seconds.  ``None`` (default) evaluates at
        each pulsar's TOA epochs; a single array is a shared grid for
        all pulsars (smooth reconstruction curves — the basis is
        analytic in time); a tuple/list gives one grid per pulsar.
    """
    times = _resolve_times(config, times_seconds)
    J, idx = _pulsar_bases_and_indices(config, times)
    return tuple(J[p] @ coefficients[idx[p]] for p in range(config.n_pulsars))


def conditional_gwb_delay_bands(
    config: PTAConfig,
    cond: ConditionalGP,
    times_seconds: Optional[TimesSpec] = None,
) -> tuple[DelayBand, ...]:
    r"""Reconstruction bands: posterior mean ± 1σ delay per pulsar.

    The pointwise delay uncertainty propagates the coefficient posterior
    through the basis: ``std_p(t) = sqrt(diag(J_p Σ_p J_p^T))``, with
    ``\Sigma_p`` the pulsar's (cross-injector) block of the joint conditional
    covariance — so the band reflects the full coupling of the
    coefficient posterior, not per-coefficient variances alone.  These
    are the mean curve and 1\sigma envelope of a GWB waveform-reconstruction
    plot.

    Parameters
    ----------
    config
        As for :func:`conditional_gwb`.
    cond
        The joint conditional from :func:`conditional_gwb`.
    times_seconds : optional
        As for :func:`conditional_gwb_delays`.
    """
    times = _resolve_times(config, times_seconds)
    J, idx = _pulsar_bases_and_indices(config, times)
    cov = conditional_covariance(cond)
    bands = []
    for p in range(config.n_pulsars):
        mean_p = J[p] @ cond.mean[idx[p]]
        cov_p = cov[jnp.ix_(idx[p], idx[p])]
        var_p = jnp.einsum("tb,bc,tc->t", J[p], cov_p, J[p])
        bands.append(DelayBand(mean=mean_p, std=jnp.sqrt(var_p)))
    return tuple(bands)
