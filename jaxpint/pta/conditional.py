"""Conditional GP posteriors: coefficient distributions given the data.

Injection samples the prior, ``a ~ N(0, \Phi)``; conditioning inverts it.
Given observed residuals ``r = F a + n`` with ``n ~ N(0, C)``, the
coefficients are Gaussian, ``a | r ~ N(\hat{a}, \Sigma)``, with

.. math::

    P \\equiv \\Sigma^{-1} = \\Phi^{-1} + F^T C^{-1} F,
    \\qquad
    \\hat a = \\Sigma\\, F^T C^{-1} r

— discovery's ``conditional`` / ``sample_conditional``.  Two levels:

- :func:`conditional_single_pulsar` — the joint posterior of **one
  pulsar's** GP coefficients (all of its ``NoiseModel``'s correlated
  components, plus any injector ``(U, \Phi)`` blocks passed as
  ``external_cov``), given the white-noise diagonal.
- :func:`conditional_gwb` — the posterior of the **correlated-signal**
  coefficients across the whole array, jointly coupled through the ORF
  prior ``\Gamma \otimes diag(S)``.  This is the inferred GWB realization: the
  posterior precision is exactly the ``Sigma_joint`` matrix the
  correlated :func:`~jaxpint.pta.likelihood.pta_logL` already factors,
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

from jaxpint.fitters import compute_time_residuals
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import GlobalParams, ParameterVector, TOAData
from jaxpint.utils import concat_woodbury_blocks
from jaxpint.pta.likelihood import (
    PTAConfig,
    _assemble_basis_overlap_joint_kpb,
    _assemble_basis_proj_residual_joint_kpb,
    _collect_per_pulsar_external_inputs,
    _n_basis_per_injector,
    _per_pulsar_intermediates,
    _phi_and_phi_inv_joint,
    _stacked_fourier_basis,
)

__all__ = [
    "ConditionalGP",
    "DelayBand",
    "conditional_single_pulsar",
    "conditional_gwb",
    "conditional_gwb_delays",
    "conditional_gwb_delay_bands",
    "conditional_covariance",
    "sample_conditional",
]


class ConditionalGP(NamedTuple):
    """Gaussian posterior of GP coefficients, ``a | r ~ N(mean, \Sigma)``.

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
    r = compute_time_residuals(timing_model, toa_data, params)
    if external_delay is not None:
        r = r - external_delay

    Ndiag, U_noise, Phi_noise = noise_model.covariance(toa_data, params)
    woodbury = concat_woodbury_blocks((U_noise, Phi_noise), external_cov)
    assert woodbury is not None  # first block is always non-None
    U, Phi = woodbury

    Ninv_U = U / Ndiag[:, None]
    return _conditional_from_blocks(jnp.diag(1.0 / Phi), U.T @ Ninv_U, Ninv_U.T @ r)


def conditional_gwb(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: PTAConfig,
) -> ConditionalGP:
    r"""Joint posterior of the correlated-signal coefficients across the array.

    The conditional counterpart of the correlated
    :func:`~jaxpint.pta.likelihood.pta_logL` branch: the same per-pulsar
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
        As for :func:`~jaxpint.pta.likelihood.pta_logL`;
        ``config.correlated_injectors`` must be non-empty.
    """
    if not config.correlated_injectors:
        raise ValueError(
            "conditional_gwb requires at least one correlated injector; "
            "for uncorrelated per-pulsar processes use "
            "conditional_single_pulsar (with the injector's (U, Phi) as "
            "external_cov)."
        )
    n_psr = config.n_pulsars
    n_basis_per_k = _n_basis_per_injector(
        config.correlated_injectors, config.toa_data_list[0]
    )

    basis_proj_residual_per_pulsar = []
    basis_overlap_per_pulsar = []
    for p in range(n_psr):
        ext_delay, ext_cov = _collect_per_pulsar_external_inputs(
            p,
            config.toa_data_list[p],
            pulsar_params[p],
            global_params,
            config.signal_injectors,
        )
        F_stack_p = _stacked_fourier_basis(
            config.correlated_injectors, config.toa_data_list[p]
        )
        _rCr, _logdet, basis_proj_residual_p, basis_overlap_p = (
            _per_pulsar_intermediates(
                config.toa_data_list[p],
                config.timing_models[p],
                config.noise_models[p],
                pulsar_params[p],
                F_stack_p,
                external_delay=ext_delay,
                external_cov=ext_cov,
            )
        )
        basis_proj_residual_per_pulsar.append(basis_proj_residual_p)
        basis_overlap_per_pulsar.append(basis_overlap_p)

    _Phi_joint, Phi_joint_inv = _phi_and_phi_inv_joint(
        config.correlated_injectors, global_params
    )
    basis_overlap_joint = _assemble_basis_overlap_joint_kpb(
        basis_overlap_per_pulsar, n_basis_per_k, n_psr
    )
    basis_proj_residual_joint = _assemble_basis_proj_residual_joint_kpb(
        basis_proj_residual_per_pulsar, n_basis_per_k, n_psr
    )
    return _conditional_from_blocks(
        Phi_joint_inv, basis_overlap_joint, basis_proj_residual_joint
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
    n_basis_per_k = _n_basis_per_injector(
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
                basis_at = getattr(cinj, "get_fourier_basis_at", None)
                if basis_at is None:
                    raise NotImplementedError(
                        f"{type(cinj).__name__} does not implement "
                        "get_fourier_basis_at(times); evaluation at "
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
