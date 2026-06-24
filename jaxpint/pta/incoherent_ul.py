"""Real-mode Bayesian CW distance upper limit with the pulsar term marginalized
over the (unknown) pulsar **distance**.

At a fixed sky position, GW frequency, and source orientation, each pulsar's CW
timing residual per unit strain ``h0`` is a 2-template combination::

    s_a(h0, Δ) / h0 = (1 - cosΔ) · e_a  +  sinΔ · ps_a

where ``e_a`` is the Earth-term residual and ``ps_a`` the pulsar-term quadrature
(the pulsar term at phase Δ=π/2), and ``Δ`` is the pulsar-term phase lag (set by
the pulsar distance ``L`` via ``Δ_p(L) = 2π f L (1+cos μ) / c``).  (The plan's
third template ``pc_a`` -- the pulsar term at Δ=0 -- equals ``-e_a`` exactly, so
this 2-template basis is the full-rank version of the same construction.)

Per pulsar we extract the timing-marginalized GLS projections and Gram::

    b_a = ((d|e_a), (d|ps_a))            # matched filter (real mode: actual residuals)
    M_a = 2x2 Gram of {e_a, ps_a}        # noise-weighted, timing-marginalized

Then with ``A(Δ) = (1-cosΔ, sinΔ)``::

    logL_a(h0, Δ) = h0 · (b_a · A(Δ)) − ½ h0² · (A(Δ)ᵀ M_a A(Δ))
    logL_a^marg(h0) = log mean_Δ exp(logL_a(h0, Δ))   # over a supplied Δ grid
    logL^marg(h0)   = Σ_a logL_a^marg(h0)             # factorizes: block-diag noise

The Δ grid encodes the distance prior: a uniform ``[0, 2π)`` grid is the
**flat-phase** limit (exact when the prior spans ≫1 phase cycle -- the realistic
case for ~10% parallaxes, since Δ_p ~ 1e4-1e6 rad), while a distance-derived grid
``Δ_p(L_i)`` for ``L_i`` uniform in ``[1/PX − kσ_L, 1/PX + kσ_L]`` keeps the
parallax information when the prior is sub-cycle.  The per-pulsar marginal MUST be
formed before summing over pulsars (logsumexp does not commute with the sum).

The posterior ``∝ exp(logL^marg(h0))`` on ``h0 ≥ 0`` is no longer a truncated
Gaussian, so the 95% upper limit is taken numerically (:func:`h0_95_grid`), then
converted to a luminosity-distance lower limit via
:func:`jaxpint.pta.cw_upper_limit.h0_to_distance`.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

from jaxpint.pta.signals.cw import _C, _KPC_TO_M


# --------------------------------------------------------------------------- b, M
def bM2_coeffs(
    logL2: Callable[[Float[Array, ""], Float[Array, ""]], Float[Array, ""]],
) -> tuple[Float[Array, " 2"], Float[Array, "2 2"]]:
    """Extract ``b`` (2-vector) and ``M`` (2x2 Gram) from a 2-amplitude logL.

    ``logL2(Ae, As)`` must be exactly quadratic in the two linear amplitudes
    (it is: each template enters the residual linearly).  For
    ``logL2 = b·A − ½ AᵀMA`` the gradient is ``b − M A``, so ``M[:,j] = grad(0) −
    grad(e_j)``.  Uses three first-order gradient evaluations (the
    finite-difference-of-gradients trick, as in
    :func:`jaxpint.pta.cw_upper_limit.quadratic_coeffs`) -- lighter on memory than
    ``jax.hessian`` through the full per-pulsar likelihood.
    """
    grad = jax.grad(logL2, argnums=(0, 1))
    z = jnp.float64(0.0)
    o = jnp.float64(1.0)
    b = jnp.asarray(grad(z, z))  # (2,) = b
    col0 = b - jnp.asarray(grad(o, z))  # M[:,0]
    col1 = b - jnp.asarray(grad(z, o))  # M[:,1]
    M = jnp.stack([col0, col1], axis=1)
    return b, 0.5 * (M + M.T)  # symmetrize tiny numerical asymmetry


def extract_pulsar_bM(
    g: Callable,
    reduced_params,
    e: Float[Array, " n_toas"],
    ps: Float[Array, " n_toas"],
) -> tuple[Float[Array, " 2"], Float[Array, "2 2"]]:
    """``(b, M)`` for one pulsar given its marginalized likelihood ``g`` and the
    two unit-strain templates ``e`` (Earth term) and ``ps`` (pulsar quadrature).

    ``g(reduced_params, external_delay=...)`` is the single-pulsar
    timing-marginalized log-likelihood (from
    :func:`jaxpint.bayes.marginalize`).  Injecting ``Ae·e + As·ps`` as the
    external delay makes ``g`` exactly quadratic in ``(Ae, As)``; differentiating
    the *actual* ``g`` inherits the correct real-mode matched-filter sign.
    """

    def logL2(Ae, As):
        return g(reduced_params, external_delay=Ae * e + As * ps)

    return bM2_coeffs(logL2)


# --------------------------------------------------------------------- phase grids
def flat_phase_grid(n_phase: int = 256) -> Float[Array, " n_phase"]:
    """Uniform midpoint grid of pulsar-term phases over ``[0, 2π)`` -- the
    flat-phase (broad-prior) limit of the distance marginalization."""
    return (jnp.arange(n_phase) + 0.5) * (2.0 * jnp.pi / n_phase)


def distance_phase_grid(
    L0_kpc: float,
    sigma_L_kpc: float,
    k: float,
    cos_mu: float,
    f_gw: float,
    n_dist: int,
) -> Float[Array, " n_dist"]:
    """Pulsar-term phases ``Δ_p(L_i)`` for ``L_i`` uniform in
    ``[L0 − kσ_L, L0 + kσ_L]`` (clipped to ``L > 0``).

    ``Δ_p(L) = 2π f L (1+cos μ) / c`` with ``L`` in kpc.  Use only when the prior
    is sub-cycle (``n_dist`` chosen ~16 points per phase cycle); for a broad prior
    use :func:`flat_phase_grid` (the exact limit) instead -- resolving a broad
    prior here would need ~1e5-1e7 points.
    """
    lo = jnp.clip(L0_kpc - k * sigma_L_kpc, 1e-6, None)
    hi = L0_kpc + k * sigma_L_kpc
    L = jnp.linspace(lo, hi, n_dist)
    return 2.0 * jnp.pi * f_gw * (L * _KPC_TO_M) * (1.0 + cos_mu) / _C


def mixed_phase_A(
    is_tight: Float[Array, " n_psr"],
    L0_kpc: Float[Array, " n_psr"],
    sigma_L_kpc: float,
    k: float,
    cos_mu: Float[Array, " n_psr"],
    f_gw: float,
    n_phase: int,
    min_pts_per_cycle: float = 16.0,
) -> Float[Array, "n_psr n_phase 2"]:
    """Per-pulsar coefficient vectors ``A(Δ)`` for the hybrid map (one sky pixel).

    Pulsars flagged ``is_tight`` (a well-measured / hypothetically tight distance)
    marginalize the phase over their narrow distance prior via
    :func:`distance_phase_grid` (localized -> coherent contribution); the rest use
    :func:`flat_phase_grid` over ``[0, 2π)`` (incoherent).  ``cos_mu`` and
    ``L0_kpc`` are per pulsar (``cos_mu`` is pixel-dependent), ``sigma_L_kpc``/``k``/
    ``f_gw`` are shared.  Stack the result over a pixel chunk in the driver.

    Adaptive: a tight prior spanning ``N_wrap = 2 k σ_L f (1+cos μ) / c`` phase
    cycles needs ``≥ min_pts_per_cycle`` grid points per cycle to resolve; once it
    exceeds what ``n_phase`` resolves the distance grid would alias, so we fall back
    to the exact flat-phase limit (a loose "tight" prior is effectively incoherent).
    Note ``L0`` cancels out of ``N_wrap``.
    """
    flat = flat_phase_grid(n_phase)  # (n_phase,)
    dist = jax.vmap(
        lambda L0, cm: distance_phase_grid(L0, sigma_L_kpc, k, cm, f_gw, n_phase),
        in_axes=(0, 0),
    )(L0_kpc, cos_mu)  # (n_psr, n_phase)
    n_wrap = 2.0 * k * sigma_L_kpc * f_gw * _KPC_TO_M * (1.0 + cos_mu) / _C
    use_dist = is_tight & (n_wrap <= n_phase / min_pts_per_cycle)
    grids = jnp.where(use_dist[:, None], dist, flat[None, :])  # (n_psr, n_phase)
    return _A_of_phase(grids)  # (n_psr, n_phase, 2)


# ------------------------------------------------------------------- marginal logL
def _A_of_phase(phase: Float[Array, " n"]) -> Float[Array, "n 2"]:
    """``A(Δ) = (1 − cosΔ, sinΔ)`` stacked over a phase grid."""
    return jnp.stack([1.0 - jnp.cos(phase), jnp.sin(phase)], axis=-1)


def earth_only_A() -> Float[Array, "1 2"]:
    """The single signal-coefficient vector for the Earth-term-only baseline.

    With the pulsar term dropped, the signal is fixed -- ``s = h0 * e`` -- so the
    coefficient vector is ``A = (1, 0)`` and there is nothing to marginalize.
    Routed through the same numerical UL as the marginalized case, this gives a
    *method-matched* Earth-term map (a controlled comparison)."""
    return jnp.array([[1.0, 0.0]])


def logL_pulsar_marg(
    h0: Float[Array, ""],
    b: Float[Array, " 2"],
    M: Float[Array, "2 2"],
    A: Float[Array, "n 2"],
) -> Float[Array, ""]:
    """Per-pulsar log-likelihood marginalized over the supplied signal-coefficient
    vectors ``A`` (``log mean_i exp(logL_a(A_i))``).

    ``A`` is the set of coefficient 2-vectors to average over: ``_A_of_phase`` of a
    phase grid for the distance-marginalized case, or :func:`earth_only_A` (a
    singleton ``(1,0)``) for the Earth-term-only baseline.
    """
    bA = A @ b  # (n,)
    AMA = jnp.einsum("ni,ij,nj->n", A, M, A)  # (n,)
    logL = h0 * bA - 0.5 * h0**2 * AMA
    return logsumexp(logL) - jnp.log(A.shape[0])


def total_logL_marg(
    h0: Float[Array, ""],
    b_stack: Float[Array, "n_psr 2"],
    M_stack: Float[Array, "n_psr 2 2"],
    A_stack: Float[Array, "n_psr n 2"],
) -> Float[Array, ""]:
    """Σ over pulsars of the per-pulsar marginalized logL (factorizes)."""
    per = jax.vmap(logL_pulsar_marg, in_axes=(None, 0, 0, 0))(
        h0, b_stack, M_stack, A_stack
    )
    return jnp.sum(per)


# ----------------------------------------------------------------------- h0 95% UL
def h0_95_grid(
    b_stack: Float[Array, "n_psr 2"],
    M_stack: Float[Array, "n_psr 2 2"],
    A_stack: Float[Array, "n_psr n 2"],
    h0_max: Float[Array, ""],
    n_h0: int = 512,
    level: float = 0.95,
) -> Float[Array, ""]:
    """95% quantile of the (improper-uniform-prior) ``h0`` posterior on ``h0 ≥ 0``.

    The phase-marginalized posterior is not a truncated Gaussian, so the quantile
    is computed numerically: evaluate ``logL^marg(h0)`` on ``[0, h0_max]``, form
    the normalized posterior weight, and interpolate the CDF.  ``h0_max`` must
    cover the 95% mass -- the driver sets it adaptively (see the module/driver).

    The distance-marginalized posterior has a power-law tail ``~ h0^{-N}`` (``N`` =
    number of pulsars), because the ``Δ≈0`` phases -- where the Earth and pulsar
    terms cancel -- give vanishing signal power; it is therefore proper only for
    ``N >= 2`` (a real PTA always satisfies this).  The Earth-term-only baseline
    (singleton ``A``) is a plain truncated Gaussian and proper for any ``N``.
    """
    h0 = jnp.linspace(0.0, h0_max, n_h0)
    logpost = jax.vmap(total_logL_marg, in_axes=(0, None, None, None))(
        h0, b_stack, M_stack, A_stack
    )
    w = jnp.exp(logpost - jnp.max(logpost))  # uniform prior on h0>=0
    cdf = jnp.cumsum(w)
    cdf = cdf / cdf[-1]
    return jnp.interp(jnp.float64(level), cdf, h0)
