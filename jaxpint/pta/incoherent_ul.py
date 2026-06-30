"""Bayesian CW distance upper limit with the pulsar term marginalized
over the (unknown) pulsar **distance**.

At a fixed sky position, GW frequency, and source orientation, each pulsar's CW
timing residual per unit strain ``h0`` is a 2-template combination::

    s_a(h0, Δ) / h0 = (1 - cosΔ) · e_a  +  sinΔ · ps_a

where ``e_a`` is the Earth-term residual and ``ps_a`` the pulsar-term quadrature
(the pulsar term at phase Δ=π/2), and ``Δ`` is the pulsar-term phase lag (set by
the pulsar distance ``L`` via ``Δ_p(L) = 2π f L (1+cos μ) / c``).

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
parallax information when the prior is sub-cycle.

Beyond the single-source upper limit, this module also holds the shared per-pulsar
CW-block machinery used by **localization**: :func:`extract_pulsar_bM` generalizes to
:func:`extract_pulsar_blocks` (the ``2S``-amplitude Gram for ``S`` sources), and
:func:`condition_on_statics` bakes ``S-1`` static sources into an effective matched
filter so one source can be scanned in O(1)-in-S -- the conditioned scan, which
reuses the same ``(b, M)`` reductions below.

"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.bayes.credible import grid_credible_upper_limit
from jaxpint.bayes.grid_marginal import grid_log_marginal, grid_log_profile
from jaxpint.pta.signals.cw import _C, _KPC_TO_M
from jaxpint.utils import quadratic_form_coeffs


def bM2_coeffs(
    logL2: Callable[[Float[Array, ""], Float[Array, ""]], Float[Array, ""]],
) -> tuple[Float[Array, " 2"], Float[Array, "2 2"]]:
    """Extract ``b`` (2-vector) and ``M`` (2x2 Gram) from a 2-amplitude logL.

    ``logL2(Ae, As)`` must be exactly quadratic in the two linear amplitudes
    (it is: each template enters the residual linearly).  The ``n=2`` case of
    :func:`jaxpint.utils.quadratic_form_coeffs`: ``b`` is the matched filter
    ``((d|e), (d|ps))`` and ``M`` the 2x2 noise-weighted Gram of the two
    templates.
    """
    return quadratic_form_coeffs(lambda A: logL2(A[0], A[1]), 2)


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
    :func:`jaxpint.bayes.marginalize_single_pulsar`).  Injecting ``Ae·e + As·ps`` as the
    external delay makes ``g`` exactly quadratic in ``(Ae, As)``; differentiating
    the *actual* ``g`` inherits the correct real-mode matched-filter sign.
    """

    def logL2(Ae, As):
        return g(reduced_params, external_delay=Ae * e + As * ps)

    return bM2_coeffs(logL2)


# ------------------------------------------------- multi-source conditioned scan
# For S coherent sources the per-pulsar logL is quadratic in all sources'
# amplitudes, logL_i(a) = a·b_i − ½ aᵀ G_i a with a = [a^0, …, a^{S-1}].  To scan
# one source with the others baked, complete the square in its block:
#     logL_i(a^0) = a^0·b_eff_i − ½ (a^0)ᵀ G_i^{00} a^0 + const_i
#     b_eff_i = b_i^0 − G_i^{0,static} a_static          # data minus statics
# The per-grid-point scan then touches only G_i^{00} and the precomputed b_eff_i
# (S enters only the once-per-pixel b_eff), and reuses the reductions below by
# feeding (b_eff, G^{00}) as (b, M).  const_i is independent of the scanned source's
# sky, so it shifts the map by a constant and is dropped.
def extract_pulsar_blocks(
    g: Callable,
    reduced_params,
    basis: Float[Array, "m n_toas"],
) -> tuple[Float[Array, " m"], Float[Array, "m m"]]:
    """Per-pulsar matched filter ``b`` and Gram ``G`` for ``m`` CW basis waveforms.

    The multi-source generalization of :func:`extract_pulsar_bM` (its ``m = 2``
    case): inject ``A @ basis`` as the external delay so ``g`` is exactly quadratic
    in the ``m`` amplitudes, and read off ``logL(A) = c + b·A - 1/2 Aᵀ G A``.

    Parameters
    ----------
    g : callable
        Single-pulsar timing-marginalized log-likelihood,
        ``g(reduced_params, external_delay=...)`` (from
        :func:`jaxpint.bayes.marginalize_single_pulsar`).
    reduced_params
        The reduced-parameter skeleton ``g`` expects.
    basis : (m, n_toas) array
        The ``m`` unit-amplitude CW templates.  For ``S`` sources this is the
        stacked ``[e_0, ps_0, e_1, ps_1, ...]`` (Earth term + pulsar quadrature per
        source), so ``m = 2S``; the scanned source's two templates come first.

    Returns
    -------
    b : (m,) array
        Matched filter ``(d | basis_k)``.
    G : (m, m) array
        Noise-weighted, timing-marginalized Gram of the basis waveforms.
    """

    def logL(A: Float[Array, " m"]) -> Float[Array, ""]:
        return g(reduced_params, external_delay=A @ basis)

    return quadratic_form_coeffs(logL, basis.shape[0])


def condition_on_statics(
    b: Float[Array, " m"],
    G: Float[Array, "m m"],
    a_static: Float[Array, " m_static"],
    n_scan: int = 2,
) -> tuple[Float[Array, " n_scan"], Float[Array, "n_scan n_scan"]]:
    """Bake the static sources into the scanned source's matched filter.

    Completes the square in the first ``n_scan`` amplitudes (the scanned source),
    holding the remaining ``m - n_scan`` amplitudes fixed at ``a_static``::

        b_eff  = b[:n_scan] - G[:n_scan, n_scan:] @ a_static
        G_scan = G[:n_scan, :n_scan]

    Feed ``(b_eff, G_scan)`` as ``(b, M)`` to the reductions below
    (:func:`total_logL_marg` etc.) to scan the source over its distance grid.  The
    static self-energy ``const`` is dropped: it is independent of the scanned
    source's sky and only shifts the map by a constant.

    Parameters
    ----------
    b : (m,) array
    G : (m, m) array
        Full multi-source matched filter / Gram (from :func:`extract_pulsar_blocks`),
        with the scanned source occupying the first ``n_scan`` entries.
    a_static : (m - n_scan,) array
        Fixed coefficients of the baked (static) sources, stacked in the same order
        as their blocks in ``b``/``G``.
    n_scan : int
        Number of amplitudes belonging to the scanned source (2 for a single CW).

    Returns
    -------
    b_eff : (n_scan,) array
        Static-corrected matched filter for the scanned source.
    G_scan : (n_scan, n_scan) array
        The scanned source's self-block Gram.
    """
    b_eff = b[:n_scan] - G[:n_scan, n_scan:] @ a_static
    G_scan = G[:n_scan, :n_scan]
    return b_eff, G_scan


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
    :func:`flat_phase_grid` over ``[0, 2π)`` (incoherent).

    Parameters
    ----------
    is_tight : (n_psr,) bool array
        Per-pulsar flag: use the narrow distance grid (coherent) vs the flat phase
        grid (incoherent).
    L0_kpc : (n_psr,) array
        Fiducial pulsar distances (kpc), used by the tight branch.
    sigma_L_kpc : float
        Distance-prior width (kpc).
    k : float
        Half-width of the distance grid in units of ``sigma_L_kpc``.
    cos_mu : (n_psr,) array
        Cosine of the angle between each pulsar and the GW source direction.
    f_gw : float
        GW frequency (Hz).
    n_phase : int
        Number of phase grid points per pulsar.
    min_pts_per_cycle : float
        A tight pulsar falls back to the flat grid unless its distance prior spans
        few enough phase cycles to keep at least this many points per cycle.

    Returns
    -------
    (n_psr, n_phase, 2) array
        Per-pulsar coefficient grid ``A(Δ) = (1 − cosΔ, sinΔ)``.
    """
    flat = flat_phase_grid(n_phase)  # (n_phase,)
    dist = jax.vmap(
        lambda L0, cm: distance_phase_grid(L0, sigma_L_kpc, k, cm, f_gw, n_phase),
        in_axes=(0, 0),
    )(L0_kpc, cos_mu)  # (n_psr, n_phase)
    n_wrap = 2.0 * k * sigma_L_kpc * f_gw * _KPC_TO_M * (1.0 + cos_mu) / _C
    use_dist = is_tight & (n_wrap <= n_phase / min_pts_per_cycle)
    grids = jnp.where(use_dist[:, None], dist, flat[None, :])  # (n_psr, n_phase)
    # A(Δ) = (1 − cosΔ, sinΔ) stacked over the per-pulsar phase grid.
    return jnp.stack(
        [1.0 - jnp.cos(grids), jnp.sin(grids)], axis=-1
    )  # (n_psr, n_phase, 2)


# ---------------------------------------------- per-pulsar logL grid + its reductions
def _pulsar_logL_grid(
    h0: Float[Array, ""],
    b: Float[Array, " 2"],
    M: Float[Array, "2 2"],
    A: Float[Array, "n 2"],
) -> Float[Array, " n"]:
    """Per-pulsar log-likelihood over the coefficient grid ``A`` (relative to the
    no-signal baseline): ``logL(Δ) = h0 (b·A(Δ)) − ½ h0² A(Δ)ᵀ M A(Δ)``.

    ``A`` is the set of coefficient 2-vectors ``(1 − cosΔ, sinΔ)`` -- a phase grid
    (e.g. :func:`mixed_phase_A`) for the distance-resolved case, or the singleton
    ``(1, 0)`` for the Earth-term-only baseline.  The marginal/profile reductions
    below consume this one scan.

    Parameters
    ----------
    h0 : scalar
        Linear strain amplitude.
    b : (2,) array
        Per-pulsar matched filter ``((d|e), (d|ps))`` (from :func:`extract_pulsar_bM`).
    M : (2, 2) array
        Per-pulsar noise-weighted Gram of the two templates.
    A : (n, 2) array
        Coefficient grid ``(1 − cosΔ, sinΔ)`` over ``n`` phase points.

    Returns
    -------
    (n,) array
        Per-grid-point log-likelihood, relative to the no-signal baseline.
    """
    bA = A @ b  # (n,)
    AMA = jnp.einsum("ni,ij,nj->n", A, M, A)  # (n,)
    return h0 * bA - 0.5 * h0**2 * AMA  # (n,)


def logL_pulsar_marg(
    h0: Float[Array, ""],
    b: Float[Array, " 2"],
    M: Float[Array, "2 2"],
    A: Float[Array, "n 2"],
) -> Float[Array, ""]:
    """Per-pulsar log-likelihood **marginalized** over the grid ``A``
    (``log mean_Δ exp logL``) -- the Bayesian reduction (integrate the
    distance/phase nuisance), via :func:`jaxpint.bayes.grid_log_marginal` over the
    flat (uniform-prior) phase grid.

    Parameters
    ----------
    h0, b, M, A
        As in :func:`_pulsar_logL_grid`.

    Returns
    -------
    scalar
        The phase-marginalized per-pulsar log-likelihood.
    """
    return grid_log_marginal(_pulsar_logL_grid(h0, b, M, A))


def logL_pulsar_profile(
    h0: Float[Array, ""],
    b: Float[Array, " 2"],
    M: Float[Array, "2 2"],
    A: Float[Array, "n 2"],
) -> Float[Array, ""]:
    """Per-pulsar **profile** log-likelihood (``max_Δ logL``) -- the frequentist
    reduction (maximize the nuisance), via :func:`jaxpint.bayes.grid_log_profile`.
    The sharper but alias-prone twin of :func:`logL_pulsar_marg`; for localization
    it is a diagnostic, not the credible-region map.

    Parameters
    ----------
    h0, b, M, A
        As in :func:`_pulsar_logL_grid`.

    Returns
    -------
    scalar
        The phase-profiled per-pulsar log-likelihood.
    """
    return grid_log_profile(_pulsar_logL_grid(h0, b, M, A))


def total_logL_marg(
    h0: Float[Array, ""],
    b_stack: Float[Array, "n_psr 2"],
    M_stack: Float[Array, "n_psr 2 2"],
    A_stack: Float[Array, "n_psr n 2"],
) -> Float[Array, ""]:
    """Σ over pulsars of the per-pulsar marginalized logL.

    The full distance-marginalized log-likelihood; block-diagonal noise makes it a
    plain sum over :func:`logL_pulsar_marg`.

    Parameters
    ----------
    h0 : scalar
        Linear strain amplitude.
    b_stack : (n_psr, 2) array
        Per-pulsar matched filters.
    M_stack : (n_psr, 2, 2) array
        Per-pulsar Grams.
    A_stack : (n_psr, n, 2) array
        Per-pulsar coefficient grids (e.g. from :func:`mixed_phase_A`).

    Returns
    -------
    scalar
        Total distance-marginalized log-likelihood, summed over pulsars.
    """
    per = jax.vmap(logL_pulsar_marg, in_axes=(None, 0, 0, 0))(
        h0, b_stack, M_stack, A_stack
    )
    return jnp.sum(per)


def total_logL_profile(
    h0: Float[Array, ""],
    b_stack: Float[Array, "n_psr 2"],
    M_stack: Float[Array, "n_psr 2 2"],
    A_stack: Float[Array, "n_psr n 2"],
) -> Float[Array, ""]:
    """Σ over pulsars of the per-pulsar profile logL -- the max-over-grid twin of
    :func:`total_logL_marg` (frequentist, alias-prone; a localization diagnostic).

    Parameters
    ----------
    h0, b_stack, M_stack, A_stack
        As in :func:`total_logL_marg`.

    Returns
    -------
    scalar
        Total profile log-likelihood, summed over pulsars.
    """
    per = jax.vmap(logL_pulsar_profile, in_axes=(None, 0, 0, 0))(
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
    # Uniform prior on h0 >= 0 -> normalized-grid credible quantile.
    return grid_credible_upper_limit(h0, logpost, level)
