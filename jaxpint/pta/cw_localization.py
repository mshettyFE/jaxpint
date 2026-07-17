"""Fisher-matrix sky-localization for a continuous-GW source.

Companion to :mod:`jaxpint.bayes.cw_upper_limit` (which gives analytic upper
limits on strain). Here we go the other way: assume a *known* signal of
amplitude ``h0`` and compute the per-pixel 2-D Fisher information for the GW
sky position ``(cos_gwtheta, gwphi)``.
The intended use is reproducing arXiv:2603.28897 (Wen et al. 2026) style
anchor-pulsar scaling plots — for each anchor configuration, build a CWInjector
with the matching ``pulsar_term_mask``, set ``h0`` per pixel so the optimal SNR
matches a target (typically 20), evaluate the sky Fisher at the truth point,
and report the 90% credible area as a function of sky direction.

The Fisher-matrix approximation is accurate at high SNR with well-localized
posteriors; it underestimates area in regimes where the pulsar-term-phase
likelihood is genuinely multi-modal (mostly the "no anchors" limit). For
order-of-magnitude scaling studies that's fine; for absolute numbers in the
no-anchor regime, prefer sampling-based methods.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.stats.regions import gaussian_credible_area
from jaxpint.types import GlobalParams

__all__ = [
    "h0_for_snr",
    "make_logL_2sky",
    "gram_at_pixel",
    "gram_block_at_pair",
    "assemble_joint_fisher",
    "per_source_credible_areas_deg2",
    "credible_area_deg2",
]


# Solid-angle conversion: (180/π)² deg² per steradian.
_STR_TO_DEG2 = (180.0 / jnp.pi) ** 2


def h0_for_snr(
    snr_target: float,
    Y: Float[Array, ""],
) -> Float[Array, ""]:
    """Calibrate ``h0`` so the optimal matched-filter SNR² equals ``snr_target²``.

    ``Y = (s_hat | s_hat)``, so need sqrt(Y) for scaling
    """
    Y = jnp.maximum(Y, jnp.finfo(jnp.float64).tiny)
    return snr_target / jnp.sqrt(Y)


def make_logL_2sky(
    g: Callable[[GlobalParams, tuple], Float[Array, ""]],
    gp: GlobalParams,
    reduced_pp: tuple,
    prefix_a: str,
    prefix_b: str,
) -> Callable[
    [Float[Array, ""], Float[Array, ""], Float[Array, " 2"], Float[Array, " 2"]],
    Float[Array, ""],
]:
    """Build the ``(h_a, h_b, sky_a, sky_b)`` log-likelihood for the Gram helpers.

    Parameters
    ----------
    g : callable
        ``(global_params, reduced_pulsar_params) -> scalar`` timing-marginalized
        PTA log-likelihood — the first return of
        :func:`jaxpint.bayes.marginalize_pta`.
    gp : GlobalParams
        Base global parameters with all fixed CW parameters already set.
    reduced_pp : tuple of ParameterVector
        The reduced per-pulsar skeletons returned alongside ``g``; passed through
        unchanged on every call.
    prefix_a, prefix_b : str
        Global-name prefixes of the two CW injectors to vary (e.g. ``"cwt"`` and
        ``"cwd"``), i.e. their parameters are ``{prefix}_h0``,
        ``{prefix}_cos_gwtheta``, ``{prefix}_gwphi``.

    Returns
    -------
    logL_2sky : callable
        ``(h_a, h_b, sky_a, sky_b) -> scalar``, ready for
        :func:`gram_block_at_pair` / :func:`gram_at_pixel`.
    """

    def logL_2sky(h_a, h_b, sky_a, sky_b):
        gp_new = (
            gp.with_value(f"{prefix_a}_h0", h_a)
            .with_value(f"{prefix_a}_cos_gwtheta", sky_a[0])
            .with_value(f"{prefix_a}_gwphi", sky_a[1])
            .with_value(f"{prefix_b}_h0", h_b)
            .with_value(f"{prefix_b}_cos_gwtheta", sky_b[0])
            .with_value(f"{prefix_b}_gwphi", sky_b[1])
        )
        return g(gp_new, reduced_pp)

    return logL_2sky


def gram_at_pixel(
    logL_2sky: Callable[
        [Float[Array, ""], Float[Array, ""], Float[Array, " 2"], Float[Array, " 2"]],
        Float[Array, ""],
    ],
    sky_pixel: Float[Array, " 2"],
) -> Float[Array, "2 2"]:
    r"""The 2x2 sky Gram matrix at ``sky_pixel`` — the same-source (diagonal) case.

    Parameters
    ----------
    logL_2sky : callable
        ``(h_a, h_b, sky_a, sky_b) -> scalar`` log-likelihood with both injectors
        active.  Non-amplitude, non-sky CW parameters (frequency, orientation)
        should be closed over in the caller, identical for the two injectors.
    sky_pixel : (2,) array
        Sky position ``(cos_gwtheta, gwphi)`` at which to evaluate.

    Returns
    -------
    Gram : (2, 2) array
        The 2x2 Gram matrix at ``sky_pixel``.  Multiply by ``h0_target**2`` to
        get the sky Fisher information.
    """
    return gram_block_at_pair(logL_2sky, sky_pixel, sky_pixel)


def gram_block_at_pair(
    logL_2sky: Callable[
        [Float[Array, ""], Float[Array, ""], Float[Array, " 2"], Float[Array, " 2"]],
        Float[Array, ""],
    ],
    sky_a_truth: Float[Array, " 2"],
    sky_b_truth: Float[Array, " 2"],
) -> Float[Array, "2 2"]:
    r"""Cross-template sky Gram block (per ``h_a h_b``) at two sky positions.

    The noise-weighted Gram of signal sky-gradients,

    .. math::
        \mathrm{Gram}_{ij} = \frac{\partial^2 Z}{\partial \theta_{a,i}\,
                                  \partial \theta_{b,j}},
        \qquad
        Z(\theta_a, \theta_b) = (\hat s(\theta_a) \,|\, \hat s(\theta_b))_N,

    the building block of the joint multi-source Fisher; the diagonal
    (``sky_a == sky_b``, both injectors on one source) is :func:`gram_at_pixel`.

    Construction: ``logL_2sky(h_a, h_b, sky_a, sky_b)`` is the timing-marginalized
    log-likelihood with *two* CW injectors active.  The Gaussian likelihood is
    *exactly bilinear* in ``(h_a, h_b)``,

    .. math::
        \log L = \text{const} + h_a X_a + h_b X_b - \tfrac{1}{2} h_a^2 Y_a
                 - \tfrac{1}{2} h_b^2 Y_b - h_a h_b\, Z(\theta_a, \theta_b),

    so the mixed amplitude derivative isolates the cross-template inner product

    .. math::
        Z(\theta_a, \theta_b)
            = -\frac{\partial^2 \log L}{\partial h_a\,\partial h_b}\bigg|_{h_a = h_b = 0},

    The two CWInjectors that ``logL_2sky`` closes over should represent the
    *two source templates whose cross-coupling this block measures*:

    * **Diagonal `G_aa`**: both injectors bound to source `a`'s parameters
      (template + data convention; identical to single-pixel Level 1).
    * **Off-diagonal `G_ab`, a != b**: injector A bound to source `a`'s
      params; injector B bound to source `b`'s.  Encodes how the data
      jointly constrains both source positions through the per-pulsar
      pulsar-term phases.

    Parameters
    ----------
    logL_2sky : callable
        ``(h_a, h_b, sky_a, sky_b) -> scalar`` log-likelihood with both
        injectors active and bound to the appropriate source parameters
        (orientation, frequency).  All non-pair injectors should be held at
        zero amplitude in the closure.
    sky_a_truth : (2,) array
        First source's true sky position ``(cos_gwtheta, gwphi)``.
    sky_b_truth : (2,) array
        Second source's true sky position.  Same as ``sky_a_truth`` for a
        diagonal block.

    Returns
    -------
    Gram_ab : (2, 2) array
        The cross Gram block.  Multiply by ``h_a_target * h_b_target`` to
        get the Fisher-information contribution to the joint Fisher matrix.
    """

    def Z(sky_a, sky_b):
        f = lambda h_a, h_b: logL_2sky(h_a, h_b, sky_a, sky_b)
        return -jax.grad(jax.grad(f, argnums=0), argnums=1)(
            jnp.float64(0.0), jnp.float64(0.0)
        )

    return jax.jacfwd(jax.jacrev(Z, argnums=0), argnums=1)(sky_a_truth, sky_b_truth)


def assemble_joint_fisher(
    gram_blocks: dict[tuple[int, int], Float[Array, "2 2"]],
    h0_targets: Float[Array, " K"],
    K: int,
) -> Float[Array, "2K 2K"]:
    r"""Stack ``K(K+1)/2`` unique Gram blocks into the symmetric joint Fisher.

    The joint Fisher matrix for K simultaneous CGW sources has structure

    .. math::
        F_{(a,i),(b,j)} = h_{0,a}\,h_{0,b}\,(\partial_i \hat s_a | \partial_j \hat s_b)_N,

    Parameters
    ----------
    gram_blocks : dict
        Mapping ``(a, b) -> (2, 2)`` Gram block, for all unique pairs with
        ``0 <= a <= b < K``.  Length must be ``K * (K + 1) / 2``.
    h0_targets : (K,) array
        Per-source calibrated amplitudes ``h_a^{target} = SNR_a / sqrt(Y_a)``.
    K : int
        Number of sources.  Must match ``len(h0_targets)`` and the unique
        block count in ``gram_blocks``.

    Returns
    -------
    F : (2K, 2K) array
        Symmetric joint Fisher information matrix.  ``F[2a:2a+2, 2b:2b+2]``
        is the ``(a, b)``-block contribution.

    Raises
    ------
    ValueError
        If ``gram_blocks`` is not exactly the ``K(K+1)/2`` upper-triangular
        pairs ``(a, b)`` with ``0 <= a <= b < K`` (a block missing, out of
        range, or otherwise not one-per-sub-matrix), or if
        ``len(h0_targets) != K``.
    """
    expected_keys = {(a, b) for a in range(K) for b in range(a, K)}
    keys = set(gram_blocks)
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        unexpected = sorted(keys - expected_keys)
        raise ValueError(
            "assemble_joint_fisher: gram_blocks must contain exactly the "
            f"{K * (K + 1) // 2} upper-triangular pairs (a, b) with "
            f"0 <= a <= b < {K} (one per sub-matrix). "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )
    if h0_targets.shape[0] != K:
        raise ValueError(
            f"assemble_joint_fisher: h0_targets has length "
            f"{h0_targets.shape[0]}, expected K = {K}."
        )

    F = jnp.zeros((2 * K, 2 * K), dtype=jnp.float64)
    for (a, b), G_ab in gram_blocks.items():
        scaled = h0_targets[a] * h0_targets[b] * G_ab
        F = F.at[2 * a : 2 * a + 2, 2 * b : 2 * b + 2].set(scaled)
        if a != b:
            F = F.at[2 * b : 2 * b + 2, 2 * a : 2 * a + 2].set(scaled.T)
    return F


def per_source_credible_areas_deg2(
    F: Float[Array, "2K 2K"],
    K: int,
    level: float = 0.9,
) -> Float[Array, " K"]:
    r"""Per-source marginal credible area from the joint Fisher matrix.

    Inverts the joint Fisher to get ``Sigma = F^{-1}``,
    slices the ``2 x 2`` marginal sky covariance per source
    ``Sigma_k = Sigma[2k:2k+2, 2k:2k+2]``, and computes its credible area from
    ``det Sigma_k`` via :func:`~jaxpint.bayes.gaussian_credible_area`.

    Returns ``inf`` for sources where the marginal covariance is degenerate
    (e.g., coincident sources, or any other case where the joint Fisher is
    near-rank-deficient).

    Parameters
    ----------
    F : (2K, 2K) array
        Joint Fisher information from :func:`assemble_joint_fisher`.
    K : int
        Number of sources (must match the Fisher matrix size).
    level : float
        Credible level (default 0.9 → 90% area).

    Returns
    -------
    areas : (K,) array
        Per-source marginal credible localization area in deg^2.
    """
    # F is positive semi definite
    Sigma = jax.scipy.linalg.cho_solve(
        jax.scipy.linalg.cho_factor(F), jnp.eye(2 * K, dtype=F.dtype)
    )
    out = []
    for k in range(K):
        Sigma_k = Sigma[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
        det_Sigma_k = jnp.linalg.det(Sigma_k)
        det_safe = jnp.asarray(jnp.where(det_Sigma_k > 0.0, det_Sigma_k, jnp.inf))
        out.append(gaussian_credible_area(det_safe, level) * _STR_TO_DEG2)
    return jnp.stack(out)


def credible_area_deg2(
    F: Float[Array, "2 2"],
    level: float = 0.9,
) -> Float[Array, ""]:
    r"""90% credible 2-D localization area in deg² from a sky Fisher matrix.

    For a 2-D Gaussian posterior ``N(0, Sigma)`` with ``Sigma = F^{-1}`` the
    ``level``-credible ellipse has area

    .. math::
        A = \pi \cdot \Delta\chi^2 \cdot \sqrt{\det \Sigma},

    Returns ``inf`` for a singular / negative-determinant Fisher (degenerate /
    unphysical posterior) rather than raising — convenient when vmapping over
    a sky grid where a handful of pixels can degenerate.
    """
    det_F = jnp.linalg.det(F)
    # det Sigma = 1 / det F; det F <= 0 → unphysical (non-positive-definite).
    det_sigma = jnp.asarray(jnp.where(det_F > 0.0, 1.0 / det_F, jnp.inf))
    return gaussian_credible_area(det_sigma, level) * _STR_TO_DEG2
