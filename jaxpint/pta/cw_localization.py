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

from typing import TYPE_CHECKING, Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.constants import C_KM_PER_S, KPC_TO_KM
from jaxpint.pta.blocks import pulsar_woodbury_blocks, whitened_quadratic
from jaxpint.stats.regions import gaussian_credible_area
from jaxpint.types import GlobalParams, ParameterVector

if TYPE_CHECKING:
    from jaxpint.pta.likelihood import PTAConfig

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


# ---------------------------------------------------------------------------
# General bilinear extraction -- the signal-agnostic fallback
#
# These functions need nothing but a callable log-likelihood, so they work for
# ANY signal (chirping sources, models without a linear amplitude).  The price
# is a fourth-order autodiff tape whose compiled graph is orders of magnitude
# larger than the direct machinery below -- prefer the direct family whenever
# ``linear_amplitude=True`` holds.  (Continued after the direct section:
# ``gram_block_at_pair`` / ``assemble_joint_fisher`` for multi-source joints.)
# ---------------------------------------------------------------------------


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
        :func:`gram_block_at_pair`.
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


# ---------------------------------------------------------------------------
# Direct (linear-amplitude) machinery
#
# ``CWInjector(linear_amplitude=True)`` makes the residual exactly linear in
# ``h0``, so the delay at unit amplitude IS the unit-strain waveform.  Every
# quantity in this section derives from ONE objective function::
#
#     neg_logpost(theta, data) = 1/2 sum_p |d_p - m_p(theta)|^2_{C_p^-1}
#                                + 1/2 sum_j ((dist_j - d0_j)/sigma_j)^2
#
# with ``theta = (cos_gwtheta, gwphi, log10_h, *nuisance, *coherent dists)``
# built by :func:`make_cw_objective`.  At a zero-residual point the Hessian is
# exactly the (Gauss-Newton) Fisher ``J^T C^-1 J`` plus the prior block, so:
#
# * expected Fisher  = ``hessian`` at truth on NOISE-FREE data
#   (:func:`marginal_sky_fisher`, :func:`gram_at_pixel`);
# * signal power     = the same whitened quadratic on the waveform itself
#   (:func:`signal_power_direct`).
# ---------------------------------------------------------------------------


def _cw_unit_waveform(inj, p, toa_data_p, pulsar_params_p, global_params, sky):
    r"""Unit-strain waveform :math:`\hat s_p` at ``sky`` -- ``delay`` at ``h0 = 1``.

    Exact rather than approximate: ``CWInjector(linear_amplitude=True)`` makes
    the residual exactly linear in the amplitude, so the delay evaluated at unit
    amplitude *is* the unit-strain waveform.
    """
    gp_ = (
        global_params.with_value(f"{inj.prefix}{inj.amp_name}", jnp.float64(1.0))
        .with_value(f"{inj.prefix}cos_gwtheta", sky[0])
        .with_value(f"{inj.prefix}gwphi", sky[1])
    )
    return inj.delay(p, toa_data_p, pulsar_params_p, gp_)


class CWObjective(NamedTuple):
    """The single objective behind the direct-formulation Fisher family.

    Fields
    ------
    neg_logpost : callable
        ``(theta, data) -> scalar`` -- whitened residual quadratic plus the
        Gaussian distance-prior term.  ``data`` is a tuple of per-pulsar
        arrays; jit/grad/hessian-safe.
    model_delays : callable
        ``theta -> tuple`` of per-pulsar model delays ``h * s_hat``.
    theta_truth : callable
        ``(sky, log10_h) -> theta`` at the stored global-parameter values and
        par-file distances.
    theta_names : tuple of str
        ``("cos_gwtheta", "gwphi", "log10_h", *nuisance, "dist_<psr>"...)``.
    n_sky, i_h : int
        Layout constants: sky occupies ``theta[:n_sky]``; ``theta[i_h]`` is
        ``log10_h``.
    """

    neg_logpost: Callable
    model_delays: Callable
    theta_truth: Callable
    theta_names: tuple
    n_sky: int
    i_h: int


def make_cw_objective(
    config: PTAConfig,
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    *,
    global_nuisance: tuple[str, ...] = (),
    dist_sigma_kpc: Sequence[float] | None = None,
    injector_index: int = 0,
) -> CWObjective:
    r"""Build the shared CW objective (see the section banner above).

    ``theta`` layout: ``[cos_gwtheta, gwphi, log10_h, *global_nuisance,
    *coherent distances]``.  Distances appear only when ``dist_sigma_kpc`` is
    given, only for pulsars whose pulsar term is in the model, and carry
    Gaussian priors centered on the par-file values -- so the prior block of
    the Hessian is exactly the ``Lambda`` of the Schur-marginalized Fisher.
    """
    inj = config.signal_injectors[injector_index]
    names = tuple(global_nuisance)
    coherent = [
        p
        for p in range(config.n_pulsars)
        if (not inj.earth_term_only) and bool(inj.pulsar_term_mask[p])
    ]
    use_dists = dist_sigma_kpc is not None and len(coherent) > 0
    if not use_dists:
        coherent = []
    # NB: param_value goes through jnp indexing, so under jit these are
    # (constant-valued) tracers -- keep them as jnp scalars, never float().
    d0 = (
        jnp.stack(
            [1.0 / pulsar_params[p].param_value(inj.dist_param) for p in coherent]
        )
        if coherent
        else jnp.zeros(0, dtype=jnp.float64)
    )
    sig_d = (
        jnp.asarray([float(dist_sigma_kpc[p]) for p in coherent])
        if coherent
        else jnp.zeros(0, dtype=jnp.float64)
    )
    n_nui = len(names)
    theta_names = (
        ("cos_gwtheta", "gwphi", "log10_h")
        + names
        + tuple(f"dist_p{p}" for p in coherent)
    )

    blocks = [
        pulsar_woodbury_blocks(config, global_params, pulsar_params[p], p)
        for p in range(config.n_pulsars)
    ]

    def _model_one(p, theta):
        gp_ = global_params
        for k, nm in enumerate(names):
            gp_ = gp_.with_value(f"{inj.prefix}{nm}", theta[3 + k])
        pp_ = pulsar_params[p]
        if p in coherent:
            j = coherent.index(p)
            pp_ = pp_.with_value(inj.dist_param, 1.0 / theta[3 + n_nui + j])
        s = _cw_unit_waveform(inj, p, config.toa_data_list[p], pp_, gp_, theta[:2])
        return (10.0 ** theta[2]) * s

    def model_delays(theta):
        return tuple(_model_one(p, theta) for p in range(config.n_pulsars))

    def neg_logpost(theta, data):
        total = jnp.float64(0.0)
        for p in range(config.n_pulsars):
            total = total + whitened_quadratic(
                blocks[p], data[p] - _model_one(p, theta)
            )
        if use_dists:
            total = total + 0.5 * jnp.sum(((theta[3 + n_nui :] - d0) / sig_d) ** 2)
        return total

    def theta_truth(sky, log10_h):
        parts = [
            jnp.asarray(sky, dtype=jnp.float64).reshape(2),
            jnp.asarray(log10_h, dtype=jnp.float64).reshape(1),
        ]
        if names:
            parts.append(
                jnp.stack(
                    [
                        jnp.asarray(
                            global_params.param_value(f"{inj.prefix}{nm}"),
                            dtype=jnp.float64,
                        )
                        for nm in names
                    ]
                )
            )
        if coherent:
            parts.append(d0)
        return jnp.concatenate(parts)

    return CWObjective(neg_logpost, model_delays, theta_truth, theta_names, 2, 2)


def _schur_to_sky(H, *, drop_h: bool):
    r"""Reduce a full-``theta`` information matrix to the 2x2 sky block.

    ``drop_h=True`` CONDITIONS on the amplitude (deletes its row/column --
    the expected-Fisher convention, where ``h0`` is a fixed calibration);
    ``drop_h=False`` MARGINALIZES it along with the rest (the bootstrap /
    posterior convention).
    """
    H = jnp.asarray(H)
    if drop_h:
        keep = jnp.asarray([0, 1] + list(range(3, H.shape[0])))
        H = H[keep][:, keep]
    F_ss, F_sn, F_nn = H[:2, :2], H[:2, 2:], H[2:, 2:]
    if F_nn.size == 0:  # static shape -> plain python branch is fine
        return F_ss
    return F_ss - F_sn @ jnp.linalg.pinv(F_nn) @ F_sn.T


def signal_power_direct(
    config: PTAConfig,
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    sky_pixel: Float[Array, " 2"],
    *,
    injector_index: int = 0,
) -> Float[Array, ""]:
    r"""Unit-strain signal power :math:`Y = (\hat s \mid \hat s)`, computed directly.

    .. math::
        Y \;=\; \sum_p \big(\hat s_p \,\big|\, \hat s_p\big)_{C_p^{-1}}

    Pair with :func:`h0_for_snr` to calibrate ``h0`` per pixel.
    """
    inj = config.signal_injectors[injector_index]
    total = jnp.float64(0.0)
    for p in range(config.n_pulsars):
        s = _cw_unit_waveform(
            inj,
            p,
            config.toa_data_list[p],
            pulsar_params[p],
            global_params,
            sky_pixel,
        )
        block = pulsar_woodbury_blocks(config, global_params, pulsar_params[p], p)
        total = total + 2.0 * whitened_quadratic(block, s)
    return total


def marginal_sky_fisher(
    config: PTAConfig,
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    sky_pixel: Float[Array, " 2"],
    *,
    h0: Float[Array, ""],
    global_nuisance: tuple[str, ...] = (),
    dist_sigma_kpc: Sequence[float] | None = None,
    injector_index: int = 0,
) -> Float[Array, "2 2"]:
    r"""Expected sky Fisher with a nuisance block marginalized out.

    The Hessian of :func:`make_cw_objective`'s objective at truth on
    noise-free data -- exactly ``J^T C^-1 J`` plus the distance-prior
    ``Lambda``, since the residual vanishes there -- Schur-complemented to the
    sky.  ``h0`` is a fixed calibration (conditioned, not marginalized),
    matching the convention of the sky-map pipelines.

    With ``global_nuisance=()`` and ``dist_sigma_kpc=None`` this reduces to
    ``h0**2 *`` the sky Gram (the plain conditional Fisher).

    .. warning::
        The Gaussian treatment of coherent distances is valid for
        :math:`\sigma_\varphi \lesssim 1` rad (:func:`phase_sigma_rad`).  In
        the comb regime the result is a central-lobe LOWER BOUND -- measured
        at ~100x under honest sampling for Wen-style anchors.
    """
    obj = make_cw_objective(
        config,
        global_params,
        pulsar_params,
        global_nuisance=global_nuisance,
        dist_sigma_kpc=dist_sigma_kpc,
        injector_index=injector_index,
    )
    th0 = obj.theta_truth(sky_pixel, jnp.log10(jnp.asarray(h0)))
    data = obj.model_delays(th0)
    H = jax.hessian(obj.neg_logpost)(th0, data)
    return _schur_to_sky(H, drop_h=True)


def gram_at_pixel(
    config: PTAConfig,
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    sky_pixel: Float[Array, " 2"],
    *,
    injector_index: int = 0,
) -> Float[Array, "2 2"]:
    r"""The 2x2 sky Gram matrix :math:`(\partial_i \hat s \mid \partial_j \hat s)`.

    Equal to :func:`marginal_sky_fisher` at ``h0 = 1`` with an empty nuisance
    block.  Multiply by ``h0**2`` for the conditional sky Fisher.

    Requires ``CWInjector(linear_amplitude=True)``.  For a signal without a
    linear amplitude (or a chirping template), the signal-agnostic bilinear
    route gives the same matrix (validated to ~4e-12) as the diagonal case of
    :func:`gram_block_at_pair`: ``gram_block_at_pair(logL_2sky, sky, sky)``.
    """
    return marginal_sky_fisher(
        config,
        global_params,
        pulsar_params,
        sky_pixel,
        h0=jnp.float64(1.0),
        injector_index=injector_index,
    )


def phase_sigma_rad(
    dist_sigma_kpc: Float[Array, ""],
    log10_fgw: float,
    cos_mu: Float[Array, ""],
) -> Float[Array, ""]:
    r"""Pulsar-term phase uncertainty :math:`\sigma_\varphi` in radians.

    .. math::
        \sigma_\varphi = 2\pi\,f_{\rm gw}\,\frac{\sigma_d}{c}\,(1+\cos\mu)
                       = 2\pi\,\frac{\sigma_d}{\lambda_{\rm GW}}\,(1+\cos\mu)
    """
    lambda_gw_kpc = (C_KM_PER_S / 10.0**log10_fgw) / KPC_TO_KM
    return 2.0 * jnp.pi * (dist_sigma_kpc / lambda_gw_kpc) * (1.0 + cos_mu)


# ---------------------------------------------------------------------------
# Bilinear family, continued: multi-source joint Fisher
# ---------------------------------------------------------------------------


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
    (``sky_a == sky_b``, both injectors on one source) is the bilinear route
    to the single-pixel Gram that :func:`gram_at_pixel` computes directly.

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
    ``det Sigma_k`` via :func:`~jaxpint.stats.gaussian_credible_area`.

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
