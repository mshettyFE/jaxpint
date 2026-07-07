"""Shared CW block extraction: quadratic (b, M) reductions of PTA likelihoods.

The one mathematical fact this module exploits: a CW timing residual is
**exactly linear** in its template amplitudes at fixed sky/frequency, so any
Gaussian PTA log-likelihood is **exactly quadratic** in them::

    logL(A) = logL(0) + b·A - 0.5 * Aᵀ M A

Everything here extracts the small ``(b, M)`` pair — the matched filter and
noise-weighted, timing-marginalized Gram — from a heavy likelihood, so that
downstream work (upper limits, detection statistics, localization scans,
sensitivity forecasts) becomes cheap algebra on the blocks.  Both inference
arms consume this module (the Bayesian upper limits in
``jaxpint.pta.cw_upper_limit`` / ``jaxpint.pta.incoherent_ul`` and the
frequentist statistics in ``jaxpint.frequentist``), which is why it lives in
``pta/`` below both.


A planned consolidation: ``jaxpint.pta.likelihood._per_pulsar_intermediates``
computes the same mathematical objects (``FᵀC⁻¹r``, ``FᵀC⁻¹F``) through the
``PTAConfig`` interface for the correlated two-tier solve; unifying it with
:func:`extract_pulsar_blocks` here would give the optimal statistic and the
HD likelihood one canonical block producer.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.utils import quadratic_form_coeffs

__all__ = [
    "EXTRACTION_ORIENTATIONS",
    "bM2_coeffs",
    "basis_quadratics",
    "default_extraction_orientations",
    "extract_pulsar_bM",
    "extract_pulsar_blocks",
    "orientation_coeffs",
    "quadratic_coeffs",
]


def quadratic_coeffs(
    logL_fn: Callable[[Float[Array, ""]], Float[Array, ""]],
    amp: float = 0.0,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Recover ``(X, Y)`` from a log-likelihood that is quadratic in amplitude.

    Assumes LogL takes form of  ``logL(A) = logL(0) + A*X - 0.5*A**2*Y``

    Parameters
    ----------
    logL_fn : callable
        Maps a scalar linear amplitude ``A`` to a scalar log-likelihood.  The
        caller closes over the (timing-marginalized) PTA likelihood and the
        unit-strain waveform.
    amp : float
        Expansion point, default 0.  ``Y`` is the same for any value, but the
        returned ``X = grad(amp) = X_true - amp*Y`` only equals the coefficient
        ``(d|s_hat)`` at ``amp = 0``.

    Returns
    -------
    X, Y : scalars
        Matched filter ``X=(d|s_hat)`` and signal power ``Y=(s_hat|s_hat)``.

    Notes
    -----
    The scalar (``n=1``) specialization of
    :func:`~jaxpint.utils.quadratic_form_coeffs`: it extracts the origin
    coefficients ``(X0, Y)`` with two first-order gradients (no second-order
    autodiff graph — much lighter on memory when vmapped over a grid), then
    shifts to the requested expansion point via ``X = X0 - amp*Y`` (``Y`` is the
    amp-invariant curvature).
    """
    b, M = quadratic_form_coeffs(lambda A: logL_fn(A[0]), 1)
    X0, Y = b[0], M[0, 0]
    return X0 - jnp.asarray(amp, dtype=jnp.float64) * Y, Y


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


# ---------------------------------------------------------------------------
# The Earth-term orientation basis (F_e-statistic basis reduction)
# ---------------------------------------------------------------------------
# At a fixed sky position the Earth-term CW residual is a linear combination of
# four orientation-independent basis waveforms {f_p*sin, f_p*cos, f_c*sin,
# f_c*cos} (see cw_delay_from_array, cw.py). For unit strain the residual is
#
#     s(omega) = sum_a c_a(omega) * basis_a ,
#
# with c(omega) an analytic function of the orientation (cos_inc, psi, phase0)
# and basis_a carrying all the data/noise/sky/time dependence. Hence the matched
# filter and signal power at ANY orientation are quadratic forms in c:
#
#     X(omega) = (d | s) = c(omega) . b ,        b_a   = (d | basis_a)
#     Y(omega) = (s | s) = c(omega) . M c(omega), M_ab = (basis_a | basis_b)
#
# So the heavy PTA likelihood is touched only to build the per-pixel 4-vector b
# and 4x4 Gram matrix M once (:func:`basis_quadratics`); every orientation in the
# marginalization grid is then two cheap contractions. The same c(omega) (and
# hence the same machinery) applies with the pulsar term at *fixed* distance --
# only the basis waveforms change. Refs: Jaranowski-Krolak-Schutz 1998;
# Ellis, Siemens & Creighton 2012 (arXiv:1204.4218); Taylor, Ellis & Gair 2014
# (arXiv:1406.5224).


def orientation_coeffs(
    cos_inc: Float[Array, ""],
    psi: Float[Array, ""],
    phase0: Float[Array, ""],
) -> Float[Array, " 4"]:
    """The 4 orientation amplitudes ``c(omega)`` for the Earth-term CW basis.

    Derived from ``cw_delay_from_array`` (cw.py): expanding the residual over the
    quadratures ``sin(2 pi f0 t)``, ``cos(2 pi f0 t)`` and the two antenna
    patterns gives ``s = h0 * sum_a c_a * basis_a`` with, writing
    ``a_i = 1 + cos_inc**2`` and ``b_i = 2 cos_inc``,

        c1 =  a_i cos2psi cos phi0 - b_i sin2psi sin phi0
        c2 =  a_i cos2psi sin phi0 + b_i sin2psi cos phi0
        c3 = -a_i sin2psi cos phi0 - b_i cos2psi sin phi0
        c4 = -a_i sin2psi sin phi0 + b_i cos2psi cos phi0

    Any overall constant (the ``-1/(2 pi f0)`` prefactor, sign) is irrelevant: it
    is absorbed consistently into ``b`` and ``M`` by :func:`basis_quadratics`,
    since the same ``c`` is used for both extraction and evaluation.

    """
    a_i = 1.0 + cos_inc**2
    b_i = 2.0 * cos_inc
    cc2 = jnp.cos(2.0 * psi)
    ss2 = jnp.sin(2.0 * psi)
    cp = jnp.cos(phase0)
    sp = jnp.sin(phase0)
    return jnp.stack(
        [
            a_i * cc2 * cp - b_i * ss2 * sp,
            a_i * cc2 * sp + b_i * ss2 * cp,
            -a_i * ss2 * cp - b_i * cc2 * sp,
            -a_i * ss2 * sp + b_i * cc2 * cp,
        ]
    )


def default_extraction_orientations(n: int = 16, seed: int = 0) -> Float[Array, "n 3"]:
    """Fixed pseudo-random ``(cos_inc, psi, phase0)`` probes spanning the cube.

    Random points (vs a coarse product grid) reliably make the design matrices
    for the ``b`` (rank 4) and ``M`` (rank 10) solves well conditioned. Fixed
    seed -> reproducible.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    cos_inc = jax.random.uniform(k1, (n,), minval=-1.0, maxval=1.0)
    psi = jax.random.uniform(k2, (n,), minval=0.0, maxval=jnp.pi)
    phase0 = jax.random.uniform(k3, (n,), minval=0.0, maxval=2.0 * jnp.pi)
    return jnp.stack([cos_inc, psi, phase0], axis=1)


EXTRACTION_ORIENTATIONS = default_extraction_orientations()


def basis_quadratics(
    logL_at_orientation: Callable[..., Float[Array, ""]],
    orientations: Float[Array, "k 3"] = EXTRACTION_ORIENTATIONS,
) -> tuple[Float[Array, "4 4"], Float[Array, " 4"]]:
    """Extract some fiducial per-pixel Gram matrix ``M`` (4x4) and data vector ``b`` (4).

    Evaluates the (heavy) likelihood only at the ``k`` probe ``orientations``,
    reusing :func:`quadratic_coeffs`, then solves the small linear systems
    ``X_k = c_k . b`` and ``Y_k = c_k . M c_k`` for ``b`` and the symmetric
    ``M``.

    Parameters
    ----------
    logL_at_orientation : callable
        ``(amp, cos_inc, psi, phase0) -> logL`` for a *fixed* sky location; ``amp`` is
        the linear strain (so logL is quadratic in it). The caller closes over
        the sky position and the timing-marginalized PTA likelihood.
    orientations : (k, 3) array
        Probe ``(cos_inc, psi, phase0)`` points; ``k >= 10`` (from symmetry of M)
        and well spread (re: design matrix c_k should be full rank and small condition number;
                         random draw of orientations seems to work fine).

    Returns
    -------
    M, b : (4, 4), (4,)
        The basis Gram matrix ``M_ab = (basis_a | basis_b)`` and the data
        projection ``b_a = (d | basis_a)``, where ``basis_a`` are the 4
        orientation-independent CW waveforms and ``(.|.)`` is the timing-
        marginalized GLS inner product (see the section-header comment). They
        satisfy ``X(omega) = c(omega) . b`` and ``Y(omega) = c(omega) . M c(omega)``
        for any orientation.
    """

    def xy_one(orient):
        f = lambda amp: logL_at_orientation(amp, orient[0], orient[1], orient[2])
        return quadratic_coeffs(f)

    Xs, Ys = jax.lax.map(xy_one, orientations)  # (k,), (k,)
    C = jax.vmap(lambda o: orientation_coeffs(o[0], o[1], o[2]))(orientations)  # (k,4)

    # b: X(omega) = c(omega) . b is linear in the 4-vector b -> stacked solve.
    b = jnp.linalg.pinv(C) @ Xs

    # We want to solve for the matrix M which satisfies: Y = c M c^T. So we have 10 variables we need to solve for
    # This part pulls out the 10 independent elements of that equation (re: upper triangular)
    # We are double counting the off diagonal elements since they repeat  in the lower half
    iu, ju = jnp.triu_indices(4)
    design = C[:, iu] * C[:, ju] * jnp.where(iu == ju, 1.0, 2.0)
    m = jnp.linalg.pinv(design) @ Ys
    # After solving the independent components, we shove them back into the full matrix (first upper, then lower half)
    M = jnp.zeros((4, 4), dtype=m.dtype).at[iu, ju].set(m).at[ju, iu].set(m)
    return M, b
