"""Analytic Bayesian 95% upper limits on continuous-GW strain (no MCMC).

The CW timing residual is *exactly linear* in the strain amplitude ``h0`` at a
fixed sky position, frequency, and source orientation (inclination,
polarization, phase).  The Gaussian PTA log-likelihood is therefore *exactly
quadratic* in ``h0``::

    logL(h0) = logL(0) + h0 * X - 0.5 * h0**2 * Y

with ``X = (d | s_hat)`` the matched filter against the unit-strain waveform and
``Y = (s_hat | s_hat)`` its noise-weighted power.

With a uniform prior on ``h0 >= 0`` the marginal posterior is a Gaussian
truncated at zero, so the 95% upper limit is closed form
(:func:`h0_95_closed_form`).  Marginalizing a grid of source orientations turns
the posterior into a Gaussian mixture in ``h0``; :func:`h0_95_marginalized`
takes its 95th percentile.  :func:`h0_to_distance` inverts the strain-distance
relation to convert a strain UL into a luminosity-distance lower limit.

Approximating the Bayesian limit of Fig. 8 in the NANOGrav 15-yr individual-SMBHB paper (arXiv:2306.16222).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.bayes.credible import (
    mixture_truncated_gaussian_upper_limit,
    truncated_gaussian_upper_limit,
)
from jaxpint.pta.signals.cw import log10_strain_from_binary

# The Earth-term, linear-amplitude CW template used here is provided by
# ``CWInjector(linear_amplitude=True, earth_term_only=True)`` in
# jaxpint/pta/signals/cw.py — its amplitude parameter "h0" enters the residual
# linearly, so pta_logL is exactly quadratic in it (see quadratic_coeffs).


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
    Uses two *first-order* gradients rather than a second derivative: since
    ``dlogL/dA = X - A*Y``, we have ``X = grad(amp)`` and
    ``Y = grad(amp) - grad(amp+1)``.  This avoids building a second-order
    autodiff graph through the full PTA likelihood — much lighter on memory,
    which matters when this is vmapped over a sky/orientation grid.
    """
    a = jnp.asarray(amp, dtype=jnp.float64)
    grad_fn = jax.grad(logL_fn)
    g_a = grad_fn(a)
    g_a1 = grad_fn(a + 1.0)
    X = g_a
    Y = g_a - g_a1
    return X, Y


def h0_95_closed_form(
    X: Float[Array, ""],
    Y: Float[Array, ""],
    level: float = 0.95,
) -> Float[Array, ""]:
    r"""Closed-form upper limit on ``h0`` for a single source orientation.

    Uniform prior on ``h0 >= 0`` and quadratic log-likelihood give a posterior
    that is ``N(mu, sigma^2)`` truncated to ``h0 >= 0``, with ``mu = X/Y`` and
    ``sigma = 1/sqrt(Y)``.  The ``level`` upper limit is

    .. math::
        h_0^{UL} = \mu + \sigma\,\Phi^{-1}\!\big(p + (1-p)\,\Phi(-\mu/\sigma)\big),

    where ``p = level``.  For a clean non-detection (``X=0`` so ``mu=0``) this is
    ``h0_UL = sigma * Phi^{-1}(0.975) ≈ 1.96 sigma`` at the default level.

    Parameters
    ----------
    X, Y : scalars
        From :func:`quadratic_coeffs`.  ``Y`` must be positive.
    level : float
        Credible level (default 0.95).

    Notes
    -----
    **Upper-bound gotcha.** A proper uniform prior is on ``[0, h_max]``, and the
    truncated-normal CDF carries an upper-edge term in its denominator::

        F(h) = [Phi((h-mu)/s) - Phi(-mu/s)] / [Phi((h_max-mu)/s) - Phi(-mu/s)].

    This function takes ``h_max -> inf``: it replaces ``Phi((h_max-mu)/s)`` with
    its limit ``1``, which is why ``q`` below carries no ``h_max`` term. The
    bound would bias the result only if the data were nearly *uninformative*
    (``Y -> 0`` so ``sigma -> inf``): the posterior would stay flat out to
    ``h_max`` and the limit would collapse to ``~level * h_max`` — set by the
    arbitrary prior bound, not the data. To impose a finite cap, replace the
    implicit ``1`` with ``phi_hi = ndtr((h_max - mu) / sigma)`` and use
    ``q = phi_lo + level * (phi_hi - phi_lo)``.
    """
    # Floor Y at the smallest positive normal float64 so a degenerate pixel
    # (Y -> 0) yields a huge-but-finite sigma -> a very weak limit, rather than
    # a NaN/inf from 1/sqrt(0). Matches h0_95_marginalized.
    Y = jnp.maximum(Y, jnp.finfo(jnp.float64).tiny)
    mu = X / Y
    sigma = 1.0 / jnp.sqrt(Y)
    return truncated_gaussian_upper_limit(mu, sigma, level)


def h0_to_distance(
    h0: Float[Array, ""],
    log10_mc: float,
    log10_fgw: float,
) -> Float[Array, ""]:
    """Convert a strain ``h0`` to luminosity distance (Mpc) for fixed chirp mass.

    Inverts :func:`jaxpint.pta.signals.cw.log10_strain_from_binary`.  Since
    ``h0 ∝ 1/D_L`` at fixed chirp mass and frequency,
    ``log10 D_L = log10_strain_from_binary(log10_mc, 0, log10_fgw) - log10(h0)``
    (the first term being ``log10 h0`` at ``D_L = 1 Mpc``).

    A 95% *upper limit* on ``h0`` maps to a 95% *lower limit* on ``D_L``.

    Parameters
    ----------
    h0 : scalar
        Strain amplitude.
    log10_mc : float
        log10 chirp mass in solar masses (e.g. 9.0 for 1e9 Msun).
    log10_fgw : float
        log10 GW frequency in Hz.
    """
    # h0 = K(M_c, f) / D_L exactly (at fixed chirp mass and frequency), so in
    # log space the distance dependence is additive:
    #     log10(h0) = log10(K) - log10(D_L_Mpc).
    # Evaluate the strain relation at D_L = 1 Mpc (log10_dist = 0) to get
    # log10(K) once, reusing log10_strain_from_binary as the single definition
    # of the constant K.
    log10_h0_at_1mpc = log10_strain_from_binary(log10_mc, 0.0, log10_fgw)
    # Solve that additive relation for the distance.
    log10_dist_mpc = log10_h0_at_1mpc - jnp.log10(h0)
    return 10.0**log10_dist_mpc


# ---------------------------------------------------------------------------
# Orientation marginalization (no MCMC) via the F_e-statistic basis reduction
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


def _default_extraction_orientations(n: int = 16, seed: int = 0) -> Float[Array, "n 3"]:
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


_EXTRACTION_ORIENTATIONS = _default_extraction_orientations()


def basis_quadratics(
    logL_at_orientation: Callable[..., Float[Array, ""]],
    orientations: Float[Array, "k 3"] = _EXTRACTION_ORIENTATIONS,
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


def h0_95_marginalized(
    Xs: Float[Array, " n"],
    Ys: Float[Array, " n"],
    level: float = 0.95,
    n_iter: int = 60,
) -> Float[Array, ""]:
    r"""95% UL on ``h0`` marginalized over an orientation grid (uniform prior).

    Each orientation ``k`` contributes a truncated-Gaussian likelihood in ``h0``
    with ``mu_k = X_k/Y_k``, ``sigma_k = 1/sqrt(Y_k)``. With a uniform prior over
    the (equally weighted) orientation grid and on ``h0 >= 0``, the marginal
    posterior is the mixture ``p(h0) ∝ sum_k exp(h0 X_k - 0.5 Y_k h0^2)``. Its
    CDF is an analytic weighted sum of normal CDFs; this returns the ``level``
    quantile by bisection.

    Reduces exactly to :func:`h0_95_closed_form` for a single orientation.

    Parameters
    ----------
    Xs, Ys : (n,) arrays
        Per-orientation matched filter and signal power (e.g. ``c_k.b`` and
        ``c_k.M c_k``). ``Ys`` must be positive.
    level : float
        Credible level (default 0.95).
    n_iter : int
        Bisection iterations (default 60 -> float64-tight).
    """
#    The constant ``logL(0)`` cancels in the normalized CDF (never exponentiated).
#    Component weights ``exp(0.5 X_k^2/Y_k)`` are stabilized by subtracting the
#    max in log-space (cancels in the CDF ratio);

    Ys = jnp.maximum(Ys, jnp.finfo(jnp.float64).tiny)
    mu = Xs / Ys
    sigma = 1.0 / jnp.sqrt(Ys)
    # Per-orientation mixture weight, up to a common additive constant.
    log_weights = 0.5 * Xs * Xs / Ys + jnp.log(sigma)
    return mixture_truncated_gaussian_upper_limit(
        mu, sigma, log_weights, level, n_iter
    )


def fstat(M: Float[Array, "4 4"], b: Float[Array, " 4"]) -> Float[Array, ""]:
    """Coherent Earth-term detection statistic ``2F = b^T M^{-1} b``.

    The likelihood maximized over the 4 basis amplitudes (``A_hat = M^{-1} b``).
    Under the null ``2F ~ chi^2_4``; a signal adds the SNR^2 (noncentral). Only
    meaningful in real mode (``b`` from actual residuals with a matching noise
    model); in expected mode ``b = 0`` so ``2F = 0`` by construction.
    """
    return b @ jnp.linalg.solve(M, b)
