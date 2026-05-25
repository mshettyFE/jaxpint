"""Analytic Bayesian 95% upper limits on continuous-GW strain (no MCMC).

The CW timing residual is *exactly linear* in the strain amplitude ``h0`` at a
fixed sky position, frequency, and source orientation (inclination,
polarization, phase).  The Gaussian PTA log-likelihood is therefore *exactly
quadratic* in ``h0``::

    logL(h0) = logL(0) + h0 * X - 0.5 * h0**2 * Y

with ``X = (d | s_hat)`` the matched filter against the unit-strain waveform and
``Y = (s_hat | s_hat)`` its noise-weighted power.  Both follow from a single
gradient/Hessian of any twice-differentiable ``logL(h0)`` (see
:func:`quadratic_coeffs`).

With a uniform prior on ``h0 >= 0`` the marginal posterior is a Gaussian
truncated at zero, so the 95% upper limit is closed form
(:func:`h0_95_closed_form`).  Marginalizing a grid of source orientations turns
the posterior into a Gaussian mixture in ``h0``; :func:`h0_95_marginalized`
takes its 95th percentile.  :func:`h0_to_distance` inverts the strain-distance
relation to convert a strain UL into a luminosity-distance lower limit.

This is the no-MCMC building block for the CGW distance-reach sky map
(``examples/cgw_distance_skymap.py``), approximating the Bayesian limit of
Fig. 8 in the NANOGrav 15-yr individual-SMBHB paper (arXiv:2306.16222).
"""
from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp, ndtr, ndtri
from jaxtyping import Array, Float

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

    For ``logL(A) = logL(0) + A*X - 0.5*A**2*Y`` the coefficients are the Taylor
    expansion *about A=0*: ``X = dlogL/dA|_0`` (the matched filter ``(d|s_hat)``)
    and ``Y = -d^2 logL/dA^2``.  Since ``logL`` is exactly quadratic, the gradient
    is ``dlogL/dA = X - A*Y``, so:

    - ``Y`` (the curvature) is constant — independent of the expansion point.
    - ``X`` is the gradient *at the origin*; the gradient at any other point is
      the shifted value ``X - A*Y``, not the coefficient.

    Hence ``amp`` must be 0 (the default) for the returned ``X`` to equal the
    matched filter that the upper-limit helpers expect; evaluating elsewhere
    returns ``X - amp*Y``.

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
    its limit ``1``, which is why ``q`` below carries no ``h_max`` term. That is
    exact to machine precision *whenever the data constrain* ``h0`` — then
    ``sigma = 1/sqrt(Y)`` is tiny and any physically sensible ``h_max`` sits many
    sigma above the posterior, so ``Phi((h_max-mu)/sigma) = 1`` in float64. The
    bound would bias the result only if the data were nearly *uninformative*
    (``Y -> 0`` so ``sigma -> inf``): the posterior would stay flat out to
    ``h_max`` and the limit would collapse to ``~level * h_max`` — set by the
    arbitrary prior bound, not the data. To impose a finite cap, replace the
    implicit ``1`` with ``phi_hi = ndtr((h_max - mu) / sigma)`` and use
    ``q = phi_lo + level * (phi_hi - phi_lo)``.
    """
    mu = X / Y
    sigma = 1.0 / jnp.sqrt(Y)
    phi_lo = ndtr(-mu / sigma)                # Phi(-mu/sigma): mass below the h0=0 wall
    # q uses the prior's UPPER edge Phi((h_max-mu)/sigma) at its h_max->inf limit
    # of 1 (see Notes "Upper-bound gotcha"); exact while the data constrain h0.
    q = level + (1.0 - level) * phi_lo
    return mu + sigma * ndtri(q)


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
    # Solve that additive relation for the distance. It is monotonically
    # decreasing in h0 (smaller strain -> larger D_L), so feeding in a 95% upper
    # limit on h0 yields a 95% lower limit on D_L.
    log10_dist_mpc = log10_h0_at_1mpc - jnp.log10(h0)
    return 10.0 ** log10_dist_mpc
