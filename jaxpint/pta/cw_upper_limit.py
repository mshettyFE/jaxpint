"""Analytic Bayesian 95% upper limits on continuous-GW strain (no MCMC).

The CW timing residual is *exactly linear* in the strain amplitude ``h0`` at a
fixed sky position, frequency, and source orientation (inclination,
polarization, phase).  The Gaussian PTA log-likelihood is therefore *exactly
quadratic* in ``h0``::

    logL(h0) = logL(0) + h0 * X - 0.5 * h0**2 * Y

with ``X = (d | s_hat)`` the matched filter against the unit-strain waveform and
``Y = (s_hat | s_hat)`` its noise-weighted power.

With the conventional improper-uniform prior on ``h0 >= 0`` the marginal
posterior is a Gaussian truncated at zero, so the 95% upper limit is closed form
(:func:`h0_95_closed_form`).  Marginalizing a grid of source orientations turns
the posterior into a Gaussian mixture in ``h0``; :func:`h0_95_marginalized`
takes its 95th percentile.

Approximating the Bayesian limit of Fig. 8 in the NANOGrav 15-yr individual-SMBHB paper (arXiv:2306.16222).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

from jaxpint.stats.regions import (
    grid_credible_upper_limit,
    mixture_truncated_gaussian_upper_limit,
    truncated_gaussian_upper_limit,
)
from jaxpint.pta.signals.cw import log10_strain_from_binary

if TYPE_CHECKING:
    # Type-only: the upper-limit code stays numpyro-free at runtime and
    # duck-types the prior (uses only ``.log_prob`` / ``.support``), so numpyro
    # is needed *only* when a caller actually passes a distribution.
    from numpyro.distributions import Distribution

# The Earth-term, linear-amplitude CW template used here is provided by
# ``CWInjector(linear_amplitude=True, earth_term_only=True)`` in
# jaxpint/pta/signals/cw.py — its amplitude parameter "h0" enters the residual
# linearly, so pta_logL is exactly quadratic in it.  The (b, M) / (X, Y) block
# extraction lives in jaxpint.pta.extraction; this module turns extracted
# blocks into Bayesian upper limits.


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
        From :func:`jaxpint.pta.extraction.quadratic_coeffs` (or the
        contractions of :func:`jaxpint.pta.extraction.basis_quadratics`
        blocks).  ``Y`` must be positive.
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


def h0_95_marginalized(
    Xs: Float[Array, " n"],
    Ys: Float[Array, " n"],
    level: float = 0.95,
    n_iter: int = 60,
    *,
    prior: Optional["Distribution"] = None,
    n_grid: int = 4096,
) -> Float[Array, ""]:
    r"""95% UL on ``h0`` marginalized over an orientation grid.

    Each orientation ``k`` contributes a truncated-Gaussian likelihood in ``h0``
    with ``mu_k = X_k/Y_k``, ``sigma_k = 1/sqrt(Y_k)``. Marginalized over the
    (equally weighted) orientation grid, the ``h0`` posterior is the mixture
    ``p(h0) ∝ prior(h0) * sum_k exp(h0 X_k - 0.5 Y_k h0^2)`` on ``h0 >= 0``.

    Two paths, selected by ``prior``:

    - ``prior=None`` (default) — the conventional **improper-uniform** prior on
      ``h0 >= 0``.  The mixture is a weighted sum of truncated Gaussians whose
      CDF is analytic; the ``level`` quantile is found by bisection in
      closed form (fast; the skymap default).  Reduces exactly to
      :func:`h0_95_closed_form` for a single orientation.
    - any distribution — its ``log_prob`` is folded into the quadrature weights
      over a uniform ``h0`` grid (a **deterministic** grid quantile, never
      sampling).  Use e.g. ``numpyro.distributions.LogUniform(lo, hi)`` for the
      log-uniform amplitude convention, or a finite ``Uniform(0, hi)`` box.

    Parameters
    ----------
    Xs, Ys : (n,) arrays
        Per-orientation matched filter and signal power (e.g. ``c_k.b`` and
        ``c_k.M c_k``). ``Ys`` must be positive.
    level : float
        Credible level (default 0.95).
    n_iter : int
        Bisection iterations for the closed-form path (default 60).
    prior : numpyro Distribution, optional
        Prior on ``h0``.  ``None`` → improper-uniform closed form.  The upper
        limit is never computed by sampling
    n_grid : int
        Number of ``h0`` grid points for the non-uniform-prior path.
    """
    #    The constant ``logL(0)`` cancels in the normalized CDF (never exponentiated).
    #    Component weights ``exp(0.5 X_k^2/Y_k)`` are stabilized by subtracting the
    #    max in log-space (cancels in the CDF ratio);

    Ys = jnp.maximum(Ys, jnp.finfo(jnp.float64).tiny)
    mu = Xs / Ys
    sigma = 1.0 / jnp.sqrt(Ys)
    if prior is None:
        # Per-orientation mixture weight, up to a common additive constant.
        log_weights = 0.5 * Xs * Xs / Ys + jnp.log(sigma)
        return mixture_truncated_gaussian_upper_limit(
            mu, sigma, log_weights, level, n_iter
        )
    return _marginalized_ul_grid(Xs, Ys, mu, sigma, prior, level, n_grid)


def _marginalized_ul_grid(
    Xs: Float[Array, " n"],
    Ys: Float[Array, " n"],
    mu: Float[Array, " n"],
    sigma: Float[Array, " n"],
    prior: "Distribution",
    level: float,
    n_grid: int,
) -> Float[Array, ""]:
    """Grid upper limit for a non-uniform ``h0`` prior (deterministic).

    Evaluates the orientation-marginalized log-likelihood
    ``logL(h0) = logsumexp_k(h0 X_k - 0.5 Y_k h0^2)`` on a uniform grid over
    ``[0, max_k mu_k + 12 max_k sigma_k]``, folds in ``prior.log_prob``, and
    inverts the CDF via :func:`grid_credible_upper_limit`.  No sampling.
    """
    h0_hi = jnp.max(mu) + 12.0 * jnp.max(sigma)
    h0 = jnp.linspace(0.0, h0_hi, n_grid)
    logL = logsumexp(
        h0[:, None] * Xs[None, :] - 0.5 * Ys[None, :] * h0[:, None] ** 2, axis=1
    )
    in_support = jnp.asarray(prior.support(h0), dtype=bool)
    log_prior = jnp.where(in_support, prior.log_prob(h0), -jnp.inf)
    return grid_credible_upper_limit(h0, logL + log_prior, level)
