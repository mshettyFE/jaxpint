"""Utility functions for JaxPINT.

All functions are JIT-compatible and operate on raw float64 arrays (no units).
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import equinox as eqx
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jaxtyping import Array, Float, Bool

if TYPE_CHECKING:
    from jaxpint.types import TOAData, ParameterVector

from jaxpint.constants import (
    ARCSEC_TO_RAD,
    C_KM_PER_S,
    DAYS_PER_JULIAN_YEAR,
    RAD_PER_MAS,
    SECS_PER_DAY,
)
from jaxpint.types.dual_float import DualFloat


# ---------------------------------------------------------------------------
# Quadratic-form (Laplace) coefficients of a likelihood
# ---------------------------------------------------------------------------


def quadratic_form_coeffs(
    logL_fn: Callable[[Float[Array, " n"]], Float[Array, ""]],
    n: int,
) -> tuple[Float[Array, " n"], Float[Array, "n n"]]:
    r"""Linear and quadratic coefficients of a log-likelihood quadratic in ``A``.

    For ``logL(A) = c + b·A - ½ Aᵀ M A`` -- exact when the residuals are linear in
    the amplitude vector ``A`` -- returns ``(b, M)``: ``b`` is the gradient at the
    origin (the matched filter) and ``M`` the symmetric curvature (Fisher / Gram;
    the log-likelihood Hessian is ``-M``).

    Uses ``n + 1`` first-order gradients via the finite-difference-of-gradients
    trick: since the gradient of a quadratic is ``∇logL(A) = b - M A``, we have
    ``b = ∇logL(0)`` and ``M[:, j] = b - ∇logL(e_j)``.  This avoids building a
    second-order autodiff graph (``jax.hessian``) through the full likelihood --
    much lighter on memory, which matters when this is vmapped over a grid.

    Parameters
    ----------
    logL_fn : callable
        ``(A,) -> scalar`` log-likelihood, with ``A`` an ``(n,)`` amplitude
        vector; must be exactly quadratic in ``A``.
    n : int
        Length of the amplitude vector ``A``.

    Returns
    -------
    b : (n,) array
        Linear coefficient ``∇logL(0)`` -- the matched filter.
    M : (n, n) array
        Symmetric quadratic coefficient (Fisher / Gram).
    """
    grad = jax.grad(logL_fn)
    zero = jnp.zeros(n, dtype=jnp.float64)
    b = grad(zero)
    cols = [b - grad(zero.at[j].set(1.0)) for j in range(n)]
    M = jnp.stack(cols, axis=1)
    return b, 0.5 * (M + M.T)  # symmetrize tiny numerical asymmetry


# ---------------------------------------------------------------------------
# Longdouble MJD split
# ---------------------------------------------------------------------------


def split_longdouble_days(
    ld_array: np.ndarray | np.longdouble,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a longdouble day value into float64 ``(int_day, frac_day)``.

    The integer/fractional split is performed in extended (longdouble)
    precision *before* casting to float64
    A single float64 cannot hold an absolute MJD (~60000) to sub-microsecond precision,
    but the two-part ``(int, frac)`` form can.  ``frac_day`` is in [0, 1).
    """
    int_part = np.floor(ld_array)
    frac_part = ld_array - int_part
    return int_part.astype(np.float64), frac_part.astype(np.float64)


# ---------------------------------------------------------------------------
# Taylor polynomial evaluation
# ---------------------------------------------------------------------------


def taylor_horner(
    x: Float[Array, " *batch"],
    coeffs: Float[Array, " n_coeffs"],
) -> Float[Array, " *batch"]:
    """Evaluate a Taylor series at *x* via the Horner scheme.

    The Taylor series is::

        coeffs[0] + coeffs[1]*x/1! + coeffs[2]*x^2/2! + ...

    Example::

        taylor_horner(2.0, jnp.array([10., 3., 4., 12.]))  # -> 40.0

    Parameters
    ----------
    x : array
        Evaluation point(s).
    coeffs : 1-D array, shape (n_coeffs,)
        Taylor coefficients.  ``coeffs[i]`` multiplies ``x**i / i!``.

    Returns
    -------
    array, same shape as *x*
    """
    return taylor_horner_deriv(x, coeffs, deriv_order=0)


def taylor_horner_deriv(
    x: Float[Array, " *batch"],
    coeffs: Float[Array, " n_coeffs"],
    deriv_order: int = 1,
) -> Float[Array, " *batch"]:
    """Evaluate the *deriv_order*-th derivative of a Taylor series.

    Uses the Horner scheme with ``jax.lax.fori_loop`` for JIT efficiency.
    (see     # https://en.wikipedia.org/wiki/Horner%27s_method)

    Example::

        taylor_horner_deriv(2.0, jnp.array([10., 3., 4., 12.]), 1)  # -> 35.0

    Parameters
    ----------
    x : array
        Evaluation point(s).
    coeffs : 1-D array, shape (n_coeffs,)
        Taylor coefficients.
    deriv_order : int
        Derivative order (non-negative).

    Returns
    -------
    array, same shape as *x*
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64)

    n_coeffs = coeffs.shape[0]
    n_terms = n_coeffs - deriv_order

    # deriv_order >= n_coeffs  →  result is zero
    if n_terms <= 0:
        return jnp.zeros_like(x)

    def body(i, result):
        coeff = coeffs[n_coeffs - 1 - i]
        fact = jnp.asarray(n_terms - i, dtype=jnp.float64)
        return result * x / fact + coeff

    return jax.lax.fori_loop(0, n_terms, body, jnp.zeros_like(x))


def taylor_horner_phase(
    dt_int_days: Float[Array, " n"],
    dt_frac_days: Float[Array, " n"],
    delay: Float[Array, " n"],
    coeffs: Float[Array, " n_coeffs"],
) -> DualFloat:
    """Evaluate a Taylor series with phase precision via int/frac Horner.

    Uses the day decomposition ``dt = dt_int_days * 86400 + dt_frac_s``
    to split each Horner multiplication into integer (exact) and
    fractional (precise) parts, avoiding the precision loss that occurs
    when a large absolute phase (~10^10 cycles) is computed as a single
    float64.

    ``coeffs`` must be **pre-divided by factorial** — i.e.
    ``coeffs[k] = F_{k-1} / k!`` for k >= 1, with ``coeffs[0] = 0`` for
    the constant phase term. (For Spindown, use
    ``Spindown._get_spin_coeffs_scaled``.) This lets the Horner loop
    multiply by ``x`` without an inner division, so the integer-part
    product ``phase_int * x_int_s`` stays exact (integer × integer below
    2^53).

    Parameters
    ----------
    dt_int_days : (n_toas,)
        Integer MJD day difference from epoch (exact).
    dt_frac_days : (n_toas,)
        Fractional MJD day difference from epoch.
    delay : (n_toas,)
        Accumulated signal delay in seconds.
    coeffs : (n_coeffs,)
        Pre-divided Taylor coefficients (see above).

    Returns
    -------
    DualFloat
        Phase in cycles, split as integer + fractional part.
    """
    dt_int_days = jnp.asarray(dt_int_days, dtype=jnp.float64)
    dt_frac_days = jnp.asarray(dt_frac_days, dtype=jnp.float64)
    delay = jnp.asarray(delay, dtype=jnp.float64)
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64)

    x_int_s = dt_int_days * SECS_PER_DAY  # integer seconds, exact
    x_frac_s = dt_frac_days * SECS_PER_DAY - delay  # sub-day seconds minus delay

    n_coeffs = coeffs.shape[0]

    def body(i, state):
        # Horner step `acc = acc * x + coeff`. The accumulator is carried as an
        # exact integer (phase_int) plus a small frac normalized to [-0.5, 0.5)
        # each iteration, with `comp` holding the frac's rounding error. x is
        # split the same way; phase_int * x_int_s is exact (int x int < 2^53).
        phase_int, phase_frac, comp = state
        coeff = coeffs[n_coeffs - 1 - i]

        new_int_base = phase_int * x_int_s
        base_frac = phase_int * x_frac_s + phase_frac * x_int_s + phase_frac * x_frac_s

        # comp is scaled by x too, to stay aligned with the accumulator bits it
        # corrects (KBN convention: comp = true - rounded).
        # See https://en.wikipedia.org/wiki/Kahan_summation_algorithm for KBN summation
        comp_x = comp * x_int_s + comp * x_frac_s

        # KBN-compensated add of (coeff + comp_x) to base_frac. coeff (~F0) dwarfs
        # the ~1e-7 frac residue, so a naive sum rounds the residue away and the
        # next step's `phase_frac * x_int_s` amplifies the loss (~7e-5 cycles);
        # new_comp captures that rounding error for the next iteration instead.
        y = coeff + comp_x
        t = base_frac + y
        # optimization_barrier stops XLA from cancelling the error-extraction
        # subtractions (exact by Sterbenz only if t is opaque) back to zero.
        t_pinned = jax.lax.optimization_barrier(t)
        new_comp = jnp.where(
            jnp.abs(base_frac) >= jnp.abs(y),
            (base_frac - t_pinned) + y,
            (y - t_pinned) + base_frac,
        )
        new_frac = t

        # Re-normalize frac into [-0.5, 0.5), carrying the integer part up.
        overflow = jnp.round(new_frac)
        return new_int_base + overflow, new_frac - overflow, new_comp

    z = jnp.zeros_like(dt_int_days)
    result_int, result_frac, result_comp = jax.lax.fori_loop(
        0,
        n_coeffs,
        body,
        (z, z, z),
    )
    # Fold the residual compensation back into the frac before normalization.
    return DualFloat.from_cycles(result_int, result_frac + result_comp)


# ---------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------


def weighted_mean(
    arrin: Float[Array, " n"],
    weights_in: Float[Array, " n"],
    inputmean: Optional[float] = None,
    calcerr: bool = False,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute weighted mean and error of *arrin*.

    Parameters
    ----------
    arrin : 1-D array
        Data values.
    weights_in : 1-D array
        Weights (typically ``1 / sigma**2``).
    inputmean : float, optional
        If given, use this as the mean instead of computing it.
    calcerr : bool
        If True, compute error from weighted scatter rather than
        ``1 / sqrt(sum(weights))``.

    Returns
    -------
    (wmean, werr)
    """
    wtot = jnp.sum(weights_in)

    if inputmean is None:
        wmean = jnp.sum(weights_in * arrin) / wtot
    else:
        wmean = jnp.asarray(inputmean, dtype=jnp.float64)

    if calcerr:
        werr = jnp.sqrt(jnp.sum(weights_in**2 * (arrin - wmean) ** 2)) / wtot
    else:
        werr = 1.0 / jnp.sqrt(wtot)

    return wmean, werr


def weighted_mean_sdev(
    arrin: Float[Array, " n"],
    weights_in: Float[Array, " n"],
    inputmean: Optional[float] = None,
    calcerr: bool = False,
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """Compute weighted mean, error, and standard deviation of *arrin*.

    Parameters
    ----------
    arrin : 1-D array
        Data values.
    weights_in : 1-D array
        Weights (typically ``1 / sigma**2``).
    inputmean : float, optional
        If given, use this as the mean instead of computing it.
    calcerr : bool
        If True, compute error from weighted scatter rather than
        ``1 / sqrt(sum(weights))``.

    Returns
    -------
    wmean : float
        Weighted mean of ``arrin``.
    werr : float
        Error on the weighted mean.
    wsdev : float
        Weighted sample standard deviation.
    """
    wmean, werr = weighted_mean(arrin, weights_in, inputmean, calcerr)

    wtot = jnp.sum(weights_in)
    wvar = jnp.sum(weights_in * (arrin - wmean) ** 2) / wtot
    wsdev = jnp.sqrt(wvar)

    return wmean, werr, wsdev


# ---------------------------------------------------------------------------
# Design matrix normalization
# ---------------------------------------------------------------------------


def normalize_designmatrix(
    M: Float[Array, "n_toas n_params"],
) -> tuple[
    Float[Array, "n_toas n_params"],
    Float[Array, " n_params"],
    Bool[Array, " n_params"],
]:
    """Column-normalize the design matrix for numerical stability.

    The normalized matrix ``Mn`` and the original ``M`` are related by
    ``M = Mn * norms`` (broadcasting over rows).  GLS expressions of the
    form ``M @ inv(M.T @ Ninv @ M) @ M.T`` are invariant under this
    rescaling.

    Columns with zero norm (degenerate parameters) are left as-is.

    Parameters
    ----------
    M : 2-D array, shape (n_toas, n_params)

    Returns
    -------
    M_normalized : 2-D array
        ``M`` divided column-wise by its column norms.
    norms : 1-D array
        Per-column 2-norms of ``M``.
    degenerate : 1-D bool array
        ``True`` for columns with zero norm — parameters that have no
        effect on the residuals.
    """
    norm = jnp.sqrt(jnp.sum(M**2, axis=0))
    degenerate = norm == 0.0
    norm = jnp.where(degenerate, 1.0, norm)
    return M / norm, norm, degenerate


# ---------------------------------------------------------------------------
# Sherman–Morrison / Woodbury inner products
# ---------------------------------------------------------------------------


def sherman_morrison_dot(
    Ndiag: Float[Array, " n"],
    v: Float[Array, " n"],
    w: Float[Array, ""],
    x: Float[Array, " n"],
    y: Float[Array, " n"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""Compute :math:`x^T C^{-1} y` where :math:`C = \mathrm{diag}(N) + w\,v\,v^T`.

    Uses the Sherman–Morrison identity to avoid forming or inverting *C*.
    See https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula

    Parameters
    ----------
    Ndiag : 1-D array
        Diagonal of *N* (positive).
    v : 1-D array
        Rank-1 update vector.
    w : scalar
        Weight of the rank-1 update.
    x, y : 1-D arrays
        Vectors for the inner product.

    Returns
    -------
    (result, logdet_C)
        The inner product and the log-determinant of *C*.
    """
    Ninv = 1.0 / Ndiag
    Ninv_v = Ninv * v
    denom = 1.0 + w * jnp.dot(v, Ninv_v)
    numer = w * jnp.dot(x, Ninv_v) * jnp.dot(y, Ninv_v)

    result = jnp.dot(x, Ninv * y) - numer / denom
    logdet_C = jnp.sum(jnp.log(Ndiag)) + jnp.log(denom)

    return result, logdet_C


def woodbury_dot(
    Ndiag: Float[Array, " n"],
    U: Float[Array, "n k"],
    Phidiag: Float[Array, " k"],
    x: Float[Array, " n"],
    y: Float[Array, " n"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""Compute :math:`x^T C^{-1} y` where :math:`C = \mathrm{diag}(N) + U\,\mathrm{diag}(\Phi)\,U^T`.

    Uses the Woodbury identity and Cholesky factorisation of the
    reduced-rank matrix :math:`\Sigma = \Phi^{-1} + U^T N^{-1} U`.

    See https://en.wikipedia.org/wiki/Woodbury_matrix_identity

    Parameters
    ----------
    Ndiag : 1-D array, shape (n,)
        Diagonal of *N* (positive).
    U : 2-D array, shape (n, k)
        Low-rank update basis.
    Phidiag : 1-D array, shape (k,)
        Diagonal of :math:`\Phi` (positive).
    x, y : 1-D arrays, shape (n,)
        Vectors for the inner product.

    Returns
    -------
    (result, logdet_C)
        The inner product and the log-determinant of *C*.
    """
    Ninv = 1.0 / Ndiag

    x_Ninv_y = jnp.sum(x * y * Ninv)
    x_Ninv_U = (x * Ninv) @ U  # (k,)
    y_Ninv_U = (y * Ninv) @ U  # (k,)

    Sigma = jnp.diag(1.0 / Phidiag) + (U.T * Ninv) @ U  # (k, k)
    Sigma_cf = jax.scipy.linalg.cho_factor(Sigma)

    x_Cinv_y = x_Ninv_y - x_Ninv_U @ jax.scipy.linalg.cho_solve(Sigma_cf, y_Ninv_U)

    logdet_N = jnp.sum(jnp.log(Ndiag))
    logdet_Phi = jnp.sum(jnp.log(Phidiag))
    # Use the Cholesky factor (already computed above) instead of jnp.linalg.slogdet:
    # det(Sigma) = det(L L^T) = (prod diag(L))^2, so logdet = 2 sum log diag(L).
    # Mathematically identical for our PD Sigma but DOES NOT break higher-order
    logdet_Sigma = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(Sigma_cf[0]))))

    logdet_C = logdet_N + logdet_Phi + logdet_Sigma

    return x_Cinv_y, logdet_C


def woodbury_dot_qr(
    Ndiag: Float[Array, " n"],
    U: Float[Array, "n k"],
    Phidiag: Float[Array, " k"],
    x: Float[Array, " n"],
    y: Float[Array, " n"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""Square-root (QR) form of :func:`woodbury_dot`: same result, more stable.

    Computes :math:`x^T C^{-1} y` and :math:`\log\det C` for
    :math:`C = \mathrm{diag}(N) + U\,\mathrm{diag}(\Phi)\,U^T`, but **never forms
    the Gram matrix** :math:`U^T N^{-1} U`. Instead it QR-factorises the stacked
    whitened design

    .. math::
        A = N^{-1/2} U\,\Phi^{1/2}, \qquad
        S = \begin{bmatrix} A \\ I_k \end{bmatrix}, \qquad
        S = Q R \;\Rightarrow\; R^T R = I_k + A^T A .

    Then with :math:`u = N^{-1/2}x`, :math:`v = N^{-1/2}y`,

    .. math::
        x^T C^{-1} y = u^T v - (R^{-T} A^T u)^T (R^{-T} A^T v), \qquad
        \log\det C = \log\det N + 2\sum_i \log|R_{ii}| .

    Why prefer this over :func:`woodbury_dot`
    -----------------------------------------
    The Cholesky form builds ``Σ = diag(1/Φ) + UᵀN⁻¹U``, whose Gram term
    **squares** the conditioning of ``N^{-1/2}U``; the QR works at the square
    root of that, so it keeps ~twice the digits when ``U`` columns are
    collinear. This matters for the analytic-marginalization block, where
    ``U`` is the timing design matrix ``M`` (genuinely collinear for
    multi-parameter binary MSPs) at ``Φ = 1e40``. On a real 13-parameter MSP
    the Cholesky form loses ~4 digits in ``xᵀC⁻¹y`` (relerr ~3e-4) while this
    form holds ~1e-9. ``S = [A; I]`` always has full column rank ``k`` (the
    ``I`` block), so the QR never fails even when ``A`` is rank-deficient or
    its columns span a huge dynamic range (mixed marginalization + red-noise
    ``Φ``). Note ``logdet(Φ)`` cancels analytically and never appears.

    Cost is ~the same order as :func:`woodbury_dot` (a QR of an
    ``(n+k, k)`` matrix vs a ``(k, k)`` Cholesky plus the Gram build). Use
    :func:`woodbury_dot` when ``U`` is well-conditioned (e.g. an orthogonal
    Fourier basis) and this when it is not.

    Parameters
    ----------
    Ndiag : 1-D array, shape (n,)
        Diagonal of *N* (positive).
    U : 2-D array, shape (n, k)
        Low-rank update basis.
    Phidiag : 1-D array, shape (k,)
        Diagonal of :math:`\Phi` (positive).
    x, y : 1-D arrays, shape (n,)
        Vectors for the inner product.

    Returns
    -------
    (result, logdet_C)
        The inner product and the log-determinant of *C*.
    """
    k = U.shape[1]
    Ninv_half = 1.0 / jnp.sqrt(Ndiag)
    A = (Ninv_half[:, None] * U) * jnp.sqrt(Phidiag)[None, :]  # (n, k)

    # R from QR of [A; I_k]: upper-triangular with R^T R = I + A^T A. The I
    # block guarantees full column rank, so R is always invertible.
    S = jnp.concatenate([A, jnp.eye(k)], axis=0)  # (n + k, k)
    R = jnp.linalg.qr(S, mode="r")  # (k, k)

    u = Ninv_half * x
    v = Ninv_half * y
    # R^{-T} (A^T u): solve the lower-triangular system R^T p = A^T u.
    p = jax.scipy.linalg.solve_triangular(R, A.T @ u, trans="T", lower=False)
    q = jax.scipy.linalg.solve_triangular(R, A.T @ v, trans="T", lower=False)

    x_Cinv_y = u @ v - p @ q
    logdet_C = jnp.sum(jnp.log(Ndiag)) + 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(R))))
    return x_Cinv_y, logdet_C


def woodbury_solve(
    Ndiag: Float[Array, " n"],
    U: Float[Array, "n k"],
    Phidiag: Float[Array, " k"],
    B: Float[Array, "n m"],
) -> Float[Array, "n m"]:
    r"""Compute :math:`C^{-1} B` where :math:`C = \mathrm{diag}(N) + U\,\mathrm{diag}(\Phi)\,U^T`.

    Uses the Woodbury identity::

        C^{-1} = N^{-1} - N^{-1} U Σ^{-1} U^T N^{-1}

    where :math:`\Sigma = \Phi^{-1} + U^T N^{-1} U`.

    Parameters
    ----------
    Ndiag : 1-D array, shape (n,)
        Diagonal of *N* (positive).
    U : 2-D array, shape (n, k)
        Low-rank update basis.
    Phidiag : 1-D array, shape (k,)
        Diagonal of :math:`\Phi` (positive).
    B : 2-D array, shape (n, m)
        Right-hand side matrix.

    Returns
    -------
    Cinv_B : array, shape (n, m)
        The product :math:`C^{-1} B`.
    """
    Ninv = 1.0 / Ndiag
    Ninv_B = Ninv[:, None] * B  # (n, m)
    Ninv_U = Ninv[:, None] * U  # (n, k)

    Sigma = jnp.diag(1.0 / Phidiag) + U.T @ Ninv_U  # (k, k)
    Sigma_cf = jax.scipy.linalg.cho_factor(Sigma)

    # Σ^{-1} (U^T N^{-1} B)
    UtNinvB = U.T @ Ninv_B  # (k, m)
    Sigma_inv_UtNinvB = jax.scipy.linalg.cho_solve(Sigma_cf, UtNinvB)  # (k, m)

    return Ninv_B - Ninv_U @ Sigma_inv_UtNinvB


def concat_woodbury_blocks(
    *blocks: Optional[tuple[Float[Array, "n_toas _k"], Float[Array, " _k"]]],
) -> Optional[tuple[Float[Array, "n_toas k"], Float[Array, " k"]]]:
    r"""Concatenate one or more low-rank Woodbury :math:`(U, \Phi)` blocks.

    Each input is either ``None`` or a ``(U, Phi)`` tuple where ``U`` has
    shape ``(n_toas, k_i)`` and ``Phi`` has shape ``(k_i,)``. ``None`` entries
    are skipped.  The result is::

        U   = jnp.concatenate([U_i  for non-None i], axis=1)   # (n_toas, sum k_i)
        Phi = jnp.concatenate([Phi_i for non-None i])           # (sum k_i,)

    A single non-None input is returned unchanged (no array copy).

    Returns
    -------
    out : tuple or None
        ``None`` if every input is ``None`` — no contribution to the noise
        covariance, which the caller may treat as "no Woodbury augmentation
        needed".  Otherwise the concatenated ``(U, Phi)`` block (a tuple of
        two arrays).

    Notes
    -----
    Used by :func:`~jaxpint.likelihood.single_pulsar_logL`,
    :func:`~jaxpint.pta.pta_logL`,
    :meth:`~jaxpint.noise.NoiseModel.covariance`,
    :func:`~jaxpint.bayes.marginalize_single_pulsar` /
    :func:`~jaxpint.bayes.marginalize_pta` and the
    correlated-likelihood paths to compose the Woodbury low-rank update from
    multiple sources (white-noise components, per-pulsar signal injectors,
    cross-pulsar GW basis, analytic-marginalization prior block, etc.).

    Callers that need a guaranteed-non-None empty-shaped result (e.g.
    ``(zeros((n_toas, 0)), zeros(0))``) should provide that fallback at the
    call site.
    """
    non_empty = [b for b in blocks if b is not None]
    if not non_empty:
        return None
    if len(non_empty) == 1:
        return non_empty[0]
    Us = [U for U, _ in non_empty]
    Phis = [Phi for _, Phi in non_empty]
    return jnp.concatenate(Us, axis=1), jnp.concatenate(Phis)


# ---------------------------------------------------------------------------
# Woodbury precompute / apply split
# ---------------------------------------------------------------------------


class WoodburyFactor(eqx.Module):
    r"""Precomputed Woodbury factorization for repeated solves.

    Splits :func:`woodbury_dot` into a parameter-independent setup
    (``Σ`` Cholesky and the two log-determinant constants) and a
    residual-dependent application :func:`apply_woodbury_dot_factor`.
    Useful for grid scans or MCMC where many evaluations share the same
    noise covariance ``C = diag(N) + U diag(Φ) U^T`` and only the
    residuals ``r`` change — the Cholesky runs once instead of per-cell.

    The factor is a JAX pytree (Equinox module), so it can be passed
    through :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`. To
    vmap the *application* over a batch of residuals while keeping the
    factor scalar, use ``in_axes=(None, 0)`` for ``(factor, r)``.

    Fields
    ------
    Ndiag : (n,)
        Diagonal of ``N``.
    U : (n, k)
        Low-rank update basis.
    Sigma_cf_factor : (k, k)
        Cholesky factor of ``Σ = Φ⁻¹ + U^T N⁻¹ U`` (the array half of
        :func:`jax.scipy.linalg.cho_factor`'s output).
    Sigma_cf_lower : bool
        ``lower`` flag from :func:`jax.scipy.linalg.cho_factor`. Static metadata.
    logdet_C : scalar
        ``log det(C) = sum log N + sum log Φ + log det Σ``, precomputed.
    """

    Ndiag: Float[Array, " n"]
    U: Float[Array, "n k"]
    Sigma_cf_factor: Float[Array, "k k"]
    Sigma_cf_lower: bool = eqx.field(static=True)
    logdet_C: Float[Array, ""]

    @property
    def Sigma_cf(self):
        """Reconstruct the ``(factor, lower)`` tuple expected by ``cho_solve``."""
        return (self.Sigma_cf_factor, self.Sigma_cf_lower)


def precompute_woodbury_factor(
    Ndiag: Float[Array, " n"],
    U: Float[Array, "n k"],
    Phidiag: Float[Array, " k"],
) -> WoodburyFactor:
    r"""Precompute the parameter-independent half of :func:`woodbury_dot`.

    Computes the Cholesky of :math:`\Sigma = \Phi^{-1} + U^T N^{-1} U`
    and the constant :math:`\log\det C` once, so that subsequent calls
    to :func:`apply_woodbury_dot_factor` only do residual-dependent
    matrix-vector work.

    Parameters
    ----------
    Ndiag : 1-D array, shape (n,)
        Diagonal of ``N`` (positive).
    U : 2-D array, shape (n, k)
        Low-rank update basis.
    Phidiag : 1-D array, shape (k,)
        Diagonal of ``Φ`` (positive).

    Returns
    -------
    factor : WoodburyFactor
        Precomputed factor; pass to :func:`apply_woodbury_dot_factor`.
    """
    Ninv = 1.0 / Ndiag
    Sigma = jnp.diag(1.0 / Phidiag) + (U.T * Ninv) @ U  # (k, k)
    Sigma_cf_factor, Sigma_cf_lower = jax.scipy.linalg.cho_factor(Sigma)

    logdet_N = jnp.sum(jnp.log(Ndiag))
    logdet_Phi = jnp.sum(jnp.log(Phidiag))
    # Use the Cholesky factor (already computed above) instead of jnp.linalg.slogdet:
    # det(Sigma) = det(L L^T) = (prod diag(L))^2, so logdet = 2 sum log diag(L).
    logdet_Sigma = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(Sigma_cf_factor))))
    logdet_C = logdet_N + logdet_Phi + logdet_Sigma

    return WoodburyFactor(
        Ndiag=Ndiag,
        U=U,
        Sigma_cf_factor=Sigma_cf_factor,
        Sigma_cf_lower=bool(Sigma_cf_lower),
        logdet_C=logdet_C,
    )


def apply_woodbury_dot_factor(
    factor: WoodburyFactor,
    x: Float[Array, " n"],
    y: Float[Array, " n"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""Compute :math:`(x^T C^{-1} y, \log\det C)` from a precomputed factor.

    Numerically equivalent to :func:`woodbury_dot` but uses the
    Cholesky factor and ``log det C`` cached in ``factor`` instead of
    recomputing them. Bit-for-bit identical when called with the same
    inputs that produced the factor.

    Vmap-friendly: the factor (containing ``(n, k)`` and ``(k, k)``
    arrays) is shared across vmapped lanes, while only the residual-
    sized vectors ``x``, ``y`` get the leading vmap axis. This is the
    cost-saving path for grid scans where ``N``, ``U``, ``Φ`` don't
    depend on the swept parameters.

    Parameters
    ----------
    factor : WoodburyFactor
        Precomputed factor from :func:`precompute_woodbury_factor`.
    x, y : 1-D arrays, shape (n,)
        Vectors for the inner product.

    Returns
    -------
    (x_Cinv_y, logdet_C)
        Same return contract as :func:`woodbury_dot`.
    """
    Ninv = 1.0 / factor.Ndiag
    x_Ninv_y = jnp.sum(x * y * Ninv)
    x_Ninv_U = (x * Ninv) @ factor.U  # (k,)
    y_Ninv_U = (y * Ninv) @ factor.U  # (k,)
    x_Cinv_y = x_Ninv_y - x_Ninv_U @ jax.scipy.linalg.cho_solve(
        factor.Sigma_cf,
        y_Ninv_U,
    )
    return x_Cinv_y, factor.logdet_C


# ---------------------------------------------------------------------------
# Ecliptic obliquity rotation
# ---------------------------------------------------------------------------


def ecl_to_icrs_rotation(obliquity_arcsec: float) -> Float[Array, "3 3"]:
    """Rotation matrix from ecliptic to ICRS (row-vector convention).

    Usage: ``L_icrs = L_ecl @ ecl_to_icrs_rotation(obl)``

    This is the transpose of astropy's ``rotation_matrix(obl, 'x')``,
    adapted for row-vector multiplication.
    """
    obl_rad = obliquity_arcsec * ARCSEC_TO_RAD
    c = jnp.cos(obl_rad)
    s = jnp.sin(obl_rad)
    return jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ]
    )


# ---------------------------------------------------------------------------
# Pulsar direction (shared by astrometry and Shapiro delay)
# ---------------------------------------------------------------------------


def compute_pulsar_direction(
    toa_data: "TOAData",
    params: "ParameterVector",
    raj_name: str,
    decj_name: str,
    pmra_name: Optional[str],
    pmdec_name: Optional[str],
    posepoch_name: Optional[str],
) -> Float[Array, "n_toas 3"]:
    """Unit vector from SSB to pulsar in ICRS Cartesian coordinates.

    Without proper motion the direction is constant; with proper motion
    a linear correction is applied per TOA.

    Parameters
    ----------
    toa_data : TOAData
        Pre-extracted TOA data (needs ``tdb_int``, ``tdb_frac``, ``n_toas``).
    params : ParameterVector
        Timing-model parameters.
    raj_name, decj_name : str
        Parameter names for RA and DEC (radians).
    pmra_name, pmdec_name : str or None
        Parameter names for proper motion (mas/yr).  None disables PM.
    posepoch_name : str or None
        Epoch parameter for proper-motion reference.
    """
    ra0 = params.param_value(raj_name)
    dec0 = params.param_value(decj_name)

    if pmra_name is not None or pmdec_name is not None:
        assert posepoch_name is not None  # POSEPOCH is set whenever PM is present
        posepoch = params.epoch_dual(posepoch_name)
        dt_yr = (toa_data.tdb - posepoch).total / DAYS_PER_JULIAN_YEAR

        if pmra_name is not None:
            pmra = params.param_value(pmra_name)  # mas/yr
            ra = ra0 + (pmra * RAD_PER_MAS / jnp.cos(dec0)) * dt_yr
        else:
            ra = jnp.broadcast_to(ra0, dt_yr.shape)

        if pmdec_name is not None:
            pmdec = params.param_value(pmdec_name)  # mas/yr
            dec = dec0 + (pmdec * RAD_PER_MAS) * dt_yr
        else:
            dec = jnp.broadcast_to(dec0, dt_yr.shape)
    else:
        ra = ra0
        dec = dec0

    cos_dec = jnp.cos(dec)
    x = jnp.cos(ra) * cos_dec
    y = jnp.sin(ra) * cos_dec
    z = jnp.sin(dec)
    L_hat = jnp.stack([x, y, z], axis=-1)

    if L_hat.ndim == 1:
        L_hat = jnp.broadcast_to(L_hat[None, :], (toa_data.n_toas, 3))

    return L_hat


def compute_pulsar_direction_ecl(
    toa_data: "TOAData",
    params: "ParameterVector",
    elong_name: str,
    elat_name: str,
    pmelong_name: Optional[str],
    pmelat_name: Optional[str],
    posepoch_name: Optional[str],
    obliquity_arcsec: float,
) -> Float[Array, "n_toas 3"]:
    """Unit vector from SSB to pulsar in ICRS, computed from ecliptic coordinates.

    Computes the direction in ecliptic frame (reusing the same lon/lat → xyz
    math as ``compute_pulsar_direction``), then rotates to ICRS.

    Parameters
    ----------
    toa_data : TOAData
    params : ParameterVector
    elong_name, elat_name : str
        Parameter names for ecliptic longitude and latitude (radians).
    pmelong_name, pmelat_name : str or None
        Proper motion parameter names (mas/yr).  None disables PM.
    posepoch_name : str or None
        Epoch parameter for proper-motion reference.
    obliquity_arcsec : float
        Obliquity of the ecliptic in arcseconds.
    """
    L_hat_ecl = compute_pulsar_direction(
        toa_data,
        params,
        raj_name=elong_name,
        decj_name=elat_name,
        pmra_name=pmelong_name,
        pmdec_name=pmelat_name,
        posepoch_name=posepoch_name,
    )
    rot = ecl_to_icrs_rotation(obliquity_arcsec)
    return L_hat_ecl @ rot


def pulsar_unit_vector(
    params: "ParameterVector",
    obliquity_arcsec: float = 84381.406,  # IAU 2006 obliquity at J2000.0
) -> Float[Array, " 3"]:
    """Static ICRS Cartesian unit vector toward a pulsar, from its sky parameters.

    The catalog direction (no proper-motion / epoch correction) -- the per-pulsar
    position input expected by e.g. :class:`jaxpint.pta.signals.cw.CWInjector`.
    The per-TOA, proper-motion-aware direction is :func:`compute_pulsar_direction`.

    Parameters
    ----------
    params : ParameterVector
        Pulsar parameters; must contain either ``RAJ``/``DECJ`` (equatorial) or
        ``ELONG``/``ELAT`` (ecliptic), in radians (PINT convention).
    obliquity_arcsec : float
        Obliquity of the ecliptic in arcseconds, used only for the ecliptic case
        to rotate to ICRS via :func:`ecl_to_icrs_rotation` (the same rotation the
        geometric-delay components use).  Default IAU 2006 at J2000.0.

    Returns
    -------
    (3,) array
        ICRS Cartesian unit vector from the SSB toward the pulsar.

    Raises
    ------
    KeyError
        If ``params`` has neither ``(RAJ, DECJ)`` nor ``(ELONG, ELAT)``.
    """
    names = params.names
    if "RAJ" in names and "DECJ" in names:
        ra, dec = params.param_value("RAJ"), params.param_value("DECJ")
        cd = jnp.cos(dec)
        return jnp.stack([jnp.cos(ra) * cd, jnp.sin(ra) * cd, jnp.sin(dec)])
    if "ELONG" in names and "ELAT" in names:
        lon, lat = params.param_value("ELONG"), params.param_value("ELAT")
        cl = jnp.cos(lat)
        v_ecl = jnp.stack([jnp.cos(lon) * cl, jnp.sin(lon) * cl, jnp.sin(lat)])
        return v_ecl @ ecl_to_icrs_rotation(obliquity_arcsec)
    raise KeyError(f"pulsar params lack (RAJ,DECJ) and (ELONG,ELAT): {tuple(names)}")


# ---------------------------------------------------------------------------
# Barycentric radio frequency (Doppler correction)
# ---------------------------------------------------------------------------
#
# PINT stores topocentric frequency and applies the barycentric Doppler
# correction on demand (``model.barycentric_radio_freq``):
#
#     f_bary = f_topo * (1 - v_obs . Lhat / c)
#
# where ``v_obs`` is the observatory velocity wrt the SSB (km/s) and ``Lhat`` is
# the unit vector toward the pulsar (from astrometry).  JaxPINT precomputes this
# once at TOAData-build time and stores the barycentric value as ``TOAData.freq``
# (see :mod:`jaxpint.loaders.native`), keeping ``TOAData`` a static,
# parameter-independent container and the Doppler/astrometry work off the jitted
# per-call likelihood path.  The cost is that the stored freq is not refreshed if
# the astrometry is later refit -- negligible in practice (sub-arcsec astrometry,
# first-order v/c ~1e-4 => worst-case <~50 ps; guarded by
# tests/test_timescale.py::test_precompute_staleness_below_ns).


def doppler_shift_freq(
    freq_topo: Float[Array, " n_toas"],
    ssb_obs_vel: Float[Array, "n_toas 3"],
    l_hat: Float[Array, "n_toas 3"],
) -> Float[Array, " n_toas"]:
    """Apply the first-order Doppler factor to a topocentric frequency.

    ``f_bary = f_topo * (1 - v.Lhat/c)``, with ``v`` in km/s.
    """
    v_dot_l = jnp.sum(ssb_obs_vel * l_hat, axis=1)  # km/s
    return freq_topo * (1.0 - v_dot_l / C_KM_PER_S)


def barycentric_radio_freq(
    toa_data: "TOAData",
    params: "ParameterVector",
    *,
    raj_name: str = "RAJ",
    decj_name: str = "DECJ",
    pmra_name: Optional[str] = None,
    pmdec_name: Optional[str] = None,
    posepoch_name: Optional[str] = None,
    ecliptic: bool = False,
    elong_name: str = "ELONG",
    elat_name: str = "ELAT",
    pmelong_name: Optional[str] = None,
    pmelat_name: Optional[str] = None,
    obliquity_arcsec: float = 0.0,
) -> Float[Array, " n_toas"]:
    """Barycentric (Doppler-corrected) radio frequency for each TOA.

    Computes ``Lhat`` from the model astrometry (equatorial by default, or
    ecliptic when ``ecliptic=True``) and applies the Doppler factor to
    ``toa_data.freq`` (which must be topocentric).
    """
    if ecliptic:
        l_hat = compute_pulsar_direction_ecl(
            toa_data,
            params,
            elong_name=elong_name,
            elat_name=elat_name,
            pmelong_name=pmelong_name,
            pmelat_name=pmelat_name,
            posepoch_name=posepoch_name,
            obliquity_arcsec=obliquity_arcsec,
        )
    else:
        l_hat = compute_pulsar_direction(
            toa_data,
            params,
            raj_name=raj_name,
            decj_name=decj_name,
            pmra_name=pmra_name,
            pmdec_name=pmdec_name,
            posepoch_name=posepoch_name,
        )
    return doppler_shift_freq(toa_data.freq, toa_data.ssb_obs_vel, l_hat)


# ---------------------------------------------------------------------------
# Fourier basis construction
# ---------------------------------------------------------------------------


def fourier_sum(
    dt_days: Float[Array, " n_toas"],
    wx_freqs: Float[Array, " n_components"],
    wx_sins: Float[Array, " n_components"],
    wx_coses: Float[Array, " n_components"],
) -> Float[Array, " n_toas"]:
    """Evaluate a Fourier sum at each TOA.

    Computes::

        result[t] = Σ_i (wx_sins[i] * sin(2π * wx_freqs[i] * dt_days[t])
                       + wx_coses[i] * cos(2π * wx_freqs[i] * dt_days[t]))

    Parameters
    ----------
    dt_days : (n_toas,)
        Time differences from the reference epoch in **days**.
    wx_freqs : (n_components,)
        Fourier frequencies in **1/day**.
    wx_sins : (n_components,)
        Sine amplitudes.
    wx_coses : (n_components,)
        Cosine amplitudes.

    Returns
    -------
    (n_toas,)
        Fourier sum evaluated at each TOA.
    """
    arg = 2.0 * jnp.pi * dt_days[:, None] * wx_freqs[None, :]  # (n_toas, n_comp)
    return jnp.sum(wx_sins * jnp.sin(arg) + wx_coses * jnp.cos(arg), axis=1)


def _group_toas_into_epochs(
    sorted_times: np.ndarray,
    sorted_indices: np.ndarray,
    dt: float,
    nmin: int,
) -> list[list[int]]:
    """Greedily group time-sorted TOAs into epochs, keeping only big-enough ones.

    A TOA joins the current epoch if it falls within *dt* seconds of that
    epoch's *first* TOA; otherwise it opens a new epoch. Returns the epochs
    (as lists of original TOA indices) that have at least *nmin* members.
    """
    epochs: list[list[int]] = []
    current: list[int] = []
    epoch_start = 0.0
    for t, idx in zip(sorted_times, sorted_indices):
        # This TOA is >= dt from the current epoch's start: close that epoch.
        if current and t - epoch_start >= dt:
            epochs.append(current)
            current = []
        # Start of new epoch. Record start time
        if not current:
            epoch_start = t
        current.append(int(idx))
    if current:
        epochs.append(current)

    return [ep for ep in epochs if len(ep) >= nmin]


def build_quantization_matrix(
    tdb_times_s: np.ndarray,
    ecorr_masks: dict[str, np.ndarray],
    dt: float = 1.0,
    nmin: int = 2,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Build the ECORR quantization matrix (NumPy, not JIT-compatible).

    Groups TOAs within *dt* seconds into epochs and creates a binary
    matrix ``U`` mapping TOAs to epochs.  Only epochs with at least
    *nmin* TOAs are kept.

    Parameters
    ----------
    tdb_times_s : (n_toas,) float64
        TOA times in TDB seconds.
    ecorr_masks : dict[str, ndarray]
        Boolean masks keyed by ECORR parameter name.
    dt, nmin : float, int
        Epoch grouping threshold (seconds) and minimum TOAs per epoch.

    Returns
    -------
    U : (n_toas, n_total_epochs)
        Binary quantization matrix.
    epoch_slices : dict[str, (int, int)]
        Column-index range for each ECORR parameter.
    """
    n_toas = len(tdb_times_s)
    columns: list[np.ndarray] = []  # one binary (n_toas,) column per kept epoch
    epoch_slices: dict[str, tuple[int, int]] = {}

    for ecorr_name in sorted(ecorr_masks):
        # Column index where this parameter's epochs will start. The running
        # column count *is* the offset, so no separate bookkeeping is needed.
        start = len(columns)

        toa_indices = np.where(ecorr_masks[ecorr_name])[0]
        if len(toa_indices) > 0:
            order = np.argsort(tdb_times_s[toa_indices])
            sorted_times = tdb_times_s[toa_indices][order]
            sorted_indices = toa_indices[order]
            for epoch in _group_toas_into_epochs(
                sorted_times, sorted_indices, dt, nmin
            ):
                col = np.zeros(n_toas, dtype=np.float64)
                col[epoch] = 1.0
                columns.append(col)

        epoch_slices[ecorr_name] = (start, len(columns))

    U = np.column_stack(columns) if columns else np.zeros((n_toas, 0), dtype=np.float64)
    return U, epoch_slices


def build_fourier_basis(
    tdb_times_s: np.ndarray,
    n_freqs: int,
    T: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an alternating sin/cos Fourier design matrix.

    Parameters
    ----------
    tdb_times_s : (n_toas,)
        TOA times in TDB seconds.
    n_freqs : int
        Number of frequency modes.
    T : float
        Time span in seconds (sets the fundamental frequency 1/T).

    Returns
    -------
    F : (n_toas, 2 * n_freqs)
        Fourier design matrix with columns
        ``[sin(2πf₁t), cos(2πf₁t), sin(2πf₂t), ...]``.
    freqs : (n_freqs,)
        Frequency array in Hz.
    freq_bin_widths : (n_freqs,)
        Δf for each frequency bin.
    """
    freqs = np.arange(1, n_freqs + 1) / T
    freq_bin_widths = np.diff(np.concatenate([[0.0], freqs]))

    phase = 2.0 * np.pi * tdb_times_s[:, None] * freqs[None, :]
    F = np.zeros((len(tdb_times_s), 2 * n_freqs))
    F[:, 0::2] = np.sin(phase)
    F[:, 1::2] = np.cos(phase)

    return F, freqs, freq_bin_widths
