"""Utility functions for JaxPINT.

Pure JAX ports of selected functions from pint.utils.
All functions are JIT-compatible and operate on raw float64 arrays (no units).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from jaxtyping import Array, Float, Bool

if TYPE_CHECKING:
    from jaxpint.types import TOAData, ParameterVector

from jaxpint.constants import ARCSEC_TO_RAD, DAYS_PER_JULIAN_YEAR, OBLIQUITY_ARCSEC, RAD_PER_MAS


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


# ---------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------

def weighted_mean(
    arrin: Float[Array, " n"],
    weights_in: Float[Array, " n"],
    inputmean: Optional[float] = None,
    calcerr: bool = False,
    sdev: bool = False,
):
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
    sdev : bool
        If True, also return the weighted standard deviation.

    Returns
    -------
    (wmean, werr) or (wmean, werr, wsdev)
    """
    wtot = jnp.sum(weights_in)

    if inputmean is None:
        wmean = jnp.sum(weights_in * arrin) / wtot
    else:
        wmean = jnp.asarray(inputmean, dtype=jnp.float64)

    if calcerr:
        werr = jnp.sqrt(jnp.sum(weights_in ** 2 * (arrin - wmean) ** 2)) / wtot
    else:
        werr = 1.0 / jnp.sqrt(wtot)

    if sdev:
        wvar = jnp.sum(weights_in * (arrin - wmean) ** 2) / wtot
        wsdev = jnp.sqrt(wvar)
        return wmean, werr, wsdev

    return wmean, werr


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
    (M_normalized, norms, degenerate)
        ``degenerate`` is a boolean mask that is True for columns with
        zero norm (i.e. parameters that have no effect on the residuals).
    """
    norm = jnp.sqrt(jnp.sum(M ** 2, axis=0))
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
    x_Ninv_U = (x * Ninv) @ U          # (k,)
    y_Ninv_U = (y * Ninv) @ U          # (k,)

    Sigma = jnp.diag(1.0 / Phidiag) + (U.T * Ninv) @ U  # (k, k)
    Sigma_cf = jax.scipy.linalg.cho_factor(Sigma)

    x_Cinv_y = x_Ninv_y - x_Ninv_U @ jax.scipy.linalg.cho_solve(
        Sigma_cf, y_Ninv_U
    )

    logdet_N = jnp.sum(jnp.log(Ndiag))
    logdet_Phi = jnp.sum(jnp.log(Phidiag))
    _, logdet_Sigma = jnp.linalg.slogdet(Sigma)

    logdet_C = logdet_N + logdet_Phi + logdet_Sigma

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
    Ninv_B = Ninv[:, None] * B              # (n, m)
    Ninv_U = Ninv[:, None] * U              # (n, k)

    Sigma = jnp.diag(1.0 / Phidiag) + U.T @ Ninv_U   # (k, k)
    Sigma_cf = jax.scipy.linalg.cho_factor(Sigma)

    # Σ^{-1} (U^T N^{-1} B)
    UtNinvB = U.T @ Ninv_B                  # (k, m)
    Sigma_inv_UtNinvB = jax.scipy.linalg.cho_solve(Sigma_cf, UtNinvB)  # (k, m)

    return Ninv_B - Ninv_U @ Sigma_inv_UtNinvB


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
    return jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, c, s],
        [0.0, -s, c],
    ])


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
        posepoch_int, posepoch_frac = params.epoch_value(posepoch_name)
        dt_int = toa_data.tdb_int - posepoch_int
        dt_frac = toa_data.tdb_frac - posepoch_frac
        dt_yr = (dt_int + dt_frac) / DAYS_PER_JULIAN_YEAR

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
        toa_data, params,
        raj_name=elong_name,
        decj_name=elat_name,
        pmra_name=pmelong_name,
        pmdec_name=pmelat_name,
        posepoch_name=posepoch_name,
    )
    rot = ecl_to_icrs_rotation(obliquity_arcsec)
    return L_hat_ecl @ rot
