"""Generalised Least Squares fitter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import woodbury_dot, woodbury_solve

from ._common import (
    compute_time_residuals,
    _subtract_weighted_mean,
    wls_step,
    compute_chi2,
)


# ---------------------------------------------------------------------------
# GLS helper functions
# ---------------------------------------------------------------------------


def _subtract_gls_weighted_mean(
    residuals: Float[Array, " n"],
    Ndiag: Float[Array, " n"],
    U: Float[Array, "n k"],
    Phidiag: Float[Array, " k"],
) -> Float[Array, " n"]:
    """Subtract the GLS-weighted mean from *residuals*.

    The GLS weighted mean is ``(1^T C^{-1} r) / (1^T C^{-1} 1)``
    where ``C = diag(N) + U diag(Phi) U^T``.
    """
    ones = jnp.ones_like(residuals)
    numerator, _ = woodbury_dot(Ndiag, U, Phidiag, ones, residuals)
    denominator, _ = woodbury_dot(Ndiag, U, Phidiag, ones, ones)
    wmean = numerator / denominator
    return residuals - wmean


def gls_step_fullcov(
    residuals: Float[Array, " n_toas"],
    Ndiag: Float[Array, " n_toas"],
    U: Float[Array, "n_toas n_epochs"],
    Phidiag: Float[Array, " n_epochs"],
    M: Float[Array, "n_toas n_free"],
    threshold: float,
) -> tuple[
    Float[Array, " n_free"],
    Float[Array, "n_free n_free"],
    Float[Array, " n_free"],
]:
    """One GLS solve via full (Woodbury) covariance inversion + SVD.

    Computes ``M^T C^{-1} M`` and ``M^T C^{-1} r`` using
    :func:`~jaxpint.utils.woodbury_solve`, then SVD-solves the
    ``(n_free, n_free)`` normal equations.

    Returns
    -------
    dpars : (n_free,)
    covariance : (n_free, n_free)
    norms : (n_free,)
    """
    # C^{-1} M  and  C^{-1} r  via Woodbury (never forms n_toas×n_toas)
    Mr = jnp.column_stack([M, residuals[:, None]])          # (n, n_free+1)
    Cinv_Mr = woodbury_solve(Ndiag, U, Phidiag, Mr)         # (n, n_free+1)
    Cinv_M = Cinv_Mr[:, :-1]                                # (n, n_free)
    Cinv_r = Cinv_Mr[:, -1]                                 # (n,)

    mtcm = M.T @ Cinv_M                                     # (n_free, n_free)
    mtcy = M.T @ Cinv_r                                     # (n_free,)

    # Normalize for numerical stability
    # normalize_designmatrix works on (n, p) — we want column norms of mtcm
    # but mtcm is square (n_free, n_free).  Use column norms directly.
    col_norms = jnp.sqrt(jnp.diag(mtcm))
    col_norms = jnp.where(col_norms == 0.0, 1.0, col_norms)
    mtcm_normalized = mtcm / col_norms / col_norms[:, None]
    mtcy_normalized = mtcy / col_norms

    # SVD solve of the (n_free, n_free) system
    U_svd, S, Vt = jnp.linalg.svd(mtcm_normalized, full_matrices=False)
    S_safe = jnp.where(S <= threshold * S[0], jnp.inf, S)

    dpars_normalized = Vt.T @ ((U_svd.T @ mtcy_normalized) / S_safe)
    dpars = dpars_normalized / col_norms

    cov_normalized = (Vt.T / S_safe**2) @ Vt
    covariance = (cov_normalized / col_norms).T / col_norms

    return dpars, covariance, col_norms


def gls_step_augmented(
    residuals: Float[Array, " n_toas"],
    Ndiag: Float[Array, " n_toas"],
    U: Float[Array, "n_toas n_epochs"],
    Phidiag: Float[Array, " n_epochs"],
    M: Float[Array, "n_toas n_free"],
    threshold: float,
) -> tuple[
    Float[Array, " n_free"],
    Float[Array, "n_free n_free"],
    Float[Array, " n_aug"],
    Float[Array, " n_epochs"],
]:
    """One GLS solve via the augmented design-matrix approach.

    Augments the design matrix as ``M_aug = [M | U]`` and solves with
    diagonal weighting ``N^{-1}`` plus a prior on noise amplitudes.

    Returns
    -------
    dpars : (n_free,)
        Timing parameter updates.
    covariance : (n_free, n_free)
        Timing parameter covariance.
    norms : (n_free + n_epochs,)
        Column norms of the augmented system (diagnostic).
    noise_realizations : (n_epochs,)
        MAP noise amplitude estimates.
    """
    n_free = M.shape[1]
    n_epochs = U.shape[1]
    n_aug = n_free + n_epochs

    # Augmented design matrix
    M_aug = jnp.concatenate([M, U], axis=1)                 # (n_toas, n_aug)

    # Diagonal weighting
    Ninv = 1.0 / Ndiag
    r_w = residuals * Ninv
    M_w = M_aug * Ninv[:, None]

    # M_aug^T N^{-1} M_aug
    mtcm = M_aug.T @ M_w                                    # (n_aug, n_aug)
    mtcy = M_aug.T @ r_w                                    # (n_aug,)

    # Add prior: uninformative (1e-40) on timing cols, 1/Phi on noise cols
    prior_inv = jnp.concatenate([
        jnp.full(n_free, 1e-40),
        1.0 / Phidiag,
    ])
    mtcm = mtcm + jnp.diag(prior_inv)

    # Normalize columns
    norms = jnp.sqrt(jnp.diag(mtcm))
    norms = jnp.where(norms == 0.0, 1.0, norms)
    mtcm_normalized = mtcm / norms / norms[:, None]
    mtcy_normalized = mtcy / norms

    # SVD solve
    U_svd, S, Vt = jnp.linalg.svd(mtcm_normalized, full_matrices=False)
    S_safe = jnp.where(S <= threshold * S[0], jnp.inf, S)

    xhat_normalized = Vt.T @ ((U_svd.T @ mtcy_normalized) / S_safe)
    xhat = xhat_normalized / norms

    xvar_normalized = (Vt.T / S_safe**2) @ Vt
    xvar = (xvar_normalized / norms).T / norms

    # Extract timing parameters and noise realizations
    dpars = xhat[:n_free]
    noise_realizations = xhat[n_free:]
    covariance = xvar[:n_free, :n_free]

    return dpars, covariance, norms, noise_realizations


def compute_gls_chi2(
    residuals: Float[Array, " n_toas"],
    Ndiag: Float[Array, " n_toas"],
    U: Float[Array, "n_toas n_epochs"],
    Phidiag: Float[Array, " n_epochs"],
) -> Float[Array, ""]:
    """GLS chi-squared: ``r^T C^{-1} r``."""
    chi2, _ = woodbury_dot(Ndiag, U, Phidiag, residuals, residuals)
    return chi2


# ---------------------------------------------------------------------------
# JIT-compiled iteration core
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _gls_iteration_core(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
    noise_model: Optional[NoiseModel],
    threshold: float,
    full_cov: bool,
) -> tuple[
    Float[Array, " n_params"],
    Float[Array, "n_free n_free"],
    Float[Array, " n_basis"],
]:
    """JIT-compiled core of one GLS Gauss-Newton iteration.

    Returns updated parameter values, covariance, and noise realizations.
    """
    free_indices = params.free_indices_array()

    if noise_model is not None:
        sigma = noise_model.scaled_sigma(toa_data, params)
        Ndiag, U, Phidiag = noise_model.covariance(toa_data, params)
    else:
        sigma = toa_data.error
        Ndiag = sigma ** 2
        U = jnp.zeros((toa_data.n_toas, 0))
        Phidiag = jnp.zeros(0)

    n_basis = U.shape[1]

    time_resid = compute_time_residuals(model, toa_data, params)
    if n_basis > 0:
        time_resid = _subtract_gls_weighted_mean(
            time_resid, Ndiag, U, Phidiag
        )
    else:
        time_resid = _subtract_weighted_mean(time_resid, sigma)

    def time_resid_fn(all_values: Float[Array, " n_params"]):
        p = eqx.tree_at(lambda pv: pv.values, params, all_values)
        return compute_time_residuals(model, toa_data, p)

    J = jax.jacobian(time_resid_fn)(params.values)
    M = -J[:, free_indices]

    noise_realizations = jnp.zeros(0)
    if full_cov:
        dpars, covariance, _norms = gls_step_fullcov(
            time_resid, Ndiag, U, Phidiag, M, threshold
        )
    elif n_basis > 0:
        dpars, covariance, _norms, noise_realizations = (
            gls_step_augmented(
                time_resid, Ndiag, U, Phidiag, M, threshold
            )
        )
    else:
        dpars, covariance, _norms = wls_step(
            time_resid, sigma, M, threshold
        )

    new_values = params.values.at[free_indices].set(
        params.values[free_indices] + dpars
    )
    return new_values, covariance, noise_realizations


# ---------------------------------------------------------------------------
# GLS result container
# ---------------------------------------------------------------------------


@dataclass
class GLSFitResult:
    """Result of a GLS fit."""

    params: ParameterVector
    covariance_matrix: Float[Array, "n_free n_free"]
    correlation_matrix: Float[Array, "n_free n_free"]
    parameter_uncertainties: Float[Array, " n_free"]
    chi2: float
    dof: int
    reduced_chi2: float
    residuals: Float[Array, " n_toas"]
    noise_realizations: Optional[Float[Array, " n_epochs"]]


# ---------------------------------------------------------------------------
# GLS fitter class
# ---------------------------------------------------------------------------


class GLSFitter:
    """Generalised Least Squares fitter.

    The fitter is an immutable configuration container.  Calling
    :meth:`fit_toas` returns a :class:`GLSFitResult` without mutating
    the fitter itself.

    Supports arbitrary correlated noise sources (ECORR, red noise, etc.)
    through the :class:`~jaxpint.noise.NoiseModel` interface.  When no
    correlated components are present the GLS fitter reduces to WLS.

    Parameters
    ----------
    model : TimingModel
        JaxPINT timing model.
    toa_data : TOAData
        Pre-extracted TOA data.
    params : ParameterVector
        Initial parameter values.
    noise_model : NoiseModel, optional
        Noise model containing white and/or correlated components.
    """

    def __init__(
        self,
        model: TimingModel,
        toa_data: TOAData,
        params: ParameterVector,
        noise_model: Optional[NoiseModel] = None,
    ):
        self.model = model
        self.toa_data = toa_data
        self.params = params
        self.noise_model = noise_model

    def _get_noise(
        self,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_basis"],
        Float[Array, " n_basis"],
    ]:
        """Return (sigma, Ndiag, U, Phidiag) for the given parameters."""
        if self.noise_model is not None:
            sigma = self.noise_model.scaled_sigma(self.toa_data, params)
            Ndiag, U, Phidiag = self.noise_model.covariance(
                self.toa_data, params
            )
        else:
            sigma = self.toa_data.error
            Ndiag = sigma ** 2
            U = jnp.zeros((self.toa_data.n_toas, 0))
            Phidiag = jnp.zeros(0)

        return sigma, Ndiag, U, Phidiag

    def _iteration(
        self,
        params: ParameterVector,
        threshold: float,
        full_cov: bool,
    ) -> tuple[
        ParameterVector,
        Float[Array, "n_free n_free"],
        Optional[Float[Array, " n_basis"]],
    ]:
        """Run one Gauss-Newton iteration.

        Returns
        -------
        (new_params, covariance, noise_realizations)
        """
        new_values, covariance, noise_real = _gls_iteration_core(
            self.model, self.toa_data, params,
            self.noise_model, threshold, full_cov,
        )
        new_params = eqx.tree_at(lambda pv: pv.values, params, new_values)
        noise_realizations = noise_real if noise_real.size > 0 else None
        return new_params, covariance, noise_realizations

    def _build_result(
        self,
        params: ParameterVector,
        covariance: Float[Array, "n_free n_free"],
        noise_realizations: Optional[Float[Array, " n_basis"]],
    ) -> GLSFitResult:
        """Compute final residuals/chi2 and return a result object."""
        sigma, Ndiag, U, Phidiag = self._get_noise(params)
        n_basis = U.shape[1]

        final_resid = compute_time_residuals(
            self.model, self.toa_data, params
        )
        if n_basis > 0:
            final_resid = _subtract_gls_weighted_mean(
                final_resid, Ndiag, U, Phidiag
            )
        else:
            final_resid = _subtract_weighted_mean(final_resid, sigma)

        if n_basis > 0:
            chi2_val = float(compute_gls_chi2(final_resid, Ndiag, U, Phidiag))
        else:
            chi2_val = float(compute_chi2(final_resid, sigma))

        dof = self.toa_data.n_toas - params.n_free

        errors = jnp.sqrt(jnp.diag(covariance))
        errors_safe = jnp.where(errors==0, 1.0, errors)
        correlation = (covariance / errors_safe).T / errors_safe

        reduced_chi2 = chi2_val / dof if dof >0 else float('nan')

        return GLSFitResult(
            params=params,
            covariance_matrix=covariance,
            correlation_matrix=correlation,
            parameter_uncertainties=errors,
            chi2=chi2_val,
            dof=dof,
            reduced_chi2=reduced_chi2,
            residuals=final_resid,
            noise_realizations=noise_realizations,
        )

    def fit_toas(
        self,
        maxiter: int = 1,
        threshold: Optional[float] = None,
        full_cov: bool = False,
    ) -> GLSFitResult:
        """Run the GLS fit.

        Parameters
        ----------
        maxiter : int
            Number of Gauss-Newton iterations.
        threshold : float, optional
            SVD threshold (default ``1e-14 * dim``).
        full_cov : bool
            If True, use Woodbury-based full covariance inversion.
            If False (default), use the augmented design-matrix approach.

        Returns
        -------
        GLSFitResult
            Fit result containing updated parameters, covariance,
            uncertainties, chi-squared, residuals, and noise realizations.
        """
        n_toas = self.toa_data.n_toas

        # Determine basis dimension for threshold calculation
        if self.noise_model is not None and self.noise_model.has_correlated:
            _, _, U_init, _ = self._get_noise(self.params)
            n_basis = U_init.shape[1]
        else:
            n_basis = 0

        if threshold is None:
            if full_cov:
                dim = self.params.n_free
            else:
                dim = self.params.n_free + n_basis
            threshold = 1e-14 * max(n_toas, dim)

        safe_maxiter =  1 if maxiter < 1 else maxiter
        params = self.params
        covariance = None
        noise_realizations = None
        for _ in range(safe_maxiter):
            params, covariance, noise_realizations = self._iteration(
                params, threshold, full_cov
            )

        return self._build_result(params, covariance, noise_realizations)
