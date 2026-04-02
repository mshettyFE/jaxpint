"""Weighted Least Squares fitter for JaxPINT.

Implements a Gauss-Newton WLS fitter that uses JAX autodiff for the
design matrix and JAX's native ``jnp.linalg.svd`` for the SVD solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.model import TimingModel
from jaxpint.phase_result import PhaseResult
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import normalize_designmatrix


# ---------------------------------------------------------------------------
# Pure JAX functions
# ---------------------------------------------------------------------------


def compute_phase_residuals(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
) -> Float[Array, " n_toas"]:
    """Compute phase residuals using nearest-pulse tracking.

    Returns the fractional part of the model phase (in cycles).
    """
    phase = model.compute_phase(toa_data, params)
    # Add delta_pulse_number before extracting fractional part
    adjusted = PhaseResult.create(
        phase.int + toa_data.delta_pulse_number,
        phase.frac,
    )
    return adjusted.frac


def compute_time_residuals(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
) -> Float[Array, " n_toas"]:
    """Compute time residuals in seconds.

    Converts phase residuals (cycles) to time by dividing by F0 (Hz).
    """
    phase_resid = compute_phase_residuals(model, toa_data, params)
    f0 = params.param_value("F0")
    return phase_resid / f0


def compute_design_matrix(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
) -> Float[Array, "n_toas n_free"]:
    """Build the design matrix via autodiff.

    Computes ``jax.jacobian`` of time residuals w.r.t. all parameter
    values, then extracts only the free-parameter columns.

    Following PINT's convention, the design matrix is negated so that
    ``M[i, j] = -d(time_resid_i) / d(param_j)``.  This ensures that
    the WLS update ``p_new = p_old + dpars`` reduces residuals.
    """

    def time_resid_fn(all_values: Float[Array, " n_params"]):
        p = eqx.tree_at(lambda pv: pv.values, params, all_values)
        return compute_time_residuals(model, toa_data, p)

    J = jax.jacobian(time_resid_fn)(params.values)  # (n_toas, n_params)
    # negative sign matches PINT convention
    return -J[:, params.free_mask_array()]



def _subtract_weighted_mean(
    residuals: Float[Array, " n"],
    sigma: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Subtract the weighted mean from *residuals*."""
    w = 1.0 / sigma**2
    wmean = jnp.sum(w * residuals) / jnp.sum(w)
    return residuals - wmean


def wls_step(
    residuals: Float[Array, " n_toas"],
    sigma: Float[Array, " n_toas"],
    M: Float[Array, "n_toas n_free"],
    threshold: float,
) -> tuple[
    Float[Array, " n_free"],
    Float[Array, "n_free n_free"],
    Float[Array, " n_free"],
]:
    """One WLS solve via SVD.

    Parameters
    ----------
    residuals : (n_toas,)
        Time residuals in seconds (mean already subtracted).
    sigma : (n_toas,)
        TOA uncertainties in seconds.
    M : (n_toas, n_free)
        Design matrix.
    threshold : float
        Singular values below ``threshold * S_max`` are discarded.

    Returns
    -------
    dpars : (n_free,)
        Parameter updates.
    covariance : (n_free, n_free)
        Parameter covariance matrix.
    norms : (n_free,)
        Column norms used for normalisation (diagnostic).
    """
    # Weight by inverse uncertainty
    r1 = residuals / sigma
    M1 = M / sigma[:, None]

    # Normalize columns for numerical stability
    M2, norms, _degenerate = normalize_designmatrix(M1)

    # SVD via JAX
    U, S, Vt = jnp.linalg.svd(M2, full_matrices=False)

    # Threshold degenerate singular values
    S_safe = jnp.where(S <= threshold * S[0], jnp.inf, S)

    # Parameter updates
    dpars = (Vt.T @ ((U.T @ r1) / S_safe)) / norms

    # Covariance matrix
    Sigma_norm = (Vt.T / S_safe**2) @ Vt
    covariance = (Sigma_norm / norms).T / norms

    return dpars, covariance, norms


def compute_chi2(
    residuals: Float[Array, " n_toas"],
    sigma: Float[Array, " n_toas"],
) -> Float[Array, ""]:
    """Weighted chi-squared."""
    return jnp.sum((residuals / sigma) ** 2)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class WLSFitResult:
    """Result of a WLS fit."""

    params: ParameterVector
    covariance_matrix: Float[Array, "n_free n_free"]
    correlation_matrix: Float[Array, "n_free n_free"]
    parameter_uncertainties: Float[Array, " n_free"]
    chi2: float
    dof: int
    reduced_chi2: float
    residuals: Float[Array, " n_toas"]


# ---------------------------------------------------------------------------
# Fitter class
# ---------------------------------------------------------------------------


class WLSFitter:
    """Weighted Least Squares fitter (Gauss-Newton with SVD).

    Parameters
    ----------
    model : TimingModel
        JaxPINT timing model.
    toa_data : TOAData
        Pre-extracted TOA data.
    params : ParameterVector
        Initial parameter values (free/frozen flags determine what is fit).
    """

    def __init__(
        self,
        model: TimingModel,
        toa_data: TOAData,
        params: ParameterVector,
    ):
        self.model = model
        self.toa_data = toa_data
        self.params = params
        self.result: Optional[WLSFitResult] = None

    def fit_toas(
        self,
        maxiter: int = 1,
        threshold: Optional[float] = None,
    ) -> float:
        """Run the WLS fit.

        Parameters
        ----------
        maxiter : int
            Number of Gauss-Newton iterations.
        threshold : float, optional
            SVD threshold (default ``1e-14 * max(n_toas, n_free)``).

        Returns
        -------
        float
            Final chi-squared value.
        """
        if threshold is None:
            threshold = 1e-14 * max(self.toa_data.n_toas, self.params.n_free)

        sigma = self.toa_data.error
        covariance = None
        norms = None

        for _ in range(maxiter):
            # 1. Time residuals
            time_resid = compute_time_residuals(
                self.model, self.toa_data, self.params
            )

            # 2. Subtract weighted mean
            time_resid = _subtract_weighted_mean(time_resid, sigma)

            # 3. Design matrix (autodiff, no mean subtraction)
            M = compute_design_matrix(self.model, self.toa_data, self.params)

            # 4. WLS SVD solve
            dpars, covariance, norms = wls_step(time_resid, sigma, M, threshold)

            # 5. Update parameters
            new_free = self.params.free_values() + dpars
            self.params = self.params.with_free_values(new_free)

        # Final residuals and chi2
        final_resid = compute_time_residuals(
            self.model, self.toa_data, self.params
        )
        final_resid = _subtract_weighted_mean(final_resid, sigma)
        chi2_val = float(compute_chi2(final_resid, sigma))
        dof = self.toa_data.n_toas - self.params.n_free

        # Uncertainties and correlation from last iteration's covariance
        errors = jnp.sqrt(jnp.diag(covariance))
        correlation = (covariance / errors).T / errors

        self.result = WLSFitResult(
            params=self.params,
            covariance_matrix=covariance,
            correlation_matrix=correlation,
            parameter_uncertainties=errors,
            chi2=chi2_val,
            dof=dof,
            reduced_chi2=chi2_val / dof,
            residuals=final_resid,
        )

        return chi2_val
