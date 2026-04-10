"""Base classes and shared pure-JAX functions for all fitters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.model import TimingModel
from jaxpint.dual_float import DualFloat
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import normalize_designmatrix


# ---------------------------------------------------------------------------
# Base fit result
# ---------------------------------------------------------------------------


@dataclass
class BaseFitResult:
    """Common fields shared by all fit results."""

    params: ParameterVector
    covariance_matrix: Float[Array, "n_free n_free"]
    correlation_matrix: Float[Array, "n_free n_free"]
    parameter_uncertainties: Float[Array, " n_free"]
    chi2: float
    dof: int
    reduced_chi2: float


# ---------------------------------------------------------------------------
# Base fitter
# ---------------------------------------------------------------------------


class BaseFitter(ABC):
    """Abstract base for all JaxPINT fitters.

    Subclasses must implement :meth:`fit_toas` (the main entry point)
    and typically also implement ``_iteration`` and ``_build_result``.

    Parameters
    ----------
    model : TimingModel
        JaxPINT timing model.
    toa_data : TOAData
        Pre-extracted TOA data.
    params : ParameterVector
        Initial parameter values (free/frozen flags determine what is fit).
    noise_model : NoiseModel, optional
        Noise model.
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

    @abstractmethod
    def fit_toas(self, maxiter: int = 1, **kwargs) -> BaseFitResult:
        """Run the fit. Subclasses narrow the return type to their specific result."""
        ...

    @staticmethod
    def _covariance_to_correlation(
        covariance: Float[Array, " n n"],
    ) -> tuple[Float[Array, " n"], Float[Array, " n n"]]:
        """Compute parameter uncertainties and correlation matrix from covariance."""
        errors = jnp.sqrt(jnp.diag(covariance))
        errors_safe = jnp.where(errors == 0, 1.0, errors)
        correlation = (covariance / errors_safe).T / errors_safe
        return errors, correlation

    @staticmethod
    def _reduced_chi2(chi2_val: float, dof: int) -> float:
        """Compute reduced chi-squared, returning NaN if dof <= 0."""
        return chi2_val / dof if dof > 0 else float("nan")


# ---------------------------------------------------------------------------
# Residuals & design matrix
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
    adjusted = DualFloat.cycles(
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
    J, M = _compute_jacobian_and_design(model, toa_data, params)
    return M


@eqx.filter_jit
def _compute_jacobian_and_design(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
) -> tuple[Float[Array, "n_toas n_params"], Float[Array, "n_toas n_free"]]:
    """JIT-compiled Jacobian and design matrix computation."""
    free_indices = params.free_indices_array()

    def time_resid_fn(all_values: Float[Array, " n_params"]):
        p = eqx.tree_at(lambda pv: pv.values, params, all_values)
        return compute_time_residuals(model, toa_data, p)

    J = jax.jacobian(time_resid_fn)(params.values)
    M = -J[:, free_indices]
    return J, M


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


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
    weighted_residuals = residuals / sigma
    weighted_design_matrix = M / sigma[:, None]

    # Normalize columns for numerical stability
    normalized_design_matrix, norms, _degenerate = normalize_designmatrix(
        weighted_design_matrix
    )

    # SVD via JAX
    U, S, Vt = jnp.linalg.svd(normalized_design_matrix, full_matrices=False)

    # Threshold degenerate singular values
    S_safe = jnp.where(S <= threshold * S[0], jnp.inf, S)

    # Parameter updates
    dpars = (Vt.T @ ((U.T @ weighted_residuals) / S_safe)) / norms

    # Covariance matrix
    cov_normalized = (Vt.T / S_safe**2) @ Vt
    covariance = (cov_normalized / norms).T / norms

    return dpars, covariance, norms


def compute_chi2(
    residuals: Float[Array, " n_toas"],
    sigma: Float[Array, " n_toas"],
) -> Float[Array, ""]:
    """Weighted chi-squared."""
    return jnp.sum((residuals / sigma) ** 2)
