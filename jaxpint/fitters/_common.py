"""Shared pure-JAX functions used by all fitters."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.model import TimingModel
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import normalize_designmatrix


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
