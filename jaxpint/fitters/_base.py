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
from jaxpint.types.dual_float import DualFloat
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
    chi2: float
    dof: int

    @property
    def parameter_uncertainties(self) -> Float[Array, " n_free"]:
        """1-sigma marginal errors: square root of the covariance diagonal."""
        return jnp.sqrt(jnp.diag(self.covariance_matrix))

    @property
    def correlation_matrix(self) -> Float[Array, "n_free n_free"]:
        """Covariance rescaled to unit diagonal: ``D^-1 C D^-1`` with
        ``D = diag(sigma)``. Zero-variance rows/cols are left unscaled."""
        errors = jnp.sqrt(jnp.diag(self.covariance_matrix))
        errors_safe = jnp.where(errors == 0, 1.0, errors)
        return (self.covariance_matrix / errors_safe).T / errors_safe

    @property
    def reduced_chi2(self) -> float:
        """``chi2 / dof``, or NaN when ``dof <= 0``."""
        return self.chi2 / self.dof if self.dof > 0 else float("nan")


# ---------------------------------------------------------------------------
# Base fitter
# ---------------------------------------------------------------------------


class BaseFitter(ABC):
    """Base for all JaxPINT fitters.

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
        """Run the fit and return a result container.

        Subclasses narrow the return type to their specific result class
        (e.g. ``WLSFitResult``, ``GLSFitResult``).

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of Gauss-Newton iterations. Default is 1.
        **kwargs
            Subclass-specific options (e.g. ``threshold``, ``full_cov``).

        Returns
        -------
        BaseFitResult
            A dataclass containing updated parameters, covariance,
            uncertainties, chi-squared, and degrees of freedom.
        """
        ...


# ---------------------------------------------------------------------------
# Residuals & design matrix
# ---------------------------------------------------------------------------


def compute_phase_residuals(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
) -> Float[Array, " n_toas"]:
    """Compute phase residuals using nearest-pulse tracking.

    Returns the fractional part of the model phase (in cycles), after
    adjusting for ``delta_pulse_number`` offsets stored in the TOA data.

    Parameters
    ----------
    model : TimingModel
        JaxPINT timing model used to compute pulse phase.
    toa_data : TOAData
        Pre-extracted TOA data containing observation times and
        ``delta_pulse_number`` corrections.
    params : ParameterVector
        Current parameter values for the timing model.

    Returns
    -------
    residuals : jax.Array, shape (n_toas,)
        Phase residuals in cycles (fractional part of adjusted phase).
    """
    phase = model.compute_phase(toa_data, params)
    adjusted = DualFloat.from_cycles(
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

    Converts phase residuals (cycles) to time by dividing by the spin
    frequency F0 (Hz).

    Parameters
    ----------
    model : TimingModel
        JaxPINT timing model.
    toa_data : TOAData
        Pre-extracted TOA data.
    params : ParameterVector
        Current parameter values (must include ``F0``).

    Returns
    -------
    residuals : jax.Array, shape (n_toas,)
        Time residuals in seconds.
    """
    phase_resid = compute_phase_residuals(model, toa_data, params)
    f0 = params.param_value("F0")
    return phase_resid / f0


def compute_design_matrix(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
    include_offset: bool = True,
) -> Float[Array, "n_toas n_cols"]:
    """Build the design matrix via autodiff.

    Computes ``jax.jacobian`` of time residuals w.r.t. all parameter
    values, then extracts only the free-parameter columns.

    Following PINT's convention, the design matrix is negated so that
    ``M[i, j] = -d(time_resid_i) / d(param_j)``.  This ensures that
    the WLS update ``p_new = p_old + dpars`` reduces residuals.

    Parameters
    ----------
    model : TimingModel
        JaxPINT timing model.
    toa_data : TOAData
        Pre-extracted TOA data.
    params : ParameterVector
        Current parameter values. The ``frozen_mask`` attribute determines
        which columns (free parameters) appear in the output.
    include_offset : bool, optional
        If True (default, matches PINT's ``incoffset=True``), prepend a
        column of ones representing the constant-residual degree of
        freedom that absolute-phase ambiguity always introduces.  This
        column is mathematically necessary for any WLS-style downstream
        use (fitting, sensitivity-curve construction, marginalization).
        Auto-suppressed to False if the model already contains an
        explicit ``PhaseOffset`` component (``model.phoff_name is not
        None``); see ``PINT/src/pint/models/timing_model.py:2404``.

    Returns
    -------
    M : jax.Array
        Negated Jacobian of time residuals with respect to free parameters,
        with shape ``(n_toas, n_free + 1)`` if the offset column is included
        (Offset column first), else ``(n_toas, n_free)``.
    """
    if model.phoff_name is not None:
        include_offset = False
    J, M = _compute_jacobian_and_design(model, toa_data, params)
    if include_offset:
        offset_col = jnp.ones((M.shape[0], 1), dtype=M.dtype)
        M = jnp.concatenate([offset_col, M], axis=1)
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
    weighted_residuals = residuals / sigma
    weighted_design_matrix = M / sigma[:, None]

    # Normalize columns for numerical stability
    normalized_design_matrix, norms, _degenerate = normalize_designmatrix(
        weighted_design_matrix
    )

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
    """Compute the weighted chi-squared statistic.

    Calculates ``sum((residuals / sigma) ** 2)``.

    Parameters
    ----------
    residuals : jax.Array, shape (n_toas,)
        Time residuals in seconds.
    sigma : jax.Array, shape (n_toas,)
        TOA uncertainties in seconds.

    Returns
    -------
    chi2 : jax.Array, shape ()
        Scalar weighted chi-squared value.
    """
    return jnp.sum((residuals / sigma) ** 2)
