"""Base classes and shared pure-JAX functions for all fitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
try:
    from beartype import beartype
except ModuleNotFoundError:  # dev-only extra; without it jaxtyped is a no-op
    beartype = None
from jaxtyping import Array, Float, jaxtyped

from jaxpint.model import TimingModel
from jaxpint.types.dual_float import DualFloat
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import normalize_designmatrix, woodbury_dot, woodbury_solve


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


@dataclass
class IterationState:
    """One Gauss-Newton step's output, passed from ``_iteration`` to ``_build_result``.

    ``noise_realizations`` is ``None`` for fitters that don't estimate them (WLS).
    """

    params: ParameterVector
    covariance: Float[Array, "n_free n_free"]
    noise_realizations: Optional[Float[Array, " n_epochs"]] = None


# ---------------------------------------------------------------------------
# Base fitter
# ---------------------------------------------------------------------------


class BaseFitter(ABC):
    """Base for all JaxPINT fitters.

    Subclasses implement :meth:`_build_result` (turns the final
    :class:`IterationState` into a fitter-specific result) and :meth:`fit_toas`,
    whose body computes the SVD threshold, runs :meth:`_gauss_newton` with a
    per-fitter step callable, and returns ``self._build_result(state)``.

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

    def _dof(self, params: ParameterVector, n_data: int) -> int:
        """Degrees of freedom: data points minus free params minus the implicit
        constant-offset column.

        The offset column is suppressed (no ``-1``) when the model carries an
        explicit ``PhaseOffset`` (``phoff_name`` set), matching PINT. ``n_data``
        is ``n_toas`` for narrowband fits and ``2 * n_toas`` for wideband.
        """
        n_offset = 0 if self.model.phoff_name is not None else 1
        return n_data - params.n_free - n_offset

    def _gauss_newton(
        self,
        maxiter: int,
        threshold: float,
        iterate: Callable[[ParameterVector, float], IterationState],
    ) -> IterationState:
        """Run ``max(1, maxiter)`` Gauss-Newton steps and return the final state.

        ``iterate(params, threshold) -> IterationState`` is the per-fitter step,
        already bound to any fitter-specific options (e.g. ``full_cov``).
        """
        params = self.params
        state: Optional[IterationState] = None
        for _ in range(max(1, maxiter)):
            state = iterate(params, threshold)
            params = state.params
        assert state is not None  # max(1, maxiter) >= 1, so the loop always runs
        return state

    @abstractmethod
    def _build_result(self, state: IterationState) -> BaseFitResult:
        """Turn the final iteration state into a fitter-specific result."""
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


@jaxtyped(typechecker=beartype)
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
        p = params.with_values(all_values)
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


@jaxtyped(typechecker=beartype)
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


# ---------------------------------------------------------------------------
# Covariance-weighted (GLS) solve primitives, shared by GLSFitter and
# WidebandGLSFitter.  The plain-diagonal counterparts above are wls_step /
# compute_chi2 / _subtract_weighted_mean.
# ---------------------------------------------------------------------------


def _subtract_cov_weighted_mean(
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


@jaxtyped(typechecker=beartype)
def _normalized_svd_solve(
    mtcm: Float[Array, "p p"],
    mtcy: Float[Array, " p"],
    threshold: float,
) -> tuple[Float[Array, " p"], Float[Array, "p p"], Float[Array, " p"]]:
    """Solve the symmetric normal system ``mtcm @ x = mtcy`` via a
    column-normalized, SVD-thresholded pseudo-inverse.

    Columns are scaled by ``sqrt(diag(mtcm))`` before the SVD to improve
    conditioning, then unscaled on the way out.  Singular values at or below
    ``threshold * S_max`` are dropped: those directions get a zero update and
    zero variance (the standard SVD-cutoff convention for degenerate params).

    Returns ``(xhat, covariance, norms)``; ``norms`` are the column norms
    (diagnostic).
    """
    norms = jnp.sqrt(jnp.diag(mtcm))
    norms = jnp.where(norms == 0.0, 1.0, norms)
    mtcm_n = mtcm / norms / norms[:, None]
    mtcy_n = mtcy / norms

    U_svd, S, Vt = jnp.linalg.svd(mtcm_n, full_matrices=False)
    S_safe = jnp.where(S <= threshold * S[0], jnp.inf, S)

    xhat = (Vt.T @ ((U_svd.T @ mtcy_n) / S_safe)) / norms
    cov_n = (Vt.T / S_safe**2) @ Vt
    covariance = (cov_n / norms).T / norms
    return xhat, covariance, norms


@jaxtyped(typechecker=beartype)
def lstsq_step_fullcov(
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
    ``(n_free, n_free)`` normal equations via :func:`_normalized_svd_solve`.

    Parameters
    ----------
    residuals : jax.Array, shape (n_toas,)
        Time residuals in seconds (GLS-weighted mean already subtracted).
    Ndiag : jax.Array, shape (n_toas,)
        Diagonal of the white-noise covariance matrix (variances).
    U : jax.Array, shape (n_toas, n_epochs)
        Basis matrix for correlated noise components.
    Phidiag : jax.Array, shape (n_epochs,)
        Diagonal covariance of the correlated-noise random effects.
    M : jax.Array, shape (n_toas, n_free)
        Design matrix (free-parameter columns only).
    threshold : float
        Singular values below ``threshold * S_max`` are discarded.

    Returns
    -------
    dpars : jax.Array, shape (n_free,)
        Parameter updates.
    covariance : jax.Array, shape (n_free, n_free)
        Parameter covariance matrix.
    norms : jax.Array, shape (n_free,)
        Column norms used for normalisation (diagnostic).
    """
    # C^{-1} M  and  C^{-1} r  via Woodbury (never forms n_toas×n_toas)
    Mr = jnp.column_stack([M, residuals[:, None]])  # (n, n_free+1)
    Cinv_Mr = woodbury_solve(Ndiag, U, Phidiag, Mr)  # (n, n_free+1)
    Cinv_M = Cinv_Mr[:, :-1]  # (n, n_free)
    Cinv_r = Cinv_Mr[:, -1]  # (n,)

    mtcm = M.T @ Cinv_M  # (n_free, n_free)
    mtcy = M.T @ Cinv_r  # (n_free,)
    return _normalized_svd_solve(mtcm, mtcy, threshold)


@jaxtyped(typechecker=beartype)
def lstsq_step_augmented(
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
    """One GLS solve via the augmented (mixed-model) normal equations.

    The timing parameters are **fixed effects** (estimated by least squares) and the
    correlated-noise amplitudes are **random effects** with covariance ``Phi``.
    Stacking ``M_aug = [M | U]`` and solving

        (M_augᵀ N⁻¹ M_aug + diag(precision)) x = M_augᵀ N⁻¹ r

    yields the timing-parameter estimates and the random-effect predictions in
    one shot. It is algebraically equivalent to :func:`lstsq_step_fullcov`,
    which instead folds ``Phi`` into the noise covariance ``C = N + U Phi Uᵀ``
    and does plain GLS ; the two give identical ``dpars``.

    Parameters
    ----------
    residuals : jax.Array, shape (n_toas,)
        Time residuals in seconds (GLS-weighted mean already subtracted).
    Ndiag : jax.Array, shape (n_toas,)
        Diagonal of the white-noise covariance matrix (variances).
    U : jax.Array, shape (n_toas, n_epochs)
        Basis matrix for the correlated-noise random effects.
    Phidiag : jax.Array, shape (n_epochs,)
        Diagonal covariance of the correlated-noise random effects (the
        noise-process variance per basis column).
    M : jax.Array, shape (n_toas, n_free)
        Design matrix (free-parameter columns only).
    threshold : float
        Singular values below ``threshold * S_max`` are discarded.

    Returns
    -------
    dpars : jax.Array, shape (n_free,)
        Timing parameter updates (fixed-effect estimates).
    covariance : jax.Array, shape (n_free, n_free)
        Timing parameter covariance.
    norms : jax.Array, shape (n_free + n_epochs,)
        Column norms of the augmented system (diagnostic).
    noise_realizations : jax.Array, shape (n_epochs,)
        BLUP (best linear unbiased predictor) of the random-effect amplitudes.
    """
    n_free = M.shape[1]

    # Augmented design matrix
    M_aug = jnp.concatenate([M, U], axis=1)  # (n_toas, n_aug)

    # Diagonal weighting
    Ninv = 1.0 / Ndiag
    r_w = residuals * Ninv
    M_w = M_aug * Ninv[:, None]

    # M_aug^T N^{-1} M_aug
    mtcm = M_aug.T @ M_w  # (n_aug, n_aug)
    mtcy = M_aug.T @ r_w  # (n_aug,)

    # Mixed-model regularization: add each column's inverse variance to the
    # normal matrix (the G⁻¹ block of Henderson's equations). Random-effect
    # (noise) columns get 1/Phi; fixed-effect (timing) columns get ~0 -- a tiny
    # 1e-40 floor only to keep the matrix positive-definite -- so the timing fit
    # is unregularized GLS.
    precision_diag = jnp.concatenate(
        [
            jnp.full(n_free, 1e-40),
            1.0 / Phidiag,
        ]
    )
    mtcm = mtcm + jnp.diag(precision_diag)

    xhat, xvar, norms = _normalized_svd_solve(mtcm, mtcy, threshold)

    # Split into fixed-effect (timing) estimates and random-effect predictions.
    dpars = xhat[:n_free]
    noise_realizations = xhat[n_free:]
    covariance = xvar[:n_free, :n_free]

    return dpars, covariance, norms, noise_realizations


@jaxtyped(typechecker=beartype)
def compute_chi2_cov(
    residuals: Float[Array, " n_toas"],
    Ndiag: Float[Array, " n_toas"],
    U: Float[Array, "n_toas n_epochs"],
    Phidiag: Float[Array, " n_epochs"],
) -> Float[Array, ""]:
    """Compute the GLS chi-squared statistic: ``r^T C^{-1} r``.

    The covariance ``C = diag(N) + U diag(Phi) U^T`` is inverted via
    the Woodbury identity without forming the full matrix.

    Parameters
    ----------
    residuals : jax.Array, shape (n_toas,)
        Time residuals in seconds.
    Ndiag : jax.Array, shape (n_toas,)
        Diagonal of the white-noise covariance matrix (variances).
    U : jax.Array, shape (n_toas, n_epochs)
        Basis matrix for correlated noise components.
    Phidiag : jax.Array, shape (n_epochs,)
        Diagonal covariance of the correlated-noise random effects.

    Returns
    -------
    chi2 : jax.Array, shape ()
        Scalar GLS chi-squared value.
    """
    chi2, _ = woodbury_dot(Ndiag, U, Phidiag, residuals, residuals)
    return chi2
