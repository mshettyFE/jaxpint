"""WLS and GLS fitters for JaxPINT.

Implements Gauss-Newton fitters that use JAX autodiff for the design
matrix and JAX's native linear-algebra routines for the solve step.

* ``WLSFitter`` — Weighted Least Squares (diagonal covariance).
* ``GLSFitter`` — Generalised Least Squares (supports ECORR via
  low-rank covariance updates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.phase_result import PhaseResult
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import normalize_designmatrix, woodbury_dot, woodbury_solve


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
    noise_model : NoiseModel, optional
        Noise model (only diagonal / white-noise part is used by WLS).
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
        self.result: Optional[WLSFitResult] = None

    def _get_sigma(self) -> Float[Array, " n_toas"]:
        """Return noise-scaled TOA uncertainties."""
        if self.noise_model is not None:
            return self.noise_model.scaled_sigma(self.toa_data, self.params)
        return self.toa_data.error

    def _iteration(self, threshold: float) -> Float[Array, "n_free n_free"]:
        """Run one Gauss-Newton iteration; returns covariance."""
        sigma = self._get_sigma()

        time_resid = compute_time_residuals(
            self.model, self.toa_data, self.params
        )
        time_resid = _subtract_weighted_mean(time_resid, sigma)

        M = compute_design_matrix(self.model, self.toa_data, self.params)
        dpars, covariance, _norms = wls_step(time_resid, sigma, M, threshold)

        new_free = self.params.free_values() + dpars
        self.params = self.params.with_free_values(new_free)
        return covariance

    def _build_result(
        self, covariance: Float[Array, "n_free n_free"],
    ) -> float:
        """Compute final residuals/chi2 and populate ``self.result``."""
        sigma = self._get_sigma()

        final_resid = compute_time_residuals(
            self.model, self.toa_data, self.params
        )
        final_resid = _subtract_weighted_mean(final_resid, sigma)
        chi2_val = float(compute_chi2(final_resid, sigma))
        dof = self.toa_data.n_toas - self.params.n_free

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

        covariance = None
        for _ in range(maxiter):
            covariance = self._iteration(threshold)

        return self._build_result(covariance)


# ---------------------------------------------------------------------------
# GLS (Generalised Least Squares) fitter
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
        self.result: Optional[GLSFitResult] = None

    def _get_noise(
        self,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_basis"],
        Float[Array, " n_basis"],
    ]:
        """Return (sigma, Ndiag, U, Phidiag) for the current parameters."""
        if self.noise_model is not None:
            sigma = self.noise_model.scaled_sigma(self.toa_data, self.params)
            Ndiag, U, Phidiag = self.noise_model.covariance(
                self.toa_data, self.params
            )
        else:
            sigma = self.toa_data.error
            Ndiag = sigma ** 2
            U = jnp.zeros((self.toa_data.n_toas, 0))
            Phidiag = jnp.zeros(0)

        return sigma, Ndiag, U, Phidiag

    def _iteration(
        self,
        threshold: float,
        full_cov: bool,
    ) -> tuple[
        Float[Array, "n_free n_free"],
        Optional[Float[Array, " n_basis"]],
    ]:
        """Run one Gauss-Newton iteration; returns (covariance, noise_realizations)."""
        sigma, Ndiag, U, Phidiag = self._get_noise()
        n_basis = U.shape[1]

        time_resid = compute_time_residuals(
            self.model, self.toa_data, self.params
        )
        if n_basis > 0:
            time_resid = _subtract_gls_weighted_mean(
                time_resid, Ndiag, U, Phidiag
            )
        else:
            time_resid = _subtract_weighted_mean(time_resid, sigma)

        M = compute_design_matrix(self.model, self.toa_data, self.params)

        # GLS solve
        noise_realizations = None
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

        new_free = self.params.free_values() + dpars
        self.params = self.params.with_free_values(new_free)
        return covariance, noise_realizations

    def _build_result(
        self,
        covariance: Float[Array, "n_free n_free"],
        noise_realizations: Optional[Float[Array, " n_basis"]],
    ) -> float:
        """Compute final residuals/chi2 and populate ``self.result``."""
        sigma, Ndiag, U, Phidiag = self._get_noise()
        n_basis = U.shape[1]

        final_resid = compute_time_residuals(
            self.model, self.toa_data, self.params
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

        dof = self.toa_data.n_toas - self.params.n_free

        errors = jnp.sqrt(jnp.diag(covariance))
        correlation = (covariance / errors).T / errors

        self.result = GLSFitResult(
            params=self.params,
            covariance_matrix=covariance,
            correlation_matrix=correlation,
            parameter_uncertainties=errors,
            chi2=chi2_val,
            dof=dof,
            reduced_chi2=chi2_val / dof,
            residuals=final_resid,
            noise_realizations=noise_realizations,
        )
        return chi2_val

    def fit_toas(
        self,
        maxiter: int = 1,
        threshold: Optional[float] = None,
        full_cov: bool = False,
    ) -> float:
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
        float
            Final chi-squared value.
        """
        n_toas = self.toa_data.n_toas

        # Determine basis dimension for threshold calculation
        if self.noise_model is not None and self.noise_model.has_correlated:
            _, _, U_init, _ = self._get_noise()
            n_basis = U_init.shape[1]
        else:
            n_basis = 0

        if threshold is None:
            if full_cov:
                dim = self.params.n_free
            else:
                dim = self.params.n_free + n_basis
            threshold = 1e-14 * max(n_toas, dim)

        covariance = None
        noise_realizations = None
        for _ in range(maxiter):
            covariance, noise_realizations = self._iteration(
                threshold, full_cov
            )

        return self._build_result(covariance, noise_realizations)
