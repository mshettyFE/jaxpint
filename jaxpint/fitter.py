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
from jaxpint.noise import EcorrNoise, ScaleToaError
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
        noise_model: Optional[ScaleToaError] = None,
    ):
        self.model = model
        self.toa_data = toa_data
        self.params = params
        self.noise_model = noise_model
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

        covariance = None
        norms = None

        for _ in range(maxiter):
            # 0. Compute scaled uncertainties (recomputed each iteration
            #    in case noise params are being fit)
            if self.noise_model is not None:
                sigma = self.noise_model.scaled_sigma(
                    self.toa_data, self.params
                )
            else:
                sigma = self.toa_data.error

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

        # Final residuals and chi2 with final noise-scaled sigma
        if self.noise_model is not None:
            sigma = self.noise_model.scaled_sigma(self.toa_data, self.params)
        else:
            sigma = self.toa_data.error

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
    M2, norms, _degen = normalize_designmatrix(mtcm)
    # normalize_designmatrix works on (n, p) — we want column norms of mtcm
    # but mtcm is square (n_free, n_free).  Use column norms directly.
    norms_col = jnp.sqrt(jnp.diag(mtcm))
    norms_col = jnp.where(norms_col == 0.0, 1.0, norms_col)
    mtcm_n = mtcm / norms_col / norms_col[:, None]
    mtcy_n = mtcy / norms_col

    # SVD solve of the (n_free, n_free) system
    Usv, S, Vt = jnp.linalg.svd(mtcm_n, full_matrices=False)
    S_safe = jnp.where(S <= threshold * S[0], jnp.inf, S)

    dpars_n = Vt.T @ ((Usv.T @ mtcy_n) / S_safe)
    dpars = dpars_n / norms_col

    Sigma_n = (Vt.T / S_safe**2) @ Vt
    covariance = (Sigma_n / norms_col).T / norms_col

    return dpars, covariance, norms_col


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
    mtcm_n = mtcm / norms / norms[:, None]
    mtcy_n = mtcy / norms

    # SVD solve
    Usv, S, Vt = jnp.linalg.svd(mtcm_n, full_matrices=False)
    S_safe = jnp.where(S <= threshold * S[0], jnp.inf, S)

    xhat_n = Vt.T @ ((Usv.T @ mtcy_n) / S_safe)
    xhat = xhat_n / norms

    xvar_n = (Vt.T / S_safe**2) @ Vt
    xvar = (xvar_n / norms).T / norms

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
    """Generalised Least Squares fitter (supports ECORR).

    When no ``ecorr_noise`` is provided the GLS fitter reduces to
    standard WLS (diagonal covariance).

    Parameters
    ----------
    model : TimingModel
        JaxPINT timing model.
    toa_data : TOAData
        Pre-extracted TOA data.
    params : ParameterVector
        Initial parameter values.
    noise_model : ScaleToaError, optional
        White noise model (EFAC/EQUAD).
    ecorr_noise : EcorrNoise, optional
        Correlated noise model (ECORR).
    """

    def __init__(
        self,
        model: TimingModel,
        toa_data: TOAData,
        params: ParameterVector,
        noise_model: Optional[ScaleToaError] = None,
        ecorr_noise: Optional[EcorrNoise] = None,
    ):
        self.model = model
        self.toa_data = toa_data
        self.params = params
        self.noise_model = noise_model
        self.ecorr_noise = ecorr_noise
        self.result: Optional[GLSFitResult] = None

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

        # Set up ECORR components (or empty arrays for pure WLS)
        if self.ecorr_noise is not None:
            U = self.ecorr_noise.quantization_matrix
        else:
            U = jnp.zeros((n_toas, 0))

        n_epochs = U.shape[1]

        if threshold is None:
            if full_cov:
                dim = self.params.n_free
            else:
                dim = self.params.n_free + n_epochs
            threshold = 1e-14 * max(n_toas, dim)

        covariance = None
        noise_realizations = None

        for _ in range(maxiter):
            # 0. White noise
            if self.noise_model is not None:
                sigma = self.noise_model.scaled_sigma(
                    self.toa_data, self.params
                )
            else:
                sigma = self.toa_data.error
            Ndiag = sigma ** 2

            # 1. ECORR weights
            if self.ecorr_noise is not None:
                Phidiag = self.ecorr_noise.ecorr_weights(self.params)
            else:
                Phidiag = jnp.zeros(0)

            # 2. Time residuals
            time_resid = compute_time_residuals(
                self.model, self.toa_data, self.params
            )

            # 3. Subtract GLS weighted mean
            if n_epochs > 0:
                time_resid = _subtract_gls_weighted_mean(
                    time_resid, Ndiag, U, Phidiag
                )
            else:
                time_resid = _subtract_weighted_mean(time_resid, sigma)

            # 4. Design matrix
            M = compute_design_matrix(
                self.model, self.toa_data, self.params
            )

            # 5. GLS solve
            if full_cov:
                dpars, covariance, _norms = gls_step_fullcov(
                    time_resid, Ndiag, U, Phidiag, M, threshold
                )
                noise_realizations = None
            else:
                if n_epochs > 0:
                    dpars, covariance, _norms, noise_realizations = (
                        gls_step_augmented(
                            time_resid, Ndiag, U, Phidiag, M, threshold
                        )
                    )
                else:
                    # Pure WLS path — no augmentation needed
                    dpars, covariance, _norms = wls_step(
                        time_resid, sigma, M, threshold
                    )
                    noise_realizations = None

            # 6. Update parameters
            new_free = self.params.free_values() + dpars
            self.params = self.params.with_free_values(new_free)

        # Final residuals and chi2
        if self.noise_model is not None:
            sigma = self.noise_model.scaled_sigma(self.toa_data, self.params)
        else:
            sigma = self.toa_data.error
        Ndiag = sigma ** 2

        if self.ecorr_noise is not None:
            Phidiag = self.ecorr_noise.ecorr_weights(self.params)
        else:
            Phidiag = jnp.zeros(0)

        final_resid = compute_time_residuals(
            self.model, self.toa_data, self.params
        )
        if n_epochs > 0:
            final_resid = _subtract_gls_weighted_mean(
                final_resid, Ndiag, U, Phidiag
            )
        else:
            final_resid = _subtract_weighted_mean(final_resid, sigma)

        if n_epochs > 0:
            chi2_val = float(compute_gls_chi2(final_resid, Ndiag, U, Phidiag))
        else:
            chi2_val = float(compute_chi2(final_resid, sigma))

        dof = n_toas - self.params.n_free

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
