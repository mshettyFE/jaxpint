"""Wideband Generalised Least Squares fitter."""
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

from ._base import (
    BaseFitter,
    BaseFitResult,
    compute_time_residuals,
    _subtract_weighted_mean,
    wls_step,
    compute_chi2,
)
from .gls import (
    _subtract_gls_weighted_mean,
    gls_step_fullcov,
    gls_step_augmented,
    compute_gls_chi2,
)


# ---------------------------------------------------------------------------
# Wideband residual / design-matrix functions
# ---------------------------------------------------------------------------


def compute_dm_residuals(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
) -> Float[Array, " n_toas"]:
    """Compute DM residuals: measured DM - model DM (pc/cm³)."""
    model_dm = model.compute_dm(toa_data, params)
    return toa_data.dm_values - model_dm


def compute_wideband_residuals(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
) -> Float[Array, " n2_toas"]:
    """Compute stacked ``[time_residuals; dm_residuals]``, shape ``(2N,)``.

    Time residuals are in seconds, DM residuals in pc/cm³.
    """
    time_resid = compute_time_residuals(model, toa_data, params)
    dm_resid = compute_dm_residuals(model, toa_data, params)
    return jnp.concatenate([time_resid, dm_resid])


def compute_wideband_design_matrix(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
) -> Float[Array, "n2_toas n_free"]:
    """Build the wideband design matrix via autodiff, shape ``(2N, n_free)``.

    Uses ``jax.jacobian`` of the combined ``[time_resid; dm_resid]``
    vector w.r.t. all parameters, then extracts free columns.
    Negated per PINT convention.
    """
    J, M = _compute_wideband_jacobian_and_design(model, toa_data, params)
    return M


@eqx.filter_jit
def _compute_wideband_jacobian_and_design(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
) -> tuple[Float[Array, "n2_toas n_params"], Float[Array, "n2_toas n_free"]]:
    """JIT-compiled wideband Jacobian and design matrix computation."""
    free_indices = params.free_indices_array()

    def combined_resid_fn(all_values: Float[Array, " n_params"]):
        p = eqx.tree_at(lambda pv: pv.values, params, all_values)
        return compute_wideband_residuals(model, toa_data, p)

    J = jax.jacobian(combined_resid_fn)(params.values)  # (2N, n_params)
    M = -J[:, free_indices]
    return J, M


# ---------------------------------------------------------------------------
# JIT-compiled iteration core
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _wideband_iteration_core(
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
    """JIT-compiled core of one wideband GLS Gauss-Newton iteration."""
    free_indices = params.free_indices_array()
    n = toa_data.n_toas

    # Noise
    if noise_model is not None:
        sigma_toa = noise_model.scaled_sigma(toa_data, params)
        Ndiag_toa, U_toa, Phi_toa, Ndiag_dm = (
            noise_model.wideband_covariance(toa_data, params)
        )
    else:
        sigma_toa = toa_data.error
        Ndiag_toa = sigma_toa ** 2
        U_toa = jnp.zeros((n, 0))
        Phi_toa = jnp.zeros(0)
        Ndiag_dm = toa_data.dm_errors ** 2

    Ndiag = jnp.concatenate([Ndiag_toa, Ndiag_dm])
    U = jnp.concatenate([U_toa, jnp.zeros((n, U_toa.shape[1]))], axis=0)
    Phidiag = Phi_toa
    n_basis = U.shape[1]

    # Residuals
    time_resid = compute_time_residuals(model, toa_data, params)
    dm_resid = compute_dm_residuals(model, toa_data, params)

    if n_basis > 0:
        time_resid = _subtract_gls_weighted_mean(
            time_resid, Ndiag_toa, U_toa, Phidiag
        )
    else:
        time_resid = _subtract_weighted_mean(time_resid, sigma_toa)

    residuals = jnp.concatenate([time_resid, dm_resid])

    # Design matrix
    def combined_resid_fn(all_values: Float[Array, " n_params"]):
        p = eqx.tree_at(lambda pv: pv.values, params, all_values)
        return compute_wideband_residuals(model, toa_data, p)

    J = jax.jacobian(combined_resid_fn)(params.values)
    M = -J[:, free_indices]

    # Solve
    noise_realizations = jnp.zeros(0)
    if full_cov:
        dpars, covariance, _norms = gls_step_fullcov(
            residuals, Ndiag, U, Phidiag, M, threshold
        )
    elif n_basis > 0:
        dpars, covariance, _norms, noise_realizations = (
            gls_step_augmented(
                residuals, Ndiag, U, Phidiag, M, threshold
            )
        )
    else:
        sigma_combined = jnp.sqrt(Ndiag)
        dpars, covariance, _norms = wls_step(
            residuals, sigma_combined, M, threshold
        )

    new_values = params.values.at[free_indices].set(
        params.values[free_indices] + dpars
    )
    return new_values, covariance, noise_realizations


# ---------------------------------------------------------------------------
# Wideband GLS result container
# ---------------------------------------------------------------------------


@dataclass
class WidebandGLSFitResult(BaseFitResult):
    """Result of a wideband GLS fit."""

    time_residuals: Float[Array, " n_toas"]
    dm_residuals: Float[Array, " n_toas"]
    noise_realizations: Optional[Float[Array, " n_epochs"]]


# ---------------------------------------------------------------------------
# Wideband GLS fitter
# ---------------------------------------------------------------------------


class WidebandGLSFitter(BaseFitter):
    """Wideband Generalised Least Squares fitter.

    Jointly fits TOA and DM residuals using a combined ``(2N,)`` residual
    vector and design matrix.  Reuses the same GLS solve routines as
    :class:`GLSFitter`.
    """

    def _get_wideband_noise(
        self,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],       # sigma_toa
        Float[Array, " n2_toas"],      # Ndiag (2N)
        Float[Array, "n2_toas n_basis"],  # U (2N, K)
        Float[Array, " n_basis"],      # Phidiag
        Float[Array, " n_toas"],       # sigma_dm
    ]:
        """Return wideband noise quantities."""
        n = self.toa_data.n_toas

        if self.noise_model is not None:
            sigma_toa = self.noise_model.scaled_sigma(self.toa_data, params)
            Ndiag_toa, U_toa, Phi_toa, Ndiag_dm = (
                self.noise_model.wideband_covariance(self.toa_data, params)
            )
            sigma_dm = self.noise_model.scaled_dm_sigma(self.toa_data, params)
        else:
            sigma_toa = self.toa_data.error
            Ndiag_toa = sigma_toa ** 2
            U_toa = jnp.zeros((n, 0))
            Phi_toa = jnp.zeros(0)
            Ndiag_dm = self.toa_data.dm_errors ** 2
            sigma_dm = self.toa_data.dm_errors

        # Stack into (2N,) diagonal and (2N, K) basis
        Ndiag = jnp.concatenate([Ndiag_toa, Ndiag_dm])
        U = jnp.concatenate([U_toa, jnp.zeros((n, U_toa.shape[1]))], axis=0)

        return sigma_toa, Ndiag, U, Phi_toa, sigma_dm

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
        """Run one Gauss-Newton iteration."""
        new_values, covariance, noise_real = _wideband_iteration_core(
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
    ) -> WidebandGLSFitResult:
        """Compute final residuals/chi2 and return a result object."""
        sigma_toa, Ndiag, U, Phidiag, sigma_dm = self._get_wideband_noise(
            params
        )
        n_basis = U.shape[1]
        n = self.toa_data.n_toas

        time_resid = compute_time_residuals(
            self.model, self.toa_data, params
        )
        dm_resid = compute_dm_residuals(self.model, self.toa_data, params)

        if n_basis > 0:
            Ndiag_toa = Ndiag[:n]
            U_toa = U[:n, :]
            time_resid = _subtract_gls_weighted_mean(
                time_resid, Ndiag_toa, U_toa, Phidiag
            )
        else:
            time_resid = _subtract_weighted_mean(time_resid, sigma_toa)

        residuals = jnp.concatenate([time_resid, dm_resid])

        if n_basis > 0:
            chi2_val = float(compute_gls_chi2(residuals, Ndiag, U, Phidiag))
        else:
            sigma_combined = jnp.sqrt(Ndiag)
            chi2_val = float(compute_chi2(residuals, sigma_combined))

        dof = 2 * n - params.n_free

        errors, correlation = self._covariance_to_correlation(covariance)
        reduced_chi2 = self._reduced_chi2(chi2_val, dof)

        return WidebandGLSFitResult(
            params=params,
            covariance_matrix=covariance,
            correlation_matrix=correlation,
            parameter_uncertainties=errors,
            chi2=chi2_val,
            dof=dof,
            reduced_chi2=reduced_chi2,
            time_residuals=time_resid,
            dm_residuals=dm_resid,
            noise_realizations=noise_realizations,
        )

    def fit_toas(
        self,
        maxiter: int = 1,
        threshold: Optional[float] = None,
        full_cov: bool = False,
    ) -> WidebandGLSFitResult:
        """Run the wideband GLS fit.

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
        WidebandGLSFitResult
        """
        n_toas = self.toa_data.n_toas

        if self.noise_model is not None and self.noise_model.has_correlated:
            _, _, U_init, _ = self.noise_model.covariance(
                self.toa_data, self.params
            )
            n_basis = U_init.shape[1]
        else:
            n_basis = 0

        if threshold is None:
            if full_cov:
                dim = self.params.n_free
            else:
                dim = self.params.n_free + n_basis
            threshold = 1e-14 * max(2 * n_toas, dim)

        safe_maxiter = 1 if maxiter < 1 else maxiter
        params = self.params
        covariance = None
        noise_realizations = None
        for _ in range(safe_maxiter):
            params, covariance, noise_realizations = self._iteration(
                params, threshold, full_cov
            )

        return self._build_result(params, covariance, noise_realizations)
