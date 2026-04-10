"""Weighted Least Squares fitter."""
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


# ---------------------------------------------------------------------------
# JIT-compiled iteration core
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _wls_iteration_core(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
    noise_model: Optional[NoiseModel],
    threshold: float,
) -> tuple[Float[Array, " n_params"], Float[Array, "n_free n_free"]]:
    """JIT-compiled core of one WLS Gauss-Newton iteration.

    Returns updated parameter values and covariance matrix.
    """
    free_indices = params.free_indices_array()

    if noise_model is not None:
        sigma = noise_model.scaled_sigma(toa_data, params)
    else:
        sigma = toa_data.error

    time_resid = compute_time_residuals(model, toa_data, params)
    time_resid = _subtract_weighted_mean(time_resid, sigma)

    def time_resid_fn(all_values: Float[Array, " n_params"]):
        p = eqx.tree_at(lambda pv: pv.values, params, all_values)
        return compute_time_residuals(model, toa_data, p)

    J = jax.jacobian(time_resid_fn)(params.values)
    M = -J[:, free_indices]

    dpars, covariance, _norms = wls_step(time_resid, sigma, M, threshold)

    new_values = params.values.at[free_indices].set(
        params.values[free_indices] + dpars
    )
    return new_values, covariance


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class WLSFitResult(BaseFitResult):
    """Result of a WLS fit."""

    residuals: Float[Array, " n_toas"]


# ---------------------------------------------------------------------------
# Fitter class
# ---------------------------------------------------------------------------


class WLSFitter(BaseFitter):
    """Weighted Least Squares fitter (Gauss-Newton with SVD).

    The fitter is an immutable configuration container.  Calling
    :meth:`fit_toas` returns a :class:`WLSFitResult` without mutating
    the fitter itself.  Only the diagonal (white-noise) part of the
    noise model is used.
    """

    def _get_sigma(self, params: ParameterVector) -> Float[Array, " n_toas"]:
        """Return noise-scaled TOA uncertainties."""
        if self.noise_model is not None:
            return self.noise_model.scaled_sigma(self.toa_data, params)
        return self.toa_data.error

    def _iteration(
        self,
        params: ParameterVector,
        threshold: float,
    ) -> tuple[ParameterVector, Float[Array, "n_free n_free"]]:
        """Run one Gauss-Newton iteration.

        Returns
        -------
        (new_params, covariance)
        """
        new_values, covariance = _wls_iteration_core(
            self.model, self.toa_data, params,
            self.noise_model, threshold,
        )
        new_params = eqx.tree_at(lambda pv: pv.values, params, new_values)
        return new_params, covariance

    def _build_result(
        self,
        params: ParameterVector,
        covariance: Float[Array, "n_free n_free"],
    ) -> WLSFitResult:
        """Compute final residuals/chi2 and return a result object."""
        sigma = self._get_sigma(params)

        final_resid = compute_time_residuals(
            self.model, self.toa_data, params
        )
        final_resid = _subtract_weighted_mean(final_resid, sigma)
        chi2_val = float(compute_chi2(final_resid, sigma))
        dof = self.toa_data.n_toas - params.n_free

        errors, correlation = self._covariance_to_correlation(covariance)
        reduced_chi2 = self._reduced_chi2(chi2_val, dof)

        return WLSFitResult(
            params=params,
            covariance_matrix=covariance,
            correlation_matrix=correlation,
            parameter_uncertainties=errors,
            chi2=chi2_val,
            dof=dof,
            reduced_chi2=reduced_chi2,
            residuals=final_resid,
        )

    def fit_toas(
        self,
        maxiter: int = 1,
        threshold: Optional[float] = None,
    ) -> WLSFitResult:
        """Run the WLS fit.

        Parameters
        ----------
        maxiter : int
            Number of Gauss-Newton iterations.
        threshold : float, optional
            SVD threshold (default ``1e-14 * max(n_toas, n_free)``).

        Returns
        -------
        WLSFitResult
            Fit result containing updated parameters, covariance,
            uncertainties, chi-squared, and residuals.
        """
        if threshold is None:
            threshold = 1e-14 * max(self.toa_data.n_toas, self.params.n_free)

        params = self.params
        covariance = None
        safe_maxiter =  1 if maxiter < 1 else maxiter

        for _ in range(safe_maxiter):
            params, covariance = self._iteration(params, threshold)

        return self._build_result(params, covariance)
