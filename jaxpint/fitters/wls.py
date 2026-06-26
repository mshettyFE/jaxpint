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
    IterationState,
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

    Returns updated parameter values and the covariance matrix for the
    timing parameters (offset row/column stripped).
    """
    free_indices = params.free_indices_array()

    if noise_model is not None:
        sigma = noise_model.scaled_sigma(toa_data, params)
    else:
        sigma = toa_data.error

    time_resid = compute_time_residuals(model, toa_data, params)

    def time_resid_fn(all_values: Float[Array, " n_params"]):
        p = params.with_values(all_values)
        return compute_time_residuals(model, toa_data, p)

    J = jax.jacobian(time_resid_fn)(params.values)
    M = -J[:, free_indices]

    include_offset = model.phoff_name is None
    if include_offset:
        offset_col = jnp.ones((M.shape[0], 1), dtype=M.dtype)
        M = jnp.concatenate([offset_col, M], axis=1)

    dpars, covariance, _norms = wls_step(time_resid, sigma, M, threshold)

    if include_offset:
        # First column is the synthetic Offset; discard its solve.
        param_updates = dpars[1:]
        covariance = covariance[1:, 1:]
    else:
        param_updates = dpars

    new_values = params.values.at[free_indices].set(
        params.values[free_indices] + param_updates
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
    ) -> IterationState:
        """Run one Gauss-Newton iteration."""
        new_values, covariance = _wls_iteration_core(
            self.model,
            self.toa_data,
            params,
            self.noise_model,
            threshold,
        )
        new_params = params.with_values(new_values)
        return IterationState(new_params, covariance)

    def _build_result(self, state: IterationState) -> WLSFitResult:
        """Compute final residuals/chi2 and return a result object.

        Final residuals are still mean-subtracted (matches PINT's
        ``Residuals(subtract_mean=True)`` default) so that reported chi^2
        is invariant to the implicit constant DOF.  The dof count subtracts
        one for the Offset column when applicable (matches PINT).
        """
        params = state.params
        covariance = state.covariance
        sigma = self._get_sigma(params)

        final_resid = compute_time_residuals(self.model, self.toa_data, params)
        final_resid = _subtract_weighted_mean(final_resid, sigma)
        chi2_val = float(compute_chi2(final_resid, sigma))

        dof = self._dof(params, self.toa_data.n_toas)

        return WLSFitResult(
            params=params,
            covariance_matrix=covariance,
            chi2=chi2_val,
            dof=dof,
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

        return self._build_result(
            self._gauss_newton(maxiter, threshold, self._iteration)
        )
