"""Weighted Least Squares fitter."""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector

from ._base import (
    _DEFAULT_MAXITER,
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
    external_delay: Optional[Float[Array, " n_toas"]] = None,
) -> tuple[Float[Array, " n_params"], Float[Array, "n_free n_free"]]:
    """JIT-compiled core of one WLS Gauss-Newton iteration.

    ``external_delay`` (seconds), when given, is subtracted from the
    residuals before the solve — e.g. a deterministic signal (CW) whose
    absorption by the timing fit should be modelled.

    Returns updated parameter values and the covariance matrix for the
    timing parameters (offset row/column stripped).
    """
    free_indices = params.free_indices_array()

    if noise_model is not None:
        sigma = noise_model.scaled_sigma(toa_data, params)
    else:
        sigma = toa_data.error

    time_resid = compute_time_residuals(model, toa_data, params)
    if external_delay is not None:
        time_resid = time_resid - external_delay

    def time_resid_fn(all_values: Float[Array, " n_params"]):
        p = params.with_values(all_values)
        return compute_time_residuals(model, toa_data, p)

    # jacfwd: n_params forward tangents. jacrev would materialize an
    # n_toas x n_toas cotangent basis and OOM high-cadence pulsars.
    J = jax.jacfwd(time_resid_fn)(params.values)
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

    # -- Differentiable-solve hooks ------------------------------------------

    def _fit_cinv(
        self, params: ParameterVector, x: Float[Array, " n_toas"]
    ) -> Float[Array, " n_toas"]:
        return x / self._get_sigma(params) ** 2

    def _core_step(
        self,
        params: ParameterVector,
        external_delay: Optional[Float[Array, " n_toas"]],
        threshold: float,
    ) -> tuple[
        Float[Array, " n_params"],
        Float[Array, "n_free n_free"],
        None,
    ]:
        new_values, covariance = _wls_iteration_core(
            self.model,
            self.toa_data,
            params,
            self.noise_model,
            threshold,
            external_delay,
        )
        return new_values, covariance, None

    def _default_threshold(self) -> float:
        return 1e-14 * max(self.toa_data.n_toas, self.params.n_free)

    # -- Public API ------------------------------------------------------------

    def fit_toas(
        self,
        maxiter: int = _DEFAULT_MAXITER,
        threshold: Optional[float] = None,
        params: Optional[ParameterVector] = None,
        external_delay: Optional[Float[Array, " n_toas"]] = None,
    ) -> WLSFitResult:
        """Run the WLS fit (differentiable end-to-end).

        Final residuals are mean-subtracted (matches PINT's
        ``Residuals(subtract_mean=True)`` default) so that reported chi^2
        is invariant to the implicit constant DOF.  The covariance is
        evaluated at the converged parameters (making it and the derived
        uncertainties differentiable).

        Parameters
        ----------
        maxiter : int
            Number of Gauss-Newton iterations.
        threshold : float, optional
            SVD threshold (default ``1e-14 * max(n_toas, n_free)``).
        params : ParameterVector, optional
            Starting parameters (default ``self.params``); the traced
            entry point for gradients w.r.t. frozen parameter values.
        external_delay : array (n_toas,), optional
            Deterministic delay (seconds) subtracted from the residuals
            before fitting, e.g. an injected CW signal.  Differentiable.

        Returns
        -------
        WLSFitResult
            Fit result containing updated parameters, covariance,
            uncertainties, chi-squared, and residuals.
        """
        if threshold is None:
            threshold = self._default_threshold()

        fitted = self.fit_params(
            params, external_delay, maxiter=maxiter, threshold=threshold
        )
        _vals, covariance, _nr = self._core_step(fitted, external_delay, threshold)
        sigma = self._get_sigma(fitted)

        final_resid = self._fit_residuals(fitted, external_delay)
        final_resid = _subtract_weighted_mean(final_resid, sigma)
        chi2_val = compute_chi2(final_resid, sigma)

        return WLSFitResult(
            params=fitted,
            covariance_matrix=covariance,
            chi2=chi2_val,
            dof=self._dof(fitted, self.toa_data.n_toas),
            residuals=final_resid,
            step_sigma=self.step_sigma(fitted, external_delay, threshold),
        )
