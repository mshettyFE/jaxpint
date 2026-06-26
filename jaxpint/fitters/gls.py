"""Generalised Least Squares fitter."""

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
    _subtract_cov_weighted_mean,
    wls_step,
    lstsq_step_fullcov,
    lstsq_step_augmented,
    compute_chi2,
    compute_chi2_cov,
)


# ---------------------------------------------------------------------------
# JIT-compiled iteration core
# ---------------------------------------------------------------------------


@eqx.filter_jit
def _gls_iteration_core(
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
    """JIT-compiled core of one GLS Gauss-Newton iteration.

    Returns updated parameter values, covariance, and noise realizations.

    """
    free_indices = params.free_indices_array()

    if noise_model is not None:
        sigma = noise_model.scaled_sigma(toa_data, params)
        Ndiag, U, Phidiag = noise_model.covariance(toa_data, params)
    else:
        sigma = toa_data.error
        Ndiag = sigma**2
        U = jnp.zeros((toa_data.n_toas, 0))
        Phidiag = jnp.zeros(0)

    n_basis = U.shape[1]

    time_resid = compute_time_residuals(model, toa_data, params)

    def time_resid_fn(all_values: Float[Array, " n_params"]):
        p = eqx.tree_at(lambda pv: pv.values, params, all_values)
        return compute_time_residuals(model, toa_data, p)

    J = jax.jacobian(time_resid_fn)(params.values)
    M = -J[:, free_indices]

    # Force constant column if not included. Same as PINT
    include_offset = model.phoff_name is None
    if include_offset:
        offset_col = jnp.ones((M.shape[0], 1), dtype=M.dtype)
        M = jnp.concatenate([offset_col, M], axis=1)

    noise_realizations = jnp.zeros(0)
    if full_cov:
        dpars, covariance, _norms = lstsq_step_fullcov(
            time_resid, Ndiag, U, Phidiag, M, threshold
        )
    elif n_basis > 0:
        dpars, covariance, _norms, noise_realizations = lstsq_step_augmented(
            time_resid, Ndiag, U, Phidiag, M, threshold
        )
    else:
        dpars, covariance, _norms = wls_step(time_resid, sigma, M, threshold)

    if include_offset:
        param_updates = dpars[1:]
        covariance = covariance[1:, 1:]
    else:
        param_updates = dpars

    new_values = params.values.at[free_indices].set(
        params.values[free_indices] + param_updates
    )
    return new_values, covariance, noise_realizations


# ---------------------------------------------------------------------------
# GLS result container
# ---------------------------------------------------------------------------


@dataclass
class GLSFitResult(BaseFitResult):
    """Result of a GLS fit."""

    residuals: Float[Array, " n_toas"]
    noise_realizations: Optional[Float[Array, " n_epochs"]]


# ---------------------------------------------------------------------------
# GLS fitter class
# ---------------------------------------------------------------------------


class GLSFitter(BaseFitter):
    """Generalised Least Squares fitter.

    The fitter is an immutable configuration container.  Calling
    :meth:`fit_toas` returns a :class:`GLSFitResult` without mutating
    the fitter itself.

    Supports arbitrary correlated noise sources (ECORR, red noise, etc.)
    through the :class:`~jaxpint.noise.NoiseModel` interface.  When no
    correlated components are present the GLS fitter reduces to WLS.
    """

    def _get_noise(
        self,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_basis"],
        Float[Array, " n_basis"],
    ]:
        """Return (sigma, Ndiag, U, Phidiag) for the given parameters."""
        if self.noise_model is not None:
            sigma = self.noise_model.scaled_sigma(self.toa_data, params)
            Ndiag, U, Phidiag = self.noise_model.covariance(self.toa_data, params)
        else:
            sigma = self.toa_data.error
            Ndiag = sigma**2
            U = jnp.zeros((self.toa_data.n_toas, 0))
            Phidiag = jnp.zeros(0)

        return sigma, Ndiag, U, Phidiag

    def _iteration(
        self,
        params: ParameterVector,
        threshold: float,
        full_cov: bool,
    ) -> IterationState:
        """Run one Gauss-Newton iteration."""
        new_values, covariance, noise_real = _gls_iteration_core(
            self.model,
            self.toa_data,
            params,
            self.noise_model,
            threshold,
            full_cov,
        )
        new_params = eqx.tree_at(lambda pv: pv.values, params, new_values)
        noise_realizations = noise_real if noise_real.size > 0 else None
        return IterationState(new_params, covariance, noise_realizations)

    def _build_result(self, state: IterationState) -> GLSFitResult:
        """Compute final residuals/chi2 and return a result object.

        Final residuals are still mean-subtracted (matches PINT's
        ``Residuals(subtract_mean=True)`` default).  The dof count
        subtracts one for the implicit Offset column when applicable.
        """
        params = state.params
        sigma, Ndiag, U, Phidiag = self._get_noise(params)
        n_basis = U.shape[1]

        final_resid = compute_time_residuals(self.model, self.toa_data, params)
        if n_basis > 0:
            final_resid = _subtract_cov_weighted_mean(final_resid, Ndiag, U, Phidiag)
        else:
            final_resid = _subtract_weighted_mean(final_resid, sigma)

        if n_basis > 0:
            chi2_val = float(compute_chi2_cov(final_resid, Ndiag, U, Phidiag))
        else:
            chi2_val = float(compute_chi2(final_resid, sigma))

        dof = self._dof(params, self.toa_data.n_toas)

        return GLSFitResult(
            params=params,
            covariance_matrix=state.covariance,
            chi2=chi2_val,
            dof=dof,
            residuals=final_resid,
            noise_realizations=state.noise_realizations,
        )

    def fit_toas(
        self,
        maxiter: int = 1,
        threshold: Optional[float] = None,
        full_cov: bool = False,
    ) -> GLSFitResult:
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
        GLSFitResult
            Fit result containing updated parameters, covariance,
            uncertainties, chi-squared, residuals, and noise realizations.
        """
        n_toas = self.toa_data.n_toas

        # Determine basis dimension for threshold calculation
        if self.noise_model is not None and self.noise_model.has_correlated:
            _, _, U_init, _ = self._get_noise(self.params)
            n_basis = U_init.shape[1]
        else:
            n_basis = 0

        if threshold is None:
            if full_cov:
                dim = self.params.n_free
            else:
                dim = self.params.n_free + n_basis
            threshold = 1e-14 * max(n_toas, dim)

        state = self._gauss_newton(
            maxiter, threshold, lambda p, t: self._iteration(p, t, full_cov)
        )
        return self._build_result(state)
