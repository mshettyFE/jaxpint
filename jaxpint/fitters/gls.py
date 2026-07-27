"""Generalised Least Squares fitter."""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import SMWhitener, woodbury_solve

from ._base import (
    _DEFAULT_MAXITER,
    BaseFitter,
    BaseFitResult,
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
    external_delay: Optional[Float[Array, " n_toas"]] = None,
) -> tuple[
    Float[Array, " n_params"],
    Float[Array, "n_free n_free"],
    Float[Array, " n_basis"],
]:
    """JIT-compiled core of one GLS Gauss-Newton iteration.

    ``external_delay`` (seconds), when given, is subtracted from the
    residuals before the solve — e.g. a deterministic signal (CW) whose
    absorption by the timing fit should be modelled.

    Returns updated parameter values, covariance, and noise realizations.

    """
    free_indices = params.free_indices_array()

    whitener = None
    if noise_model is not None:
        sigma = noise_model.scaled_sigma(toa_data, params)
        Ndiag, U, Phidiag = noise_model.covariance(toa_data, params)
        if noise_model.ecorr_kernel is not None:
            # Kernel ECORR: pre-whiten the whole GLS system (residuals,
            # design matrix, GP basis) — every fitted quantity is a
            # C⁻¹-inner product and therefore whitening-invariant.  Noise
            # parameters are frozen during the fit, so the whitener is
            # constant across iterations (rebuilt per traced call, O(n)).
            whitener = noise_model.ecorr_kernel.ops(Ndiag, params)
            U = whitener.whiten(U)
            Ndiag = jnp.ones_like(Ndiag)
            sigma = jnp.ones_like(sigma)
    else:
        sigma = toa_data.error
        Ndiag = sigma**2
        U = jnp.zeros((toa_data.n_toas, 0))
        Phidiag = jnp.zeros(0)

    n_basis = U.shape[1]

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

    # Force constant column if not included. Same as PINT
    include_offset = model.phoff_name is None
    if include_offset:
        offset_col = jnp.ones((M.shape[0], 1), dtype=M.dtype)
        M = jnp.concatenate([offset_col, M], axis=1)

    if whitener is not None:
        # After the offset column: the constant column must be whitened too.
        time_resid = whitener.whiten(time_resid)
        M = whitener.whiten(M)

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
        "Optional[SMWhitener]",
    ]:
        """Return ``(sigma, Ndiag, U, Phidiag, whitener)``.

        With kernel ECORR the returned blocks are pre-whitened
        (``Ndiag = sigma = 1``, ``U`` whitened) and ``whitener`` carries
        the transform; otherwise ``whitener`` is ``None``.
        """
        if self.noise_model is not None:
            sigma = self.noise_model.scaled_sigma(self.toa_data, params)
            Ndiag, U, Phidiag = self.noise_model.covariance(self.toa_data, params)
            if self.noise_model.ecorr_kernel is not None:
                whitener = self.noise_model.ecorr_kernel.ops(Ndiag, params)
                return (
                    jnp.ones_like(sigma),
                    jnp.ones_like(Ndiag),
                    whitener.whiten(U),
                    Phidiag,
                    whitener,
                )
        else:
            sigma = self.toa_data.error
            Ndiag = sigma**2
            U = jnp.zeros((self.toa_data.n_toas, 0))
            Phidiag = jnp.zeros(0)

        return sigma, Ndiag, U, Phidiag, None

    # -- Differentiable-solve hooks ------------------------------------------

    def _fit_cinv(
        self, params: ParameterVector, x: Float[Array, " n_toas"]
    ) -> Float[Array, " n_toas"]:
        _sigma, Ndiag, U, Phidiag, whitener = self._get_noise(params)
        if whitener is not None:
            # Data-space solve through the whitened system:
            # C^-1 x = W^T (W C W^T)^-1 W x, with W C Wᵀ = I + \bar{U} \Psi \bar{U^T}
            xw = whitener.whiten(x)
            if U.shape[1] == 0:
                return whitener.whiten_t(xw)
            return whitener.whiten_t(
                woodbury_solve(Ndiag, U, Phidiag, xw[:, None])[:, 0]
            )
        if U.shape[1] == 0:
            return x / Ndiag
        return woodbury_solve(Ndiag, U, Phidiag, x[:, None])[:, 0]

    def _core_step(
        self,
        params: ParameterVector,
        external_delay: Optional[Float[Array, " n_toas"]],
        threshold: float,
        full_cov: bool = False,
    ) -> tuple[
        Float[Array, " n_params"],
        Float[Array, "n_free n_free"],
        Optional[Float[Array, " n_basis"]],
    ]:
        new_values, covariance, noise_real = _gls_iteration_core(
            self.model,
            self.toa_data,
            params,
            self.noise_model,
            threshold,
            full_cov,
            external_delay,
        )
        noise_realizations = noise_real if noise_real.size > 0 else None
        return new_values, covariance, noise_realizations

    def _default_threshold(self, full_cov: bool = False) -> float:
        if self.noise_model is not None and self.noise_model.has_correlated:
            _, _, U_init, _, _ = self._get_noise(self.params)
            n_basis = U_init.shape[1]
        else:
            n_basis = 0
        dim = self.params.n_free if full_cov else self.params.n_free + n_basis
        return 1e-14 * max(self.toa_data.n_toas, dim)

    # -- Public API ------------------------------------------------------------

    def fit_toas(
        self,
        maxiter: int = _DEFAULT_MAXITER,
        threshold: Optional[float] = None,
        full_cov: bool = False,
        params: Optional[ParameterVector] = None,
        external_delay: Optional[Float[Array, " n_toas"]] = None,
    ) -> GLSFitResult:
        """Run the GLS fit (differentiable end-to-end).

        Final residuals are mean-subtracted (matches PINT's
        ``Residuals(subtract_mean=True)`` default).  The covariance and
        noise realizations are evaluated at the converged parameters
        (making them, and the derived uncertainties, differentiable).

        Parameters
        ----------
        maxiter : int
            Number of Gauss-Newton iterations.
        threshold : float, optional
            SVD threshold (default ``1e-14 * dim``).
        full_cov : bool
            If True, use Woodbury-based full covariance inversion.
            If False (default), use the augmented design-matrix approach.
        params : ParameterVector, optional
            Starting parameters (default ``self.params``); the traced
            entry point for gradients w.r.t. frozen parameter values.
        external_delay : array (n_toas,), optional
            Deterministic delay (seconds) subtracted from the residuals
            before fitting, e.g. an injected CW signal.  Differentiable.

        Returns
        -------
        GLSFitResult
            Fit result containing updated parameters, covariance,
            uncertainties, chi-squared, residuals, and noise realizations.
        """
        if threshold is None:
            threshold = self._default_threshold(full_cov=full_cov)

        fitted = self.fit_params(
            params,
            external_delay,
            maxiter=maxiter,
            threshold=threshold,
            full_cov=full_cov,
        )
        _vals, covariance, noise_realizations = self._core_step(
            fitted, external_delay, threshold, full_cov=full_cov
        )

        sigma, Ndiag, U, Phidiag, whitener = self._get_noise(fitted)
        n_basis = U.shape[1]

        final_resid = self._fit_residuals(fitted, external_delay)
        if whitener is not None:
            # Kernel ECORR: mean-subtract and take chi^2 in *data* space
            # using the kernel-aware C^-1 (the returned residuals stay in
            # data space, matching the basis-form output contract).
            ones = jnp.ones_like(final_resid)
            z1 = self._fit_cinv(fitted, final_resid)
            z0 = self._fit_cinv(fitted, ones)
            final_resid = final_resid - jnp.sum(z1) / jnp.sum(z0)
            chi2_val = final_resid @ self._fit_cinv(fitted, final_resid)
        elif n_basis > 0:
            final_resid = _subtract_cov_weighted_mean(final_resid, Ndiag, U, Phidiag)
            chi2_val = compute_chi2_cov(final_resid, Ndiag, U, Phidiag)
        else:
            final_resid = _subtract_weighted_mean(final_resid, sigma)
            chi2_val = compute_chi2(final_resid, sigma)

        return GLSFitResult(
            params=fitted,
            covariance_matrix=covariance,
            chi2=chi2_val,
            dof=self._dof(fitted, self.toa_data.n_toas),
            residuals=final_resid,
            noise_realizations=noise_realizations,
            step_sigma=self.step_sigma(
                fitted, external_delay, threshold, full_cov=full_cov
            ),
        )
