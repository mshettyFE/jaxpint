"""Wideband Generalised Least Squares fitter."""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import woodbury_solve

from ._base import (
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
# Wideband residual / design-matrix functions
# ---------------------------------------------------------------------------


def compute_dm_residuals(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
) -> Float[Array, " n_toas"]:
    """Compute DM residuals: measured DM - model DM (pc/cm³)."""
    model_dm = model.compute_dm(toa_data, params)
    assert toa_data.dm_values is not None  # wideband data always carries DM
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
    include_offset: bool = True,
) -> Float[Array, "n2_toas n_cols"]:
    """Build the wideband design matrix via autodiff.

    Uses ``jax.jacfwd`` of the combined ``[time_resid; dm_resid]``
    vector w.r.t. all parameters, then extracts free columns.
    Negated per PINT convention.

    Parameters
    ----------
    include_offset : bool, optional
        If True (default, matches PINT's ``incoffset=True``), prepend an
        Offset column whose entries are 1 in the time-residual half
        (rows ``[0, n_toas)``) and 0 in the DM-residual half (rows
        ``[n_toas, 2*n_toas)``).  This mirrors PINT's
        ``wideband_designmatrix``.

    Returns
    -------
    M : jax.Array
        Negated Jacobian, shape ``(2*n_toas, n_free + 1)`` when offset is
        included (Offset column first), else ``(2*n_toas, n_free)``.
    """
    if model.phoff_name is not None:
        include_offset = False
    J, M = _compute_wideband_jacobian_and_design(model, toa_data, params)
    if include_offset:
        n = toa_data.n_toas
        offset_col = jnp.concatenate(
            [jnp.ones(n, dtype=M.dtype), jnp.zeros(n, dtype=M.dtype)]
        )[:, None]
        M = jnp.concatenate([offset_col, M], axis=1)
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
        p = params.with_values(all_values)
        return compute_wideband_residuals(model, toa_data, p)

    # jacfwd: n_params forward tangents (jacrev would materialize a
    # 2N x 2N cotangent basis and OOM high-cadence pulsars). Matches the
    # narrowband helper's mode, so the wideband top-block equals the
    # narrowband design matrix bitwise.
    J = jax.jacfwd(combined_resid_fn)(params.values)  # (2N, n_params)
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
    external_delay: Optional[Float[Array, " n_toas"]] = None,
) -> tuple[
    Float[Array, " n_params"],
    Float[Array, "n_free n_free"],
    Float[Array, " n_basis"],
]:
    """JIT-compiled core of one wideband GLS Gauss-Newton iteration.

    ``external_delay`` (seconds) is subtracted from the time-residual
    half only; DM residuals are unaffected.
    """
    free_indices = params.free_indices_array()
    n = toa_data.n_toas

    if noise_model is not None:
        sigma_toa = noise_model.scaled_sigma(toa_data, params)
        Ndiag_toa, U_toa, Phi_toa, Ndiag_dm = noise_model.wideband_covariance(
            toa_data, params
        )
    else:
        sigma_toa = toa_data.error
        Ndiag_toa = sigma_toa**2
        U_toa = jnp.zeros((n, 0))
        Phi_toa = jnp.zeros(0)
        assert toa_data.dm_errors is not None  # wideband data always carries DM
        Ndiag_dm = toa_data.dm_errors**2

    Ndiag = jnp.concatenate([Ndiag_toa, Ndiag_dm])
    U = jnp.concatenate([U_toa, jnp.zeros((n, U_toa.shape[1]))], axis=0)
    Phidiag = Phi_toa
    n_basis = U.shape[1]

    time_resid = compute_time_residuals(model, toa_data, params)
    if external_delay is not None:
        time_resid = time_resid - external_delay
    dm_resid = compute_dm_residuals(model, toa_data, params)
    residuals = jnp.concatenate([time_resid, dm_resid])

    def combined_resid_fn(all_values: Float[Array, " n_params"]):
        p = params.with_values(all_values)
        return compute_wideband_residuals(model, toa_data, p)

    # jacfwd: n_params forward tangents (jacrev would materialize a
    # 2N x 2N cotangent basis and OOM high-cadence pulsars).
    J = jax.jacfwd(combined_resid_fn)(params.values)
    M = -J[:, free_indices]

    include_offset = model.phoff_name is None
    if include_offset:
        # Offset column: 1 for time rows, 0 for DM rows.  Mirrors PINT's
        # wideband_designmatrix: dm_designmatrix sets the Offset column to
        # zero, while designmatrix sets it to one.
        offset_col = jnp.concatenate(
            [jnp.ones(n, dtype=M.dtype), jnp.zeros(n, dtype=M.dtype)]
        )[:, None]
        M = jnp.concatenate([offset_col, M], axis=1)

    noise_realizations = jnp.zeros(0)
    if full_cov:
        dpars, covariance, _norms = lstsq_step_fullcov(
            residuals, Ndiag, U, Phidiag, M, threshold
        )
    elif n_basis > 0:
        dpars, covariance, _norms, noise_realizations = lstsq_step_augmented(
            residuals, Ndiag, U, Phidiag, M, threshold
        )
    else:
        sigma_combined = jnp.sqrt(Ndiag)
        dpars, covariance, _norms = wls_step(residuals, sigma_combined, M, threshold)

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
# Wideband GLS result container
# ---------------------------------------------------------------------------


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
        Float[Array, " n_toas"],  # sigma_toa
        Float[Array, " n2_toas"],  # Ndiag (2N)
        Float[Array, "n2_toas n_basis"],  # U (2N, K)
        Float[Array, " n_basis"],  # Phidiag
        Float[Array, " n_toas"],  # sigma_dm
    ]:
        """Return wideband noise quantities."""
        n = self.toa_data.n_toas

        if self.noise_model is not None:
            sigma_toa = self.noise_model.scaled_sigma(self.toa_data, params)
            Ndiag_toa, U_toa, Phi_toa, Ndiag_dm = self.noise_model.wideband_covariance(
                self.toa_data, params
            )
            sigma_dm = self.noise_model.scaled_dm_sigma(self.toa_data, params)
        else:
            sigma_toa = self.toa_data.error
            Ndiag_toa = sigma_toa**2
            U_toa = jnp.zeros((n, 0))
            Phi_toa = jnp.zeros(0)
            assert (
                self.toa_data.dm_errors is not None
            )  # wideband data always carries DM
            Ndiag_dm = self.toa_data.dm_errors**2
            sigma_dm = self.toa_data.dm_errors

        # Stack into (2N,) diagonal and (2N, K) basis
        Ndiag = jnp.concatenate([Ndiag_toa, Ndiag_dm])
        U = jnp.concatenate([U_toa, jnp.zeros((n, U_toa.shape[1]))], axis=0)

        return sigma_toa, Ndiag, U, Phi_toa, sigma_dm

    # -- Differentiable-solve hooks ------------------------------------------

    def _fit_residuals(
        self,
        params: ParameterVector,
        external_delay: Optional[Float[Array, " n_toas"]],
    ) -> Float[Array, " n2_toas"]:
        """Stacked ``[time; dm]`` residuals; the delay hits the time half."""
        r = compute_wideband_residuals(self.model, self.toa_data, params)
        if external_delay is not None:
            r = r - jnp.concatenate([external_delay, jnp.zeros_like(external_delay)])
        return r

    def _offset_vector(self) -> Optional[Float[Array, " n2_toas"]]:
        """Offset column: 1 for time rows, 0 for DM rows (mirrors PINT)."""
        if self.model.phoff_name is not None:
            return None
        n = self.toa_data.n_toas
        return jnp.concatenate([jnp.ones(n), jnp.zeros(n)])

    def _fit_cinv(
        self, params: ParameterVector, x: Float[Array, " n2_toas"]
    ) -> Float[Array, " n2_toas"]:
        _s_toa, Ndiag, U, Phidiag, _s_dm = self._get_wideband_noise(params)
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
        new_values, covariance, noise_real = _wideband_iteration_core(
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
            # Read n_basis from the SAME wideband path the solve uses
            # (_get_wideband_noise -> wideband_covariance), NOT the narrowband
            # covariance(): the two agree only because wideband_covariance
            # currently delegates to it, so reading covariance() here would
            # silently miscalibrate the SVD threshold the day that delegation
            # changes.  Mirrors GLSFitter._default_threshold (which uses _get_noise).
            _s, _Nd, U_init, _Phi, _sdm = self._get_wideband_noise(self.params)
            n_basis = U_init.shape[1]
        else:
            n_basis = 0
        dim = self.params.n_free if full_cov else self.params.n_free + n_basis
        return 1e-14 * max(2 * self.toa_data.n_toas, dim)

    # -- Public API ------------------------------------------------------------

    def fit_toas(
        self,
        maxiter: int = 1,
        threshold: Optional[float] = None,
        full_cov: bool = False,
        params: Optional[ParameterVector] = None,
        external_delay: Optional[Float[Array, " n_toas"]] = None,
    ) -> WidebandGLSFitResult:
        """Run the wideband GLS fit (differentiable end-to-end).

        The covariance and noise realizations are evaluated at the
        converged parameters.

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
            Deterministic delay (seconds) subtracted from the time
            residuals before fitting.  Differentiable.

        Returns
        -------
        WidebandGLSFitResult
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

        sigma_toa, Ndiag, U, Phidiag, _sigma_dm = self._get_wideband_noise(fitted)
        n_basis = U.shape[1]
        n = self.toa_data.n_toas

        time_resid = compute_time_residuals(self.model, self.toa_data, fitted)
        if external_delay is not None:
            time_resid = time_resid - external_delay
        dm_resid = compute_dm_residuals(self.model, self.toa_data, fitted)

        if n_basis > 0:
            Ndiag_toa = Ndiag[:n]
            U_toa = U[:n, :]
            time_resid = _subtract_cov_weighted_mean(
                time_resid, Ndiag_toa, U_toa, Phidiag
            )
        else:
            time_resid = _subtract_weighted_mean(time_resid, sigma_toa)

        residuals = jnp.concatenate([time_resid, dm_resid])

        if n_basis > 0:
            chi2_val = compute_chi2_cov(residuals, Ndiag, U, Phidiag)
        else:
            chi2_val = compute_chi2(residuals, jnp.sqrt(Ndiag))

        return WidebandGLSFitResult(
            params=fitted,
            covariance_matrix=covariance,
            chi2=chi2_val,
            dof=self._dof(fitted, 2 * n),
            time_residuals=time_resid,
            dm_residuals=dm_resid,
            noise_realizations=noise_realizations,
        )
