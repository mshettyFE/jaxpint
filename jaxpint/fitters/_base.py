"""Base classes and shared pure-JAX functions for all fitters.

:meth:`BaseFitter.fit_params` wraps the Gauss-Newton iteration in a
``jax.custom_vjp`` so that gradients of the *fitted* parameters with
respect to frozen parameters or an ``external_delay`` never
backpropagate through the iteration loop -- the backward pass is one
linear solve (``_solve_gn_normal``) plus one VJP of the stationarity
map (``_optimality``), via the implicit function theorem.  The full
derivation, the code-to-math mapping, and the convergence caveat
(:meth:`BaseFitter.fit_gap`) live in the
:doc:`differentiable-fitting guide </guides/differentiable_fitting>`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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


class BaseFitResult(eqx.Module):
    """Common fields shared by all fit results."""

    params: ParameterVector
    covariance_matrix: Float[Array, "n_free n_free"]
    chi2: Float[Array, ""]
    dof: int = eqx.field(static=True)

    @property
    def parameter_uncertainties(self) -> Float[Array, " n_free"]:
        """Square root of the covariance diagonal (1-sigma marginal errors)."""
        return jnp.sqrt(jnp.diag(self.covariance_matrix))

    @property
    def correlation_matrix(self) -> Float[Array, "n_free n_free"]:
        """Correlation matrix -- covariance rescaled to unit diagonal, ``D^-1 C D^-1`` with
        ``D = diag(sigma)``. Zero-variance rows/cols are left unscaled."""
        errors = jnp.sqrt(jnp.diag(self.covariance_matrix))
        errors_safe = jnp.where(errors == 0, 1.0, errors)
        return (self.covariance_matrix / errors_safe).T / errors_safe

    @property
    def reduced_chi2(self) -> Float[Array, ""]:
        """``chi2 / dof`` (0-d array), or NaN when ``dof <= 0``."""
        return self.chi2 / self.dof if self.dof > 0 else jnp.asarray(jnp.nan)


# ---------------------------------------------------------------------------
# Base fitter
# ---------------------------------------------------------------------------


class BaseFitter(ABC):
    """Base for all JaxPINT fitters — differentiable by default.

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

    # -- Per-fitter hooks for the differentiable solve ----------------------

    @abstractmethod
    def _fit_cinv(
        self, params: ParameterVector, x: Float[Array, " n"]
    ) -> Float[Array, " n"]:
        """Apply the noise-covariance solve ``C^{-1} x`` at *params*."""
        ...

    @abstractmethod
    def _core_step(
        self,
        params: ParameterVector,
        external_delay: Optional[Float[Array, " n_toas"]],
        threshold: float,
        **opts,
    ) -> tuple[
        Float[Array, " n_params"],
        Float[Array, "n_free n_free"],
        Optional[Float[Array, " n_epochs"]],
    ]:
        """One Gauss-Newton step at *params*.

        Returns ``(new_values, covariance, noise_realizations)``.  The
        covariance is evaluated at *params* and has offset column stripped.
        ``noise_realizations`` is
        ``None`` for fitters that don't estimate them.
        """
        ...

    @abstractmethod
    def _default_threshold(self, **opts) -> float:
        """Default SVD threshold, matching :meth:`fit_toas`'s convention."""
        ...

    def _fit_residuals(
        self,
        params: ParameterVector,
        external_delay: Optional[Float[Array, " n_toas"]],
    ) -> Float[Array, " n"]:
        """Residual vector the fit minimizes (narrowband default).

        Wideband overrides this with the stacked ``[time; dm]`` layout.
        """
        r = compute_time_residuals(self.model, self.toa_data, params)
        return r if external_delay is None else r - external_delay

    def _offset_vector(self) -> Optional[Float[Array, " n"]]:
        """Synthetic-Offset column, or ``None`` with an explicit PhaseOffset."""
        if self.model.phoff_name is not None:
            return None
        return jnp.ones(self.toa_data.n_toas)

    # -- Shared differentiable machinery -------------------------------------

    def _fit_design_free(self, params: ParameterVector) -> Float[Array, "n n_free"]:
        """``M = -dr/dy`` restricted to the free columns."""

        def resid_fn(values: Float[Array, " n_params"]):
            return self._fit_residuals(params.with_values(values), None)

        # jacfwd: n_params forward tangents (jacrev would materialize
        # an n x n cotangent basis and OOM high-cadence pulsars).
        J = jax.jacfwd(resid_fn)(params.values)
        return -J[:, params.free_indices_array()]

    def _optimality(
        self,
        M_star: Float[Array, "n n_free"],
        params: ParameterVector,
        external_delay: Optional[Float[Array, " n_toas"]],
    ) -> Float[Array, " n_free"]:
        """Stationarity residual ``G = M*^T C^{-1} (I - P) r``.

        At convergence the free parameters satisfy the stationarity
        condition

        .. math::

            G(y, \\theta) = M^T C^{-1} (I - P)\\, r(y, \\theta) = 0

        where ``M = -dr/dy`` restricted to the free columns, ``C`` is the
        fitter's noise covariance, ``theta`` collects the frozen
        parameter values and any ``external_delay``, and ``r`` are the
        full residuals.  ``P`` is the ``C``-weighted projector onto the
        synthetic-Offset direction: subtracting the ``C``-weighted mean
        reproduces the offset column of the forward solve.  ``M_star``
        is held constant (Gauss-Newton freezing).
        """
        r = self._fit_residuals(params, external_delay)
        z = self._fit_cinv(params, r)
        o = self._offset_vector()
        if o is not None:
            zo = self._fit_cinv(params, o)
            z = z - zo * ((o @ z) / (o @ zo))
        return M_star.T @ z

    def _solve_gn_normal(
        self,
        params: ParameterVector,
        M_star: Float[Array, "n n_free"],
        v_free: Float[Array, " n_free"],
        threshold: float,
    ) -> Float[Array, " n_free"]:
        """Solve ``H u = v_free`` with ``H = M_ms^T C^{-1} M_ms``.

        ``M_ms`` removes each column's ``C``-weighted mean, and the solve uses the
        same column normalization and relative SVD threshold as the
        forward step (:func:`_normalized_svd_solve`), so the backward pass
        truncates the same degenerate directions.  Note: the covariance
        returned by the *augmented* GLS solve is not accurate enough here
        (its timing block carries the 1e-40 mixed-model ridge), which is
        why the backward pass builds ``H`` directly instead of reusing the
        core's covariance.
        """
        o = self._offset_vector()
        if o is not None:
            zo = self._fit_cinv(params, o)
            col_wmean = (zo @ M_star) / (o @ zo)
            M_ms = M_star - o[:, None] * col_wmean[None, :]
        else:
            M_ms = M_star
        cinv_M = jax.vmap(
            lambda col: self._fit_cinv(params, col), in_axes=1, out_axes=1
        )(M_ms)
        H = M_ms.T @ cinv_M
        u, _cov, _norms = _normalized_svd_solve(H, v_free, threshold)
        return u

    def fit_params(
        self,
        params: Optional[ParameterVector] = None,
        external_delay: Optional[Float[Array, " n_toas"]] = None,
        *,
        maxiter: int = 1,
        threshold: Optional[float] = None,
        **core_opts,
    ) -> ParameterVector:
        """Differentiable Gauss-Newton fit returning only the fitted parameters.

        Equivalent to ``fit_toas(...).params`` but skips result
        construction — the lean entry point for eager gradient/vmap loops.

        The implicit gradients assume a *converged* fit — verify with
        :meth:`fit_gap` when starting far from the solution or using a
        small ``maxiter``.

        Parameters
        ----------
        params
            Starting parameters; defaults to ``self.params``.  Pass a
            traced ``ParameterVector`` (via ``with_values``) to
            differentiate with respect to frozen parameter values.
        external_delay : array (n_toas,), optional
            Deterministic delay (seconds) subtracted from the (time)
            residuals before fitting, e.g. an injected CW signal.
        maxiter, threshold
            As for :meth:`fit_toas`.
        **core_opts
            Fitter-specific solve options (e.g. ``full_cov`` for GLS).
        """
        skeleton = self.params if params is None else params
        if skeleton.n_free == 0:
            raise ValueError("fit_params: params has no free parameters.")
        thr = self._default_threshold(**core_opts) if threshold is None else threshold
        free_idx = skeleton.free_indices_array()
        n_iter = max(1, maxiter)

        def _iterate(values, ext):
            for _ in range(n_iter):
                values, _cov, _nr = self._core_step(
                    skeleton.with_values(values), ext, thr, **core_opts
                )
            return values

        @jax.custom_vjp
        def _fixed_point(values0, ext):
            return _iterate(values0, ext)

        def _fp_fwd(values0, ext):
            y_star = _iterate(values0, ext)
            return y_star, (y_star, ext)

        def _fp_bwd(res, v):
            y_star, ext = res
            pv_star = skeleton.with_values(y_star)
            M_star = self._fit_design_free(pv_star)
            # IFT: dy*/dtheta = H^{-1} dG/dtheta.
            u = self._solve_gn_normal(pv_star, M_star, v[free_idx], thr)
            _G, vjp_fn = jax.vjp(
                lambda vals, e: self._optimality(M_star, skeleton.with_values(vals), e),
                y_star,
                ext,
            )
            g_values, g_ext = vjp_fn(u)
            # theta is the *frozen* entries only; the free entries of
            # values0 are just the iteration seed and get zero gradient.
            g_values = g_values.at[free_idx].set(0.0)
            # Frozen entries of the output pass through from values0.
            g_values = g_values + v.at[free_idx].set(0.0)
            return g_values, g_ext

        _fixed_point.defvjp(_fp_fwd, _fp_bwd)

        return skeleton.with_values(_fixed_point(skeleton.values, external_delay))

    def fit_gap(
        self,
        params: Optional[ParameterVector] = None,
        external_delay: Optional[Float[Array, " n_toas"]] = None,
        threshold: Optional[float] = None,
        **core_opts,
    ) -> Float[Array, " n_free"]:
        """Free-parameter update of one further Gauss-Newton step.

        Convergence diagnostic: ~0 (in parameter units) at a true fixed
        point.  The implicit gradients of :meth:`fit_params` /
        :meth:`fit_toas` are trustworthy when the gap is a small fraction
        of each parameter's posterior sigma.
        """
        skeleton = self.params if params is None else params
        thr = self._default_threshold(**core_opts) if threshold is None else threshold
        new_values, _cov, _nr = self._core_step(
            skeleton, external_delay, thr, **core_opts
        )
        idx = skeleton.free_indices_array()
        return new_values[idx] - skeleton.values[idx]


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
    """JIT-compiled Jacobian and design matrix computation.

    jacfwd: n_params forward tangents. jacrev would materialize an
    n_toas x n_toas cotangent basis and OOM high-cadence pulsars. Forward
    mode differs from reverse only in physically-negligible near-cancelling
    entries (|value| below the ~1e-8 absolute floor), which no consumer of
    this design matrix observes.
    """
    free_indices = params.free_indices_array()

    def time_resid_fn(all_values: Float[Array, " n_params"]):
        p = params.with_values(all_values)
        return compute_time_residuals(model, toa_data, p)

    J = jax.jacfwd(time_resid_fn)(params.values)
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
    and does plain GLS; the two give identical ``dpars``.

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
