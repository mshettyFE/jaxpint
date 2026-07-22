"""Post-fit diagnostics: whitened residuals, normality tests, and the F-test.

Whitening is pure JAX (JIT-safe); the statistical tests are deliberately
numpy/scipy-side -- they are human-facing diagnostics with no gradient or
tracing story, and scipy is a guaranteed dependency (jax itself requires
``scipy>=1.13``).

Wideband whitening (the stacked ``[time; dm]`` layout) is NOT covered yet: the
stacking lives inside ``WidebandGLSFitter``'s private hooks, and pulling it
through this interface deserves its own design rather than a bolt-on.
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jaxpint.utils import woodbury_solve

__all__ = [
    "whiten_residuals",
    "whiten_wideband_residuals",
    "normality_tests",
    "NormalityReport",
    "ftest",
    "ftest_results",
    "FTestResult",
]


def whiten_residuals(
    residuals: Float[Array, " n_toas"],
    toa_data,
    params,
    noise_model,
    *,
    noise_realizations: Optional[Float[Array, " n_basis"]] = None,
) -> Float[Array, " n_toas"]:
    """Whiten residuals under a noise model: ``(r - U b) / sigma_scaled``.

    ``b`` is the conditional mean of the correlated-noise (GP) coefficients
    given the residuals, ``b = Phi U^T C^{-1} r``. A GLS fit already computes
    it (``GLSFitResult.noise_realizations``); pass it via *noise_realizations*
    to whiten exactly what the fit saw. Without it, ``b`` is computed here with
    one Woodbury solve -- which is what makes this function usable on any
    residual vector (marginalized-likelihood workflows, injected-signal
    checks), not only as a fit post-processor.

    With no correlated components the whole correlated term vanishes and this
    reduces to ``r / sigma_scaled``.

    Notes
    -----
    *residuals* are used as given -- typically the mean-subtracted residuals a
    fit reports. No re-demeaning happens here (matching PINT, which whitens
    its already-demeaned residuals).
    """
    r = jnp.asarray(residuals)
    sigma = noise_model.scaled_sigma(toa_data, params)
    Ndiag, U, Phidiag = noise_model.covariance(toa_data, params)
    return _whiten(r, sigma, Ndiag, U, Phidiag, noise_realizations)


def _whiten(r, sigma, Ndiag, U, Phidiag, noise_realizations):
    """Core whitening on an explicit decomposition: ``(r - U b) / sigma``.

    Shared by the narrowband and wideband entry points so the formula (and
    its conditional-mean computation) exists exactly once.
    """
    if U.shape[1] == 0:
        return r / sigma
    if noise_realizations is None:
        cinv_r = woodbury_solve(Ndiag, U, Phidiag, r[:, None])[:, 0]
        noise_realizations = Phidiag * (U.T @ cinv_r)
    return (r - U @ jnp.asarray(noise_realizations)) / sigma


def whiten_wideband_residuals(
    time_residuals: Float[Array, " n_toas"],
    dm_residuals: Float[Array, " n_toas"],
    toa_data,
    params,
    noise_model,
    *,
    noise_realizations: Optional[Float[Array, " n_basis"]] = None,
) -> tuple[Float[Array, " n_toas"], Float[Array, " n_toas"]]:
    """Whiten wideband residuals; returns ``(whitened_time, whitened_dm)``.

    The wideband counterpart of :func:`whiten_residuals`, over the stacked
    ``[time; dm]`` system assembled by
    :func:`jaxpint.fitters.wideband.stack_wideband_noise` -- the same stacking
    the wideband GLS fit uses, so a ``WidebandGLSFitResult``'s
    ``noise_realizations`` drops straight in. The return matches the result's
    ``time_residuals``/``dm_residuals`` split (PINT's stacked
    ``calc_wideband_whitened_resids`` is the concatenation of the two).

    Note the DM caveat documented on ``stack_wideband_noise``: the correlated
    basis has no DM-block rows (DM measurements are white-only in this noise
    model), so DM whitening divides by ``scaled_dm_sigma`` and nothing else.
    *noise_model* may be ``None`` for raw-error whitening, mirroring the
    fitter.
    """
    from .wideband import stack_wideband_noise

    r = jnp.concatenate([jnp.asarray(time_residuals), jnp.asarray(dm_residuals)])
    sigma_toa, Ndiag, U, Phidiag, sigma_dm = stack_wideband_noise(
        noise_model, toa_data, params
    )
    sigma = jnp.concatenate([sigma_toa, sigma_dm])
    w = _whiten(r, sigma, Ndiag, U, Phidiag, noise_realizations)
    n = time_residuals.shape[0]
    return w[:n], w[n:]


class NormalityReport(NamedTuple):
    """Result of :func:`normality_tests`.

    ``ks_stat``/``ks_p``: Kolmogorov-Smirnov statistic and p-value against
    N(0, 1). ``ad_stat``: Anderson-Darling A^2 against N(0, 1);
    ``ad_critical`` maps significance level to the case-0 asymptotic critical
    value (Stephens 1974) -- ``ad_stat`` above ``ad_critical[0.01]`` rejects
    normality at 1%. A p-value for case-0 A^2 has no closed form worth
    hand-rolling, so critical values are reported instead of inventing one.
    """

    ks_stat: float
    ks_p: float
    ad_stat: float
    ad_critical: dict


# Case-0 (fully specified null) asymptotic critical values for A^2,
# Stephens (1974), Table 1. NOT scipy.stats.anderson's values -- those are
# case 3 (mean and variance estimated from the data), a different null.
_AD_CASE0_CRITICAL = {0.15: 1.610, 0.10: 1.933, 0.05: 2.492, 0.025: 3.070, 0.01: 3.857}


def normality_tests(whitened) -> NormalityReport:
    """KS + Anderson-Darling of whitened residuals against N(0, 1).

    The null is *fully specified*: whitening fixes the scale, so there is no
    estimated-parameter correction. That is why the A^2 statistic is computed
    directly here rather than via ``scipy.stats.anderson``, whose null is
    "normal with estimated mean/variance" and whose critical values would be
    silently wrong for this question.
    """
    from scipy import stats

    w = np.sort(np.asarray(whitened, dtype=np.float64))
    n = w.size
    if n < 8:
        raise ValueError(f"normality_tests needs >= 8 residuals, got {n}")

    ks_stat, ks_p = stats.kstest(w, "norm")

    # A^2 = -n - (1/n) sum (2i-1) [ln F(w_i) + ln(1 - F(w_{n+1-i}))]
    cdf = stats.norm.cdf(w)
    eps = np.finfo(np.float64).tiny
    cdf = np.clip(cdf, eps, 1.0 - 1e-16)
    i = np.arange(1, n + 1)
    a2 = -n - np.mean((2 * i - 1) * (np.log(cdf) + np.log1p(-cdf[::-1])))

    return NormalityReport(
        ks_stat=float(ks_stat),
        ks_p=float(ks_p),
        ad_stat=float(a2),
        ad_critical=dict(_AD_CASE0_CRITICAL),
    )


class FTestResult(NamedTuple):
    """``f_stat`` and the probability that the chi2 improvement is chance.

    Small ``p`` -> the extra parameters are warranted; ``p`` near 1 -> the
    richer model should likely be rejected. ``f_stat`` is NaN when the test
    could not be performed (equal dof) and 0.0 when the richer model fit no
    better (where ``p`` is pinned to 1.0), mirroring PINT's ``FTest``.
    """

    f_stat: float
    p: float


def ftest(
    chi2_simple: float, dof_simple: int, chi2_complex: float, dof_complex: int
) -> FTestResult:
    """Nested-model F-test, Sherpa/PINT convention.

    *simple* is the model with fewer free parameters (larger dof). Follows
    ``pint.utils.FTest`` including its edge conventions: equal dof -> warn and
    return NaN; a richer model that fits no better -> p = 1.0.
    """
    from scipy.special import fdtrc

    delta_dof = dof_simple - dof_complex
    if delta_dof == 0:
        warnings.warn("ftest: models have equal degrees of freedom", stacklevel=2)
        return FTestResult(f_stat=math.nan, p=math.nan)
    if delta_dof < 0:
        raise ValueError(
            "ftest: the simple model must have MORE degrees of freedom "
            f"(fewer free parameters); got dof_simple={dof_simple} < "
            f"dof_complex={dof_complex}. Swap the arguments."
        )
    delta_chi2 = float(chi2_simple) - float(chi2_complex)
    if delta_chi2 <= 0:
        warnings.warn(
            "ftest: the richer model did not improve chi2; p = 1", stacklevel=2
        )
        return FTestResult(f_stat=0.0, p=1.0)
    f_stat = (delta_chi2 / delta_dof) / (float(chi2_complex) / dof_complex)
    return FTestResult(
        f_stat=float(f_stat), p=float(fdtrc(delta_dof, dof_complex, f_stat))
    )


def ftest_results(simple, complex) -> FTestResult:
    """:func:`ftest` on two fit results (``.chi2`` / ``.dof``)."""
    return ftest(
        float(simple.chi2), int(simple.dof), float(complex.chi2), int(complex.dof)
    )
