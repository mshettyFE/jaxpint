"""Tests for jaxpint.frequentist.stats (frequentist F-stat detection stats)."""

import numpy as np
from scipy.stats import chi2, ncx2

from jaxpint.frequentist.stats import (
    chi2_threshold,
    detection_probability,
    h0_min_from_lambda,
)


def test_chi2_threshold_is_the_null_upper_quantile():
    # the threshold is the value the null chi^2(dof) exceeds with probability fap
    for dof, fap in [(4, 1e-3), (4, 1e-2), (2, 1e-4)]:
        thr = chi2_threshold(fap, dof)
        assert np.isclose(chi2.sf(thr, dof), fap, rtol=1e-6)


def test_detection_probability_null_equals_fap():
    # at lambda = 0 the signal law is the null, so P_det at the fap-threshold is fap
    dof, fap = 4, 1e-3
    thr = chi2_threshold(fap, dof)
    assert np.isclose(detection_probability(thr, 0.0, dof), fap, rtol=1e-6)


def test_detection_probability_matches_monte_carlo():
    # independent oracle: the fraction of ncx2 draws above the threshold
    dof, lam, thr = 4, 20.0, 15.0
    draws = ncx2.rvs(dof, lam, size=400_000, random_state=np.random.default_rng(0))
    assert np.isclose(
        detection_probability(thr, lam, dof), (draws > thr).mean(), atol=3e-3
    )


def test_detection_probability_monotone_in_lambda():
    thr = chi2_threshold(1e-3, 4)
    p = detection_probability(thr, np.array([0.0, 5.0, 20.0, 60.0]), 4)
    assert np.all(np.diff(p) > 0)


def test_h0_min_recovers_beta():
    # at the returned h0_min, the orientation-averaged detection prob equals beta
    thr, beta = chi2_threshold(1e-3, 4), 0.95
    lam1 = np.random.default_rng(1).uniform(1e26, 1e27, size=(3, 64))
    h0 = h0_min_from_lambda(thr, lam1, dof=4, beta=beta)
    pdet = detection_probability(thr, (h0[:, None] ** 2) * lam1, 4).mean(1)
    assert np.allclose(pdet, beta, atol=2e-3)


def test_h0_min_scales_as_inverse_sqrt_lambda():
    # P_det depends on h0^2 * lambda, so h0_min ∝ 1/sqrt(lambda)
    thr = chi2_threshold(1e-3, 4)
    lam1 = np.full((1, 32), 1e26)
    h0a = h0_min_from_lambda(thr, lam1, 4)
    h0b = h0_min_from_lambda(thr, 4.0 * lam1, 4)
    assert np.isclose(h0b, h0a / 2.0, rtol=1e-2)


def test_h0_min_scalar_input_returns_scalar():
    thr = chi2_threshold(1e-3, 4)
    h0 = h0_min_from_lambda(thr, np.full(48, 1e26), 4)  # 1-D lam1 -> scalar
    assert np.ndim(h0) == 0
