"""Unit tests for the downhill line search in ``BaseFitter.fit_params``.

The line search (chi2 acceptance, lambda shrinking, rollback -- the traced port
of PINT's DownhillFitter loop) is exercised against a stub fitter whose
``_core_step`` proposes a *controlled multiple* of the exact Newton step on a
1-D quadratic. That makes every branch deterministic:

    overshoot = 1.0   exact step        -> accepted at lambda = 1, untouched
    overshoot = 2.5   recoverable       -> rejected at 1, accepted at 1/1.5
    overshoot = -1.0  wrong direction   -> no lambda helps, rollback + stall

Real-data behaviour (that the search is a no-op on a healthy fit) is pinned
separately at the bottom against NGC6440E, PINT-free via the native loader.

Layout note: these are deliberately NOT in test_fitter.py -- that module does
``pytest.importorskip("pint")`` at import time, which would silently skip all
of this in a PINT-less environment. Nothing here needs PINT.
"""

from __future__ import annotations

import pathlib
import types

import jax.numpy as jnp
import numpy as np

import jaxpint.fitters._base as fitter_base
from jaxpint.fitters._base import BaseFitter

from .helpers import make_params


class _ControlledStepFitter(BaseFitter):
    """1-D quadratic chi2 whose Gauss-Newton proposal overshoots on purpose.

    chi2(x) = ((x - target) / 1e-3)^2 via a single weighted residual. The exact
    Newton step is ``target - x``; ``_core_step`` proposes ``overshoot`` times
    it, so the line search's behaviour is a pure function of ``overshoot``.

    ``BaseFitter.__init__`` is bypassed: no TimingModel or TOAData exists here,
    and the only model attribute the base class touches on this path is
    ``phoff_name`` (setting it suppresses the synthetic-Offset column, keeping
    the quadratic exactly quadratic).
    """

    _SIGMA_R = 1e-3

    def __init__(self, params, *, target=3.0, overshoot=1.0, nan_at=None):
        self.params = params
        self.target = target
        self.overshoot = overshoot
        self.nan_at = nan_at
        self.model = types.SimpleNamespace(phoff_name="PHOFF")
        self.toa_data = None
        self.noise_model = None

    def _fit_residuals(self, params, external_delay):
        r = (params.values[0] - self.target) / self._SIGMA_R
        if self.nan_at is not None:
            r = jnp.where(params.values[0] == self.nan_at, jnp.nan, r)
        return jnp.array([r])

    def _fit_cinv(self, params, x):
        return x

    def _default_threshold(self):
        return 1e-14

    def _core_step(self, params, external_delay, threshold):
        x = params.values[0]
        proposal = x + self.overshoot * (self.target - x)
        return params.values.at[0].set(proposal), jnp.eye(1), None

    def fit_toas(self, maxiter=fitter_base._DEFAULT_MAXITER, **kwargs):
        raise NotImplementedError("unit stub; use fit_params")


def _fit(fitter, maxiter):
    return float(fitter.fit_params(maxiter=maxiter).values[0])


def test_exact_step_is_accepted_untouched():
    """A chi2-decreasing proposal passes at lambda = 1, bit-exact.

    Landing on the target *exactly* is the assertion that lambda was 1: any
    shrunk step would leave a (1 - lambda) remainder.
    """
    f = _ControlledStepFitter(make_params(("X",), [0.0]), target=3.0, overshoot=1.0)
    assert _fit(f, maxiter=10) == 3.0


def test_overshooting_step_is_shrunk_and_converges():
    """overshoot = 2.5: rejected at lambda = 1, accepted at 1/1.5.
    """
    f = _ControlledStepFitter(make_params(("X",), [0.0]), target=3.0, overshoot=2.5)
    assert abs(_fit(f, maxiter=30) - 3.0) < 1e-3


def test_undamped_gauss_newton_diverges_on_the_same_problem(monkeypatch):
    """The contrast case: with the acceptance test disabled, overshoot = 2.5
    multiplies the offset by 1.5 every iteration and the fit runs away.

    This is the test that justifies the machinery's existence -- if it ever
    fails (the undamped fit converging), the line search is dead weight and
    the comparison in test_overshooting_step_is_shrunk_and_converges proves
    nothing.
    """
    monkeypatch.setattr(fitter_base, "_MAX_CHI2_INCREASE", float("inf"))
    f = _ControlledStepFitter(make_params(("X",), [0.0]), target=3.0, overshoot=2.5)
    assert abs(_fit(f, maxiter=30) - 3.0) > 1e3  # 3 * 1.5^30 ~ 5.7e5


def test_unusable_step_rolls_back_to_start():
    """overshoot = -1 walks *away* from the minimum: no lambda can help, so the
    search exhausts to _MIN_LAMBDA, stalls, and returns the start unmoved.

    Bit-exact equality is the rollback claim -- a partially applied step would
    show up here. Even the smallest lambda (1e-4) raises chi2 by ~800 at this
    weighting, far over the 1e-2 allowance, so every trial is genuinely
    rejected (at O(1) chi2 scale the same geometry was *accepted* as sub-
    allowance drift -- see the class docstring).
    """
    f = _ControlledStepFitter(make_params(("X",), [1.0]), target=3.0, overshoot=-1.0)
    assert _fit(f, maxiter=10) == 1.0


def test_stalled_fit_reports_not_converged():
    """The rollback must not masquerade as success: the remaining step at the
    returned point is the full (rejected) proposal, far above tolerance."""
    f = _ControlledStepFitter(make_params(("X",), [1.0]), target=3.0, overshoot=-1.0)
    fitted = f.fit_params(maxiter=10)
    assert float(f.step_sigma(fitted)) > fitter_base._STEP_SIGMA_TOL


def test_nan_start_is_escaped():
    """A NaN chi2 at the current point must not veto every finite trial.

    NaN <= x is False, so without the explicit escape clause the line search
    would reject all lambdas and stall *at the NaN point*. The clause accepts
    any finite trial as an improvement over NaN.
    """
    f = _ControlledStepFitter(
        make_params(("X",), [1.0]), target=3.0, overshoot=1.0, nan_at=1.0
    )
    x = _fit(f, maxiter=10)
    assert np.isfinite(x) and x == 3.0


# ---------------------------------------------------------------------------
# Real-data no-op check
# ---------------------------------------------------------------------------

_DATA = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"


def test_line_search_is_a_no_op_on_a_healthy_fit(monkeypatch):
    """On NGC6440E, damped and undamped fits agree bit-for-bit.

    Every proposal on this dataset decreases chi2, so lambda stays 1 and the
    accepted steps are the plain Gauss-Newton ones. This is the guarantee that
    made an off-switch unnecessary: the guard only changes results where the
    unguarded fitter was already misbehaving. If this test starts failing, the
    line search has begun rejecting healthy steps and that guarantee is gone.
    """
    from .helpers import ngc6440e_native_fitter

    fitter, _parsed = ngc6440e_native_fitter()

    damped = fitter.fit_toas()
    monkeypatch.setattr(fitter_base, "_MAX_CHI2_INCREASE", float("inf"))
    undamped = fitter.fit_toas()

    np.testing.assert_array_equal(
        np.asarray(damped.params.values), np.asarray(undamped.params.values)
    )
    assert float(damped.chi2) == float(undamped.chi2)
