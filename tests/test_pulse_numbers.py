"""Tests for absolute pulse-number tracking.

The feature exists for exactly one reason, and the marquee test demonstrates
it: with nearest-pulse tracking, a cold start converges *cleanly* into a
cycle-slipped solution (chi2 ~ 2e6, the trap pinned in
test_fitter.py::test_convergence_does_not_imply_correctness); with pulse
numbers frozen from a trusted model, the same cold start recovers the true
solution (chi2 59.57).

PINT-free except the two parity checks (function-level importorskip).
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import jaxpint.par as jpar
from jaxpint import WLSFitter, build_model, native
from jaxpint.fitters import (
    compute_phase_residuals,
    compute_pulse_numbers,
    compute_time_residuals,
)

_DATA = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"


@pytest.fixture(scope="module")
def ngc(request):
    parsed = jpar.get_model(str(_DATA / "NGC6440E.par"))
    toa_data = native.get_TOAs(str(_DATA / "NGC6440E.tim"), parsed)
    model, nm = build_model(parsed, toa_data)
    return parsed, toa_data, model, nm


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------


class TestMechanics:
    def test_tracked_equals_nearest_near_solution(self, ngc):
        """At a good model, both modes agree to float precision.

        Pulse numbers computed from the same model just make the "nearest"
        assignment explicit, so the residuals must be identical -- any
        difference is an arithmetic error in the tracked path.
        """
        parsed, toa_data, model, _ = ngc
        pn = compute_pulse_numbers(model, toa_data, parsed.params)
        td_pn = toa_data.with_pulse_numbers(pn)
        r_near = compute_phase_residuals(model, toa_data, parsed.params)
        r_track = compute_phase_residuals(model, td_pn, parsed.params)
        npt.assert_allclose(np.asarray(r_track), np.asarray(r_near), atol=1e-9)

    def test_tracked_residuals_unwrapped_far_from_solution(self, ngc):
        """Far from the solution, tracked residuals exceed one cycle.

        This is the defining property: nearest-mode residuals are confined to
        (-0.5, 0.5] cycles no matter how wrong the model is, while tracked
        residuals grow without wrapping -- the gradient signal that makes
        recovery possible.
        """
        parsed, toa_data, model, _ = ngc
        pn = compute_pulse_numbers(model, toa_data, parsed.params)
        td_pn = toa_data.with_pulse_numbers(pn)

        names = list(parsed.params.names)
        v = np.asarray(parsed.params.values).copy()
        v[names.index("F0")] += 1e-6
        cold = parsed.params.with_values(jnp.asarray(v))

        r_near = np.asarray(compute_phase_residuals(model, toa_data, cold))
        r_track = np.asarray(compute_phase_residuals(model, td_pn, cold))
        assert np.abs(r_near).max() <= 0.5 + 1e-12  # wrapped, by construction
        assert np.abs(r_track).max() > 10.0  # unwrapped: many cycles of error

    def test_auto_mode_is_presence_based(self, ngc):
        """track_mode=None: tracking iff the TOAData carries pulse numbers."""
        parsed, toa_data, model, _ = ngc
        pn = compute_pulse_numbers(model, toa_data, parsed.params)
        td_pn = toa_data.with_pulse_numbers(pn + 1.0)  # shift all by one turn
        # auto on td_pn -> tracked: the +1 shows up as exactly +1 cycle
        r = np.asarray(compute_phase_residuals(model, td_pn, parsed.params))
        assert np.allclose(np.round(-np.median(r)), 1.0)
        # explicit nearest on the same data ignores the pulse numbers
        r_near = np.asarray(
            compute_phase_residuals(model, td_pn, parsed.params, "nearest")
        )
        assert np.abs(r_near).max() <= 0.5 + 1e-12

    def test_requesting_tracking_without_pn_raises(self, ngc):
        parsed, toa_data, model, _ = ngc
        with pytest.raises(ValueError, match="carries no pulse numbers"):
            compute_phase_residuals(
                model, toa_data, parsed.params, "use_pulse_numbers"
            )
        with pytest.raises(ValueError, match="track_mode must be"):
            compute_phase_residuals(model, toa_data, parsed.params, "sideways")

    def test_with_pulse_numbers_validates(self, ngc):
        _, toa_data, _, _ = ngc
        n = toa_data.n_toas
        with pytest.raises(ValueError, match="shape"):
            toa_data.with_pulse_numbers(np.zeros(n + 1))
        with pytest.raises(ValueError, match="finite"):
            toa_data.with_pulse_numbers(np.full(n, np.nan))
        with pytest.raises(ValueError, match="integer-valued"):
            toa_data.with_pulse_numbers(np.full(n, 1.5))

    def test_jit_and_grad_safe(self, ngc):
        """Tracking must trace: resolution is static, arithmetic is traced."""
        parsed, toa_data, model, _ = ngc
        pn = compute_pulse_numbers(model, toa_data, parsed.params)
        td_pn = toa_data.with_pulse_numbers(pn)

        f = jax.jit(lambda td, p: compute_time_residuals(model, td, p))
        r_jit = f(td_pn, parsed.params)
        r_eager = compute_time_residuals(model, td_pn, parsed.params)
        # JIT fuses/reorders the (phase.int - N) cancellation (~1e11 - 1e11),
        # shifting the result by ~2e-8 relative (~20 ps absolute) -- the same
        # reordering effect the likelihood JIT test tolerates. Not an error.
        npt.assert_allclose(np.asarray(r_jit), np.asarray(r_eager), rtol=1e-7)

        g = jax.grad(
            lambda v: jnp.sum(
                compute_time_residuals(
                    model, td_pn, parsed.params.with_values(v)
                )
                ** 2
            )
        )(parsed.params.values)
        assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# The marquee: cycle-slip rescue
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cold_start_recovers_with_pulse_numbers(ngc):
    """The trap documented on BaseFitResult.converged, and its cure.

    Same cold start as test_convergence_does_not_imply_correctness (F0
    perturbed by 1e-6 Hz): nearest-mode converges to a cycle-slipped chi2 of
    ~2e6; with pulse numbers frozen from the true model, the identical start
    recovers chi2 59.57 -- the correct solution.
    """
    parsed, toa_data, model, nm = ngc
    names = list(parsed.params.names)
    v = np.asarray(parsed.params.values).copy()
    v[names.index("F0")] += 1e-6
    cold = parsed.params.with_values(jnp.asarray(v))

    # Without tracking: the documented trap (pinned in test_fitter; asserted
    # loosely here only to prove the two runs differ for the stated reason).
    trapped = WLSFitter(model, toa_data, cold, noise_model=nm).fit_toas()
    assert float(trapped.chi2) > 1e5

    # With tracking: same cold start, same fitter, correct solution.
    pn = compute_pulse_numbers(model, toa_data, parsed.params)
    td_pn = toa_data.with_pulse_numbers(pn)
    rescued = WLSFitter(model, td_pn, cold, noise_model=nm).fit_toas()
    assert bool(rescued.converged)
    assert float(rescued.chi2) == pytest.approx(59.5747, abs=0.01)


# ---------------------------------------------------------------------------
# Sources: .tim -pn flags, writer round-trip, PINT parity
# ---------------------------------------------------------------------------


def test_tim_pn_flags_populate_pulse_numbers():
    """ecorr_fit_test.tim carries -pn on every TOA; the loader must use them."""
    parsed = jpar.get_model(str(_DATA / "ecorr_fit_test.par"))
    td = native.get_TOAs(str(_DATA / "ecorr_fit_test.tim"), parsed)
    assert td.pulse_number is not None
    pn = np.asarray(td.pulse_number)
    assert np.all(pn == np.round(pn))
    assert np.all(np.isfinite(pn))


def test_partial_pn_flags_dropped_with_warning(tmp_path):
    """All-or-nothing: partial -pn coverage warns and yields None."""
    tim = tmp_path / "partial.tim"
    tim.write_text(
        "FORMAT 1\nMODE 1\n"
        "a 1400.0 55000.5 1.0 gbt -pn 100\n"
        "b 1400.0 55001.5 1.0 gbt\n"
    )
    with pytest.warns(UserWarning, match="cover only 1/2"):
        td = native.get_TOAs(str(tim))
    assert td.pulse_number is None


def test_writer_round_trips_pn(tmp_path):
    """TOAData -> write_tim -> native re-read preserves the pulse numbers."""
    from jaxpint.native import toa_data_to_raw
    from jaxpint.tim import write_tim

    parsed = jpar.get_model(str(_DATA / "ecorr_fit_test.par"))
    td = native.get_TOAs(str(_DATA / "ecorr_fit_test.tim"), parsed)
    assert td.pulse_number is not None
    out = tmp_path / "rt.tim"
    write_tim(toa_data_to_raw(td), out)
    td2 = native.get_TOAs(str(out), parsed)
    assert td2.pulse_number is not None
    npt.assert_array_equal(np.asarray(td2.pulse_number), np.asarray(td.pulse_number))


@pytest.mark.slow
def test_pn_parity_vs_pint():
    """Bridge and native agree with PINT's pulse_number column, exactly."""
    pytest.importorskip("pint")
    import pint.models
    import pint.toa

    from jaxpint.bridge import pint_toas_to_jax

    m = pint.models.get_model(str(_DATA / "ecorr_fit_test.par"))
    toas = pint.toa.get_TOAs(str(_DATA / "ecorr_fit_test.tim"), model=m)
    assert "pulse_number" in toas.table.colnames

    bridged = pint_toas_to_jax(toas, model=m)
    assert bridged.pulse_number is not None
    npt.assert_array_equal(
        np.asarray(bridged.pulse_number), np.asarray(toas.table["pulse_number"])
    )

    parsed = jpar.get_model(str(_DATA / "ecorr_fit_test.par"))
    td = native.get_TOAs(str(_DATA / "ecorr_fit_test.tim"), parsed)
    npt.assert_array_equal(
        np.asarray(td.pulse_number), np.asarray(toas.table["pulse_number"])
    )


@pytest.mark.slow
def test_tracked_residual_parity_vs_pint():
    """Tracked residuals match PINT's track_mode='use_pulse_numbers'."""
    pytest.importorskip("pint")
    import pint.models
    import pint.residuals
    import pint.toa

    from jaxpint.bridge import (
        build_timing_model,
        pint_model_to_params,
        pint_toas_to_jax,
    )

    m = pint.models.get_model(str(_DATA / "ecorr_fit_test.par"))
    toas = pint.toa.get_TOAs(str(_DATA / "ecorr_fit_test.tim"), model=m)
    pres = pint.residuals.Residuals(
        toas, m, track_mode="use_pulse_numbers", subtract_mean=False
    )
    r_pint = pres.time_resids.to("s").value

    toa_data = pint_toas_to_jax(toas, model=m)
    params = pint_model_to_params(m).params
    model, _nm = build_timing_model(m, toas)
    r_ours = np.asarray(
        compute_time_residuals(model, toa_data, params, "use_pulse_numbers")
    )
    npt.assert_allclose(r_ours, r_pint, atol=5e-8)


# ---------------------------------------------------------------------------
# TrackMode enum + the par TRACK parameter
# ---------------------------------------------------------------------------


class TestTrackModeEnum:
    def test_enum_and_string_are_interchangeable(self, ngc):
        """StrEnum contract: members == their strings, both forms accepted."""
        from jaxpint.fitters import TrackMode

        parsed, toa_data, model, _ = ngc
        pn = compute_pulse_numbers(model, toa_data, parsed.params)
        td_pn = toa_data.with_pulse_numbers(pn)
        r_str = compute_phase_residuals(model, td_pn, parsed.params, "nearest")
        r_enum = compute_phase_residuals(
            model, td_pn, parsed.params, TrackMode.NEAREST
        )
        npt.assert_array_equal(np.asarray(r_str), np.asarray(r_enum))

    def test_invalid_mode_lists_the_valid_set(self, ngc):
        parsed, toa_data, model, _ = ngc
        with pytest.raises(ValueError, match="'nearest', 'use_pulse_numbers'"):
            compute_phase_residuals(model, toa_data, parsed.params, "sideways")


class TestTrackParParameter:
    """The par's TRACK line, honoured at TOAData construction (PINT semantics:
    "0" forbids tracking, "-2" demands it)."""

    def _par(self, track):
        return (
            "PSR J0000+0000\nPEPOCH 55000\nF0 100.0 1\nDM 15.0\n"
            f"TRACK {track}\n"
        )

    def _tim(self, pn=True):
        lines = ["FORMAT 1", "MODE 1"]
        for i in range(3):
            flag = f" -pn {100 + i}" if pn else ""
            lines.append(f"t{i} 1400.0 {55000 + i}.5 1.0 gbt{flag}")
        return "\n".join(lines) + "\n"

    def test_track_zero_strips_pn_flags(self, tmp_path):
        import io

        par = jpar.get_model(io.StringIO(self._par("0")))
        tim = tmp_path / "t.tim"
        tim.write_text(self._tim(pn=True))
        with pytest.warns(UserWarning, match="TRACK 0 forbids"):
            td = native.get_TOAs(str(tim), par)
        assert td.pulse_number is None

    def test_track_minus_two_warns_when_unfulfillable(self, tmp_path):
        import io

        par = jpar.get_model(io.StringIO(self._par("-2")))
        tim = tmp_path / "t.tim"
        tim.write_text(self._tim(pn=False))
        with pytest.warns(UserWarning, match="TRACK -2 requests"):
            td = native.get_TOAs(str(tim), par)
        assert td.pulse_number is None

    def test_track_minus_two_with_pn_is_silent(self, tmp_path):
        import io
        import warnings as _w

        par = jpar.get_model(io.StringIO(self._par("-2")))
        tim = tmp_path / "t.tim"
        tim.write_text(self._tim(pn=True))
        with _w.catch_warnings():
            _w.simplefilter("error")
            td = native.get_TOAs(str(tim), par)
        assert td.pulse_number is not None
        npt.assert_array_equal(np.asarray(td.pulse_number), [100.0, 101.0, 102.0])


# ---------------------------------------------------------------------------
# One-call convenience + the refreeze fixed point
# ---------------------------------------------------------------------------


class TestOneCallConvenience:
    def test_equals_the_two_step_form(self, ngc):
        parsed, toa_data, model, _ = ngc
        two_step = toa_data.with_pulse_numbers(
            compute_pulse_numbers(model, toa_data, parsed.params)
        )
        one_call = toa_data.with_computed_pulse_numbers(model, parsed.params)
        npt.assert_array_equal(
            np.asarray(one_call.pulse_number), np.asarray(two_step.pulse_number)
        )

    @pytest.mark.slow
    def test_refreeze_at_converged_fit_is_a_noop(self, ngc):
        """Recomputing pulse numbers at the fitted params reproduces them.

        The fixed-point property that bounds the "forgot to refreeze" hazard:
        whenever every tracked residual is under half a turn (i.e. any
        converged, correct fit), round(phase at fitted params) == the frozen
        numbers, so refreezing changes nothing. Forgetting only matters when
        the original assignments were partly *wrong* -- the iterative
        phase-connection scenario, which is exactly when you are editing
        assignments deliberately. PINT has the same property and likewise
        never auto-refreezes after a fit.
        """
        parsed, toa_data, model, nm = ngc
        td_pn = toa_data.with_computed_pulse_numbers(model, parsed.params)
        fit = WLSFitter(model, td_pn, parsed.params, noise_model=nm).fit_toas()
        assert bool(fit.converged)
        refrozen = toa_data.with_computed_pulse_numbers(model, fit.params)
        npt.assert_array_equal(
            np.asarray(refrozen.pulse_number), np.asarray(td_pn.pulse_number)
        )


# ---------------------------------------------------------------------------
# Phase-connection editing verbs
# ---------------------------------------------------------------------------


class TestEditingVerbs:
    """The manual-connection moves, pinned against the residual convention:
    residual = phase + delta_pulse_number - pulse_number."""

    def test_add_phase_turns_raises_selected_residuals(self, ngc):
        parsed, toa_data, model, _ = ngc
        td_pn = toa_data.with_computed_pulse_numbers(model, parsed.params)
        mjd = np.asarray(toa_data.mjd_int) + np.asarray(toa_data.mjd_frac)
        cut = float(np.median(mjd))

        before = np.asarray(compute_phase_residuals(model, td_pn, parsed.params))
        after = np.asarray(
            compute_phase_residuals(
                model, td_pn.add_phase_turns(2, after_mjd=cut), parsed.params
            )
        )
        sel = mjd > cut
        npt.assert_allclose(after[sel] - before[sel], 2.0, atol=1e-9)
        npt.assert_allclose(after[~sel], before[~sel], atol=1e-12)

    def test_shift_pulse_numbers_lowers_selected_residuals(self, ngc):
        parsed, toa_data, model, _ = ngc
        td_pn = toa_data.with_computed_pulse_numbers(model, parsed.params)
        mjd = np.asarray(toa_data.mjd_int) + np.asarray(toa_data.mjd_frac)
        cut = float(np.median(mjd))

        before = np.asarray(compute_phase_residuals(model, td_pn, parsed.params))
        after = np.asarray(
            compute_phase_residuals(
                model, td_pn.shift_pulse_numbers(2, after_mjd=cut), parsed.params
            )
        )
        sel = mjd > cut
        npt.assert_allclose(after[sel] - before[sel], -2.0, atol=1e-9)
        npt.assert_allclose(after[~sel], before[~sel], atol=1e-12)

    def test_matched_edits_cancel(self, ngc):
        """+k turns and +k assignment on the same TOAs: exact cancellation.

        This is the convention check in executable form -- if either sign
        flips, this fails before any user hits it interactively.
        """
        parsed, toa_data, model, _ = ngc
        td_pn = toa_data.with_computed_pulse_numbers(model, parsed.params)
        mjd = np.asarray(toa_data.mjd_int) + np.asarray(toa_data.mjd_frac)
        cut = float(np.median(mjd))

        base = np.asarray(compute_phase_residuals(model, td_pn, parsed.params))
        edited = td_pn.add_phase_turns(3, after_mjd=cut).shift_pulse_numbers(
            3, after_mjd=cut
        )
        both = np.asarray(compute_phase_residuals(model, edited, parsed.params))
        npt.assert_allclose(both, base, atol=1e-9)

    def test_mask_selection_and_validation(self, ngc):
        _, toa_data, _, _ = ngc
        n = toa_data.n_toas
        m = np.zeros(n, dtype=bool)
        m[:3] = True
        out = toa_data.add_phase_turns(0.25, mask=m)
        dpn = np.asarray(out.delta_pulse_number) - np.asarray(
            toa_data.delta_pulse_number
        )
        npt.assert_allclose(dpn[:3], 0.25)
        npt.assert_allclose(dpn[3:], 0.0)

        with pytest.raises(ValueError, match="exactly one of"):
            toa_data.add_phase_turns(1)
        with pytest.raises(ValueError, match="exactly one of"):
            toa_data.add_phase_turns(1, after_mjd=53000.0, mask=m)
        with pytest.raises(ValueError, match="bool array"):
            toa_data.add_phase_turns(1, mask=np.ones(n))  # float, not bool

    def test_shift_requires_pn_and_whole_turns(self, ngc):
        parsed, toa_data, model, _ = ngc
        with pytest.raises(ValueError, match="no pulse numbers"):
            toa_data.shift_pulse_numbers(1, after_mjd=53000.0)
        td_pn = toa_data.with_computed_pulse_numbers(model, parsed.params)
        with pytest.raises(ValueError, match="whole rotations"):
            td_pn.shift_pulse_numbers(0.5, after_mjd=53000.0)

    def test_without_pulse_numbers_reverts_to_nearest(self, ngc):
        parsed, toa_data, model, _ = ngc
        td_pn = toa_data.with_computed_pulse_numbers(model, parsed.params)
        stripped = td_pn.without_pulse_numbers()
        assert stripped.pulse_number is None
        r = np.asarray(compute_phase_residuals(model, stripped, parsed.params))
        assert np.abs(r).max() <= 0.5 + 1e-12  # nearest mode again
