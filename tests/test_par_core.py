"""Unit tests for the shared, PINT-free ``jaxpint.par`` core.

Exercises :func:`jaxpint.par.core.raw_params_to_result` in isolation (no PINT
objects involved) -- unit coercion, mask handling, pair splitting, alias
synthesis, the int-valued-float dual exposure, and the non-finite guard -- plus
the source-level invariant that ``jaxpint.par`` imports no PINT.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from jaxpint.par import ParamKind, RawParam, raw_params_to_result
from jaxpint.par.registry import Component

_REPO = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Shared core in isolation (no PINT objects involved)
# ---------------------------------------------------------------------------


def test_raw_params_to_result_basic_kinds():
    raw = [
        RawParam("F0", ParamKind.FLOAT, value=300.0, unit="Hz", frozen=False),
        RawParam("RAJ", ParamKind.ANGLE, value=1.234, frozen=True),
        RawParam("PEPOCH", ParamKind.MJD, mjd_split=(55000.0, 0.25), frozen=True),
        RawParam("PLANET_SHAPIRO", ParamKind.BOOL, bool_value=True),
        RawParam("NHARMS", ParamKind.INT, int_value=7),
        RawParam("ECL", ParamKind.STR, str_value="IERS2010"),
    ]
    res = raw_params_to_result(raw, component_set={Component.SPINDOWN}, binary_model=None)

    pv = res.params
    assert pv.names == ("F0", "RAJ", "PEPOCH")
    # F0 unchanged, RAJ stays radians, PEPOCH stores fractional day only.
    np.testing.assert_array_equal(np.asarray(pv.values), np.array([300.0, 1.234, 0.25]))
    assert pv.units == ("Hz", "rad", "day")
    assert pv.frozen_mask == (False, True, True)
    assert pv.epoch_int_values == {"PEPOCH": 55000.0}
    assert res.bool_params == {"PLANET_SHAPIRO": True}
    assert res.int_params == {"NHARMS": 7}
    assert res.metadata == {"ECL": "IERS2010"}
    assert res.component_set == {Component.SPINDOWN}


def test_raw_params_to_result_deg_to_rad():
    """deg is swapped for rad in compound units (deg -> rad, deg/yr -> rad/yr).

    This mirrors the original bridge behaviour exactly: only the angular base is
    converted; the time base (``/yr``) is left untouched.
    """
    raw = [
        RawParam("OM", ParamKind.FLOAT, value=180.0, unit="deg"),
        RawParam("OMDOT", ParamKind.FLOAT, value=1.0, unit="deg / yr"),
    ]
    res = raw_params_to_result(raw, component_set=set())
    om, omdot = np.asarray(res.params.values)
    assert np.isclose(om, np.pi)            # 180 deg -> pi rad
    assert res.params.units[0] == "rad"
    # 1 deg/yr -> pi/180 rad/yr (deg base swapped for rad; /yr preserved)
    assert np.isclose(omdot, np.pi / 180.0)
    assert "rad" in res.params.units[1] and "yr" in res.params.units[1]


def test_raw_params_to_result_equad_us_to_s():
    """EQUAD/ECORR mask params convert microseconds -> seconds."""
    raw = [
        RawParam(
            "EQUAD1", ParamKind.MASK, value=0.5, unit="us", frozen=False,
            mask_key="-fe", mask_key_value="430",
        ),
    ]
    res = raw_params_to_result(raw, component_set=set())
    assert np.isclose(float(res.params.values[0]), 0.5e-6)
    assert res.params.units == ("s",)
    mi = res.mask_info["EQUAD1"]
    assert (mi.key, mi.key_value, mi.key_value2) == ("-fe", "430", None)


def test_raw_params_to_result_pair_split():
    raw = [
        RawParam("WAVE1", ParamKind.PAIR, value_pair=(1.0, 2.0), unit="s", frozen=True),
    ]
    res = raw_params_to_result(raw, component_set=set())
    assert res.params.names == ("WAVE1_A", "WAVE1_B")
    np.testing.assert_array_equal(np.asarray(res.params.values), np.array([1.0, 2.0]))
    assert res.params.units == ("s", "s")


def test_raw_params_to_result_alias_rnamp_and_fb():
    """RNAMP/RNIDX -> TNREDAMP/TNREDGAM and FB0/FB1 -> PB/PBDOT synthesis."""
    raw = [
        RawParam("RNAMP", ParamKind.FLOAT, value=1e-13, unit="", frozen=True),
        RawParam("RNIDX", ParamKind.FLOAT, value=-3.0, unit="", frozen=True),
        RawParam("FB0", ParamKind.FLOAT, value=1e-5, unit="Hz", frozen=False),
        RawParam("FB1", ParamKind.FLOAT, value=-1e-20, unit="Hz / s", frozen=False),
    ]
    res = raw_params_to_result(raw, component_set=set())
    names = res.params.names
    assert {"TNREDAMP", "TNREDGAM", "PB", "PBDOT"}.issubset(set(names))
    idx = {n: i for i, n in enumerate(names)}
    vals = np.asarray(res.params.values)
    assert np.isclose(vals[idx["TNREDGAM"]], 3.0)            # -RNIDX
    assert np.isclose(vals[idx["PB"]], 1.0 / (1e-5 * 86400)) # 1/(FB0*86400)
    assert np.isclose(vals[idx["PBDOT"]], 1e-20 / (1e-5) ** 2)


def test_raw_params_to_result_int_valued_floats_dual_exposed():
    """Semantically-int floats land in BOTH the vector and int_params."""
    raw = [RawParam("TNREDC", ParamKind.FLOAT, value=30.0, unit="")]
    res = raw_params_to_result(raw, component_set=set())
    assert "TNREDC" in res.params.names
    assert res.int_params == {"TNREDC": 30}


def test_raw_params_to_result_nonfinite_guard():
    raw = [RawParam("F0", ParamKind.FLOAT, value=float("inf"), unit="Hz")]
    with pytest.raises(ValueError, match="Non-finite"):
        raw_params_to_result(raw, component_set=set())


def test_metadata_extra_merged():
    res = raw_params_to_result(
        [], component_set=set(), metadata_extra={"_SWX_THETA0_RAD": "0.5"}
    )
    assert res.metadata == {"_SWX_THETA0_RAD": "0.5"}


# ---------------------------------------------------------------------------
# UNITS guard
#
# Without it a ``UNITS TCB`` par loads clean and is silently treated as TDB --
# a wrong answer rather than an error, across every par file in IPTA DR1 and
# EPTA DR2.
# ---------------------------------------------------------------------------


def _units_raw(value: str) -> list[RawParam]:
    return [RawParam("UNITS", ParamKind.STR, str_value=value)]


def test_units_tdb_accepted():
    res = raw_params_to_result(_units_raw("TDB"), component_set=set())
    assert res.metadata["UNITS"] == "TDB"


def test_units_absent_accepted():
    """No UNITS line means TDB -- TEMPO1 predates the distinction."""
    res = raw_params_to_result([], component_set=set())
    assert "UNITS" not in res.metadata


@pytest.mark.parametrize("value", ["tdb", " TDB ", "TDB\t"])
def test_units_tdb_tolerates_whitespace_and_case(value):
    """Real par files use tabs and trailing spaces (e.g. PINT's slug.par)."""
    raw_params_to_result(_units_raw(value), component_set=set())


def test_units_tcb_rejected_with_actionable_message():
    with pytest.raises(NotImplementedError, match="UNITS TCB") as exc:
        raw_params_to_result(_units_raw("TCB"), component_set=set())
    # The remedy matters as much as the rejection: TCB pars are the norm in
    # IPTA/EPTA, so the error has to say what to do next.
    assert "tcb2tdb" in str(exc.value)


def test_units_unrecognized_rejected():
    """A typo'd timescale is a corrupt par file, not a default."""
    with pytest.raises(ValueError, match="unrecognized UNITS value"):
        raw_params_to_result(_units_raw("NONSENSE"), component_set=set())


# ---------------------------------------------------------------------------
# Duplicate mask-selector validation (parity with PINT's *.validate())
# ---------------------------------------------------------------------------


def _mask(name, key, key_value, value=1.0, unit=""):
    return RawParam(
        name, ParamKind.MASK, value=value, unit=unit,
        mask_key=key, mask_key_value=key_value,
    )


@pytest.mark.parametrize(
    "family,unit",
    [("EFAC", ""), ("EQUAD", "us"), ("ECORR", "us"), ("DMEFAC", ""), ("DMEQUAD", "")],
)
def test_duplicate_mask_selector_raises_per_family(family, unit):
    """Two params of one family selecting identical TOAs is an error, not a silent
    double-application (PINT: "'EFACs' have duplicated keys and key values.")."""
    raw = [
        _mask(f"{family}1", "-f", "430_ASP", unit=unit),
        _mask(f"{family}2", "-f", "430_ASP", unit=unit),
    ]
    with pytest.raises(ValueError, match="have duplicated keys and key values"):
        raw_params_to_result(raw, component_set=set())


def test_distinct_backends_do_not_collide():
    """The must-not-overfire case: real multi-backend pars are legal."""
    res = raw_params_to_result(
        [_mask("EFAC1", "-f", "430_ASP"), _mask("EFAC2", "-f", "L-wide_PUPPI")],
        component_set=set(),
    )
    assert set(res.mask_info) == {"EFAC1", "EFAC2"}


def test_same_selector_different_families_do_not_collide():
    """EFAC and EQUAD may (and normally do) share a selector."""
    res = raw_params_to_result(
        [_mask("EFAC1", "-f", "A"), _mask("EQUAD1", "-f", "A", unit="us")],
        component_set=set(),
    )
    assert set(res.mask_info) == {"EFAC1", "EQUAD1"}


def test_duplicate_detection_is_superset_of_pint_dash_variant():
    """``-f A`` and ``f A`` select identical TOAs, so they are a duplicate.

    PINT compares ``(key, key_value)`` verbatim and misses this; JaxPINT
    normalizes the key, making the check a strict superset.  Deliberate.
    """
    raw = [_mask("EFAC1", "-f", "A"), _mask("EFAC2", "f", "A")]
    with pytest.raises(ValueError, match="have duplicated keys and key values"):
        raw_params_to_result(raw, component_set=set())


# ---------------------------------------------------------------------------
# TNEQ -> EQUAD synthesis (parity with PINT's ScaleToaError.setup)
# ---------------------------------------------------------------------------


def test_tneq_converts_to_equad_in_seconds():
    """TNEQ is log10(seconds); EQUAD is microseconds -> stored as seconds."""
    res = raw_params_to_result(
        [RawParam("TNEQ1", ParamKind.MASK, value=-6.5, unit="dex(s)",
                  mask_key="-f", mask_key_value="L-wide")],
        component_set=set(),
    )
    assert "TNEQ1" not in res.params.names  # source convention, not a parameter
    assert "TNEQ1" not in res.mask_info
    np.testing.assert_allclose(
        float(res.params.param_value("EQUAD1")), 10.0**-6.5, rtol=1e-12
    )
    assert res.mask_info["EQUAD1"].key == "-f"
    assert res.mask_info["EQUAD1"].key_value == "L-wide"


def test_tneq_uncertainty_propagates_by_delta_method():
    """A dex sigma has a well-defined linear equivalent: sigma_x = x * ln(10) * sigma_y.

    PINT drops the uncertainty when converting TNEQ (it copies quantity/key/
    key_value only).  Propagating cannot break numerical parity -- a *parameter*
    uncertainty never enters the likelihood or delay path -- and dropping it
    would leave an arbitrary asymmetry against an explicitly-written EQUAD.
    """
    import math

    res = raw_params_to_result(
        [RawParam("TNEQ1", ParamKind.MASK, value=-6.5, uncertainty=0.1,
                  unit="dex(s)", mask_key="-f", mask_key_value="A")],
        component_set=set(),
    )
    value = float(res.params.param_value("EQUAD1"))
    sigma = float(res.params.param_uncertainty("EQUAD1"))

    np.testing.assert_allclose(value, 10.0**-6.5, rtol=1e-12)
    np.testing.assert_allclose(sigma, value * math.log(10.0) * 0.1, rtol=1e-12)
    # the delta-method signature: fractional sigma is ln(10) * sigma_dex,
    # independent of the value itself
    np.testing.assert_allclose(sigma / value, math.log(10.0) * 0.1, rtol=1e-12)


def test_tneq_without_uncertainty_stays_nan():
    res = raw_params_to_result(
        [RawParam("TNEQ1", ParamKind.MASK, value=-6.5, unit="dex(s)",
                  mask_key="-f", mask_key_value="A")],
        component_set=set(),
    )
    assert np.isnan(float(res.params.param_uncertainty("EQUAD1")))


def test_tneq_multi_backend():
    res = raw_params_to_result(
        [RawParam("TNEQ1", ParamKind.MASK, value=-6.5, unit="dex(s)",
                  mask_key="-f", mask_key_value="A"),
         RawParam("TNEQ2", ParamKind.MASK, value=-7.0, unit="dex(s)",
                  mask_key="-f", mask_key_value="B")],
        component_set=set(),
    )
    np.testing.assert_allclose(
        float(res.params.param_value("EQUAD1")), 10.0**-6.5, rtol=1e-12
    )
    np.testing.assert_allclose(
        float(res.params.param_value("EQUAD2")), 10.0**-7.0, rtol=1e-12
    )
    assert {res.mask_info[n].key_value for n in ("EQUAD1", "EQUAD2")} == {"A", "B"}


def test_tneq_index_collision_keeps_both_and_does_not_duplicate_names(caplog):
    """A TNEQ whose index collides with an EQUAD on a *different* selector.

    PINT reuses the TNEQ's index and silently overwrites that EQUAD's value AND
    key, destroying the user's parameter.  Reusing the index here would instead
    emit a duplicate name into the ParameterVector, which ``names_with_prefix``
    would apply twice.  We allocate a fresh index, keep both, and warn.
    """
    raw = [
        _mask("EQUAD1", "-f", "BACKEND_A", value=5.0, unit="us"),
        RawParam("TNEQ1", ParamKind.MASK, value=-6.0, unit="dex(s)",
                 mask_key="-f", mask_key_value="BACKEND_B"),
    ]
    res = raw_params_to_result(raw, component_set=set())

    equads = [n for n in res.params.names if n.startswith("EQUAD")]
    assert len(equads) == len(set(equads)), f"duplicate names emitted: {equads}"
    selectors = {
        res.mask_info[n].key_value for n in res.mask_info if n.startswith("EQUAD")
    }
    assert selectors == {"BACKEND_A", "BACKEND_B"}, "a user EQUAD was destroyed"
    # the user's value survives untouched
    by_sel = {res.mask_info[n].key_value: n for n in res.mask_info}
    np.testing.assert_allclose(
        float(res.params.param_value(by_sel["BACKEND_A"])), 5.0e-6, rtol=1e-12
    )
    np.testing.assert_allclose(
        float(res.params.param_value(by_sel["BACKEND_B"])), 10.0**-6.0, rtol=1e-12
    )
    assert any("would map to EQUAD1" in r.getMessage() for r in caplog.records), \
        "the PINT divergence was not announced"


def test_explicit_equad_wins_over_tneq_on_same_selector():
    """PINT's setup() prefers an explicit EQUAD; the TNEQ is dropped, and the
    result must not trip the duplicate-selector check."""
    res = raw_params_to_result(
        [RawParam("TNEQ1", ParamKind.MASK, value=-6.5, unit="dex(s)",
                  mask_key="-f", mask_key_value="A"),
         _mask("EQUAD1", "-f", "A", value=0.8, unit="us")],
        component_set=set(),
    )
    assert "TNEQ1" not in res.params.names
    np.testing.assert_allclose(
        float(res.params.param_value("EQUAD1")), 0.8e-6, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Source-level PINT-free invariant
# ---------------------------------------------------------------------------


def test_par_subpackage_has_no_pint_imports():
    par_dir = _REPO / "jaxpint" / "par"
    offenders = []
    for py in par_dir.rglob("*.py"):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("import pint") or s.startswith("from pint"):
                offenders.append(f"{py.relative_to(_REPO)}:{i}: {s}")
    assert not offenders, "PINT imported in jaxpint/par/:\n" + "\n".join(offenders)

def test_synthesize_pb_from_fb0():
    """FB0 only → PB synthesized as 1 / (FB0 * 86400) days, appended to the list."""
    from jaxpint.par import ParamKind, RawParam
    from jaxpint.par.aliases import synthesize_pb_from_fb

    fb0_hz = 8.3387216e-5  # J0023+0923 value
    raw = [RawParam("FB0", ParamKind.FLOAT, value=fb0_hz, unit="Hz", frozen=False)]
    synthesize_pb_from_fb(raw)
    synth = raw[1:]

    assert [r.name for r in synth] == ["PB"]
    assert abs(synth[0].value - 1.0 / (fb0_hz * 86400.0)) < 1e-15
    assert synth[0].unit == "d"
    assert synth[0].frozen is False


def test_synthesize_pbdot_from_fb1():
    """FB0 and FB1 set → both PB and PBDOT synthesized."""
    from jaxpint.par import ParamKind, RawParam
    from jaxpint.par.aliases import synthesize_pb_from_fb

    fb0_hz = 8.3387216e-5
    fb1 = 3.6553667e-20
    raw = [
        RawParam("FB0", ParamKind.FLOAT, value=fb0_hz, unit="Hz", frozen=False),
        RawParam("FB1", ParamKind.FLOAT, value=fb1, unit="Hz / s", frozen=True),
    ]
    synthesize_pb_from_fb(raw)
    synth = raw[2:]

    assert [r.name for r in synth] == ["PB", "PBDOT"]
    assert abs(synth[0].value - 1.0 / (fb0_hz * 86400.0)) < 1e-15
    assert abs(synth[1].value - (-fb1 / (fb0_hz * fb0_hz))) < 1e-25
    assert [r.unit for r in synth] == ["d", "s / s"]
    assert [r.frozen for r in synth] == [False, True]


def test_synthesize_pb_skips_when_pb_set():
    """If PB is already present, synthesis must not fire (would otherwise
    duplicate PB in the parameter vector)."""
    from jaxpint.par import ParamKind, RawParam
    from jaxpint.par.aliases import synthesize_pb_from_fb

    raw = [
        RawParam("FB0", ParamKind.FLOAT, value=8.3387216e-5, unit="Hz"),
        RawParam("PB", ParamKind.FLOAT, value=0.139, unit="d"),  # days
    ]
    synthesize_pb_from_fb(raw)

    assert [r.name for r in raw] == ["FB0", "PB"]  # nothing appended


def test_synthesize_pbdot_skips_when_pbdot_set():
    """PB synthesized, but PBDOT already present → don't overwrite."""
    from jaxpint.par import ParamKind, RawParam
    from jaxpint.par.aliases import synthesize_pb_from_fb

    raw = [
        RawParam("FB0", ParamKind.FLOAT, value=8.3387216e-5, unit="Hz"),
        RawParam("FB1", ParamKind.FLOAT, value=3.6553667e-20, unit="Hz / s"),
        RawParam("PBDOT", ParamKind.FLOAT, value=1e-12, unit="s / s"),
    ]
    synthesize_pb_from_fb(raw)
    synth = raw[3:]

    assert [r.name for r in synth] == ["PB"]


def test_synthesize_pb_noop_without_fb0():
    """No FB0 → no synthesis, regardless of other state."""
    from jaxpint.par.aliases import synthesize_pb_from_fb

    raw = []
    synthesize_pb_from_fb(raw)

    assert raw == []
