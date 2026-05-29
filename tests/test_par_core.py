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
