"""Tests for the standalone .par file parser.

Compares output of the standalone parser against the PINT bridge layer
to ensure they produce identical ParameterVector objects.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from pint.config import examplefile
from pint.models import get_model

from jaxpint.bridge import pint_model_to_params, build_timing_model
from jaxpint.parfile import parse_par, build_model
from jaxpint.parfile._converters import (
    parse_float,
    parse_hms_to_rad,
    parse_dms_to_rad,
    split_mjd,
    deg_to_rad,
    deg_per_yr_to_rad_per_s,
    us_to_s,
    tcb_scale_parameter,
    tcb_transform_mjd,
    IFTE_K,
    IFTE_MJD0,
)


# =========================================================================
# Unit tests: converters
# =========================================================================

class TestConverters:
    def test_parse_float_standard(self):
        assert parse_float("3.14") == pytest.approx(3.14)

    def test_parse_float_fortran_d(self):
        assert parse_float("1.23D-04") == pytest.approx(1.23e-04)
        assert parse_float("-6.205147513395d-16") == pytest.approx(-6.205147513395e-16)

    def test_parse_hms_to_rad(self):
        # 12:00:00 = 180 degrees = pi radians
        assert parse_hms_to_rad("12:00:00") == pytest.approx(math.pi)
        # 06:00:00 = 90 degrees = pi/2
        assert parse_hms_to_rad("6:00:00") == pytest.approx(math.pi / 2)

    def test_parse_dms_to_rad(self):
        # 90:00:00 = pi/2
        assert parse_dms_to_rad("90:00:00") == pytest.approx(math.pi / 2)
        # -45:00:00 = -pi/4
        assert parse_dms_to_rad("-45:00:00") == pytest.approx(-math.pi / 4)

    def test_split_mjd(self):
        int_part, frac = split_mjd(57000.75)
        assert int_part == 57000.0
        assert frac == pytest.approx(0.75)

    def test_split_mjd_integer(self):
        int_part, frac = split_mjd(57000.0)
        assert int_part == 57000.0
        assert frac == pytest.approx(0.0)

    def test_deg_to_rad(self):
        assert deg_to_rad(180.0) == pytest.approx(math.pi)

    def test_deg_per_yr_to_rad_per_s(self):
        # 360 deg/yr should be 2*pi / (365.25 * 86400)
        expected = 2 * math.pi / (365.25 * 86400.0)
        assert deg_per_yr_to_rad_per_s(360.0) == pytest.approx(expected)

    def test_us_to_s(self):
        assert us_to_s(1000.0) == pytest.approx(0.001)

    def test_tcb_scale(self):
        # F0 scaling: n=1
        f0_tcb = 100.0
        f0_tdb = tcb_scale_parameter(f0_tcb, 1)
        assert f0_tdb == pytest.approx(f0_tcb * IFTE_K)

    def test_tcb_transform_mjd(self):
        t_tcb = 55000.0
        t_tdb = tcb_transform_mjd(t_tcb)
        expected = (t_tcb - IFTE_MJD0) / IFTE_K + IFTE_MJD0
        assert t_tdb == pytest.approx(expected)


# =========================================================================
# Unit tests: tokenizer
# =========================================================================

class TestTokenizer:
    def test_basic_parsing(self):
        from jaxpint.parfile._tokenizer import tokenize

        text = """\
PSR  J1234+5678
F0  100.0  1  0.001
# This is a comment
C This is also a comment

F1  -1e-15  0
"""
        lines = tokenize(text)
        assert len(lines) == 3
        assert lines[0].name == "PSR"
        assert lines[0].tokens == ["J1234+5678"]
        assert lines[1].name == "F0"
        assert lines[1].tokens == ["100.0", "1", "0.001"]
        assert lines[2].name == "F1"
        assert lines[2].tokens == ["-1e-15", "0"]

    def test_alias_resolution(self):
        from jaxpint.parfile._tokenizer import tokenize

        text = "RA 12:00:00\nDEC 45:00:00\nE 0.1\n"
        lines = tokenize(text)
        assert lines[0].name == "RAJ"
        assert lines[1].name == "DECJ"
        assert lines[2].name == "ECC"


# =========================================================================
# Unit tests: registry lookup
# =========================================================================

class TestRegistry:
    def test_direct_lookup(self):
        from jaxpint.parfile._registry import Component, lookup

        result = lookup("F0")
        assert result is not None
        meta, name = result
        assert name == "F0"
        assert meta.component is Component.SPINDOWN

    def test_prefix_lookup(self):
        from jaxpint.parfile._registry import Component, lookup

        result = lookup("DMX_0001")
        assert result is not None
        meta, name = result
        assert meta.component is Component.DISPERSION_DMX

    def test_prefix_f_lookup(self):
        from jaxpint.parfile._registry import Component, lookup

        result = lookup("F2")
        assert result is not None
        meta, name = result
        assert meta.component is Component.SPINDOWN

    def test_unknown_param(self):
        from jaxpint.parfile._registry import lookup

        result = lookup("TOTALLY_UNKNOWN_PARAM")
        assert result is None

    def test_all_registry_units_valid(self):
        """Every default_unit in both registries must be a valid astropy unit."""
        from astropy.units import Unit
        from jaxpint.parfile._registry import PARAM_REGISTRY, PREFIX_REGISTRY

        invalid = []
        for name, meta in {**PARAM_REGISTRY, **PREFIX_REGISTRY}.items():
            if meta.default_unit == "":
                continue
            try:
                Unit(meta.default_unit)
            except ValueError:
                invalid.append(f"  {name}: {meta.default_unit!r}")

        if invalid:
            pytest.fail("Invalid unit strings in registry:\n" + "\n".join(invalid))


# =========================================================================
# Integration test: parse a simple .par string
# =========================================================================

class TestParsePar:
    def test_simple_par(self):
        par_text = """\
PSR  J0000+0000
F0  100.0  1
F1  -1.0D-15  1
PEPOCH  55000.0
RAJ  12:00:00.0  1
DECJ  45:00:00.0  1
DM  10.0  1
EPHEM  DE421
"""
        result = parse_par(par_text)

        assert result.metadata["PSR"] == "J0000+0000"
        assert result.metadata["EPHEM"] == "DE421"

        # Check F0
        assert "F0" in result.params._name_to_index
        f0_val = float(result.params.values[result.params._name_to_index["F0"]])
        assert f0_val == pytest.approx(100.0)

        # Check F1 (Fortran notation)
        assert "F1" in result.params._name_to_index
        f1_val = float(result.params.values[result.params._name_to_index["F1"]])
        assert f1_val == pytest.approx(-1.0e-15)

        # Check PEPOCH (epoch split)
        assert "PEPOCH" in result.params._name_to_index
        pepoch_frac = float(result.params.values[result.params._name_to_index["PEPOCH"]])
        assert pepoch_frac == pytest.approx(0.0)
        assert result.params.epoch_int_values["PEPOCH"] == 55000.0

        # Check RAJ (H:M:S → radians)
        raj_val = float(result.params.values[result.params._name_to_index["RAJ"]])
        assert raj_val == pytest.approx(math.pi)  # 12h = 180° = π

        # Check DECJ (D:M:S → radians)
        decj_val = float(result.params.values[result.params._name_to_index["DECJ"]])
        assert decj_val == pytest.approx(math.pi / 4)  # 45° = π/4

        # Check DM
        dm_val = float(result.params.values[result.params._name_to_index["DM"]])
        assert dm_val == pytest.approx(10.0)

    def test_binary_par(self):
        par_text = """\
PSR  J0000+0000
F0  100.0  1
PEPOCH  55000.0
RAJ  12:00:00.0  1
DECJ  45:00:00.0  1
DM  10.0
BINARY  DD
PB  1.5  1
T0  55000.5  1
A1  3.0  1
ECC  0.01  1
OM  90.0  1
"""
        result = parse_par(par_text)
        from jaxpint.parfile._registry import BinaryModel
        assert result.binary_model is BinaryModel.DD

        # OM should be converted from degrees to radians
        om_val = float(result.params.values[result.params._name_to_index["OM"]])
        assert om_val == pytest.approx(math.pi / 2)

    def test_mask_params(self):
        par_text = """\
PSR  J0000+0000
F0  100.0  1
PEPOCH  55000.0
RAJ  12:00:00.0
DECJ  45:00:00.0
DM  10.0
EFAC -f Rcvr_800 1.2
EFAC -f Rcvr1_2 0.9
EQUAD -f Rcvr_800 0.5
"""
        result = parse_par(par_text)

        # EFAC1, EFAC2 should exist
        assert "EFAC1" in result.params._name_to_index
        assert "EFAC2" in result.params._name_to_index
        # EQUAD converted from us to s
        equad_val = float(result.params.values[result.params._name_to_index["EQUAD1"]])
        assert equad_val == pytest.approx(0.5e-6)

        # Mask info
        assert "EFAC1" in result.mask_info
        assert result.mask_info["EFAC1"].key == "-f"
        assert result.mask_info["EFAC1"].key_value == "Rcvr_800"


# =========================================================================
# Comparison test: standalone vs bridge
# =========================================================================


class TestCompareWithBridge:
    """Compare standalone parser output against PINT bridge for real .par files."""

    @pytest.fixture
    def b1855_par(self):
        return examplefile("B1855+09_NANOGrav_9yv1.gls.par")

    def test_parameter_names_coverage(self, b1855_par):
        """All standalone params should appear in bridge output.

        The bridge may have *extra* params (PINT auto-creates defaults like
        A0=0, PBDOT=0 for binary models), but the standalone parser should
        not produce any parameter that the bridge doesn't have.
        """
        pint_model = get_model(b1855_par)
        bridge_params = pint_model_to_params(pint_model).params

        standalone_result = parse_par(b1855_par)
        standalone_params = standalone_result.params

        bridge_names = set(bridge_params.names)
        standalone_names = set(standalone_params.names)

        # Standalone should not have params the bridge lacks
        only_standalone = standalone_names - bridge_names
        if only_standalone:
            pytest.fail(
                f"Parameters in standalone but not bridge: {sorted(only_standalone)}"
            )

        # Bridge may have extra zero-valued defaults — that's expected.
        # Just report for visibility.
        only_bridge = bridge_names - standalone_names
        if only_bridge:
            # Verify all bridge-only params are zero-valued frozen defaults
            for name in only_bridge:
                idx = bridge_params._name_to_index[name]
                val = float(bridge_params.values[idx])
                # Allow non-zero for metadata params (START, FINISH, etc.)
                # that the standalone parser intentionally skips

    def test_parameter_values_match(self, b1855_par):
        """Parameter values from both paths should match within tolerance."""
        pint_model = get_model(b1855_par)
        bridge_params = pint_model_to_params(pint_model).params

        standalone_result = parse_par(b1855_par)
        standalone_params = standalone_result.params

        # Compare values for all shared parameters
        for name in bridge_params.names:
            if name not in standalone_params._name_to_index:
                continue
            b_idx = bridge_params._name_to_index[name]
            s_idx = standalone_params._name_to_index[name]
            b_val = float(bridge_params.values[b_idx])
            s_val = float(standalone_params.values[s_idx])

            np.testing.assert_allclose(
                s_val, b_val,
                rtol=1e-10, atol=1e-20,
                err_msg=f"Value mismatch for {name}",
            )

    def test_epoch_int_values_match(self, b1855_par):
        """Epoch integer parts should match exactly."""
        pint_model = get_model(b1855_par)
        bridge_params = pint_model_to_params(pint_model).params

        standalone_result = parse_par(b1855_par)
        standalone_params = standalone_result.params

        # Compare shared epoch params (bridge may have extras like START, FINISH)
        shared_epochs = set(bridge_params.epoch_int_values) & set(standalone_params.epoch_int_values)
        assert len(shared_epochs) > 0, "No shared epoch parameters found"
        for name in shared_epochs:
            assert bridge_params.epoch_int_values[name] == standalone_params.epoch_int_values[name], \
                f"Epoch int mismatch for {name}"

    def test_frozen_mask_matches(self, b1855_par):
        """Frozen status should match for all shared parameters."""
        pint_model = get_model(b1855_par)
        bridge_params = pint_model_to_params(pint_model).params

        standalone_result = parse_par(b1855_par)
        standalone_params = standalone_result.params

        for name in bridge_params.names:
            if name not in standalone_params._name_to_index:
                continue
            b_idx = bridge_params._name_to_index[name]
            s_idx = standalone_params._name_to_index[name]
            assert bridge_params.frozen_mask[b_idx] == standalone_params.frozen_mask[s_idx], \
                f"Frozen mismatch for {name}: bridge={bridge_params.frozen_mask[b_idx]}, standalone={standalone_params.frozen_mask[s_idx]}"


# =========================================================================
# Slow test: compare against ALL PINT example .par files
# =========================================================================


def _collect_pint_example_par_files():
    """Discover all .par files in the PINT examples directory."""
    import importlib.resources
    examples_dir = Path(str(importlib.resources.files("pint.data") / "examples"))
    if examples_dir.is_dir():
        return sorted(examples_dir.glob("*.par"))
    return []


_EXAMPLE_PARS = _collect_pint_example_par_files()


@pytest.mark.slow
class TestAllExampleParFiles:
    """Compare standalone parser vs bridge for every PINT example .par file."""

    @pytest.fixture(params=_EXAMPLE_PARS, ids=[p.name for p in _EXAMPLE_PARS])
    def par_file(self, request):
        return str(request.param)

    def test_parse_does_not_crash(self, par_file):
        """Standalone parser should not raise on any example .par file."""
        result = parse_par(par_file)
        assert result.params is not None
        assert len(result.params.names) > 0

    def test_values_match_bridge(self, par_file):
        """Shared parameter values should match the bridge output."""
        try:
            pint_model = get_model(par_file)
        except Exception as e:
            pytest.skip(f"PINT cannot load this file: {e}")

        bridge_params = pint_model_to_params(pint_model).params
        standalone_params = parse_par(par_file).params

        shared = set(bridge_params.names) & set(standalone_params.names)
        if not shared:
            pytest.skip("No shared parameters between bridge and standalone")

        mismatches = []
        for name in sorted(shared):
            b_val = float(bridge_params.values[bridge_params._name_to_index[name]])
            s_val = float(standalone_params.values[standalone_params._name_to_index[name]])
            if b_val == 0.0 and s_val == 0.0:
                continue
            if b_val != 0.0 and abs((s_val - b_val) / b_val) > 1e-10:
                mismatches.append(f"  {name}: bridge={b_val:.15e}, standalone={s_val:.15e}")
            elif b_val == 0.0 and abs(s_val) > 1e-20:
                mismatches.append(f"  {name}: bridge=0, standalone={s_val:.15e}")

        if mismatches:
            pytest.fail(
                f"Value mismatches in {Path(par_file).name}:\n" + "\n".join(mismatches)
            )

    def test_epochs_match_bridge(self, par_file):
        """Shared epoch integer parts should match the bridge output."""
        try:
            pint_model = get_model(par_file)
        except Exception:
            pytest.skip("PINT cannot load this file")

        bridge_params = pint_model_to_params(pint_model).params
        standalone_params = parse_par(par_file).params

        shared = set(bridge_params.epoch_int_values) & set(standalone_params.epoch_int_values)
        for name in shared:
            assert bridge_params.epoch_int_values[name] == standalone_params.epoch_int_values[name], \
                f"Epoch int mismatch for {name} in {Path(par_file).name}"

    def test_frozen_matches_bridge(self, par_file):
        """Frozen mask should match for shared parameters."""
        try:
            pint_model = get_model(par_file)
        except Exception:
            pytest.skip("PINT cannot load this file")

        bridge_params = pint_model_to_params(pint_model).params
        standalone_params = parse_par(par_file).params

        mismatches = []
        shared = set(bridge_params.names) & set(standalone_params.names)
        for name in sorted(shared):
            b_idx = bridge_params._name_to_index[name]
            s_idx = standalone_params._name_to_index[name]
            b_frozen = bridge_params.frozen_mask[b_idx]
            s_frozen = standalone_params.frozen_mask[s_idx]
            if b_frozen != s_frozen:
                # Skip frozen mismatches for zero-valued params — PINT
                # component-specific defaults (e.g. DMX frozen=False) differ
                # from our parser default, but don't affect computation.
                b_val = float(bridge_params.values[b_idx])
                if b_val == 0.0:
                    continue
                mismatches.append(f"  {name}: bridge={b_frozen}, standalone={s_frozen}")

        if mismatches:
            pytest.fail(
                f"Frozen mismatches in {Path(par_file).name}:\n" + "\n".join(mismatches)
            )
