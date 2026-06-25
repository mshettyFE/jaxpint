"""Tests for component indexing and decompose_* methods on TimingModel and NoiseModel."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.components import _make_component_names
from jaxpint.delay.dispersion_dm import DispersionDM
from jaxpint.types.dual_float import DualFloat
from jaxpint.model import TimingModel
from jaxpint.noise import ScaleToaError
from jaxpint.noise.noise_model import NoiseModel
from jaxpint.noise.ecorr import EcorrNoise
from jaxpint.phase.spin import Spindown

from tests.helpers import make_gbt_toa_data, make_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_model():
    """TimingModel with DispersionDM delay + Spindown phase."""
    dm = DispersionDM(dm_param_names=("DM",))
    spin = Spindown(spin_param_names=("F0",))
    return TimingModel(
        delay_components=(dm,),
        phase_components=(spin,),
        dispersion_components=(dm,),
    )


def _make_params():
    return make_params(
        ["F0", "PEPOCH", "DM", "DMEPOCH"],
        [200.0, 0.0, 15.0, 0.0],
        units=("Hz", "day", "pc cm^-3", "day"),
        epoch_int_values={"PEPOCH": 59000.0, "DMEPOCH": 59000.0},
    )


def _make_duplicate_model():
    """TimingModel with two DispersionDM components (for name disambiguation)."""
    dm1 = DispersionDM(dm_param_names=("DM",))
    dm2 = DispersionDM(dm_param_names=("DM",))
    spin = Spindown(spin_param_names=("F0",))
    return TimingModel(
        delay_components=(dm1, dm2),
        phase_components=(spin,),
        dispersion_components=(dm1, dm2),
    )


# ===========================================================================
# _make_component_names
# ===========================================================================


class TestMakeComponentNames:
    """Tests for the _make_component_names helper."""

    def test_unique_names(self):
        dm = DispersionDM(dm_param_names=("DM",))
        spin = Spindown(spin_param_names=("F0",))
        names = _make_component_names((dm, spin))
        assert names == ("DispersionDM", "Spindown")

    def test_duplicate_names_get_suffix(self):
        dm1 = DispersionDM(dm_param_names=("DM",))
        dm2 = DispersionDM(dm_param_names=("DM",))
        names = _make_component_names((dm1, dm2))
        assert names == ("DispersionDM_0", "DispersionDM_1")

    def test_mixed_unique_and_duplicate(self):
        dm1 = DispersionDM(dm_param_names=("DM",))
        dm2 = DispersionDM(dm_param_names=("DM",))
        spin = Spindown(spin_param_names=("F0",))
        names = _make_component_names((dm1, spin, dm2))
        assert names == ("DispersionDM_0", "Spindown", "DispersionDM_1")

    def test_empty_tuple(self):
        names = _make_component_names(())
        assert names == ()


# ===========================================================================
# TimingModel indexing
# ===========================================================================


class TestTimingModelIndexing:
    """Tests for TimingModel.__getitem__ and related properties."""

    def test_components_property(self):
        model = _make_simple_model()
        comps = model.components
        assert len(comps) == 2
        assert isinstance(comps[0], DispersionDM)
        assert isinstance(comps[1], Spindown)

    def test_component_names_property(self):
        model = _make_simple_model()
        assert model.component_names == ("DispersionDM", "Spindown")

    def test_getitem_by_name(self):
        model = _make_simple_model()
        comp = model["DispersionDM"]
        assert isinstance(comp, DispersionDM)

    def test_getitem_by_name_phase(self):
        model = _make_simple_model()
        comp = model["Spindown"]
        assert isinstance(comp, Spindown)

    def test_getitem_by_int(self):
        model = _make_simple_model()
        assert isinstance(model[0], DispersionDM)
        assert isinstance(model[1], Spindown)

    def test_getitem_by_negative_int(self):
        model = _make_simple_model()
        assert isinstance(model[-1], Spindown)

    def test_getitem_by_slice(self):
        model = _make_simple_model()
        result = model[0:2]
        assert len(result) == 2
        assert isinstance(result[0], DispersionDM)
        assert isinstance(result[1], Spindown)

    def test_getitem_keyerror(self):
        model = _make_simple_model()
        with pytest.raises(KeyError, match="NotAComponent"):
            model["NotAComponent"]

    def test_getitem_typeerror(self):
        model = _make_simple_model()
        with pytest.raises(TypeError):
            model[3.14]

    def test_duplicate_name_indexing(self):
        model = _make_duplicate_model()
        assert model.component_names[:2] == ("DispersionDM_0", "DispersionDM_1")
        comp0 = model["DispersionDM_0"]
        comp1 = model["DispersionDM_1"]
        assert isinstance(comp0, DispersionDM)
        assert isinstance(comp1, DispersionDM)

    def test_empty_model(self):
        model = TimingModel(delay_components=(), phase_components=())
        assert model.components == ()
        assert model.component_names == ()
        with pytest.raises(KeyError):
            model["Spindown"]


# ===========================================================================
# TimingModel decompose methods
# ===========================================================================


class TestDecomposeDelay:
    """Tests for TimingModel.decompose_delay."""

    def test_single_component(self):
        model = _make_simple_model()
        toa_data = make_gbt_toa_data(freq=1400.0)
        params = _make_params()

        result = model.decompose_delay(toa_data, params)

        assert isinstance(result, dict)
        assert "DispersionDM" in result
        assert result["DispersionDM"].shape == (5,)

    def test_sum_matches_compute_delay(self):
        """Sum of decomposed delays equals compute_delay output."""
        model = _make_simple_model()
        toa_data = make_gbt_toa_data(freq=1400.0)
        params = _make_params()

        decomposed = model.decompose_delay(toa_data, params)
        total = model.compute_delay(toa_data, params)

        summed = sum(decomposed.values())
        np.testing.assert_allclose(summed, total, rtol=1e-14)

    def test_duplicate_components(self):
        model = _make_duplicate_model()
        toa_data = make_gbt_toa_data(freq=1400.0)
        params = _make_params()

        result = model.decompose_delay(toa_data, params)

        assert "DispersionDM_0" in result
        assert "DispersionDM_1" in result

    def test_empty_delay(self):
        model = TimingModel(delay_components=(), phase_components=())
        toa_data = make_gbt_toa_data()
        params = _make_params()

        result = model.decompose_delay(toa_data, params)
        assert result == {}


class TestDecomposePhase:
    """Tests for TimingModel.decompose_phase."""

    def test_single_component(self):
        model = _make_simple_model()
        toa_data = make_gbt_toa_data(freq=1400.0)
        params = _make_params()

        result = model.decompose_phase(toa_data, params)

        assert isinstance(result, dict)
        assert "Spindown" in result
        assert isinstance(result["Spindown"], DualFloat)

    def test_sum_matches_phase_components(self):
        """Sum of decomposed phases equals _sum_phase_components output."""
        model = _make_simple_model()
        toa_data = make_gbt_toa_data(freq=1400.0)
        params = _make_params()

        decomposed = model.decompose_phase(toa_data, params)
        delay = model.compute_delay(toa_data, params)
        expected = model._sum_phase_components(toa_data, params, delay)

        total_int = jnp.zeros(toa_data.n_toas)
        total_frac = jnp.zeros(toa_data.n_toas)
        for phase in decomposed.values():
            total_int = total_int + phase.int
            total_frac = total_frac + phase.frac

        np.testing.assert_allclose(
            total_int + total_frac,
            expected.int + expected.frac,
            rtol=1e-12,
        )

    def test_empty_phase(self):
        model = TimingModel(delay_components=(), phase_components=())
        toa_data = make_gbt_toa_data()
        params = _make_params()

        result = model.decompose_phase(toa_data, params)
        assert result == {}


class TestDecomposeDm:
    """Tests for TimingModel.decompose_dm."""

    def test_single_component(self):
        model = _make_simple_model()
        toa_data = make_gbt_toa_data(freq=1400.0)
        params = _make_params()

        result = model.decompose_dm(toa_data, params)

        assert isinstance(result, dict)
        assert "DispersionDM" in result
        assert result["DispersionDM"].shape == (5,)

    def test_sum_matches_compute_dm(self):
        """Sum of decomposed DMs equals compute_dm output."""
        model = _make_simple_model()
        toa_data = make_gbt_toa_data(freq=1400.0)
        params = _make_params()

        decomposed = model.decompose_dm(toa_data, params)
        total = model.compute_dm(toa_data, params)

        summed = sum(decomposed.values())
        np.testing.assert_allclose(summed, total, rtol=1e-14)

    def test_empty_dispersion(self):
        spin = Spindown(spin_param_names=("F0",))
        model = TimingModel(
            delay_components=(), phase_components=(spin,),
            dispersion_components=(),
        )
        toa_data = make_gbt_toa_data()
        params = _make_params()

        result = model.decompose_dm(toa_data, params)
        assert result == {}


# ===========================================================================
# NoiseModel indexing
# ===========================================================================


class TestNoiseModelIndexing:
    """Tests for NoiseModel.__getitem__ and related properties."""

    def test_components_includes_white_noise(self):
        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        noise = NoiseModel(white_noise=white, correlated=())

        assert len(noise.components) == 1
        assert isinstance(noise.components[0], ScaleToaError)

    def test_components_excludes_none(self):
        noise = NoiseModel(white_noise=None, correlated=())
        assert noise.components == ()

    def test_component_names(self):
        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        noise = NoiseModel(white_noise=white, correlated=())

        assert noise.component_names == ("ScaleToaError",)

    def test_getitem_by_name(self):
        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        noise = NoiseModel(white_noise=white, correlated=())

        comp = noise["ScaleToaError"]
        assert isinstance(comp, ScaleToaError)

    def test_getitem_by_int(self):
        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        noise = NoiseModel(white_noise=white, correlated=())

        assert isinstance(noise[0], ScaleToaError)

    def test_getitem_keyerror(self):
        noise = NoiseModel(white_noise=None, correlated=())
        with pytest.raises(KeyError):
            noise["ScaleToaError"]

    def test_getitem_typeerror(self):
        noise = NoiseModel(white_noise=None, correlated=())
        with pytest.raises(TypeError):
            noise[3.14]

    def test_correlated_components_included(self):
        white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        ecorr = EcorrNoise(
            ecorr_names=("ECORR1",),
            quantization_matrix=jnp.eye(3),
            ecorr_epoch_slices=((0, 1), (1, 2), (2, 3)),
        )
        noise = NoiseModel(white_noise=white, correlated=(ecorr,))

        assert len(noise.components) == 2
        assert noise.component_names == ("ScaleToaError", "EcorrNoise")
        assert isinstance(noise["EcorrNoise"], EcorrNoise)
