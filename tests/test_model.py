"""Tests for jaxpint.model: TimingModel orchestration layer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from jaxpint.types import TOAData
from jaxpint.phase.spin import Spindown
from jaxpint.delay.dispersion_dm import DispersionDM
from jaxpint.model import TimingModel, _reconstruct_tzr_toa
from jaxpint.model_builder import _validate_referenced_params, _validate_flag_masks
from jaxpint.par.result import ParResult, MaskInfo
from jaxpint.noise import NoiseModel
from tests.helpers import (
    make_gbt_toa_data, make_spindown_params, make_params, make_toa_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_full_params(
    f0=200.0, f1=None, dm=15.0,
    pepoch_int=59000.0, pepoch_frac=0.0,
    dmepoch_int=59000.0, dmepoch_frac=0.0,
):
    names = ["F0"]; values = [f0]; units = ["Hz"]

    if f1 is not None:
        names += ["F1"]; values += [f1]; units += ["Hz/s"]

    names += ["PEPOCH"]; values += [pepoch_frac]; units += ["day"]
    names += ["DM"]; values += [dm]; units += ["pc cm^-3"]
    names += ["DMEPOCH"]; values += [dmepoch_frac]; units += ["day"]

    return make_params(names, values, units=tuple(units),
                       epoch_int_values={"PEPOCH": pepoch_int, "DMEPOCH": dmepoch_int})


# ===========================================================================
# compute_delay
# ===========================================================================


class TestComputeDelay:
    """Tests for TimingModel.compute_delay."""

    def test_no_delay_components(self):
        """No delay components returns zeros."""
        model = TimingModel(delay_components=(), phase_components=())
        toa_data = make_gbt_toa_data()
        params = make_spindown_params()

        delay = model.compute_delay(toa_data, params)

        assert delay.shape == (5,)
        np.testing.assert_allclose(delay, 0.0)

    def test_single_dispersion_delay(self):
        """Single DispersionDM component produces expected delay."""
        dm_comp = DispersionDM(dm_param_names=("DM",))
        model = TimingModel(delay_components=(dm_comp,), phase_components=())

        toa_data = make_gbt_toa_data(freq=1400.0)
        params = _make_full_params(dm=15.0)

        delay = model.compute_delay(toa_data, params)

        # Expected: DM * K_DM / freq^2
        from jaxpint.constants import DMCONST
        expected = 15.0 * DMCONST / 1400.0**2
        np.testing.assert_allclose(delay, expected, rtol=1e-12)

    def test_delay_accumulates_sequentially(self):
        """Multiple delay components accumulate."""
        dm1 = DispersionDM(dm_param_names=("DM",))
        # Same component twice to test accumulation via lax.switch
        model = TimingModel(delay_components=(dm1, dm1), phase_components=())

        toa_data = make_gbt_toa_data(freq=1400.0)
        params = _make_full_params(dm=15.0)

        delay = model.compute_delay(toa_data, params)

        # DispersionDM ignores the accumulated delay input,
        # so two identical components should give 2x the delay
        from jaxpint.constants import DMCONST
        single = 15.0 * DMCONST / 1400.0**2
        np.testing.assert_allclose(delay, 2 * single, rtol=1e-12)

    def test_delay_jit_compatible(self):
        """compute_delay works under jax.jit."""
        dm_comp = DispersionDM(dm_param_names=("DM",))
        model = TimingModel(delay_components=(dm_comp,), phase_components=())

        toa_data = make_gbt_toa_data()
        params = _make_full_params()

        delay_eager = model.compute_delay(toa_data, params)
        delay_jit = jax.jit(model.compute_delay)(toa_data, params)

        np.testing.assert_allclose(delay_jit, delay_eager, rtol=1e-14)


class TestComputePhaseRelative:
    """Tests for compute_phase without TZR subtraction."""

    def test_spindown_only_no_tzr(self):
        """Spindown-only model without TZR gives raw phase."""
        spin = Spindown(spin_param_names=("F0",))
        model = TimingModel(delay_components=(), phase_components=(spin,))

        toa_data = make_gbt_toa_data(n_toas=3, tdb_int=59001.0, tdb_frac=jnp.array([0.0, 0.25, 0.5]))
        params = make_spindown_params(f0=200.0, pepoch_int=59000.0, pepoch_frac=0.0)

        phase = model.compute_phase(toa_data, params)

        # dt = (59001 - 59000 + frac) * 86400 seconds
        dt = (1.0 + jnp.array([0.0, 0.25, 0.5])) * 86400.0
        expected_phase = 200.0 * dt  # F0 * dt

        np.testing.assert_allclose(
            phase.int + phase.frac, expected_phase, rtol=1e-10
        )

    def test_spindown_with_dispersion_no_tzr(self):
        """Spindown + DispersionDM: delay affects the phase."""
        spin = Spindown(spin_param_names=("F0",))
        dm_comp = DispersionDM(dm_param_names=("DM",))
        model = TimingModel(
            delay_components=(dm_comp,),
            phase_components=(spin,),
        )

        toa_data = make_gbt_toa_data(n_toas=3, tdb_int=59001.0, tdb_frac=jnp.array([0.0, 0.25, 0.5]))
        params = _make_full_params(f0=200.0, dm=15.0)

        phase = model.compute_phase(toa_data, params)

        # dt = (tdb - pepoch) * 86400 - delay
        from jaxpint.constants import DMCONST
        dm_delay = 15.0 * DMCONST / 1400.0**2
        dt = (1.0 + jnp.array([0.0, 0.25, 0.5])) * 86400.0 - dm_delay
        expected_phase = 200.0 * dt

        np.testing.assert_allclose(
            phase.int + phase.frac, expected_phase, rtol=1e-10
        )

    def test_no_phase_components(self):
        """Model with no phase components returns zero phase."""
        model = TimingModel(delay_components=(), phase_components=())
        toa_data = make_gbt_toa_data()
        params = make_spindown_params()

        phase = model.compute_phase(toa_data, params)

        np.testing.assert_allclose(phase.int, 0.0)
        np.testing.assert_allclose(phase.frac, 0.0)


# ===========================================================================
# compute_phase — with absolute phase (TZR subtraction)
# ===========================================================================


class TestComputePhaseAbsolute:
    """Tests for compute_phase with TZR reference phase subtraction."""

    def test_phase_at_tzr_epoch_is_zero(self):
        """Phase at the TZR TOA time should be ~zero after subtraction."""
        spin = Spindown(spin_param_names=("F0",))
        model = TimingModel(delay_components=(), phase_components=(spin,))

        # TZR at exactly the same time as one of the TOAs
        tzr_int, tzr_frac = 59001.0, 0.25
        toa_data = make_gbt_toa_data(
            n_toas=3,
            tdb_int=59001.0,
            tdb_frac=jnp.array([0.0, 0.25, 0.5]),
            tzr_tdb_int=tzr_int,
            tzr_tdb_frac=tzr_frac,
            tzr_freq=1400.0,
        )
        params = make_spindown_params(f0=200.0, pepoch_int=59000.0)

        phase = model.compute_phase(toa_data, params)

        # TOA index 1 has the same time as TZR, so its phase should be ~0
        np.testing.assert_allclose(
            phase.int[1] + phase.frac[1], 0.0, atol=1e-10
        )

    def test_absolute_phase_differences_preserved(self):
        """Phase differences between TOAs are preserved after TZR subtraction."""
        spin = Spindown(spin_param_names=("F0",))
        model = TimingModel(delay_components=(), phase_components=(spin,))

        toa_data_no_tzr = make_gbt_toa_data(
            n_toas=3,
            tdb_int=59001.0,
            tdb_frac=jnp.array([0.0, 0.25, 0.5]),
        )
        toa_data_with_tzr = make_gbt_toa_data(
            n_toas=3,
            tdb_int=59001.0,
            tdb_frac=jnp.array([0.0, 0.25, 0.5]),
            tzr_tdb_int=59001.0,
            tzr_tdb_frac=0.0,
            tzr_freq=1400.0,
        )
        params = make_spindown_params(f0=200.0, pepoch_int=59000.0)

        phase_no_tzr = model.compute_phase(toa_data_no_tzr, params)
        phase_with_tzr = model.compute_phase(toa_data_with_tzr, params)

        # Phase differences should be identical
        diff_no_tzr = (phase_no_tzr.int[1:] + phase_no_tzr.frac[1:]) - (phase_no_tzr.int[0] + phase_no_tzr.frac[0])
        diff_with_tzr = (phase_with_tzr.int[1:] + phase_with_tzr.frac[1:]) - (phase_with_tzr.int[0] + phase_with_tzr.frac[0])

        np.testing.assert_allclose(diff_with_tzr, diff_no_tzr, rtol=1e-12)

    def test_tzr_with_dispersion(self):
        """TZR subtraction accounts for dispersion delay at TZR freq."""
        spin = Spindown(spin_param_names=("F0",))
        dm_comp = DispersionDM(dm_param_names=("DM",))
        model = TimingModel(
            delay_components=(dm_comp,),
            phase_components=(spin,),
        )

        toa_data = make_gbt_toa_data(
            n_toas=3,
            tdb_int=59001.0,
            tdb_frac=jnp.array([0.0, 0.25, 0.5]),
            freq=1400.0,
            tzr_tdb_int=59001.0,
            tzr_tdb_frac=0.25,
            tzr_freq=1400.0,
        )
        params = _make_full_params(f0=200.0, dm=15.0)

        phase = model.compute_phase(toa_data, params)

        # TOA index 1 has same time and freq as TZR → phase ≈ 0
        np.testing.assert_allclose(
            phase.int[1] + phase.frac[1], 0.0, atol=1e-10
        )


# ===========================================================================
# TZR TOA construction
# ===========================================================================


class TestBuildTzrToaData:
    """Tests for _reconstruct_tzr_toa helper."""

    def test_shape(self):
        """TZR TOAData has n_toas=1 and correct array shapes."""
        toa_data = make_gbt_toa_data(
            tzr_tdb_int=59001.0,
            tzr_tdb_frac=0.5,
            tzr_freq=1400.0,
        )
        tzr = _reconstruct_tzr_toa(toa_data)

        assert tzr.n_toas == 1
        assert tzr.tdb_int.shape == (1,)
        assert tzr.tdb_frac.shape == (1,)
        assert tzr.freq.shape == (1,)
        assert tzr.ssb_obs_pos.shape == (1, 3)

    def test_tdb_values(self):
        """TZR TOAData has correct TDB values."""
        toa_data = make_gbt_toa_data(
            tzr_tdb_int=59001.0,
            tzr_tdb_frac=0.5,
            tzr_freq=1400.0,
        )
        tzr = _reconstruct_tzr_toa(toa_data)

        np.testing.assert_allclose(tzr.tdb_int[0], 59001.0)
        np.testing.assert_allclose(tzr.tdb_frac[0], 0.5)

    def test_freq_value(self):
        """TZR TOAData uses the TZR frequency."""
        toa_data = make_gbt_toa_data(
            tzr_tdb_int=59001.0,
            tzr_tdb_frac=0.5,
            tzr_freq=2000.0,
        )
        tzr = _reconstruct_tzr_toa(toa_data)

        np.testing.assert_allclose(tzr.freq[0], 2000.0)

    def test_no_tzr_fields(self):
        """TZR TOAData has no TZR fields itself (no recursion)."""
        toa_data = make_gbt_toa_data(
            tzr_tdb_int=59001.0,
            tzr_tdb_frac=0.5,
            tzr_freq=1400.0,
        )
        tzr = _reconstruct_tzr_toa(toa_data)

        assert tzr.tzr_tdb_int is None
        assert tzr.tzr_tdb_frac is None


# ===========================================================================
# JIT and autodiff
# ===========================================================================


class TestJitAndGrad:
    """JIT compilation and gradient tests."""

    def test_compute_phase_jit(self):
        """compute_phase runs under jax.jit and matches eager."""
        spin = Spindown(spin_param_names=("F0",))
        model = TimingModel(delay_components=(), phase_components=(spin,))
        toa_data = make_gbt_toa_data(
            n_toas=3,
            tzr_tdb_int=59000.0,
            tzr_tdb_frac=0.5,
            tzr_freq=1400.0,
        )
        params = make_spindown_params(f0=200.0)

        eager = model.compute_phase(toa_data, params)
        jitted = jax.jit(model.compute_phase)(toa_data, params)

        np.testing.assert_allclose(jitted.int, eager.int, rtol=1e-14)
        np.testing.assert_allclose(jitted.frac, eager.frac, rtol=1e-14)

    def test_compute_phase_grad(self):
        """Gradient through compute_phase w.r.t. params produces finite values."""
        spin = Spindown(spin_param_names=("F0",))
        model = TimingModel(delay_components=(), phase_components=(spin,))
        toa_data = make_gbt_toa_data(
            n_toas=3,
            tdb_int=59001.0,
            tzr_tdb_int=59000.0,
            tzr_tdb_frac=0.5,
            tzr_freq=1400.0,
        )
        params = make_spindown_params(f0=200.0)

        def scalar_phase(p):
            ph = model.compute_phase(toa_data, p)
            return jnp.sum(ph.frac)

        grad = jax.grad(scalar_phase)(params)

        # grad is a ParameterVector-like pytree; only .values has gradients
        assert jnp.all(jnp.isfinite(grad.values))
        # dphase/dF0 should be nonzero (F0 directly multiplies dt)
        f0_idx = params.param_index("F0")
        assert grad.values[f0_idx] != 0.0

    def test_compute_phase_grad_with_delay(self):
        """Gradient flows through delay + phase components."""
        spin = Spindown(spin_param_names=("F0",))
        dm_comp = DispersionDM(dm_param_names=("DM",))
        model = TimingModel(
            delay_components=(dm_comp,),
            phase_components=(spin,),
        )
        toa_data = make_gbt_toa_data(
            n_toas=3,
            tdb_int=59001.0,
            tzr_tdb_int=59000.0,
            tzr_tdb_frac=0.5,
            tzr_freq=1400.0,
        )
        params = _make_full_params(f0=200.0, dm=15.0)

        def scalar_phase(p):
            ph = model.compute_phase(toa_data, p)
            return jnp.sum(ph.frac)

        grad = jax.grad(scalar_phase)(params)

        assert jnp.all(jnp.isfinite(grad.values))
        # dphase/dDM should be nonzero (DM affects delay → affects phase)
        dm_idx = params.param_index("DM")
        assert grad.values[dm_idx] != 0.0

    def test_no_retrace_same_structure(self):
        """Repeated calls with same-shaped data don't retrace."""
        spin = Spindown(spin_param_names=("F0",))
        model = TimingModel(delay_components=(), phase_components=(spin,))

        toa_data = make_gbt_toa_data(n_toas=3)
        params = make_spindown_params(f0=200.0)

        fn = jax.jit(model.compute_phase)
        fn(toa_data, params)  # first call compiles one variant
        # Same structure, different values -> cache hit, must not recompile.
        fn(toa_data, params.with_value("F0", 300.0))
        assert fn._cache_size() == 1, (
            f"recompiled on same-structure inputs: {fn._cache_size()} cached variants"
        )


# ===========================================================================
# Comparison against PINT
# ===========================================================================


class TestVsPINT:
    """Compare JaxPINT model output against PINT."""

    @pytest.fixture(scope="class")
    def ngc6440e(self):
        """Load NGC6440E dataset via PINT."""
        import pint.models as models
        import pint.toa as toa
        from pint.config import examplefile

        model = models.get_model(examplefile("NGC6440E.par"))
        toas = toa.get_TOAs(examplefile("NGC6440E.tim"), ephem="DE421")
        return model, toas

    @pytest.mark.slow
    def test_phase_finite_and_bounded(self, ngc6440e):
        """Spindown + DispersionDM phase is finite and within expected magnitude.

        This is a smoke/regression check on the phase pipeline, NOT a PINT
        parity test: PINT's absolute phase includes astrometry and other
        components this minimal model omits, so the totals are not directly
        comparable.  (Full PINT phase parity is covered by the bridge/native
        parity tests.)
        """
        pint_model, toas = ngc6440e
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params

        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params

        # Build JaxPINT model with only Spindown + DispersionDM.
        # NGC6440E has no DMEPOCH; use PEPOCH as the DM reference epoch
        # (matches PINT's fallback behaviour for constant-DM pulsars).
        spin = Spindown(spin_param_names=("F0", "F1"))
        dm_comp = DispersionDM(dm_param_names=("DM",), dmepoch_name="PEPOCH")
        jax_model = TimingModel(
            delay_components=(dm_comp,),
            phase_components=(spin,),
        )

        jax_phase = jax_model.compute_phase(toa_data, params)

        total_phase = jax_phase.int + jax_phase.frac
        assert jnp.all(jnp.isfinite(total_phase))
        # ~200 Hz pulsar over ~300 days; pulse numbers relative to the TZR
        # epoch are of order 1e9, comfortably under 1e11.
        assert jnp.max(jnp.abs(total_phase)) < 1e11

    @pytest.mark.slow
    def test_build_timing_model_factory(self, ngc6440e):
        """build_timing_model creates correct component types."""
        pint_model, toas = ngc6440e
        from jaxpint.bridge import build_timing_model

        jax_model, _noise = build_timing_model(pint_model)

        # Should have one Spindown phase component
        assert len(jax_model.phase_components) == 1
        assert isinstance(jax_model.phase_components[0], Spindown)
        assert jax_model.phase_components[0].spin_param_names == ("F0", "F1")

        # Should have AstrometryEquatorial + SolarSystemShapiroDelay + DispersionDM delay components
        from jaxpint.delay.astrometry import AstrometryEquatorial
        from jaxpint.delay.shapiro import SolarSystemShapiroDelay

        assert len(jax_model.delay_components) == 3
        delay_types = {type(c).__name__ for c in jax_model.delay_components}
        assert delay_types == {"AstrometryEquatorial", "SolarSystemShapiroDelay", "DispersionDM"}

        # Find components by type for detailed assertions
        astro = [c for c in jax_model.delay_components if isinstance(c, AstrometryEquatorial)][0]
        shapiro = [c for c in jax_model.delay_components if isinstance(c, SolarSystemShapiroDelay)][0]
        dm = [c for c in jax_model.delay_components if isinstance(c, DispersionDM)][0]

        assert astro.raj_name == "RAJ"
        assert astro.decj_name == "DECJ"
        assert shapiro.planet_shapiro is False
        assert dm.dm_param_names == ("DM",)
        # NGC6440E has no DMEPOCH, should fall back to PEPOCH
        assert dm.dmepoch_name == "PEPOCH"

    @pytest.mark.slow
    def test_build_timing_model_phase_matches(self, ngc6440e):
        """Full bridge-built model matches PINT's absolute phase (incl. pulse numbers).

        Unlike the mean-subtracted residual parity tests, comparing
        ``abs_phase=True`` phase catches TZR/pulse-numbering bugs: a
        constant integer-turn offset is invisible to residuals but fails
        here.
        """
        pint_model, toas = ngc6440e
        from jaxpint.bridge import build_timing_model, pint_toas_to_jax, pint_model_to_params

        jax_model, _noise = build_timing_model(pint_model)
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params

        jax_phase = jax_model.compute_phase(toa_data, params)

        # PINT reference: longdouble absolute phase (TZR-referenced).
        pint_phase = pint_model.phase(toas, abs_phase=True)

        # Never collapse int + frac at ~1e9 cycles (float64 ulp there is
        # ~1e-7 cycles).  Difference the parts first: the int-part
        # difference is small, so the sum below is exact.  This is also
        # robust to the two implementations landing on opposite sides of a
        # frac = +/-0.5 boundary (int off by one, frac compensating).
        dint = np.asarray(jax_phase.int) - pint_phase.int.astype(np.float64)
        dfrac = np.asarray(jax_phase.frac) - pint_phase.frac.astype(np.float64)
        dphase = dint + dfrac

        # Delay-path differences (float64 astrometry/Shapiro/DM vs PINT
        # longdouble) are ~ns-level; at F0 ~ 61.5 Hz that is ~1e-7 cycles.
        # Measured max|dphase| on 2026-07-23: 5.8e-8 cycles, with the int
        # parts agreeing exactly.
        np.testing.assert_allclose(dphase, 0.0, atol=2e-7)

    @pytest.mark.slow
    def test_spindown_phase_matches_pint_component(self, ngc6440e):
        """Spindown phase alone matches PINT's spindown_phase function."""
        pint_model, toas = ngc6440e
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params

        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params

        spin = Spindown(spin_param_names=("F0", "F1"))

        # JaxPINT: Spindown only, no delay, no TZR (actual tdb_int values)
        toa_data_no_tzr = TOAData(
            mjd_int=toa_data.tdb_int,
            mjd_frac=toa_data.tdb_frac,
            tdb_int=toa_data.tdb_int,
            tdb_frac=toa_data.tdb_frac,
            error=toa_data.error,
            freq=toa_data.freq,
            delta_pulse_number=toa_data.delta_pulse_number,
            ssb_obs_pos=toa_data.ssb_obs_pos,
            ssb_obs_vel=toa_data.ssb_obs_vel,
            obs_sun_pos=toa_data.obs_sun_pos,
            obs_indices=toa_data.obs_indices,
            flag_masks=toa_data.flag_masks,
            planet_positions=toa_data.planet_positions,
            dm_values=toa_data.dm_values,
            dm_errors=toa_data.dm_errors,
            tropo_alt=toa_data.tropo_alt, tropo_alt_valid=toa_data.tropo_alt_valid,
            obs_geodetic_lat=toa_data.obs_geodetic_lat, obs_height_km=toa_data.obs_height_km,
            n_toas=toa_data.n_toas,
            obs_names=toa_data.obs_names,
            tzr_tdb_int=None,
            tzr_tdb_frac=None,
            tzr_freq=None,
            tzr_ssb_obs_pos=None,
        )

        jax_phase = spin(toa_data_no_tzr, params, jnp.zeros(toa_data.n_toas))

        # PINT: spindown_phase with zero delay
        import astropy.units as u
        pint_spin = pint_model.components["Spindown"]
        pint_phase_quantity = pint_spin.spindown_phase(toas, delay=np.zeros(toas.ntoas) * u.s)
        pint_phase = pint_phase_quantity.value.astype(np.float64)

        jax_total = np.asarray(jax_phase.int + jax_phase.frac)

        # Should match within ~0.01 cycles (float64 precision over the data span)
        np.testing.assert_allclose(jax_total, pint_phase, rtol=1e-8)


class TestValidateReferencedParams:
    """build_model's build-time check that every component's referenced
    parameter names exist in the ParameterVector."""

    @staticmethod
    def _empty_noise():
        return NoiseModel(white_noise=None, correlated=())

    def test_missing_name_raises(self):
        # pepoch_name points at a parameter absent from the vector (F0 present).
        spin = Spindown(spin_param_names=("F0",), pepoch_name="MISSING_PEPOCH")
        model = TimingModel(delay_components=(), phase_components=(spin,))
        params = make_params(["F0"], [200.0], units=("Hz",))
        with pytest.raises(ValueError, match="MISSING_PEPOCH"):
            _validate_referenced_params(model, self._empty_noise(), params)

    def test_aggregates_all_missing(self):
        # Two components, each with one absent name -> a single error listing both.
        spin = Spindown(spin_param_names=("F0",), pepoch_name="MISSING_PEPOCH")
        dm = DispersionDM(dm_param_names=("DM",), dmepoch_name="MISSING_DMEPOCH")
        model = TimingModel(
            delay_components=(dm,),
            phase_components=(spin,),
            dispersion_components=(dm,),
        )
        params = make_params(["F0", "DM"], [200.0, 15.0], units=("Hz", "pc cm^-3"))
        with pytest.raises(ValueError) as exc:
            _validate_referenced_params(model, self._empty_noise(), params)
        msg = str(exc.value)
        assert "MISSING_PEPOCH" in msg and "MISSING_DMEPOCH" in msg

    def test_valid_model_does_not_raise(self):
        # All referenced names present (incl. epoch params PEPOCH/DMEPOCH).
        spin = Spindown(spin_param_names=("F0",))          # pepoch_name -> "PEPOCH"
        dm = DispersionDM(dm_param_names=("DM",))          # dmepoch_name -> "DMEPOCH"
        model = TimingModel(
            delay_components=(dm,),
            phase_components=(spin,),
            dispersion_components=(dm,),
        )
        params = _make_full_params()
        _validate_referenced_params(model, self._empty_noise(), params)  # no raise

    def test_phoff_name_validated(self):
        # PHOFF is the one name-bearing field on TimingModel itself.
        spin = Spindown(spin_param_names=("F0",))
        model = TimingModel(
            delay_components=(),
            phase_components=(spin,),
            phoff_name="PHOFF",
        )
        params = _make_full_params()  # has F0/PEPOCH/DM/DMEPOCH but not PHOFF
        with pytest.raises(ValueError, match="PHOFF"):
            _validate_referenced_params(model, self._empty_noise(), params)


class TestValidateFlagMasks:
    """build_model's build-time check that the TOAData carries a flag mask for
    every masked parameter declared in the par."""

    @staticmethod
    def _par(mask_names):
        names = list(mask_names) or ["F0"]
        return ParResult(
            params=make_params(names, [1.0] * len(names), units=("",) * len(names)),
            mask_info={
                n: MaskInfo(name=n, key="-fe", key_value="430") for n in mask_names
            },
        )

    def test_missing_mask_raises(self):
        # par declares EFAC1 as a mask param, but the TOAData carries no mask for it.
        par = self._par(["EFAC1"])
        toa = make_toa_data(n_toas=4, flag_masks={})
        with pytest.raises(ValueError, match="EFAC1"):
            _validate_flag_masks(par, toa)

    def test_present_mask_ok(self):
        par = self._par(["EFAC1"])
        toa = make_toa_data(n_toas=4, flag_masks={"EFAC1": jnp.ones(4, dtype=bool)})
        _validate_flag_masks(par, toa)  # no raise

    def test_no_masks_ok(self):
        par = self._par([])
        toa = make_toa_data(n_toas=4, flag_masks={})
        _validate_flag_masks(par, toa)  # no raise


# ===========================================================================
# param_is_set: free-at-zero parameters stay wired into the model
# ===========================================================================


class TestFreeAtZeroParams:
    """A free parameter at value 0 (e.g. ``PMRA 0 1``) must stay connected to
    its component.  ``param_is_set`` formerly required a nonzero value, so
    such parameters ended up referenced by no component: their design-matrix
    column was identically zero, the SVD cutoff truncated it, and the fit
    silently never moved them — while fitting PM/PHOFF from an initial 0 is
    standard practice (and works in PINT).  Frozen zero-valued parameters
    remain unset."""

    _PAR = """\
PSR           J0000+0000
EPHEM         DE421
CLK           TT(BIPM2019)
UNITS         TDB
RAJ           04:37:15.0
DECJ          -47:15:09.0
PMRA          0 1
PMDEC         0 1
PX            0
PEPOCH        54000
POSEPOCH      54000
F0            100.0 1
F1            -1e-15 1
DM            15.0 1
PHOFF         0 1
TZRMJD        54000
TZRFRQ        1400
TZRSITE       @
"""

    def _build(self):
        import io

        import jaxpint.par as jpar
        from jaxpint import build_model

        par = jpar.get_model(io.StringIO(self._PAR))
        model, _ = build_model(par)
        return par, model

    def test_param_is_set_semantics(self):
        import io

        import jaxpint.par as jpar
        from jaxpint._build_context import param_is_set

        par = jpar.get_model(io.StringIO(self._PAR))
        assert param_is_set(par, "PMRA")  # zero but free -> set
        assert param_is_set(par, "PHOFF")  # zero but free -> set
        assert param_is_set(par, "F0")  # nonzero -> set
        assert not param_is_set(par, "PX")  # zero AND frozen -> unset
        assert not param_is_set(par, "NOT_A_PARAM")

    def test_free_at_zero_pm_wired_into_astrometry(self):
        from jaxpint.delay.astrometry import AstrometryEquatorial

        _, model = self._build()
        astro = [
            c for c in model.delay_components
            if isinstance(c, AstrometryEquatorial)
        ]
        assert len(astro) == 1
        assert astro[0].pmra_name == "PMRA"
        assert astro[0].pmdec_name == "PMDEC"
        # PX is zero and frozen -> stays disconnected.
        assert astro[0].px_name is None

    def test_free_at_zero_phoff_wired(self):
        _, model = self._build()
        assert model.phoff_name == "PHOFF"

    def test_free_at_zero_pm_has_nonzero_design_column(self):
        """End-to-end: d(delay)/d(PMRA) is not identically zero at PMRA=0."""
        par, model = self._build()
        params = par.params
        assert "PMRA" in params.free_names()

        from tests.helpers import make_toa_data

        # Spread over a year so proper motion has a lever arm; realistic
        # ssb_obs_pos is not needed for a nonzero-column check, but the
        # default zeros would make the astrometric delay vanish — patch in
        # an Earth-orbit-scale position.
        import equinox as eqx

        t = np.linspace(54001.0, 54365.0, 12)
        toa_data = make_toa_data(t_mjd=t)
        phase = 2.0 * np.pi * (t - t[0]) / 365.25
        obs = np.stack(
            [1.496e8 * np.cos(phase), 1.496e8 * np.sin(phase), np.zeros_like(phase)],
            axis=1,
        )
        toa_data = eqx.tree_at(
            lambda td: td.ssb_obs_pos, toa_data, jnp.asarray(obs)
        )

        def delay_sum(p):
            return model.compute_delay(toa_data, p).sum()

        grads = jax.grad(delay_sum)(params)
        pmra_idx = params.param_index("PMRA")
        assert jnp.isfinite(grads.values[pmra_idx])
        assert float(jnp.abs(grads.values[pmra_idx])) > 0.0

    def test_free_at_zero_px_has_finite_nonzero_gradient(self):
        """PX free at 0 now stays wired; the parallax term is linear in PX,
        so d(delay)/d(PX) at PX=0 must be finite and nonzero (the former
        1/PX formulation gave 0*inf = nan there)."""
        import io

        import equinox as eqx

        import jaxpint.par as jpar
        from jaxpint import build_model
        from tests.helpers import make_toa_data

        par_text = self._PAR.replace("PX            0", "PX            0 1")
        par = jpar.get_model(io.StringIO(par_text))
        model, _ = build_model(par)
        params = par.params
        assert "PX" in params.free_names()

        t = np.linspace(54001.0, 54365.0, 12)
        toa_data = make_toa_data(t_mjd=t)
        phase = 2.0 * np.pi * (t - t[0]) / 365.25
        obs = np.stack(
            [1.496e8 * np.cos(phase), 1.496e8 * np.sin(phase), np.zeros_like(phase)],
            axis=1,
        )
        toa_data = eqx.tree_at(lambda td: td.ssb_obs_pos, toa_data, jnp.asarray(obs))

        def delay_sum(p):
            return model.compute_delay(toa_data, p).sum()

        grads = jax.grad(delay_sum)(params)
        px_grad = grads.values[params.param_index("PX")]
        assert jnp.isfinite(px_grad)
        assert float(jnp.abs(px_grad)) > 0.0
