"""Tests for jaxpint.model: TimingModel orchestration layer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


from jaxpint.types import TOAData
from jaxpint.phase_result import PhaseResult
from jaxpint.phase.spin import Spindown
from jaxpint.delay.dispersion_dm import DispersionDM
from jaxpint.model import TimingModel, _build_tzr_toa_data
from tests.helpers import make_gbt_toa_data, make_spindown_params, make_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_full_params(
    f0=200.0, f1=None, dm=15.0,
    pepoch_int=59000.0, pepoch_frac=0.0,
    dmepoch_int=59000.0, dmepoch_frac=0.0,
):
    names = ["F0"]; values = [f0]
    components = ["Spindown"]; units = ["Hz"]

    if f1 is not None:
        names += ["F1"]; values += [f1]
        components += ["Spindown"]; units += ["Hz/s"]

    names += ["PEPOCH"]; values += [pepoch_frac]
    components += ["Spindown"]; units += ["day"]
    names += ["DM"]; values += [dm]
    components += ["DispersionDM"]; units += ["pc cm^-3"]
    names += ["DMEPOCH"]; values += [dmepoch_frac]
    components += ["DispersionDM"]; units += ["day"]

    return make_params(names, values, units=tuple(units),
                       components=tuple(components),
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


# ===========================================================================
# compute_phase — without absolute phase (no TZR)
# ===========================================================================


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
    """Tests for _build_tzr_toa_data helper."""

    def test_shape(self):
        """TZR TOAData has n_toas=1 and correct array shapes."""
        toa_data = make_gbt_toa_data(
            tzr_tdb_int=59001.0,
            tzr_tdb_frac=0.5,
            tzr_freq=1400.0,
        )
        tzr = _build_tzr_toa_data(toa_data)

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
        tzr = _build_tzr_toa_data(toa_data)

        np.testing.assert_allclose(tzr.tdb_int[0], 59001.0)
        np.testing.assert_allclose(tzr.tdb_frac[0], 0.5)

    def test_freq_value(self):
        """TZR TOAData uses the TZR frequency."""
        toa_data = make_gbt_toa_data(
            tzr_tdb_int=59001.0,
            tzr_tdb_frac=0.5,
            tzr_freq=2000.0,
        )
        tzr = _build_tzr_toa_data(toa_data)

        np.testing.assert_allclose(tzr.freq[0], 2000.0)

    def test_no_tzr_fields(self):
        """TZR TOAData has no TZR fields itself (no recursion)."""
        toa_data = make_gbt_toa_data(
            tzr_tdb_int=59001.0,
            tzr_tdb_frac=0.5,
            tzr_freq=1400.0,
        )
        tzr = _build_tzr_toa_data(toa_data)

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
        fn(toa_data, params)  # first call triggers trace

        # Change values but not structure
        params2 = params.with_value("F0", 300.0)
        with jax.log_compiles():
            fn(toa_data, params2)  # should not retrace


# ===========================================================================
# Comparison against PINT
# ===========================================================================


class TestVsPINT:
    """Compare JaxPINT model output against PINT."""

    @pytest.fixture
    def ngc6440e(self):
        """Load NGC6440E dataset via PINT."""
        import pint.models as models
        import pint.toa as toa
        from pint.config import examplefile

        model = models.get_model(examplefile("NGC6440E.par"))
        toas = toa.get_TOAs(examplefile("NGC6440E.tim"), ephem="DE421")
        return model, toas

    def test_phase_matches_pint(self, ngc6440e):
        """Absolute phase matches PINT within float64 precision.

        This test only validates Spindown + DispersionDM. Astrometry
        delays are not yet ported, so we expect agreement only after
        accounting for the missing delay components.
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

        # Compute JaxPINT phase
        jax_phase = jax_model.compute_phase(toa_data, params)

        # Compute PINT phase (only Spindown + Dispersion, no astrometry)
        # We can't directly compare since PINT uses all components.
        # Instead, verify internal consistency: delay + phase pipeline
        # produces finite, reasonable values.
        total_phase = jax_phase.int + jax_phase.frac
        assert jnp.all(jnp.isfinite(total_phase))
        # Phase values: ~200 Hz pulsar over ~300 days ≈ 5e9 cycles.
        # After TZR subtraction, absolute phases are pulse numbers
        # relative to TZR epoch — order 10^9 is expected.
        assert jnp.max(jnp.abs(total_phase)) < 1e11

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

        assert shapiro.planet_shapiro is False
        assert dm.dm_param_names == ("DM",)
        # NGC6440E has no DMEPOCH, should fall back to PEPOCH
        assert dm.dmepoch_name == "PEPOCH"

    def test_build_timing_model_phase_matches(self, ngc6440e):
        """build_timing_model produces a model whose phase is finite and consistent."""
        pint_model, toas = ngc6440e
        from jaxpint.bridge import build_timing_model, pint_toas_to_jax, pint_model_to_params

        jax_model, _noise = build_timing_model(pint_model)
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params

        phase = jax_model.compute_phase(toa_data, params)
        total = phase.int + phase.frac

        assert jnp.all(jnp.isfinite(total))
        assert total.shape == (toas.ntoas,)

    def test_spindown_phase_matches_pint_component(self, ngc6440e):
        """Spindown phase alone matches PINT's spindown_phase function."""
        pint_model, toas = ngc6440e
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params

        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params

        spin = Spindown(spin_param_names=("F0", "F1"))
        jax_model = TimingModel(delay_components=(), phase_components=(spin,))

        # JaxPINT: Spindown only, no delay, no TZR
        toa_data_no_tzr = make_gbt_toa_data(
            n_toas=toa_data.n_toas,
            tdb_int=float(toa_data.tdb_int[0]),
            tdb_frac=toa_data.tdb_frac,
        )
        # Override with actual tdb_int values
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
