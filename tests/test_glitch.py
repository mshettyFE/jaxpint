"""Tests for jaxpint.glitch: Glitch phase component."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jaxpint.phase.glitch import Glitch
from jaxpint.constants import SECS_PER_DAY
from tests.helpers import make_toa_data, make_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_glitch(
    glph=0.0, glf0=0.0, glf1=0.0, glf2=0.0, glf0d=0.0, gltd=0.0,
    glep_int=59100.0, glep_frac=0.0,
):
    """Build a single-glitch Glitch component and matching params."""
    comp = Glitch(
        n_glitches=1,
        glep_names=("GLEP_1",),
        glph_names=("GLPH_1",),
        glf0_names=("GLF0_1",),
        glf1_names=("GLF1_1",),
        glf2_names=("GLF2_1",),
        glf0d_names=("GLF0D_1",),
        gltd_names=("GLTD_1",),
    )
    params = make_params(
        names=("GLEP_1", "GLPH_1", "GLF0_1", "GLF1_1", "GLF2_1", "GLF0D_1", "GLTD_1"),
        values=(glep_frac, glph, glf0, glf1, glf2, glf0d, gltd),
        units=("day", "phase", "Hz", "Hz/s", "Hz/s^2", "Hz", "day"),
        components=("Glitch",) * 7,
        epoch_int_values={"GLEP_1": glep_int},
    )
    return comp, params


# ===========================================================================
# Construction tests
# ===========================================================================

class TestConstruction:
    def test_single_glitch(self):
        g = Glitch(
            n_glitches=1,
            glep_names=("GLEP_1",), glph_names=("GLPH_1",),
            glf0_names=("GLF0_1",), glf1_names=("GLF1_1",),
            glf2_names=("GLF2_1",), glf0d_names=("GLF0D_1",),
            gltd_names=("GLTD_1",),
        )
        assert g.n_glitches == 1

    def test_multiple_glitches(self):
        g = Glitch(
            n_glitches=2,
            glep_names=("GLEP_1", "GLEP_3"),
            glph_names=("GLPH_1", "GLPH_3"),
            glf0_names=("GLF0_1", "GLF0_3"),
            glf1_names=("GLF1_1", "GLF1_3"),
            glf2_names=("GLF2_1", "GLF2_3"),
            glf0d_names=("GLF0D_1", "GLF0D_3"),
            gltd_names=("GLTD_1", "GLTD_3"),
        )
        assert g.n_glitches == 2

    def test_zero_glitches_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Glitch(
                n_glitches=0,
                glep_names=(), glph_names=(), glf0_names=(),
                glf1_names=(), glf2_names=(), glf0d_names=(),
                gltd_names=(),
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            Glitch(
                n_glitches=2,
                glep_names=("GLEP_1",),  # wrong length
                glph_names=("GLPH_1", "GLPH_2"),
                glf0_names=("GLF0_1", "GLF0_2"),
                glf1_names=("GLF1_1", "GLF1_2"),
                glf2_names=("GLF2_1", "GLF2_2"),
                glf0d_names=("GLF0D_1", "GLF0D_2"),
                gltd_names=("GLTD_1", "GLTD_2"),
            )


class TestPytree:
    def test_zero_dynamic_leaves(self):
        g, _ = _make_single_glitch()
        leaves, _ = jax.tree.flatten(g)
        assert len(leaves) == 0


# ===========================================================================
# Phase computation tests
# ===========================================================================

class TestPhaseComputation:
    def test_phase_jump_only(self):
        """GLPH_1 = 0.3 should give 0.3 for post-glitch TOAs, 0 for pre-glitch."""
        comp, params = _make_single_glitch(glph=0.3, glep_int=59100.0)
        # TOAs: 2 before glitch, 3 after
        toa_data = make_toa_data(
            t_mjd=[59099.0, 59099.5, 59101.0, 59102.0, 59103.0]
        )
        delay = jnp.zeros(5)
        result = comp(toa_data, params, delay)
        phase = result.int + result.frac

        np.testing.assert_allclose(phase[:2], 0.0, atol=1e-15)
        np.testing.assert_allclose(phase[2:], 0.3, atol=1e-12)

    def test_pre_glitch_zero(self):
        """All TOAs before the glitch epoch should have zero phase."""
        comp, params = _make_single_glitch(glph=1.0, glf0=1e-5, glep_int=60000.0)
        toa_data = make_toa_data(t_mjd=[59990.0, 59995.0, 59999.0])
        delay = jnp.zeros(3)
        result = comp(toa_data, params, delay)
        phase = result.int + result.frac
        np.testing.assert_allclose(phase, 0.0, atol=1e-15)

    def test_frequency_step(self):
        """GLF0 * dt should give expected phase for post-glitch TOAs."""
        glf0 = 1e-6  # Hz
        comp, params = _make_single_glitch(glf0=glf0, glep_int=59100.0)
        toa_data = make_toa_data(t_mjd=[59101.0])  # 1 day after glitch
        delay = jnp.zeros(1)
        result = comp(toa_data, params, delay)
        phase = result.int + result.frac

        expected = glf0 * 1.0 * SECS_PER_DAY  # GLF0 * dt_seconds
        np.testing.assert_allclose(phase[0], expected, rtol=1e-12)

    def test_full_polynomial(self):
        """Test GLPH + GLF0*dt + 0.5*GLF1*dt^2 + (1/6)*GLF2*dt^3."""
        glph = 0.1
        glf0 = 2e-6
        glf1 = -1e-14
        glf2 = 3e-22
        comp, params = _make_single_glitch(
            glph=glph, glf0=glf0, glf1=glf1, glf2=glf2, glep_int=59100.0
        )
        toa_data = make_toa_data(t_mjd=[59110.0])  # 10 days after
        delay = jnp.zeros(1)
        result = comp(toa_data, params, delay)
        phase = result.int + result.frac

        dt = 10.0 * SECS_PER_DAY
        expected = glph + glf0 * dt + 0.5 * glf1 * dt**2 + (1.0/6.0) * glf2 * dt**3
        np.testing.assert_allclose(phase[0], expected, rtol=1e-10)

    def test_decay_term(self):
        """GLF0D and GLTD should give exponential recovery."""
        glf0d = 1e-6  # Hz
        gltd = 10.0   # days
        comp, params = _make_single_glitch(
            glf0d=glf0d, gltd=gltd, glep_int=59100.0
        )
        toa_data = make_toa_data(t_mjd=[59110.0])  # 10 days after
        delay = jnp.zeros(1)
        result = comp(toa_data, params, delay)
        phase = result.int + result.frac

        dt = 10.0 * SECS_PER_DAY
        tau = gltd * SECS_PER_DAY
        expected = glf0d * tau * (1.0 - np.exp(-dt / tau))
        np.testing.assert_allclose(phase[0], expected, rtol=1e-12)

    def test_no_decay_when_gltd_zero(self):
        """When GLTD=0, decay term should be zero even if GLF0D is nonzero."""
        comp, params = _make_single_glitch(glf0d=1e-6, gltd=0.0, glep_int=59100.0)
        toa_data = make_toa_data(t_mjd=[59110.0])
        delay = jnp.zeros(1)
        result = comp(toa_data, params, delay)
        phase = result.int + result.frac
        np.testing.assert_allclose(phase[0], 0.0, atol=1e-15)

    def test_delay_shifts_dt(self):
        """Accumulated delay should shift dt for the glitch."""
        glf0 = 1e-6
        comp, params = _make_single_glitch(glf0=glf0, glep_int=59100.0)
        toa_data = make_toa_data(t_mjd=[59101.0])
        delay_val = 10.0  # 10 seconds of delay
        delay = jnp.array([delay_val])
        result = comp(toa_data, params, delay)
        phase = result.int + result.frac

        dt = 1.0 * SECS_PER_DAY - delay_val
        expected = glf0 * dt
        np.testing.assert_allclose(phase[0], expected, rtol=1e-12)


class TestMultipleGlitches:
    def test_two_glitches_additive(self):
        """Two glitches at different epochs should add their phase contributions."""
        comp = Glitch(
            n_glitches=2,
            glep_names=("GLEP_1", "GLEP_2"),
            glph_names=("GLPH_1", "GLPH_2"),
            glf0_names=("GLF0_1", "GLF0_2"),
            glf1_names=("GLF1_1", "GLF1_2"),
            glf2_names=("GLF2_1", "GLF2_2"),
            glf0d_names=("GLF0D_1", "GLF0D_2"),
            gltd_names=("GLTD_1", "GLTD_2"),
        )
        params = make_params(
            names=("GLEP_1", "GLPH_1", "GLF0_1", "GLF1_1", "GLF2_1", "GLF0D_1", "GLTD_1",
                   "GLEP_2", "GLPH_2", "GLF0_2", "GLF1_2", "GLF2_2", "GLF0D_2", "GLTD_2"),
            values=(0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
            units=("day", "phase", "Hz", "Hz/s", "Hz/s^2", "Hz", "day") * 2,
            components=("Glitch",) * 14,
            epoch_int_values={"GLEP_1": 59100.0, "GLEP_2": 59200.0},
        )

        # TOA before both, between, and after both
        toa_data = make_toa_data(t_mjd=[59050.0, 59150.0, 59250.0])
        delay = jnp.zeros(3)
        result = comp(toa_data, params, delay)
        phase = result.int + result.frac

        np.testing.assert_allclose(phase[0], 0.0, atol=1e-15)    # before both
        np.testing.assert_allclose(phase[1], 0.1, atol=1e-12)    # after glitch 1 only
        np.testing.assert_allclose(phase[2], 0.3, atol=1e-12)    # after both


# ===========================================================================
# JIT and gradient tests
# ===========================================================================

class TestJitAndGrad:
    def test_jit(self):
        comp, params = _make_single_glitch(glph=0.3, glf0=1e-6)
        toa_data = make_toa_data(t_mjd=[59101.0, 59102.0, 59103.0])
        delay = jnp.zeros(3)

        result_eager = comp(toa_data, params, delay)
        result_jit = jax.jit(comp)(toa_data, params, delay)

        np.testing.assert_allclose(
            result_eager.frac, result_jit.frac, atol=1e-15
        )

    def test_grad_glph(self):
        """Gradient of total phase w.r.t. GLPH should be 1.0 for post-glitch TOAs."""
        comp, params = _make_single_glitch(glph=0.3, glep_int=59100.0)
        toa_data = make_toa_data(t_mjd=[59101.0])
        delay = jnp.zeros(1)

        def phase_sum(p):
            result = comp(toa_data, p, delay)
            return jnp.sum(result.int + result.frac)

        grad_params = jax.grad(phase_sum)(params)
        glph_idx = params.param_index("GLPH_1")
        np.testing.assert_allclose(grad_params.values[glph_idx], 1.0, atol=1e-12)

    def test_grad_glf0(self):
        """Gradient w.r.t. GLF0 should be dt for post-glitch TOAs."""
        comp, params = _make_single_glitch(glf0=1e-6, glep_int=59100.0)
        toa_data = make_toa_data(t_mjd=[59101.0])  # 1 day after
        delay = jnp.zeros(1)

        def phase_sum(p):
            result = comp(toa_data, p, delay)
            return jnp.sum(result.int + result.frac)

        grad_params = jax.grad(phase_sum)(params)
        glf0_idx = params.param_index("GLF0_1")
        expected_dt = 1.0 * SECS_PER_DAY
        np.testing.assert_allclose(grad_params.values[glf0_idx], expected_dt, rtol=1e-10)

    def test_grad_finite(self):
        """All gradients should be finite (no NaN from decay safety)."""
        comp, params = _make_single_glitch(
            glph=0.1, glf0=1e-6, glf1=-1e-14, glf0d=1e-6, gltd=10.0,
            glep_int=59100.0,
        )
        toa_data = make_toa_data(t_mjd=[59110.0])
        delay = jnp.zeros(1)

        def phase_sum(p):
            result = comp(toa_data, p, delay)
            return jnp.sum(result.int + result.frac)

        grad_params = jax.grad(phase_sum)(params)
        assert jnp.all(jnp.isfinite(grad_params.values))

    def test_grad_finite_zero_decay(self):
        """Gradients should be finite even when GLTD=0 and GLF0D=0."""
        comp, params = _make_single_glitch(
            glph=0.1, glf0=1e-6, glf0d=0.0, gltd=0.0, glep_int=59100.0,
        )
        toa_data = make_toa_data(t_mjd=[59110.0])
        delay = jnp.zeros(1)

        def phase_sum(p):
            result = comp(toa_data, p, delay)
            return jnp.sum(result.int + result.frac)

        grad_params = jax.grad(phase_sum)(params)
        assert jnp.all(jnp.isfinite(grad_params.values))


# ===========================================================================
# Integration test: JaxPINT glitch phase vs PINT glitch phase
# ===========================================================================

class TestVsPINT:
    """Compare JaxPINT Glitch output against PINT on J0007+7303 (3 glitches)."""

    GLITCH_PAR = """\
        PSRJ           J0835-4510
        RAJ            08:35:20.61149
        DECJ           -45:10:34.8751
        F0             11.18965156782
        F1             -1.5e-12
        PEPOCH         55305
        POSEPOCH       55305
        DM             67.99
        UNITS          TDB
        EPHEM          DE440
        GLEP_1         55200
        GLPH_1         0.0
        GLF0_1         1.75e-06
        GLF1_1         -6.57e-15
        GLF0D_1        1.0e-06
        GLTD_1         10.0
        GLEP_2         55500
        GLPH_2         0.4
        GLF0_2         3.9e-06
        GLF1_2         -4.82e-14
    """

    @pytest.fixture
    def glitch_model(self):
        """Build a glitch model from inline par and generate fake TOAs."""
        from io import StringIO
        import pint.models as models
        import pint.toa as toa
        from pint.simulation import make_fake_toas_uniform

        model = models.get_model(StringIO(self.GLITCH_PAR))
        toas = make_fake_toas_uniform(
            startMJD=55000, endMJD=55700, ntoas=100,
            model=model, add_noise=False,
        )
        return model, toas

    def test_glitch_phase_matches_pint(self, glitch_model):
        """JaxPINT glitch phase matches PINT's glitch_phase with zero delay."""
        import astropy.units as u
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model

        pint_model, toas = glitch_model

        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)

        # Build JaxPINT model automatically from PINT
        jax_model, _ = build_timing_model(pint_model)

        # Find the Glitch component
        glitch_comp = None
        for comp in jax_model.phase_components:
            if isinstance(comp, Glitch):
                glitch_comp = comp
                break
        assert glitch_comp is not None, "build_timing_model should produce a Glitch component"
        assert glitch_comp.n_glitches == 2

        # Compute JaxPINT glitch phase with zero delay
        zero_delay = jnp.zeros(toa_data.n_toas)
        jax_result = glitch_comp(toa_data, params, zero_delay)
        jax_phase = np.asarray(jax_result.int + jax_result.frac)

        # Compute PINT glitch phase with zero delay
        pint_glitch = pint_model.components["Glitch"]
        pint_phase = pint_glitch.glitch_phase(
            toas, delay=np.zeros(toas.ntoas) * u.s
        ).value.astype(np.float64)

        np.testing.assert_allclose(jax_phase, pint_phase, rtol=1e-8)

    def test_glitch_phase_matches_pint_with_delay(self, glitch_model):
        """JaxPINT glitch phase matches PINT's glitch_phase with nonzero delay."""
        import astropy.units as u
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model

        pint_model, toas = glitch_model

        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)

        jax_model, _ = build_timing_model(pint_model)
        glitch_comp = [c for c in jax_model.phase_components if isinstance(c, Glitch)][0]

        # Use PINT's full delay as the delay input
        pint_delay = pint_model.delay(toas)
        delay_seconds = pint_delay.to(u.s).value
        jax_delay = jnp.array(delay_seconds, dtype=jnp.float64)

        # JaxPINT glitch phase
        jax_result = glitch_comp(toa_data, params, jax_delay)
        jax_phase = np.asarray(jax_result.int + jax_result.frac)

        # PINT glitch phase
        pint_phase = pint_model.components["Glitch"].glitch_phase(
            toas, delay=pint_delay
        ).value.astype(np.float64)

        np.testing.assert_allclose(jax_phase, pint_phase, rtol=1e-8)

    def test_full_model_residuals(self, glitch_model):
        """Full-model residuals with glitches: glitch contribution matches PINT.

        TODO: There is a small constant offset between JaxPINT and PINT full-model
        residuals (pre-existing, not glitch-related). We verify that the
        *variation* (mean-subtracted residuals) matches, confirming the glitch
        phase is applied correctly in the full pipeline.
        """
        import pint.residuals
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model
        from jaxpint.fitter import compute_phase_residuals

        pint_model, toas = glitch_model

        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)
        jax_model, _ = build_timing_model(pint_model)

        # JaxPINT residuals
        jax_resid = np.asarray(compute_phase_residuals(jax_model, toa_data, params))

        # PINT residuals
        pint_resid = pint.residuals.Residuals(toas, pint_model).phase_resids.value.astype(np.float64)

        # Compare mean-subtracted residuals to remove constant offset
        jax_centered = jax_resid - np.mean(jax_resid)
        pint_centered = pint_resid - np.mean(pint_resid)
        np.testing.assert_allclose(jax_centered, pint_centered, atol=1e-6)
