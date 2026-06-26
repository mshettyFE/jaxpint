"""Glitch phase component: models pulsar spin-down glitches.

Each Glitch contributes phase only for TOAs after its epoch (GLEP_N):

    phase = GLPH + GLF0*dt + 0.5*GLF1*dt^2 + (1/6)*GLF2*dt^3
          + GLF0D * tau * (1 - exp(-dt / tau))

where dt = (t_TDB - GLEP_N) in seconds minus accumulated delay,
and tau = GLTD_N in seconds.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl, PhaseComponent
from jaxpint.constants import SECS_PER_DAY
from jaxpint.types.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector


class Glitch(PhaseComponent):
    """Pulsar glitch phase component.

    Parameters
    ----------
    n_glitches : int
        Number of glitches in the model.
    glep_names : tuple[str, ...]
        MJD epoch parameter names, one per glitch (e.g. ``("GLEP_1",)``).
    glph_names, glf0_names, glf1_names, glf2_names : tuple[str, ...]
        Phase jump, frequency, frequency-derivative, and second
        frequency-derivative parameter names.
    glf0d_names, gltd_names : tuple[str, ...]
        Decaying-frequency and decay-time-constant parameter names.

    Raises
    ------
    ValueError
        If ``n_glitches`` is less than 1.
    ValueError
        If the length of any glitch parameter name tuple does not match
        ``n_glitches``.
    """

    PARAMS = (
        ParamDecl("GLEP_1", kind="mjd", prefix="GLEP_"),
        ParamDecl("GLPH_1", prefix="GLPH_"),
        ParamDecl("GLF0_1", prefix="GLF0_"),
        ParamDecl("GLF1_1", prefix="GLF1_"),
        ParamDecl("GLF2_1", prefix="GLF2_"),
        ParamDecl("GLF0D_1", prefix="GLF0D_"),
        ParamDecl("GLTD_1", prefix="GLTD_"),
    )

    n_glitches: int = eqx.field(static=True)
    glep_names: tuple[str, ...] = eqx.field(static=True)
    glph_names: tuple[str, ...] = eqx.field(static=True)
    glf0_names: tuple[str, ...] = eqx.field(static=True)
    glf1_names: tuple[str, ...] = eqx.field(static=True)
    glf2_names: tuple[str, ...] = eqx.field(static=True)
    glf0d_names: tuple[str, ...] = eqx.field(static=True)
    gltd_names: tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self):
        if self.n_glitches < 1:
            raise ValueError("Glitch requires at least one glitch")
        for attr in (
            "glep_names",
            "glph_names",
            "glf0_names",
            "glf1_names",
            "glf2_names",
            "glf0d_names",
            "gltd_names",
        ):
            if len(getattr(self, attr)) != self.n_glitches:
                raise ValueError(
                    f"Length of {attr} ({len(getattr(self, attr))}) "
                    f"does not match n_glitches ({self.n_glitches})"
                )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> DualFloat:
        """Compute glitch phase contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times, etc.).
        params : ParameterVector
            Timing-model parameters containing glitch parameters.
        delay : array, shape (n_toas,)
            Accumulated signal delay in **seconds**.

        Returns
        -------
        DualFloat
            Glitch phase in cycles (dimensionless), split as int + frac.
        """
        phase = jnp.zeros(toa_data.n_toas)

        for i in range(self.n_glitches):
            glep = params.epoch_dual(self.glep_names[i])
            dt_dual = toa_data.tdb - glep
            dt = dt_dual.total * SECS_PER_DAY - delay

            glph = params.param_value(self.glph_names[i])
            glf0 = params.param_value(self.glf0_names[i])
            glf1 = params.param_value(self.glf1_names[i])
            glf2 = params.param_value(self.glf2_names[i])
            glf0d = params.param_value(self.glf0d_names[i])
            gltd = params.param_value(self.gltd_names[i]) * SECS_PER_DAY

            glitch_phase = (
                glph + glf0 * dt + 0.5 * glf1 * dt**2 + (1.0 / 6.0) * glf2 * dt**3
            )

            # Exponential decay term (safe against gltd == 0)
            safe_gltd = jnp.where(gltd != 0.0, gltd, 1.0)
            decay = glf0d * safe_gltd * (1.0 - jnp.exp(-dt / safe_gltd))
            decay = jnp.where(gltd != 0.0, decay, 0.0)
            glitch_phase = glitch_phase + decay

            # Only apply for TOAs after the glitch epoch
            phase = phase + jnp.where(dt > 0.0, glitch_phase, 0.0)

        return DualFloat.from_cycles(jnp.zeros(toa_data.n_toas), phase)
