"""ELL1 binary delay model and variants for low-eccentricity orbits.

Uses rectangular eccentricity parameters (EPS1, EPS2) and TASC instead
of (ECC, OM, T0).  No Kepler equation — the Roemer delay is expanded
analytically to O(e^3).

Reference
---------
Lange et al. (2001), MNRAS, 326 (1), 274-282.
Zhu et al. (2019), MNRAS, 482, 3249.
PINT ``stand_alone_psr_binaries/ELL1_model.py``.
"""

from __future__ import annotations

from typing import Optional

import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent
from jaxpint.types import TOAData, ParameterVector
from jaxpint.binary.common import (
    TSUN,
    SECS_PER_DAY,
    compute_tt0,
    compute_orbital_phase,
)


# ---------------------------------------------------------------------------
# ELL1 Roemer delay expansion (O(e^3))
# ---------------------------------------------------------------------------

def _ell1_roemer_da1(Phi, eps1, eps2):
    """ELL1 Roemer delay / a1, including 3rd-order eccentricity corrections.

    Zhu et al. (2019) Eq. 1, with typo correction from TEMPO source.
    """
    sin_Phi = jnp.sin(Phi)
    sin_2Phi = jnp.sin(2.0 * Phi)
    cos_2Phi = jnp.cos(2.0 * Phi)
    sin_3Phi = jnp.sin(3.0 * Phi)
    cos_3Phi = jnp.cos(3.0 * Phi)
    sin_4Phi = jnp.sin(4.0 * Phi)
    cos_4Phi = jnp.cos(4.0 * Phi)

    return (
        sin_Phi
        + 0.5 * (eps2 * sin_2Phi - eps1 * cos_2Phi)
        - (1.0 / 8) * (
            5 * eps2**2 * sin_Phi
            - 3 * eps2**2 * sin_3Phi
            - 2 * eps2 * eps1 * jnp.cos(Phi)
            + 6 * eps2 * eps1 * cos_3Phi
            + 3 * eps1**2 * sin_Phi
            + 3 * eps1**2 * sin_3Phi
        )
        - (1.0 / 12) * (
            5 * eps2**3 * sin_2Phi
            + 3 * eps1**2 * eps2 * sin_2Phi
            - 6 * eps1 * eps2**2 * cos_2Phi
            - 4 * eps1**3 * cos_2Phi
            - 4 * eps2**3 * sin_4Phi
            + 12 * eps1**2 * eps2 * sin_4Phi
            + 12 * eps1 * eps2**2 * cos_4Phi
            - 4 * eps1**3 * cos_4Phi
        )
    )


# Scalar versions for computing Drep and Drepp via jax.grad.
# jax.grad operates on scalar-output functions, so we define the Roemer
# delay at a single orbital phase and vmap over the TOA axis.

def _roemer_scalar(Phi, eps1, eps2):
    """Roemer delay / a1 at a single Phi (scalar)."""
    return _ell1_roemer_da1(Phi, eps1, eps2)

_droemer_dPhi_scalar = jax.grad(_roemer_scalar, argnums=0)
_d2roemer_dPhi2_scalar = jax.grad(_droemer_dPhi_scalar, argnums=0)

# vmap over the TOA axis (first arg = Phi), broadcasting eps1/eps2.
_droemer_dPhi = jax.vmap(_droemer_dPhi_scalar, in_axes=(0, 0, 0))
_d2roemer_dPhi2 = jax.vmap(_d2roemer_dPhi2_scalar, in_axes=(0, 0, 0))


class BinaryELL1(DelayComponent):
    """ELL1 binary delay model for low-eccentricity orbits.

    Uses rectangular eccentricity (EPS1, EPS2) and TASC epoch.
    Shapiro delay parameterization controlled by ``shapiro_mode``:

    - ``"standard"``: Uses ``SINI`` and ``M2`` directly.
    - ``"none"``: No Shapiro delay.
    """

    pb_name: str = eqx.field(static=True, default="PB")
    tasc_name: str = eqx.field(static=True, default="TASC")
    a1_name: str = eqx.field(static=True, default="A1")
    eps1_name: str = eqx.field(static=True, default="EPS1")
    eps2_name: str = eqx.field(static=True, default="EPS2")

    # Optional secular derivatives
    pbdot_name: Optional[str] = eqx.field(static=True, default=None)
    a1dot_name: Optional[str] = eqx.field(static=True, default=None)
    eps1dot_name: Optional[str] = eqx.field(static=True, default=None)
    eps2dot_name: Optional[str] = eqx.field(static=True, default=None)
    xpbdot_name: Optional[str] = eqx.field(static=True, default=None)

    # Shapiro delay parameters
    shapiro_mode: str = eqx.field(static=True, default="standard")
    m2_name: Optional[str] = eqx.field(static=True, default=None)
    sini_name: Optional[str] = eqx.field(static=True, default=None)

    # ELL1H parameters
    h3_name: Optional[str] = eqx.field(static=True, default=None)
    h4_name: Optional[str] = eqx.field(static=True, default=None)
    stigma_name: Optional[str] = eqx.field(static=True, default=None)
    nharms: int = eqx.field(static=True, default=7)

    # ELL1k parameters
    omdot_name: Optional[str] = eqx.field(static=True, default=None)
    lnedot_name: Optional[str] = eqx.field(static=True, default=None)

    def _get_sini_m2(self, params: ParameterVector):
        """Compute sin(i) and M2 based on Shapiro parameterization."""
        if self.shapiro_mode == "standard":
            sini = params.param_value(self.sini_name) if self.sini_name else 0.0
            m2 = params.param_value(self.m2_name) if self.m2_name else 0.0
        elif self.shapiro_mode == "h3stigma":
            h3 = params.param_value(self.h3_name)
            stigma = params.param_value(self.stigma_name)
            sini = 2.0 * stigma / (1.0 + stigma ** 2)
            m2 = h3 / (stigma ** 3 * TSUN)
        elif self.shapiro_mode == "h3h4":
            h3 = params.param_value(self.h3_name)
            h4 = params.param_value(self.h4_name)
            stigma = h4 / h3
            sini = 2.0 * stigma / (1.0 + stigma ** 2)
            m2 = h3 / (stigma ** 3 * TSUN)
        else:
            sini = 0.0
            m2 = 0.0
        return sini, m2

    def _compute_eps(self, params, ttasc_s):
        """Compute time-dependent eps1, eps2."""
        eps1_0 = params.param_value(self.eps1_name)
        eps2_0 = params.param_value(self.eps2_name)

        if self.omdot_name is not None and self.lnedot_name is not None:
            # ELL1k model: Susobhanan et al. 2018, Eq. 15
            omdot_rad_per_s = params.param_value(self.omdot_name)  # rad/s (bridge converts)
            lnedot = params.param_value(self.lnedot_name)  # 1/yr
            lnedot_per_s = lnedot / (365.25 * 86400.0)
            dt = ttasc_s
            growth = 1.0 + lnedot_per_s * dt
            cos_omdot_dt = jnp.cos(omdot_rad_per_s * dt)
            sin_omdot_dt = jnp.sin(omdot_rad_per_s * dt)
            eps1 = growth * (eps1_0 * cos_omdot_dt + eps2_0 * sin_omdot_dt)
            eps2 = growth * (eps2_0 * cos_omdot_dt - eps1_0 * sin_omdot_dt)
        else:
            eps1dot = params.param_value(self.eps1dot_name) if self.eps1dot_name else 0.0
            eps2dot = params.param_value(self.eps2dot_name) if self.eps2dot_name else 0.0
            eps1 = eps1_0 + eps1dot * ttasc_s
            eps2 = eps2_0 + eps2dot * ttasc_s

        return eps1, eps2

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute ELL1 binary delay."""
        # --- Extract parameters ---
        pb_d = params.param_value(self.pb_name)
        tasc_int, tasc_frac = params.epoch_value(self.tasc_name)
        a1_ls = params.param_value(self.a1_name)

        pbdot = params.param_value(self.pbdot_name) if self.pbdot_name else 0.0
        a1dot = params.param_value(self.a1dot_name) if self.a1dot_name else 0.0

        sini, m2 = self._get_sini_m2(params)

        # --- Time since TASC ---
        ttasc_s = compute_tt0(toa_data.tdb_int, toa_data.tdb_frac, tasc_int, tasc_frac)

        # --- Time-dependent orbital elements ---
        a1 = a1_ls + a1dot * ttasc_s  # light-seconds = seconds
        eps1, eps2 = self._compute_eps(params, ttasc_s)

        # --- Orbital phase (precision-preserving via int/frac day split) ---
        pb_prime_s = pb_d * SECS_PER_DAY + pbdot * ttasc_s
        Phi = compute_orbital_phase(
            toa_data.tdb_int, toa_data.tdb_frac, tasc_int, tasc_frac,
            pb_d, pbdot,
        )

        # --- ELL1 Roemer delay (O(e^3) expansion) ---
        Dre = a1 * _ell1_roemer_da1(Phi, eps1, eps2)

        # --- Drep and Drepp via JAX autodiff of the Roemer delay ---
        Drep = a1 * _droemer_dPhi(Phi, eps1, eps2)
        Drepp = a1 * _d2roemer_dPhi2(Phi, eps1, eps2)

        # --- Inverse timing (simpler than DD: no e*sinE/(1-e*cosE) term) ---
        nhat = 2.0 * jnp.pi / pb_prime_s
        delay_inverse = Dre * (
            1.0 - nhat * Drep
            + (nhat * Drep) ** 2
            + 0.5 * nhat ** 2 * Dre * Drepp
        )

        # --- Shapiro delay ---
        if self.shapiro_mode == "none":
            delay_shapiro = 0.0
        else:
            TM2 = m2 * TSUN
            delay_shapiro = -2.0 * TM2 * jnp.log(1.0 - sini * jnp.sin(Phi))

        return delay_inverse + delay_shapiro


class BinaryELL1H(BinaryELL1):
    """ELL1 model with H3/STIGMA or H3/H4 Shapiro delay (ELL1H).

    Uses harmonic decomposition from Freire & Wex (2010).
    """

    shapiro_mode: str = eqx.field(static=True, default="h3stigma")


class BinaryELL1k(BinaryELL1):
    """ELL1 model with OMDOT/LNEDOT for short-period binaries (ELL1k).

    Susobhanan et al. (2018) — uses OMDOT and LNEDOT instead of EPS1DOT/EPS2DOT.
    """
    pass
