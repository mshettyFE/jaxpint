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

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.binary._param_decls import BINARY_CORE
from jaxpint.types import TOAData, ParameterVector
from jaxpint.constants import SECS_PER_DAY, TSUN
from jaxpint.binary.common import (
    compute_tt0,
    compute_orbital_phase,
    ell1h_fourier_shapiro,
    get_sini_m2,
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
        - (1.0 / 8)
        * (
            5 * eps2**2 * sin_Phi
            - 3 * eps2**2 * sin_3Phi
            - 2 * eps2 * eps1 * jnp.cos(Phi)
            + 6 * eps2 * eps1 * cos_3Phi
            + 3 * eps1**2 * sin_Phi
            + 3 * eps1**2 * sin_3Phi
        )
        - (1.0 / 12)
        * (
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
    - ``"h3stigma"``: Uses ``H3`` and ``STIGMA`` (derives sini, m2).
    - ``"h3h4"``: Uses ``H3`` and ``H4`` (derives ``STIGMA = H4/H3``).
    - ``"h3nharms"``: ELL1H with ``H3`` only — Freire-Wex 2010 Eq. 19
    - ``"none"``: No Shapiro delay.
    """

    PARAMS = (
        *BINARY_CORE,
        ParamDecl("TASC", kind="mjd"),
        ParamDecl("EPS1"),
        ParamDecl("EPS1DOT"),
        ParamDecl("EPS2"),
        ParamDecl("EPS2DOT"),
        ParamDecl("M2"),
        ParamDecl("SINI"),
        ParamDecl("H3"),
        ParamDecl("H4"),
        ParamDecl("STIGMA", aliases=("STIG", "VARSIGMA")),
        ParamDecl("NHARMS", kind="int"),
        ParamDecl("LNEDOT"),
    )

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

    def _compute_eps(self, params, ttasc_s):
        """Compute time-dependent eps1, eps2."""
        eps1_0 = params.param_value(self.eps1_name)
        eps2_0 = params.param_value(self.eps2_name)

        if self.omdot_name is not None and self.lnedot_name is not None:
            # ELL1k model: Susobhanan et al. 2018, Eq. 15
            omdot_rad_per_s = params.param_value(
                self.omdot_name
            )  # rad/s (bridge converts)
            lnedot = params.param_value(self.lnedot_name)  # 1/yr
            lnedot_per_s = lnedot / (365.25 * 86400.0)
            dt = ttasc_s
            growth = 1.0 + lnedot_per_s * dt
            cos_omdot_dt = jnp.cos(omdot_rad_per_s * dt)
            sin_omdot_dt = jnp.sin(omdot_rad_per_s * dt)
            eps1 = growth * (eps1_0 * cos_omdot_dt + eps2_0 * sin_omdot_dt)
            eps2 = growth * (eps2_0 * cos_omdot_dt - eps1_0 * sin_omdot_dt)
        else:
            eps1dot = params.param_value_or(self.eps1dot_name)
            eps2dot = params.param_value_or(self.eps2dot_name)
            eps1 = eps1_0 + eps1dot * ttasc_s
            eps2 = eps2_0 + eps2dot * ttasc_s

        return eps1, eps2

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute ELL1 binary delay for low-eccentricity orbits.

        Uses a third-order analytic expansion in rectangular eccentricity
        (EPS1, EPS2) instead of solving Kepler's equation, plus an
        optional Shapiro delay term.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times, etc.).
        params : ParameterVector
            Timing-model parameters containing ELL1 orbital elements
            (PB, TASC, A1, EPS1, EPS2) and optional Shapiro parameters.
        delay : array, shape (n_toas,)
            Accumulated signal delay in seconds, used to correct
            the time of arrival to emission time.

        Returns
        -------
        array, shape (n_toas,)
            Binary delay in seconds.
        """
        # --- Extract parameters ---
        pb_d = params.param_value(self.pb_name)
        tasc = params.epoch_dual(self.tasc_name)
        a1_ls = params.param_value(self.a1_name)

        pbdot = params.param_value_or(self.pbdot_name)
        a1dot = params.param_value_or(self.a1dot_name)

        sini, m2 = get_sini_m2(
            params,
            self.shapiro_mode,
            self.sini_name,
            self.m2_name,
            h3_name=self.h3_name,
            stigma_name=self.stigma_name,
            h4_name=self.h4_name,
        )

        # --- Time since TASC (corrected for accumulated delay) ---
        ttasc_s = compute_tt0(toa_data.tdb, tasc, delay=delay)

        # --- Time-dependent orbital elements ---
        a1 = a1_ls + a1dot * ttasc_s  # light-seconds = seconds
        eps1, eps2 = self._compute_eps(params, ttasc_s)

        # --- Orbital phase (precision-preserving via int/frac day split) ---
        pb_prime_s = pb_d * SECS_PER_DAY + pbdot * ttasc_s
        Phi = compute_orbital_phase(
            toa_data.tdb,
            tasc,
            pb_d,
            pbdot,
            delay=delay,
        )

        # --- ELL1 Roemer delay (O(e^3) expansion) ---
        Dre = a1 * _ell1_roemer_da1(Phi, eps1, eps2)

        # --- Drep and Drepp via JAX autodiff of the Roemer delay ---
        Drep = a1 * _droemer_dPhi(Phi, eps1, eps2)
        Drepp = a1 * _d2roemer_dPhi2(Phi, eps1, eps2)

        # --- Inverse timing (simpler than DD: no e*sinE/(1-e*cosE) term) ---
        nhat = 2.0 * jnp.pi / pb_prime_s
        delay_inverse = Dre * (
            1.0 - nhat * Drep + (nhat * Drep) ** 2 + 0.5 * nhat**2 * Dre * Drepp
        )

        # --- Shapiro delay ---
        if self.shapiro_mode == "none":
            delay_shapiro = 0.0
        elif self.shapiro_mode == "h3nharms":
            assert self.h3_name is not None
            h3 = params.param_value(self.h3_name)
            delay_shapiro = ell1h_fourier_shapiro(h3, 0.0, Phi, self.nharms)
        else:
            TM2 = m2 * TSUN
            delay_shapiro = -2.0 * TM2 * jnp.log(1.0 - sini * jnp.sin(Phi))

        return delay_inverse + delay_shapiro
