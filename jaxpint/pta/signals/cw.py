"""Continuous gravitational wave signal model.

Provides antenna pattern functions, the CW timing delay (Earth + pulsar
term), and the :class:`CWInjector` adapter for the ``SignalInjector``
protocol.  Ported from Discovery's ``deterministic.py``.

References
----------
.. [1] Ellis, Siemens & Creighton (2012), "Optimal strategies for
   continuous gravitational wave detection in pulsar timing arrays",
   ApJ 756, 175.  Eqs. 1--3 (antenna patterns F+, Fx).
.. [2] Sesana & Vecchio (2010), "Measuring the parameters of massive
   black hole binary systems with pulsar timing array observations of
   gravitational waves", PRD 81, 104008.  Eq. 5 (CW timing residual).
.. [3] Ellis (2013), "A Bayesian analysis pipeline for continuous GW
   sources in the PTA band", CQG 30, 224004.  Eq. 4 (phase-averaging
   decomposition of Earth + pulsar terms).
.. [4] Detweiler (1979), "Pulsar timing measurements and the search for
   gravitational waves", ApJ 234, 1100.  Eq. 5 (strain-to-residual
   scaling alpha = h / (2*pi*f)).
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.types import TOAData, ParameterVector
from jaxpint.pta.likelihood import SignalInjector

# Speed of light (m/s) and kpc → metres conversion
_C: float = 299_792_458.0
_KPC_TO_M: float = 3.085_677_581e19

# Reference epoch: MJD of J2000.0, in seconds
_TREF: float = 86400.0 * 51544.5


# ---------------------------------------------------------------------------
# Antenna patterns
# ---------------------------------------------------------------------------


def fplus_fcross(
    pos: Float[Array, "3"],
    gwtheta: Float[Array, ""],
    gwphi: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute F+ and Fx antenna pattern response for a single pulsar.

    Implements Eqs. 1--3 of Ellis, Siemens & Creighton (2012) [1]_.

    Parameters
    ----------
    pos : (3,) array
        Unit vector pointing to the pulsar.
    gwtheta : scalar
        GW source colatitude (radians, ``pi/2 - dec``).
    gwphi : scalar
        GW source right ascension (radians).

    Returns
    -------
    fplus, fcross : scalars
        Antenna pattern coefficients.

    References
    ----------
    .. [1] Ellis, Siemens & Creighton (2012), ApJ 756, 175.
    """
    x, y, z = pos[0], pos[1], pos[2]

    sin_phi = jnp.sin(gwphi)
    cos_phi = jnp.cos(gwphi)
    sin_theta = jnp.sin(gwtheta)
    cos_theta = jnp.cos(gwtheta)

    m_dot_pos = sin_phi * x - cos_phi * y
    n_dot_pos = (
        -cos_theta * cos_phi * x
        - cos_theta * sin_phi * y
        + sin_theta * z
    )
    omhat_dot_pos = (
        -sin_theta * cos_phi * x
        - sin_theta * sin_phi * y
        - cos_theta * z
    )

    denom = 1.0 + omhat_dot_pos

    fplus = 0.5 * (m_dot_pos**2 - n_dot_pos**2) / denom
    fcross = (m_dot_pos * n_dot_pos) / denom
    return fplus, fcross


# ---------------------------------------------------------------------------
# CW timing delay
# ---------------------------------------------------------------------------


def cw_delay(
    toa_data: TOAData,
    pos: Float[Array, "3"],
    pulsar_dist: Float[Array, ""],
    global_params,
    prefix: str = "cw0_",
) -> Float[Array, " n_toas"]:
    """CW-induced timing delay for one pulsar (Earth + pulsar term).

    Implements the timing residual from Sesana & Vecchio (2010) [1]_ Eq. 5,
    using the phase-averaging decomposition of Ellis (2013) [2]_ Eq. 4.
    The strain-to-residual scaling ``alpha = h / (2*pi*f)`` follows from
    Detweiler (1979) [3]_ Eq. 5.

    The pulsar-term phase depends on pulsar distance, which is what makes
    the Fisher matrix informative for distance constraints.

    Parameters
    ----------
    toa_data : TOAData
        Pulse time-of-arrival data (uses TDB timestamps).
    pos : (3,) array
        Unit vector pointing to the pulsar.
    pulsar_dist : scalar
        Pulsar distance in kpc.
    global_params : GlobalParams
        Shared PTA parameters (accessed by prefixed name).
    prefix : str
        Naming prefix for this CW source in *global_params*.

    Returns
    -------
    delay : (n_toas,) array
        CW timing residual in seconds.

    References
    ----------
    .. [1] Sesana & Vecchio (2010), PRD 81, 104008.
    .. [2] Ellis (2013), CQG 30, 224004.
    .. [3] Detweiler (1979), ApJ 234, 1100.
    """
    # Extract CW source parameters via named access
    h0 = 10.0 ** global_params.param_value(f"{prefix}log10_h")
    f0 = 10.0 ** global_params.param_value(f"{prefix}log10_fgw")
    gwtheta = jnp.arccos(global_params.param_value(f"{prefix}cos_gwtheta"))
    gwphi = global_params.param_value(f"{prefix}gwphi")
    inc = jnp.arccos(global_params.param_value(f"{prefix}cos_inc"))
    psi = global_params.param_value(f"{prefix}psi")
    phase0 = global_params.param_value(f"{prefix}phase0")

    # Antenna patterns
    fp, fc = fplus_fcross(pos, gwtheta, gwphi)

    # TOA times in seconds (float64)
    toas_s = (
        toa_data.tdb_int.astype(jnp.float64) * 86400.0
        + toa_data.tdb_frac * 86400.0
    )

    # Earth-term phase
    phase_earth = phase0 + 2.0 * jnp.pi * f0 * (toas_s - _TREF)

    # GW propagation direction (towards the source)
    sin_theta = jnp.sin(gwtheta)
    cos_theta = jnp.cos(gwtheta)
    omhat = jnp.array(
        [
            -sin_theta * jnp.cos(gwphi),
            -sin_theta * jnp.sin(gwphi),
            -cos_theta,
        ]
    )

    # Pulsar-term phase offset from light travel time
    cos_mu = jnp.dot(omhat, pos)
    dist_m = pulsar_dist * _KPC_TO_M
    phase_pulsar = phase_earth - (
        2.0 * jnp.pi * f0 * dist_m / _C * (1.0 + cos_mu)
    )

    # Amplitude scaling
    alpha = h0 / (2.0 * jnp.pi * f0)

    # Phase averaging (Earth + pulsar terms combined)
    phi_avg = 0.5 * (phase_earth + phase_pulsar)
    phi_diff = 0.5 * (phase_earth - phase_pulsar)

    delta_sin = 2.0 * jnp.cos(phi_avg) * jnp.sin(phi_diff)
    delta_cos = -2.0 * jnp.sin(phi_avg) * jnp.sin(phi_diff)

    # Polarisation tensor components
    cos_inc = jnp.cos(inc)
    At = -(1.0 + cos_inc**2) * delta_sin
    Bt = 2.0 * cos_inc * delta_cos

    cos2psi = jnp.cos(2.0 * psi)
    sin2psi = jnp.sin(2.0 * psi)
    rplus = alpha * (-At * cos2psi + Bt * sin2psi)
    rcross = alpha * (At * sin2psi + Bt * cos2psi)

    return -fp * rplus - fc * rcross


# ---------------------------------------------------------------------------
# CW injector
# ---------------------------------------------------------------------------

CW_PARAM_DEFAULTS: dict[str, float] = {
    "log10_h": -14.0,  # log10 strain amplitude
    "cos_gwtheta": 0.0,  # cos(GW source colatitude)
    "gwphi": 0.0,  # GW source RA (rad)
    "log10_fgw": -8.0,  # log10 GW frequency (Hz)
    "cos_inc": 0.0,  # cos(orbital inclination)
    "psi": 0.0,  # polarisation angle (rad)
    "phase0": 0.0,  # Earth-term orbital phase (rad)
}


class CWInjector(SignalInjector):
    """Injects a single continuous gravitational wave source.

    Subclasses :class:`~jaxpint.pta.likelihood.SignalInjector`.  Uses a naming *prefix*
    (e.g. ``'cw0_'``, ``'cw1_'``) so that multiple CW sources can coexist
    in the same :class:`~jaxpint.pta.params.GlobalParams`.

    Parameters
    ----------
    pulsar_positions : (n_psr, 3) array
        Unit vectors pointing to each pulsar.
    dist_param_name : str
        Name of the distance parameter in each pulsar's
        :class:`~jaxpint.types.ParameterVector` (default ``'PX'``).
    prefix : str
        Naming prefix for this source in :class:`GlobalParams`.
    initial_values : dict, optional
        Override default initial values.  Keys must be in
        :data:`CW_PARAM_DEFAULTS`.
    """

    param_defaults = CW_PARAM_DEFAULTS

    def __init__(
        self,
        pulsar_positions: Float[Array, "n_psr 3"],
        dist_param_name: str = "PX",
        prefix: str = "cw0_",
        initial_values: Optional[dict[str, float]] = None,
    ):
        self.positions = pulsar_positions
        self.dist_param = dist_param_name
        self.prefix = prefix

        self.param_spec: dict[str, float] = dict(CW_PARAM_DEFAULTS)
        if initial_values is not None:
            unknown = set(initial_values) - set(CW_PARAM_DEFAULTS)
            if unknown:
                raise ValueError(
                    f"Unknown CW parameters: {unknown}. "
                    f"Valid parameters: {list(CW_PARAM_DEFAULTS.keys())}"
                )
            self.param_spec.update(initial_values)

    # -- SignalInjector protocol ------------------------------------------------

    def register_params(self, global_params):
        """Register CW source parameters into *global_params*."""
        names = [f"{self.prefix}{n}" for n in self.param_spec]
        values = list(self.param_spec.values())
        return global_params.add_params(names, values)

    def delay(self, p, toa_data, pulsar_params, global_params):
        """Compute CW delay for pulsar *p*."""
        return cw_delay(
            toa_data,
            self.positions[p],
            pulsar_params.param_value(self.dist_param),
            global_params,
            prefix=self.prefix,
        )

    # covariance() inherited from SignalInjector — returns None (CW is deterministic)
