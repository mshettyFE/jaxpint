"""Continuous gravitational wave signal model.

Provides antenna pattern functions, the CW timing delay (Earth + pulsar
term), and the :class:`CWInjector` adapter for the ``SignalInjector``
protocol.  Ported from Discovery's ``deterministic.py``.

References
----------
.. [cw_mod_esc12] Ellis, Siemens & Creighton (2012), "Optimal strategies for
   continuous gravitational wave detection in pulsar timing arrays",
   ApJ 756, 175.  Eqs. 1--3 (antenna patterns F+, Fx).
.. [cw_mod_sv10] Sesana & Vecchio (2010), "Measuring the parameters of massive
   black hole binary systems with pulsar timing array observations of
   gravitational waves", PRD 81, 104008.  Eq. 5 (CW timing residual).
.. [cw_mod_e13] Ellis (2013), "A Bayesian analysis pipeline for continuous GW
   sources in the PTA band", CQG 30, 224004.  Eq. 4 (phase-averaging
   decomposition of Earth + pulsar terms).
.. [cw_mod_d79] Detweiler (1979), "Pulsar timing measurements and the search for
   gravitational waves", ApJ 234, 1100.  Eq. 5 (strain-to-residual
   scaling alpha = h / (2*pi*f)).
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.types import TOAData, ParameterVector
from jaxpint.pta.likelihood import SignalInjector

# Speed of light (m/s) and kpc → metres conversion
_C: float = 299_792_458.0
_KPC_TO_M: float = 3.085_677_581e19
_MPC_TO_M: float = 1.0e3 * _KPC_TO_M  # Mpc → metres
# GM_sun / c^3 in seconds (chirp mass in geometric units; matches constants.TSUN)
_TSUN: float = 4.925_490_948e-6

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

    Implements Eqs. 4--10 of Ellis, Siemens & Creighton (2012) [cw_esc12]_.

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
    fplus : scalar
        Plus-polarization antenna pattern coefficient.
    fcross : scalar
        Cross-polarization antenna pattern coefficient.

    References
    ----------
    .. [cw_esc12] Ellis, Siemens & Creighton (2012), ApJ 756, 175.
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
    earth_term_only: bool = False,
) -> Float[Array, " n_toas"]:
    """CW-induced timing delay for one pulsar (Earth + pulsar term).

    Implements the timing residual from Sesana & Vecchio (2010) [cw_sv10]_ Eq. 5,
    using the phase-averaging decomposition of Ellis (2013) [cw_e13]_ Eq. 4.
    The strain-to-residual scaling ``alpha = h / (2*pi*f)`` follows from
    Detweiler (1979) [cw_d79]_ Eq. 5.

    The pulsar-term phase depends on pulsar distance, which is what makes
    the Fisher matrix informative for distance constraints.

    Parameters
    ----------
    toa_data : TOAData
        Pulse time-of-arrival data (uses TDB timestamps).
    pos : (3,) array
        Unit vector pointing to the pulsar.
    pulsar_dist : scalar
        Pulsar parallax in mas (types.py convention). Converted internally
        to physical distance in kpc for the Ellis+2012 pulsar-term phase.
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
    .. [cw_sv10] Sesana & Vecchio (2010), PRD 81, 104008.
    .. [cw_e13] Ellis (2013), CQG 30, 224004.
    .. [cw_d79] Detweiler (1979), ApJ 234, 1100.
    """
    # Extract CW source parameters and delegate to cw_delay_from_array
    cw_params = jnp.array([
        global_params.param_value(f"{prefix}{name}")
        for name in CW_PARAM_DEFAULTS
    ])
    return cw_delay_from_array(
        toa_data, pos, pulsar_dist, cw_params, earth_term_only=earth_term_only
    )


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


def log10_strain_from_binary(
    log10_mc: Float[Array, ""],
    log10_dist: Float[Array, ""],
    log10_fgw: Float[Array, ""],
) -> Float[Array, ""]:
    r"""``log10`` GW strain ``h0`` of a circular SMBH binary (Earth-term amplitude).

    Reparameterizes the CW strain in terms of the physical source properties,
    so callers can sweep / fit chirp mass and luminosity distance instead of the
    raw ``log10_h``.  The result drops straight into the CW ``log10_h`` slot;
    :func:`cw_delay_from_array` then applies the Detweiler (1979) strain-to-residual
    scaling ``alpha = h0 / (2*pi*f)`` internally.

    .. math::
        h_0 = \frac{2\,(G\mathcal{M}/c^3)^{5/3}\,c\,(\pi f)^{2/3}}{D_L}

    i.e. ``h0 ∝ M_c^{5/3} f^{2/3} / D_L``.  Note this is the *strain*; the induced
    timing-residual amplitude carries an extra ``1/(2*pi*f)`` (applied in the
    delay), giving the familiar ``∝ M_c^{5/3} f^{-1/3} / D_L`` residual scaling.

    Parameters
    ----------
    log10_mc : scalar
        ``log10`` chirp mass in solar masses.
    log10_dist : scalar
        ``log10`` luminosity distance in Mpc.
    log10_fgw : scalar
        ``log10`` GW frequency in Hz.

    Returns
    -------
    scalar
        ``log10(h0)`` (dimensionless strain), suitable for the ``log10_h``
        parameter of :class:`CWInjector`.

    References
    ----------
    .. [cw_strain_thorne87] Thorne (1987), in *300 Years of Gravitation*;
       see also Ellis, Siemens & Creighton (2012), ApJ 756, 175.
    """
    mc_sec = (10.0 ** log10_mc) * _TSUN          # G M_c / c^3 in seconds
    dist_m = (10.0 ** log10_dist) * _MPC_TO_M    # luminosity distance in metres
    fgw = 10.0 ** log10_fgw                      # GW frequency in Hz
    h0 = (
        2.0 * _C * mc_sec ** (5.0 / 3.0)
        * (jnp.pi * fgw) ** (2.0 / 3.0)
        / dist_m
    )
    return jnp.log10(h0)


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
        Name of the parallax parameter in each pulsar's
        :class:`~jaxpint.types.ParameterVector` (default ``'PX'``, in mas).
        The pulsar-term phase is computed from distance
        ``L_kpc = 1 / PX_mas`` internally (Ellis+2012).
    prefix : str
        Naming prefix for this source in :class:`GlobalParams`.
    initial_values : dict, optional
        Override default initial values.  Keys must be in
        ``CW_PARAM_DEFAULTS``.
    """

    param_defaults = CW_PARAM_DEFAULTS

    def __init__(
        self,
        pulsar_positions: Float[Array, "n_psr 3"],
        dist_param_name: str = "PX",
        prefix: str = "cw0_",
        initial_values: Optional[dict[str, float]] = None,
        earth_term_only: bool = False,
    ):
        self.positions = pulsar_positions
        self.dist_param = dist_param_name
        self.prefix = prefix
        self.earth_term_only = earth_term_only

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
        """Register CW source parameters into *global_params*.

        Parameters
        ----------
        global_params : GlobalParams
            Mutable accumulator of shared PTA parameters.

        Returns
        -------
        GlobalParams
            Updated copy with this CW source's parameters appended.
        """
        names = [f"{self.prefix}{n}" for n in self.param_spec]
        values = list(self.param_spec.values())
        return global_params.add_params(names, values)

    def delay(self, p, toa_data, pulsar_params, global_params):
        """Compute CW delay for pulsar *p*.

        Parameters
        ----------
        p : int
            Pulsar index within the PTA.
        toa_data : TOAData
            Pulse time-of-arrival data for pulsar *p*.
        pulsar_params : ParameterVector
            Timing and noise parameters for pulsar *p*.
        global_params : GlobalParams
            Shared PTA parameters containing this source's CW values.

        Returns
        -------
        (n_toas,) array
            CW timing residual in seconds.
        """
        return cw_delay(
            toa_data,
            self.positions[p],
            pulsar_params.param_value(self.dist_param),
            global_params,
            prefix=self.prefix,
            earth_term_only=self.earth_term_only,
        )

    # covariance() inherited from SignalInjector — returns None (CW is deterministic)


# ---------------------------------------------------------------------------
# Vectorized CW delay (vmappable over sources)
# ---------------------------------------------------------------------------

# Canonical parameter order matching CW_PARAM_DEFAULTS keys
_CW_PARAM_NAMES: tuple[str, ...] = tuple(CW_PARAM_DEFAULTS.keys())
_N_CW_PARAMS: int = len(_CW_PARAM_NAMES)


def cw_delay_from_array(
    toa_data: TOAData,
    pos: Float[Array, "3"],
    pulsar_dist: Float[Array, ""],
    cw_params: Float[Array, " 7"],
    earth_term_only: bool = False,
) -> Float[Array, " n_toas"]:
    """CW timing delay using a flat parameter array (vmappable over sources).

    Identical physics to :func:`cw_delay` but takes CW parameters as a
    positional ``(7,)`` array instead of named lookups into
    :class:`~jaxpint.pta.params.GlobalParams`.

    Parameter order (matching :data:`CW_PARAM_DEFAULTS` key order):
        0: log10_h, 1: cos_gwtheta, 2: gwphi, 3: log10_fgw,
        4: cos_inc, 5: psi, 6: phase0

    Parameters
    ----------
    toa_data : TOAData
        Pulse time-of-arrival data (uses TDB timestamps).
    pos : (3,) array
        Unit vector pointing to the pulsar.
    pulsar_dist : scalar
        Pulsar parallax in mas (types.py convention). Converted internally
        to physical distance in kpc for the Ellis+2012 pulsar-term phase.
        Unused when ``earth_term_only`` is True.
    cw_params : (7,) array
        Flat CW parameter vector in canonical order.
    earth_term_only : bool
        If True, keep only the Earth-term sinusoid and drop the pulsar term.
        The full residual uses ``delta_sin = sin(phase_earth) - sin(phase_pulsar)``
        and ``delta_cos = cos(phase_earth) - cos(phase_pulsar)`` (written here via
        the sum-to-product identity); dropping the pulsar term leaves
        ``delta_sin = sin(phase_earth)``, ``delta_cos = cos(phase_earth)``.  This
        removes the dependence on ``pulsar_dist`` (whose ~10% uncertainty makes the
        pulsar-term phase unconstrained at PTA frequencies), and is the standard
        simplification for upper-limit / sensitivity sky maps.  Default False
        preserves the full Earth + pulsar term behavior.

    Returns
    -------
    delay : (n_toas,) array
        CW timing residual in seconds.
    """
    h0 = 10.0 ** cw_params[0]
    gwtheta = jnp.arccos(cw_params[1])
    gwphi = cw_params[2]
    f0 = 10.0 ** cw_params[3]
    inc = jnp.arccos(cw_params[4])
    psi = cw_params[5]
    phase0 = cw_params[6]

    fp, fc = fplus_fcross(pos, gwtheta, gwphi)

    toas_s = (
        toa_data.tdb_int.astype(jnp.float64) * 86400.0
        + toa_data.tdb_frac * 86400.0
    )

    # Main contribution of pulsar distance is the phase as measured at earth and pulsar
    phase_earth = phase0 + 2.0 * jnp.pi * f0 * (toas_s - _TREF)

    alpha = h0 / (2.0 * jnp.pi * f0)

    if earth_term_only:
        # Drop the pulsar term: delta_sin/delta_cos with the pulsar-term
        # sinusoid removed (see docstring). No dependence on pulsar_dist.
        delta_sin = jnp.sin(phase_earth)
        delta_cos = jnp.cos(phase_earth)
    else:
        sin_theta = jnp.sin(gwtheta)
        cos_theta = jnp.cos(gwtheta)
        omhat = jnp.array([
            -sin_theta * jnp.cos(gwphi),
            -sin_theta * jnp.sin(gwphi),
            -cos_theta,
        ])

        cos_mu = jnp.dot(omhat, pos)
        # pulsar_dist is parallax in mas (types.py convention).
        # Ellis+2012 (arXiv:1204.4218) writes the pulsar-term phase in terms of
        # the physical distance L; convert mas -> kpc via L_kpc = 1 / PX_mas.
        dist_m = (1.0 / pulsar_dist) * _KPC_TO_M
        # Pulsar phase gets delayed by light vacuum time
        phase_pulsar = phase_earth - (
            2.0 * jnp.pi * f0 * dist_m / _C * (1.0 + cos_mu)
        )

        # Sum to product formula  to convert sin(t_{p}) - sin(t_{e}) = sin((t_{p}+t_{e})/2)cos((t_{p}-t_{e})/2)
        # Similar for cosine differences as well
        phi_avg = 0.5 * (phase_earth + phase_pulsar)
        phi_diff = 0.5 * (phase_earth - phase_pulsar)

        delta_sin = 2.0 * jnp.cos(phi_avg) * jnp.sin(phi_diff)
        delta_cos = -2.0 * jnp.sin(phi_avg) * jnp.sin(phi_diff)

    cos_inc = jnp.cos(inc)
    At = -(1.0 + cos_inc**2) * delta_sin
    Bt = 2.0 * cos_inc * delta_cos

    cos2psi = jnp.cos(2.0 * psi)
    sin2psi = jnp.sin(2.0 * psi)
    rplus = alpha * (-At * cos2psi + Bt * sin2psi)
    rcross = alpha * (At * sin2psi + Bt * cos2psi)

    return -fp * rplus - fc * rcross


def sum_cw_delays(
    toa_data: TOAData,
    pos: Float[Array, "3"],
    pulsar_dist: Float[Array, ""],
    cw_params_stack: Float[Array, "n_cw 7"],
    earth_term_only: bool = False,
) -> Float[Array, " n_toas"]:
    """Sum CW delays from multiple sources via vmap.

    Parameters
    ----------
    toa_data : TOAData
        Pulse time-of-arrival data.
    pos : (3,) array
        Pulsar unit vector.
    pulsar_dist : scalar
        Pulsar parallax in mas (types.py convention). Converted internally
        to physical distance in kpc for the Ellis+2012 pulsar-term phase.
    cw_params_stack : (n_cw, 7) array
        Stacked CW parameters for all sources.

    Returns
    -------
    delay : (n_toas,) array
        Total CW delay summed over all sources.
    """
    per_source = jax.vmap(
        lambda p: cw_delay_from_array(
            toa_data, pos, pulsar_dist, p, earth_term_only=earth_term_only
        )
    )(cw_params_stack)  # (n_cw, n_toas)
    return jnp.sum(per_source, axis=0)


# ---------------------------------------------------------------------------
# Vectorized CW injector (single trace for M sources)
# ---------------------------------------------------------------------------


class CWInjectorStack(SignalInjector):
    """Vectorized injector for multiple CW sources.

    Replaces M separate :class:`CWInjector` instances with a single object
    that uses ``jax.vmap`` over sources.  JIT compilation time is O(1) in
    the number of sources instead of O(M).

    Implements the :class:`~jaxpint.pta.likelihood.SignalInjector` protocol,
    so it works as a drop-in replacement in :class:`PTAConfig`.

    Parameters
    ----------
    pulsar_positions : (n_psr, 3) array
        Unit vectors pointing to each pulsar.
    n_sources : int
        Number of CW sources.
    dist_param_name : str
        Name of the parallax parameter in each pulsar's
        :class:`~jaxpint.types.ParameterVector` (default ``'PX'``, in mas).
        The pulsar-term phase is computed from distance
        ``L_kpc = 1 / PX_mas`` internally (Ellis+2012).
    initial_values : dict, optional
        Override default initial values (applied to all sources).
        Keys must be in :data:`CW_PARAM_DEFAULTS`.
    per_source_values : list of dict, optional
        Per-source overrides (length must equal *n_sources*).
        Takes precedence over *initial_values* for each source.

    Examples
    --------
    >>> # Before: M separate injectors (M JIT traces)
    >>> injectors = [CWInjector(positions, prefix=f"cw{i}_") for i in range(M)]
    >>>
    >>> # After: one stacked injector (1 JIT trace)
    >>> injector = CWInjectorStack(positions, n_sources=M)
    >>> config = PTAConfig(..., signal_injectors=(injector,))
    """

    def __init__(
        self,
        pulsar_positions: Float[Array, "n_psr 3"],
        n_sources: int,
        dist_param_name: str = "PX",
        initial_values: Optional[dict[str, float]] = None,
        per_source_values: Optional[list[dict[str, float]]] = None,
        earth_term_only: bool = False,
    ):
        self.positions = pulsar_positions
        self.dist_param = dist_param_name
        self.n_sources = n_sources
        self.prefixes = tuple(f"cw{i}_" for i in range(n_sources))
        self.earth_term_only = earth_term_only

        # Build per-source param specs
        self.param_specs: list[dict[str, float]] = []
        for m in range(n_sources):
            spec = dict(CW_PARAM_DEFAULTS)
            if initial_values is not None:
                unknown = set(initial_values) - set(CW_PARAM_DEFAULTS)
                if unknown:
                    raise ValueError(
                        f"Unknown CW parameters: {unknown}. "
                        f"Valid: {list(CW_PARAM_DEFAULTS.keys())}"
                    )
                spec.update(initial_values)
            if per_source_values is not None:
                if len(per_source_values) != n_sources:
                    raise ValueError(
                        f"per_source_values length {len(per_source_values)} "
                        f"!= n_sources {n_sources}"
                    )
                unknown = set(per_source_values[m]) - set(CW_PARAM_DEFAULTS)
                if unknown:
                    raise ValueError(
                        f"Unknown CW parameters in source {m}: {unknown}"
                    )
                spec.update(per_source_values[m])
            self.param_specs.append(spec)

        # _param_indices will be set during register_params
        self._param_indices: Optional[jnp.ndarray] = None

    def register_params(self, global_params):
        """Register all CW sources' parameters into *global_params*.

        Parameters
        ----------
        global_params : GlobalParams
            Mutable accumulator of shared PTA parameters.

        Returns
        -------
        GlobalParams
            Updated copy with all CW sources' parameters appended.
        """
        indices = []
        for m in range(self.n_sources):
            prefix = self.prefixes[m]
            spec = self.param_specs[m]
            names = [f"{prefix}{n}" for n in _CW_PARAM_NAMES]
            values = [spec[n] for n in _CW_PARAM_NAMES]
            offset = global_params.n_params
            global_params = global_params.add_params(names, values)
            indices.append(list(range(offset, offset + _N_CW_PARAMS)))

        self._param_indices = jnp.array(indices, dtype=jnp.int32)
        return global_params

    def delay(self, p, toa_data, pulsar_params, global_params):
        """Compute total CW delay for pulsar *p* (vmapped over sources).

        Parameters
        ----------
        p : int
            Pulsar index within the PTA.
        toa_data : TOAData
            Pulse time-of-arrival data for pulsar *p*.
        pulsar_params : ParameterVector
            Timing and noise parameters for pulsar *p*.
        global_params : GlobalParams
            Shared PTA parameters containing all CW source values.

        Returns
        -------
        (n_toas,) array
            Total CW timing residual summed over all sources, in seconds.
        """
        cw_stack = global_params.values[self._param_indices]  # (n_sources, 7)
        return sum_cw_delays(
            toa_data,
            self.positions[p],
            pulsar_params.param_value(self.dist_param),
            cw_stack,
            earth_term_only=self.earth_term_only,
        )

    # covariance() inherited from SignalInjector — returns None (CW is deterministic)
