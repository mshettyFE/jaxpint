"""Shared orbital mechanics for JaxPINT binary models.

Pure functions used by BT, DD, and ELL1 model families.  All inputs and
outputs are dimensionless JAX arrays in the unit conventions documented
on each function.

Reference
---------
PINT ``binary_generic.py``, ``binary_orbits.py``.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from jaxpint.binary.kepler import solve_kepler
from jaxpint.constants import SECS_PER_DAY, TSUN
from jaxpint.types.dual_float import DualFloat
from jaxpint.types import ParameterVector


# ---------------------------------------------------------------------------
# Time since epoch  (precision-preserving)
# ---------------------------------------------------------------------------


def compute_tt0(
    tdb: DualFloat,
    epoch: DualFloat,
    delay: Optional[Float[Array, " n_toas"]] = None,
) -> Float[Array, " n_toas"]:
    """Time from epoch to each TOA in seconds, using DualFloat precision.

    Parameters
    ----------
    tdb : DualFloat
        TDB MJD of each TOA (int day + fractional day).
    epoch : DualFloat
        Reference epoch (T0 or TASC) as int day + fractional day.
    delay : array, optional
        Accumulated delay from prior components (seconds).  Subtracted
        from TDB to form the barycentric time used by the binary model,
        matching PINT's ``barycentric_time = tdbld - acc_delay``.

    Returns
    -------
    array
        ``(tdb - delay - epoch)`` in seconds, shape ``(n_toas,)``.
    """
    dt = tdb - epoch
    tt0 = dt.total * SECS_PER_DAY
    if delay is not None:
        tt0 = tt0 - delay
    return tt0


# ---------------------------------------------------------------------------
# Orbit phase (OrbitPB parameterization)
# ---------------------------------------------------------------------------


def compute_orbits_pb(
    tt0_s: Float[Array, " n_toas"],
    pb_d: ArrayLike,
    pbdot: ArrayLike = 0.0,
    xpbdot: ArrayLike = 0.0,
) -> Float[Array, " n_toas"]:
    """Orbital phase (number of orbits since T0) using the PB parameterization.

    Parameters
    ----------
    tt0_s : array
        Time since T0 in seconds.
    pb_d : float
        Binary period PB in days.
    pbdot : float
        Time derivative of PB (dimensionless, s/s).
    xpbdot : float
        Excess PBDOT (dimensionless, s/s).

    Returns
    -------
    array
        Number of orbits (dimensionless).  Fractional part gives orbital phase.
    """
    pb_s = pb_d * SECS_PER_DAY
    ratio = tt0_s / pb_s
    return ratio - 0.5 * (pbdot + xpbdot) * ratio**2


def compute_mean_anomaly(orbits: Float[Array, " n_toas"]) -> Float[Array, " n_toas"]:
    """Mean anomaly in radians from orbital phase count.

    Parameters
    ----------
    orbits : array
        Number of orbits (from ``compute_orbits_pb``).

    Returns
    -------
    array
        Mean anomaly in [0, 2*pi), in radians.
    """
    phase = orbits - jnp.floor(orbits)
    return 2.0 * jnp.pi * phase


def compute_orbital_phase(
    tdb: DualFloat,
    epoch: DualFloat,
    pb_d: ArrayLike,
    pbdot: ArrayLike = 0.0,
    xpbdot: ArrayLike = 0.0,
    delay: Optional[Float[Array, " n_toas"]] = None,
) -> Float[Array, " n_toas"]:
    """Orbital phase in [0, 2*pi) using DualFloat precision.

    Avoids the precision loss that occurs when a large orbit count
    (e.g. 167.083...) is computed as a single float64 and the fractional
    part is extracted.  Instead, computes ``(dt_int mod PB)`` in day-space
    where the integer day difference is exact, then adds the fractional
    day contribution.

    Parameters
    ----------
    tdb : DualFloat
        TDB MJD of each TOA (int day + fractional day).
    epoch : DualFloat
        Reference epoch (T0 or TASC) as int day + fractional day.
    pb_d : float
        Binary period PB in days.
    pbdot : float
        Time derivative of PB (dimensionless, s/s).
    xpbdot : float
        Excess PBDOT (dimensionless, s/s).
    delay : array, optional
        Accumulated delay from prior components (seconds).  Subtracted
        from TDB to form the barycentric time, matching PINT.

    Returns
    -------
    array
        Orbital phase (mean anomaly) in [0, 2*pi), in radians.
    """
    dt = tdb - epoch
    dt_int_days = dt.int  # exact integer days
    dt_frac_days = dt.frac  # fractional day, full precision

    # Subtract accumulated delay (convert seconds → days) from fractional part.
    if delay is not None:
        dt_frac_days = dt_frac_days - delay / SECS_PER_DAY

    # Integer orbits from integer days, remainder in days.
    # Because dt_int_days is an exact integer, the subtraction
    # dt_int_days - n_orbits*pb_d preserves ~14 digits.
    n_orbits = jnp.floor(dt_int_days / pb_d)
    rem_int_days = dt_int_days - n_orbits * pb_d

    # Total sub-orbit remainder (both parts are O(pb_d), full precision).
    rem_days = rem_int_days + dt_frac_days
    extra = jnp.floor(rem_days / pb_d)
    rem_days = rem_days - extra * pb_d

    # Fractional orbit with full precision.
    frac_orbit = rem_days / pb_d

    # PBDOT/XPBDOT correction (small perturbation).
    # Computed from the full tt0/PB ratio — precision here is not critical
    # because the correction itself is tiny (order PBDOT * N_orbits^2).
    tt0_s = (dt_int_days + dt_frac_days) * SECS_PER_DAY
    pb_s = pb_d * SECS_PER_DAY
    ratio = tt0_s / pb_s
    pbdot_corr = -0.5 * (pbdot + xpbdot) * ratio**2

    frac_total = frac_orbit + pbdot_corr
    frac_total = frac_total - jnp.floor(frac_total)

    return 2.0 * jnp.pi * frac_total


# ---------------------------------------------------------------------------
# Time-dependent orbital elements
# ---------------------------------------------------------------------------


def compute_ecc(
    ecc0: ArrayLike, edot: ArrayLike, tt0_s: Float[Array, " n_toas"]
) -> Float[Array, " n_toas"]:
    """Time-dependent eccentricity: ``ecc(t) = ECC + EDOT * tt0``.

    Parameters
    ----------
    ecc0 : float
        Reference eccentricity ECC (dimensionless).
    edot : float
        Time derivative EDOT (1/s).
    tt0_s : array
        Time since reference epoch in seconds.

    Returns
    -------
    Float[Array, " n_toas"]
        Eccentricity at each TOA (dimensionless).
    """
    return jnp.asarray(ecc0 + edot * tt0_s)


def compute_a1(
    a1_0: ArrayLike, a1dot: ArrayLike, tt0_s: Float[Array, " n_toas"]
) -> Float[Array, " n_toas"]:
    """Time-dependent projected semi-major axis: ``a1(t) = A1 + A1DOT * tt0``.

    Parameters
    ----------
    a1_0 : float
        Reference A1 in light-seconds.
    a1dot : float
        Time derivative A1DOT (ls/s).
    tt0_s : array
        Time since reference epoch in seconds.

    Returns
    -------
    Float[Array, " n_toas"]
        Projected semi-major axis at each TOA, in light-seconds.
    """
    return jnp.asarray(a1_0 + a1dot * tt0_s)


def compute_omega_bt(
    om_rad: ArrayLike,
    omdot_rad_per_s: ArrayLike,
    tt0_s: Float[Array, " n_toas"],
) -> Float[Array, " n_toas"]:
    """Longitude of periastron for BT model: ``omega = OM + OMDOT * tt0``.

    Parameters
    ----------
    om_rad : float
        Reference OM in radians (bridge converts from degrees).
    omdot_rad_per_s : float
        OMDOT in rad/s (bridge converts from deg/yr).
    tt0_s : array
        Time since T0 in seconds.

    Returns
    -------
    array
        omega in radians.
    """
    return jnp.asarray(om_rad + omdot_rad_per_s * tt0_s)


def _bt_delay_formula(
    a1: Float[Array, " n_toas"],
    ecc: Float[Array, " n_toas"],
    omega: Float[Array, " n_toas"],
    gamma: ArrayLike,
    pb_d: ArrayLike,
    pbdot: ArrayLike,
    tt0_s: Float[Array, " n_toas"],
    E: Float[Array, " n_toas"],
) -> Float[Array, " n_toas"]:
    """Core Blandford-Teukolsky binary delay formula.

    Shared by :class:`~jaxpint.binary.bt.BinaryBT` and
    :class:`~jaxpint.binary.bt_piecewise.BinaryBTPiecewise`. Inputs may be
    scalar or per-TOA; broadcasting handles both.

    Parameters
    ----------
    a1, ecc, omega : array
        Time-dependent (or constant, broadcast) projected semi-major axis
        in light-seconds, eccentricity, and longitude of periastron in
        radians.
    gamma : float
        GAMMA parameter (s); 0 disables the relativistic Einstein delay
        term.
    pb_d : float
        Orbital period in days.
    pbdot : float
        First time derivative of PB (s/s).
    tt0_s : array
        Time from periastron in seconds (per TOA).
    E : array
        Eccentric anomaly in radians (per TOA).

    Returns
    -------
    Float[Array, " n_toas"]
        Binary delay in seconds.
    """
    sinE = jnp.sin(E)
    cosE = jnp.cos(E)
    sin_omega = jnp.sin(omega)
    cos_omega = jnp.cos(omega)
    sqrt_1me2 = jnp.sqrt(1.0 - ecc**2)

    L1 = a1 * sin_omega * (cosE - ecc)
    L2 = (a1 * cos_omega * sqrt_1me2 + gamma) * sinE

    pb_s = (pb_d + pbdot * tt0_s / SECS_PER_DAY) * SECS_PER_DAY
    num = a1 * cos_omega * sqrt_1me2 * cosE - a1 * sin_omega * sinE
    den = 1.0 - ecc * cosE
    R = 1.0 - 2.0 * jnp.pi * num / (den * pb_s)

    return (L1 + L2) * R


# ---------------------------------------------------------------------------
# Eccentric and true anomaly
# ---------------------------------------------------------------------------


def compute_eccentric_anomaly(
    ecc: Float[Array, " n_toas"],
    mean_anomaly: Float[Array, " n_toas"],
) -> Float[Array, " n_toas"]:
    """Solve Kepler's equation for eccentric anomaly.

    Thin wrapper around :func:`solve_kepler`.

    Parameters
    ----------
    ecc : Float[Array, " n_toas"]
        Per-TOA eccentricity (dimensionless).
    mean_anomaly : Float[Array, " n_toas"]
        Per-TOA mean anomaly in radians.

    Returns
    -------
    Float[Array, " n_toas"]
        Eccentric anomaly at each TOA, in radians.
    """
    return solve_kepler(mean_anomaly, ecc)


def compute_true_anomaly(
    E: Float[Array, " n_toas"],
    ecc: Float[Array, " n_toas"],
    orbits: Float[Array, " n_toas"],
    mean_anomaly: Float[Array, " n_toas"],
) -> Float[Array, " n_toas"]:
    """True anomaly from eccentric anomaly, tracking full orbit count.

    Uses the standard formula::

        nu = 2 * arctan(sqrt((1+e)/(1-e)) * tan(E/2))

    then adjusts to track cumulative orbits (matching PINT convention).

    Parameters
    ----------
    E : array
        Eccentric anomaly in radians.
    ecc : array
        Eccentricity (may be time-dependent).
    orbits : array
        Orbital phase count (from ``compute_orbits_pb``).
    mean_anomaly : array
        Mean anomaly in radians.

    Returns
    -------
    array
        True anomaly in radians (cumulative, not folded to [0, 2*pi)).
    """
    nu_raw = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + ecc) * jnp.sin(E / 2.0),
        jnp.sqrt(1.0 - ecc) * jnp.cos(E / 2.0),
    )
    # Normalize to [0, 2*pi)
    nu_raw = jnp.where(nu_raw < 0.0, nu_raw + 2.0 * jnp.pi, nu_raw)
    # Add full orbit count to track cumulative phase (matches PINT's nu2)
    nu = 2.0 * jnp.pi * orbits + nu_raw - mean_anomaly
    return nu


# ---------------------------------------------------------------------------
# DD-specific omega
# ---------------------------------------------------------------------------


def compute_omega_dd(
    om_rad: ArrayLike,
    omdot_rad_per_s: ArrayLike,
    nu: Float[Array, " n_toas"],
    pb_d: ArrayLike,
    pbdot: ArrayLike = 0.0,
    tt0_s: Optional[Float[Array, " n_toas"]] = None,
) -> Float[Array, " n_toas"]:
    """Longitude of periastron for DD model: ``omega = OM + nu * k``.

    In the DD model, periastron advance is parameterized as advance per orbit
    ``k = OMDOT / n`` where ``n = 2*pi / PB_prime`` and PB_prime is the
    instantaneous binary period ``PB + PBDOT * tt0``.

    Parameters
    ----------
    om_rad : float
        Reference OM in radians.
    omdot_rad_per_s : float
        OMDOT in rad/s (bridge converts from deg/yr).
    nu : array
        True anomaly in radians (cumulative).
    pb_d : float
        Binary period PB in days.
    pbdot : float
        Time derivative of PB (s/s).
    tt0_s : array or None
        Time since T0 in seconds. Used to compute instantaneous period.

    Returns
    -------
    array
        omega in radians.
    """
    # Instantaneous period: PB + PBDOT * tt0 (matching PINT's self.pb())
    pb_s = pb_d * SECS_PER_DAY
    if tt0_s is not None:
        pb_prime_s = pb_s + pbdot * tt0_s
    else:
        pb_prime_s = pb_s
    n = 2.0 * jnp.pi / pb_prime_s  # mean motion (rad/s)
    k = omdot_rad_per_s / n  # advance of periastron per orbit (rad/rad)
    return jnp.asarray(om_rad + nu * k)


# ---------------------------------------------------------------------------
# DD inverse timing formula
# ---------------------------------------------------------------------------


def dd_inverse_timing(
    Dre: Float[Array, " n_toas"],
    Drep: Float[Array, " n_toas"],
    Drepp: Float[Array, " n_toas"],
    nhat: Float[Array, " n_toas"],
    ecc: Float[Array, " n_toas"],
    sinE: Float[Array, " n_toas"],
    cosE: Float[Array, " n_toas"],
) -> Float[Array, " n_toas"]:
    """DD inverse timing formula (Damour & Deruelle 1986, eq. [46]-[52]).

    Corrects the combined Roemer + Einstein delay from proper time to
    coordinate time, to second order.

    Parameters
    ----------
    Dre : array
        Combined Roemer + Einstein delay (seconds).
    Drep : array
        First derivative of Dre w.r.t. eccentric anomaly parameter u.
    Drepp : array
        Second derivative of Dre w.r.t. u.
    nhat : array
        ``n / (1 - e*cos(E))`` where ``n = 2*pi/PB``.
    ecc, sinE, cosE : array
        Eccentricity and trig functions of eccentric anomaly.

    Returns
    -------
    array
        Inverse timing delay in seconds.
    """
    brace = (
        1.0
        - nhat * Drep
        + (nhat * Drep) ** 2
        + 0.5 * nhat**2 * Dre * Drepp
        - 0.5 * ecc * sinE / (1.0 - ecc * cosE) * nhat**2 * Dre * Drep
    )
    return Dre * brace


# ---------------------------------------------------------------------------
# Shapiro delay helpers
# ---------------------------------------------------------------------------


def dd_shapiro_delay(
    ecc: Float[Array, " n_toas"],
    cosE: Float[Array, " n_toas"],
    sinE: Float[Array, " n_toas"],
    sin_omega: Float[Array, " n_toas"],
    cos_omega: Float[Array, " n_toas"],
    sini: float,
    m2_msun: float,
) -> Float[Array, " n_toas"]:
    """Shapiro delay for DD-family models.

    Parameters
    ----------
    ecc, cosE, sinE : array
        Eccentricity and trig functions of eccentric anomaly.
    sin_omega, cos_omega : array
        Trig functions of longitude of periastron.
    sini : float or array
        Sine of orbital inclination.
    m2_msun : float
        Companion mass in solar masses.

    Returns
    -------
    array
        Shapiro delay in seconds.
    """
    TM2 = m2_msun * TSUN
    arg = (
        1.0
        - ecc * cosE
        - sini * (sin_omega * (cosE - ecc) + jnp.sqrt(1.0 - ecc**2) * cos_omega * sinE)
    )
    return -2.0 * TM2 * jnp.log(arg)


def dd_aberration_delay(
    A0: ArrayLike,
    B0: ArrayLike,
    sin_omega: Float[Array, " n_toas"],
    cos_omega: Float[Array, " n_toas"],
    nu: Float[Array, " n_toas"],
    omega: Float[Array, " n_toas"],
    ecc: Float[Array, " n_toas"],
) -> Float[Array, " n_toas"]:
    """Aberration delay for DD-family models.

    Parameters
    ----------
    A0, B0 : float
        Aberration coefficients in seconds.
    sin_omega, cos_omega : array
        Trig functions of longitude of periastron.
    nu : array
        True anomaly in radians.
    omega : array
        Longitude of periastron in radians.
    ecc : array
        Eccentricity.

    Returns
    -------
    array
        Aberration delay in seconds.
    """
    omg_plus_nu = omega + nu
    return jnp.asarray(
        A0 * (jnp.sin(omg_plus_nu) + ecc * sin_omega)
        + B0 * (jnp.cos(omg_plus_nu) + ecc * cos_omega)
    )


# ---------------------------------------------------------------------------
# Shapiro delay parameterization dispatch
# ---------------------------------------------------------------------------


def ell1h_fourier_shapiro(h3, stigma, phi, nharms: int):
    """Freire & Wex (2010) Eq. 19 — approximate Shapiro delay for ELL1H.

    Sums Fourier harmonics k = 3..nharms of the orbital phase. The first
    two harmonics are absorbed into the ELL1 Roemer delay, so the series
    starts at k=3. Used by ``BinaryELL1`` when the par file specifies H3
    (and optionally NHARMS) but no STIGMA or H4.

    PINT reference: ``delayS3p_H3_STIGMA_approximate`` and
    ``fourier_component`` in
    ``pint.models.stand_alone_psr_binaries.ELL1H_model``.

    Parameters
    ----------
    h3 : scalar or array
        Shapiro amplitude (seconds).
    stigma : scalar or array
        Orthometric ratio. 0 for the H3-only case.
    phi : array
        Orbital phase (radians).
    nharms : int
        Maximum harmonic index (static). PINT's default is 7.

    Returns
    -------
    array
        Shapiro delay in seconds.
    """
    total = jnp.zeros_like(phi)
    for k in range(3, nharms + 1):
        if k % 2 == 0:
            pwr = (k + 2) // 2
            basis = jnp.cos(k * phi)
        else:
            pwr = (k + 1) // 2
            basis = jnp.sin(k * phi)
        coeff = ((-1) ** pwr) * (2.0 / k) * stigma ** (k - 3)
        total = total + coeff * basis
    return -2.0 * h3 * total


def get_sini_m2(
    params: ParameterVector,
    shapiro_mode: str,
    sini_name: str | None = None,
    m2_name: str | None = None,
    shapmax_name: str | None = None,
    h3_name: str | None = None,
    stigma_name: str | None = None,
    h4_name: str | None = None,
) -> tuple:
    """Compute sin(i) and companion mass M2 from the Shapiro parameterization.

    Supports four modes:

    - ``"standard"``: Uses ``SINI`` and ``M2`` directly.
    - ``"shapmax"``: Uses ``SHAPMAX = -ln(1 - sin(i))`` and ``M2``.
    - ``"h3stigma"``: Uses orthometric parameters ``H3`` and ``STIGMA``.
    - ``"h3h4"``: Uses ``H3`` and ``H4`` (derives ``STIGMA = H4/H3``).

    Any other mode (including ``"none"``) returns ``(0.0, 0.0)``.

    Parameters
    ----------
    params : ParameterVector
    shapiro_mode : str
    sini_name, m2_name, shapmax_name, h3_name, stigma_name, h4_name : str or None
        Parameter names for the relevant mode.

    Returns
    -------
    (sini, m2)
    """
    if shapiro_mode == "standard":
        sini = params.param_value_or(sini_name)
        m2 = params.param_value_or(m2_name)
    elif shapiro_mode == "shapmax":
        assert shapmax_name is not None
        shapmax = params.param_value(shapmax_name)
        sini = 1.0 - jnp.exp(-shapmax)
        m2 = params.param_value_or(m2_name)
    elif shapiro_mode == "h3stigma":
        assert h3_name is not None and stigma_name is not None
        h3 = params.param_value(h3_name)
        stigma = params.param_value(stigma_name)
        sini = 2.0 * stigma / (1.0 + stigma**2)
        m2 = h3 / (stigma**3 * TSUN)
    elif shapiro_mode == "h3h4":
        assert h3_name is not None and h4_name is not None
        h3 = params.param_value(h3_name)
        h4 = params.param_value(h4_name)
        stigma = h4 / h3
        sini = 2.0 * stigma / (1.0 + stigma**2)
        m2 = h3 / (stigma**3 * TSUN)
    else:
        sini = 0.0
        m2 = 0.0
    return sini, m2


# ---------------------------------------------------------------------------
# DD core delay computation (shared by DD, DDK, DDGR)
# ---------------------------------------------------------------------------


def dd_core_delay(
    E: Float[Array, " n_toas"],
    ecc: Float[Array, " n_toas"],
    omega: Float[Array, " n_toas"],
    nu: Float[Array, " n_toas"],
    a1: Float[Array, " n_toas"],
    tt0_s: Float[Array, " n_toas"],
    pb_d: ArrayLike,
    pbdot: ArrayLike,
    gamma: ArrayLike,
    dr: ArrayLike,
    dth: ArrayLike,
    A0: ArrayLike,
    B0: ArrayLike,
    sini,
    m2,
) -> Float[Array, " n_toas"]:
    """Compute the DD delay from pre-computed orbital intermediates.

    Computes the three DD delay terms (inverse timing, Shapiro, aberration)
    from the eccentric anomaly, longitude of periastron, and orbital
    elements.  Shared by BinaryDD, BinaryDDK, and BinaryDDGR.

    Parameters
    ----------
    E : array
        Eccentric anomaly in radians.
    ecc : array
        (Time-dependent) eccentricity.
    omega : array
        Longitude of periastron in radians.
    nu : array
        True anomaly in radians (cumulative).
    a1 : array
        (Time-dependent) projected semi-major axis in light-seconds.
    tt0_s : array
        Time since epoch in seconds.
    pb_d : float
        Binary period in days.
    pbdot : float
        Time derivative of PB (s/s).
    gamma, dr, dth : float
        DD-specific parameters.
    A0, B0 : float
        Aberration coefficients in seconds.
    sini : float or array
        Sine of orbital inclination.
    m2 : float
        Companion mass in solar masses.

    Returns
    -------
    array, shape (n_toas,)
        Total DD binary delay in seconds.
    """
    sinE = jnp.sin(E)
    cosE = jnp.cos(E)
    sin_omega = jnp.sin(omega)
    cos_omega = jnp.cos(omega)

    # DD-specific eccentricities
    er = ecc * (1.0 + dr)
    eTheta = ecc * (1.0 + dth)

    # DD intermediate quantities (D&D eqs. [46]-[47])
    alpha = a1 * sin_omega
    beta = a1 * jnp.sqrt(1.0 - eTheta**2) * cos_omega

    # Roemer + Einstein delay (Dre, eq. [48])
    Dre = alpha * (cosE - er) + (beta + gamma) * sinE

    # Dre derivatives w.r.t. u (eqs. [49]-[50])
    Drep = -alpha * sinE + (beta + gamma) * cosE
    Drepp = -alpha * cosE - (beta + gamma) * sinE

    # nhat (eq. [51]) — uses instantaneous period
    pb_prime_s = pb_d * SECS_PER_DAY + pbdot * tt0_s
    nhat = jnp.asarray(2.0 * jnp.pi / pb_prime_s / (1.0 - ecc * cosE))

    # 1. Inverse timing delay (eq. [52])
    delay_inverse = dd_inverse_timing(Dre, Drep, Drepp, nhat, ecc, sinE, cosE)

    # 2. Shapiro delay (eq. [26])
    delay_shapiro = dd_shapiro_delay(ecc, cosE, sinE, sin_omega, cos_omega, sini, m2)

    # 3. Aberration delay (eq. [27])
    delay_aberration = dd_aberration_delay(A0, B0, sin_omega, cos_omega, nu, omega, ecc)

    return delay_inverse + delay_shapiro + delay_aberration
