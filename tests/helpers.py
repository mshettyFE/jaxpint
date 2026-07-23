"""Shared test helpers for constructing TOAData and ParameterVector."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import build_fourier_basis as _build_fourier_basis


# ---------------------------------------------------------------------------
# Presets (thin wrappers with common defaults for specific test domains)
# ---------------------------------------------------------------------------

def make_binary_toa_data(t_mjd, *, tzr_tdb_int=54000.0):
    """TOAData preset for binary model tests (with TZR fields)."""
    return make_toa_data(
        t_mjd=t_mjd,
        tzr_tdb_int=jnp.array(tzr_tdb_int),
        tzr_tdb_frac=jnp.array(0.5),
        tzr_freq=jnp.array(jnp.inf),
        tzr_ssb_obs_pos=jnp.zeros(3),
    )


def make_binary_params(param_names, param_values, epoch_int_values=None):
    """ParameterVector preset for binary model tests."""
    return make_params(
        param_names, param_values,
        epoch_int_values=epoch_int_values or {},
    )


def ddk_earth_obs_pos_km(t_mjd):
    """Earth-orbit SSB observatory position (km), circular J2000 approximation.

    Shared by the DDK binary tests (``test_binary_common.py`` /
    ``test_binary_ddk.py``), which both need a realistic ``ssb_obs_pos`` to
    exercise the Kopeikin (K96) terms.
    """
    t_mjd = np.asarray(t_mjd)
    au_km = 149597870.7
    phase = 2 * np.pi * (t_mjd - 51544.5) / 365.25
    return np.column_stack([
        au_km * np.cos(phase),
        au_km * np.sin(phase),
        np.zeros_like(phase),
    ])


def make_gbt_toa_data(
    n_toas=5, *, tdb_int=59000.0, tdb_frac=None, freq=1400.0,
    t_mjd=None,
    tzr_tdb_int=None, tzr_tdb_frac=None, tzr_freq=None,
):
    """TOAData preset with GBT observatory and no planet positions."""
    return make_toa_data(
        n_toas, t_mjd=t_mjd,
        tdb_int=tdb_int, tdb_frac=tdb_frac, freq=freq,
        obs_names=("GBT",), planet_positions=None,
        tzr_tdb_int=tzr_tdb_int, tzr_tdb_frac=tzr_tdb_frac,
        tzr_freq=tzr_freq,
    )


def make_spindown_params(f0=200.0, f1=None, f2=None,
                         pepoch_int=59000.0, pepoch_frac=0.0):
    """ParameterVector preset for Spindown component tests."""
    names = ["F0"]; values = [f0]; units = ["Hz"]

    if f1 is not None:
        names += ["F1"]; values += [f1]; units += ["Hz/s"]
    if f2 is not None:
        names += ["F2"]; values += [f2]; units += ["Hz/s"]

    names += ["PEPOCH"]; values += [pepoch_frac]; units += ["day"]

    return make_params(names, values, units=tuple(units),
                       epoch_int_values={"PEPOCH": pepoch_int})


def make_dispersion_dm_params(dm=15.0, dm1=None, dm2=None,
                              dmepoch_int=59000.0, dmepoch_frac=0.0):
    """ParameterVector preset for DispersionDM component tests."""
    names = ["DM"]; values = [dm]; units = ["pc cm^-3"]

    if dm1 is not None:
        names += ["DM1"]; values += [dm1]; units += ["pc cm^-3/yr"]
    if dm2 is not None:
        names += ["DM2"]; values += [dm2]; units += ["pc cm^-3/yr"]

    names += ["DMEPOCH"]; values += [dmepoch_frac]; units += ["day"]

    return make_params(names, values, units=tuple(units),
                       epoch_int_values={"DMEPOCH": dmepoch_int})


def make_dmx_params(dmx_values, dmxr1_mjds, dmxr2_mjds, frozen_dmx=None):
    """ParameterVector preset for DispersionDMX bin parameters."""
    n = len(dmx_values)
    names = []
    values = []
    units = []
    epoch_int_values = {}
    frozen_mask = []

    for i in range(n):
        idx = f"{i + 1:04d}"
        names.append(f"DMX_{idx}")
        values.append(dmx_values[i])
        units.append("pc cm^-3")
        frozen_mask.append(False if frozen_dmx is None else frozen_dmx[i])

        names.append(f"DMXR1_{idx}")
        r1_int = float(int(dmxr1_mjds[i]))
        r1_frac = dmxr1_mjds[i] - r1_int
        values.append(r1_frac)
        units.append("day")
        epoch_int_values[f"DMXR1_{idx}"] = r1_int
        frozen_mask.append(True)

        names.append(f"DMXR2_{idx}")
        r2_int = float(int(dmxr2_mjds[i]))
        r2_frac = dmxr2_mjds[i] - r2_int
        values.append(r2_frac)
        units.append("day")
        epoch_int_values[f"DMXR2_{idx}"] = r2_int
        frozen_mask.append(True)

    return make_params(
        names, values,
        units=tuple(units),
        epoch_int_values=epoch_int_values,
        frozen_mask=tuple(frozen_mask),
    )


def make_noise_params(names, values, frozen=None):
    """ParameterVector preset for noise component tests (frozen by default)."""
    if frozen is None:
        frozen = [True] * len(names)
    return make_params(names, values, frozen_mask=tuple(frozen))


def make_params_with_frozen_names(names, values, frozen_names=(), units=None):
    """ParameterVector from names with a frozen_names set instead of mask."""
    frozen_mask = tuple(n in frozen_names for n in names)
    if units is None:
        units = tuple("s" for _ in names)
    return make_params(names, values, frozen_mask=frozen_mask, units=units)


def make_toa_data(
    n_toas=5,
    *,
    t_mjd=None,
    tdb_int=59000.0,
    tdb_frac=None,
    error=1e-6,
    freq=1400.0,
    flag_masks=None,
    planet_positions=None,
    dm_values=None,
    dm_errors=None,
    tropo_alt=None,
    tropo_alt_valid=None,
    obs_geodetic_lat=None,
    obs_height_km=None,
    obs_names=("fake",),
    tzr_tdb_int=None,
    tzr_tdb_frac=None,
    tzr_freq=None,
    tzr_ssb_obs_pos=None,
    tzr_obs_sun_pos=None,
):
    """Build a minimal TOAData for tests.

    Two modes:
    - Pass ``t_mjd`` (array of MJD values) to split into int/frac automatically.
    - Or pass ``n_toas`` with optional ``tdb_int``/``tdb_frac`` for linspace-style.
    """
    if t_mjd is not None:
        t_np = np.asarray(t_mjd)
        n_toas = len(t_np)
        tdb_int_arr = jnp.array(np.floor(t_np))
        tdb_frac_arr = jnp.array(t_np - np.floor(t_np))
    else:
        if tdb_frac is None:
            tdb_frac_arr = jnp.linspace(0.1, 0.9, n_toas)
        else:
            tdb_frac_arr = jnp.broadcast_to(jnp.asarray(tdb_frac), (n_toas,))
        tdb_int_arr = jnp.full(n_toas, tdb_int)

    error_arr = jnp.broadcast_to(jnp.asarray(error, dtype=jnp.float64), (n_toas,))
    freq_arr = jnp.broadcast_to(jnp.asarray(freq), (n_toas,))

    if flag_masks is None:
        flag_masks = {}
    else:
        flag_masks = {
            k: jnp.asarray(v, dtype=jnp.bool_) for k, v in flag_masks.items()
        }

    if planet_positions is None:
        planet_positions = {}

    return TOAData(
        mjd_int=tdb_int_arr,
        mjd_frac=tdb_frac_arr,
        tdb_int=tdb_int_arr,
        tdb_frac=tdb_frac_arr,
        error=error_arr,
        freq=freq_arr,
        delta_pulse_number=jnp.zeros(n_toas),
        ssb_obs_pos=jnp.zeros((n_toas, 3)),
        ssb_obs_vel=jnp.zeros((n_toas, 3)),
        obs_sun_pos=jnp.zeros((n_toas, 3)),
        obs_indices=jnp.zeros(n_toas, dtype=jnp.int32),
        flag_masks=flag_masks,
        planet_positions=planet_positions,
        dm_values=dm_values,
        dm_errors=dm_errors,
        tropo_alt=tropo_alt,
        tropo_alt_valid=tropo_alt_valid,
        obs_geodetic_lat=obs_geodetic_lat,
        obs_height_km=obs_height_km,
        n_toas=int(n_toas),
        obs_names=tuple(str(s) for s in obs_names),
        tzr_tdb_int=float(tzr_tdb_int) if tzr_tdb_int is not None else None,
        tzr_tdb_frac=float(tzr_tdb_frac) if tzr_tdb_frac is not None else None,
        tzr_freq=float(tzr_freq) if tzr_freq is not None else None,
        tzr_ssb_obs_pos=tzr_ssb_obs_pos,
        tzr_obs_sun_pos=tzr_obs_sun_pos,
        # This synthetic data has zero solar-system geometry (ssb_obs_pos = 0), so
        # the barycentric arrival times equal TDB exactly; TDB here IS the
        # enterprise-convention choice, not an approximation to it.
        basis_seconds=tdb_int_arr * 86400.0 + tdb_frac_arr * 86400.0,
        basis_coord="tdb",
    )


def make_simple_pulsar(
    n_toas,
    f0,
    f1,
    *,
    pepoch_int=59000.0,
    tdb_int=None,
    error=1e-6,
    seed=0,
):
    """Build a simple spindown pulsar with white noise (for PTA tests).

    Returns ``(toa_data, timing_model, noise_model, params)``.

    Parameters
    ----------
    n_toas : int
    f0, f1 : float
        Spin frequency and spin-down rate.
    pepoch_int : float
        Integer MJD of PEPOCH (defaults to 59000).
    tdb_int : float, optional
        Integer MJD for TOA TDB times. Defaults to ``pepoch_int``.
    error : float
        Per-TOA error in seconds.
    seed : int
        RNG seed for the random TOA fractional spacings.
    """
    # Deferred imports keep this helper usable from tests that don't
    # otherwise pull in the model/noise/phase machinery.
    from jaxpint.model import TimingModel
    from jaxpint.noise import NoiseModel
    from jaxpint.noise.white import ScaleToaError
    from jaxpint.phase.spin import Spindown

    rng = np.random.default_rng(seed)
    if tdb_int is None:
        tdb_int = pepoch_int

    tdb_frac = jnp.array(np.sort(rng.uniform(0.0, 1.0, n_toas)))
    efac_mask = jnp.ones(n_toas, dtype=jnp.bool_)
    equad_mask = jnp.ones(n_toas, dtype=jnp.bool_)

    toa_data = make_toa_data(
        n_toas,
        tdb_int=tdb_int,
        tdb_frac=tdb_frac,
        error=error,
        flag_masks={"EFAC1": efac_mask, "EQUAD1": equad_mask},
        tzr_tdb_int=pepoch_int,
        tzr_tdb_frac=0.5,
        tzr_freq=jnp.inf,
        tzr_ssb_obs_pos=jnp.zeros(3),
        tzr_obs_sun_pos=jnp.zeros(3),
    )

    spindown = Spindown(spin_param_names=("F0", "F1"), pepoch_name="PEPOCH")
    timing_model = TimingModel(
        delay_components=(),
        phase_components=(spindown,),
        phoff_name=None,
    )

    white_noise = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
    noise_model = NoiseModel(white_noise=white_noise, correlated=())

    params = make_params(
        names=("F0", "F1", "PEPOCH", "EFAC1", "EQUAD1"),
        values=(f0, f1, 0.0, 1.0, 0.0),
        frozen_mask=(False, False, True, True, True),
        epoch_int_values={"PEPOCH": pepoch_int},
    )

    return toa_data, timing_model, noise_model, params


def make_fourier_basis(n_toas, n_freqs, T):
    """Build a Fourier basis on a uniform time grid.

    Thin wrapper around :func:`jaxpint.utils.build_fourier_basis` that
    returns ``(F, freqs, df, t)`` with ``F``, ``freqs``, ``df`` as JAX
    arrays and ``t`` as a numpy time vector.
    """
    t = np.linspace(0.0, T, n_toas)
    F, freqs, df = _build_fourier_basis(t, n_freqs, T)
    return jnp.asarray(F), jnp.asarray(freqs), jnp.asarray(df), t


def make_params(
    names,
    values,
    *,
    frozen_mask=None,
    units=None,
    epoch_int_values=None,
):
    """Build a minimal ParameterVector for tests.

    Parameters
    ----------
    names : tuple of str
    values : sequence of float
    frozen_mask : tuple of bool, optional
        Defaults to all False.
    units : tuple of str, optional
        Defaults to all empty strings.
    epoch_int_values : dict, optional
    """
    names = tuple(names)
    n = len(names)

    if frozen_mask is None:
        frozen_mask = (False,) * n
    if units is None:
        units = ("",) * n
    if epoch_int_values is None:
        epoch_int_values = {}
    else:
        epoch_int_values = {k: float(v) for k, v in epoch_int_values.items()}

    return ParameterVector(
        values=jnp.array(values, dtype=jnp.float64),
        frozen_mask=tuple(frozen_mask),
        names=names,
        units=tuple(units),
        epoch_int_values=epoch_int_values,
    )


# ---------------------------------------------------------------------------
# End-to-end GLS whitening (shared by the PL-noise red/DM/chrom/SW tests)
# ---------------------------------------------------------------------------

def run_gls_whitening(
    correlated,
    *,
    param_names,
    param_values,
    param_units,
    freq=1400.0,
    obs_sun_pos=None,
    n_toas=200,
    seed=2024,
    delay_components=(),
    extra_free_params=(),
):
    """Simulate white + one correlated noise component, GLS-fit, and whiten.

    Shared driver for the power-law-noise end-to-end GLS tests (red / DM /
    chromatic / solar wind).  The caller builds the correlated ``component``
    (construction is noise-type-specific) and passes its extra parameter
    ``param_names``/``param_values``/``param_units``; the spindown +
    white-noise scaffolding, fake-TOA simulation (fixed ``seed``), fit, and
    whitening are identical across noise types.

    ``freq`` may be a scalar or a per-TOA array; ``obs_sun_pos`` (when given)
    replaces the TOA-data Sun positions, as the SW model needs.

    Returns ``(whitened_residuals, result)``.
    """
    import copy

    import jax

    from jaxpint.fitters import GLSFitter
    from jaxpint.model import TimingModel
    from jaxpint.noise import NoiseModel, ScaleToaError
    from jaxpint.phase.spin import Spindown
    from jaxpint.simulation import make_fake_toas

    T = 3.0 * 365.25 * 86400.0
    mask = np.ones(n_toas, dtype=bool)
    t_mjd = np.linspace(53000.0, 53000.0 + T / 86400.0, n_toas)
    toa_data = make_toa_data(
        t_mjd=t_mjd,
        error=1e-6,
        freq=freq,
        flag_masks={"EFAC1": mask},
        tzr_tdb_int=53000.0,
        tzr_tdb_frac=0.0,
        tzr_freq=1400.0,
        tzr_ssb_obs_pos=np.zeros(3),
    )
    if obs_sun_pos is not None:
        import equinox as eqx

        toa_data = eqx.tree_at(
            lambda t: t.obs_sun_pos, toa_data, jnp.asarray(obs_sun_pos)
        )

    white = ScaleToaError(efac_names=("EFAC1",), equad_names=())
    noise_model = NoiseModel(white_noise=white, correlated=(correlated,))

    spin = Spindown(spin_param_names=("F0", "F1"))
    timing_model = TimingModel(
        delay_components=tuple(delay_components), phase_components=(spin,)
    )

    # Extra free timing params (e.g. DM) fit alongside F0/F1; each is a
    # ``(name, value, unit)`` triple.  ``param_names`` are the frozen
    # correlated-noise params.
    free_names = tuple(p[0] for p in extra_free_params)
    free_values = [p[1] for p in extra_free_params]
    free_units = tuple(p[2] for p in extra_free_params)

    params = make_params(
        ("F0", "F1", "PEPOCH", *free_names, "EFAC1", *param_names),
        [100.0, -1e-15, 0.0, *free_values, 1.0, *param_values],
        units=("Hz", "Hz/s", "day", *free_units, "", *param_units),
        frozen_mask=(
            (False, False, True)
            + (False,) * len(free_names)
            + (True,)
            + (True,) * len(param_names)
        ),
        epoch_int_values={"PEPOCH": 53000.0},
    )

    key = jax.random.PRNGKey(seed)
    fake_toa_data = make_fake_toas(
        timing_model, toa_data, params, key,
        noise_components=[white, correlated],
    )

    fit_params = copy.deepcopy(params)
    fitter = GLSFitter(
        timing_model, fake_toa_data, fit_params, noise_model=noise_model,
    )
    result = fitter.fit_toas(maxiter=3)
    sigma = noise_model.scaled_sigma(fake_toa_data, result.params)

    # Whiten: subtract the correlated-noise realization, divide by scaled sigma.
    if noise_model.has_correlated and result.noise_realizations is not None:
        _, U, _ = noise_model.covariance(fake_toa_data, result.params)
        rc = U @ result.noise_realizations
        whitened = (result.residuals - rc) / sigma
    else:
        whitened = result.residuals / sigma

    return whitened, result


def assert_covariance_matches_pint(
    jax_fit,
    pint_fitter,
    *,
    uncert_rtol: float,
    corr_atol: float,
    exclude: tuple = (),
):
    """Assert a JaxPINT fit's parameter covariance matches a PINT fitter's.

    Aligns by parameter name (PINT's labeled ``parameter_covariance_matrix``
    with the Offset row/column dropped), converts JaxPINT uncertainties to
    PINT's native units via astropy (e.g. rad -> deg for angles), and
    compares:

    - per-parameter uncertainties (sqrt-diag) at ``uncert_rtol``
    - the correlation matrix (dimensionless, unit-invariant) at ``corr_atol``

    ``exclude`` names parameters left out of the comparison, e.g. members
    of a near-degenerate pair where JaxPINT's SVD-cutoff convention (zero
    variance along dropped directions) and PINT's Cholesky (huge marginal
    variance along the same directions) legitimately disagree.
    """
    import astropy.units as u

    pcov = pint_fitter.parameter_covariance_matrix
    plabels = [lbl for lbl, _ in pcov.labels[0]]
    pmat = np.asarray(pcov.matrix, dtype=np.float64)

    params = jax_fit.params
    free_idx = np.asarray(params.free_indices_array())
    names = list(params.free_names())
    keep = [k for k, n in enumerate(names) if n not in exclude]
    assert keep, "exclude list removed every parameter"

    jcov = np.asarray(jax_fit.covariance_matrix)[np.ix_(keep, keep)]

    # JaxPINT internal units -> PINT native units (dimensionless_angles
    # handles rad -> deg and the rad/s -> mas/yr proper-motion cases).
    # When the unit strings already agree (including PINT-only units such
    # as "ls" for A1, which astropy cannot parse), no conversion is needed.
    def _scale(k):
        jax_unit_str = params.units[free_idx[k]]
        pint_unit = getattr(pint_fitter.model, names[k]).units
        if pint_unit is None or jax_unit_str == str(pint_unit):
            return 1.0
        return float(
            (1.0 * u.Unit(jax_unit_str))
            .to(pint_unit, equivalencies=u.dimensionless_angles())
            .value
        )

    scales = np.array([_scale(k) for k in keep])

    idx = [plabels.index(names[k]) for k in keep]
    psub = pmat[np.ix_(idx, idx)]

    jerr = np.sqrt(np.diag(jcov)) * scales
    perr = np.sqrt(np.diag(psub))
    np.testing.assert_allclose(
        jerr, perr, rtol=uncert_rtol,
        err_msg=f"uncertainty mismatch; params={[names[k] for k in keep]}",
    )

    jcorr = jcov / np.outer(np.sqrt(np.diag(jcov)), np.sqrt(np.diag(jcov)))
    pcorr = psub / np.outer(perr, perr)
    np.testing.assert_allclose(
        jcorr, pcorr, atol=corr_atol, err_msg="correlation-matrix mismatch"
    )
