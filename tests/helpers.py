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


def make_binary_params(param_names, param_values, component, epoch_int_values=None):
    """ParameterVector preset for binary model tests."""
    return make_params(
        param_names, param_values,
        components=component,
        epoch_int_values=epoch_int_values or {},
    )


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
    names = ["F0"]; values = [f0]
    components = ["Spindown"]; units = ["Hz"]

    if f1 is not None:
        names += ["F1"]; values += [f1]
        components += ["Spindown"]; units += ["Hz/s"]
    if f2 is not None:
        names += ["F2"]; values += [f2]
        components += ["Spindown"]; units += ["Hz/s"]

    names += ["PEPOCH"]; values += [pepoch_frac]
    components += ["Spindown"]; units += ["day"]

    return make_params(names, values, units=tuple(units),
                       components=tuple(components),
                       epoch_int_values={"PEPOCH": pepoch_int})


def make_dispersion_dm_params(dm=15.0, dm1=None, dm2=None,
                              dmepoch_int=59000.0, dmepoch_frac=0.0):
    """ParameterVector preset for DispersionDM component tests."""
    names = ["DM"]; values = [dm]
    components = ["DispersionDM"]; units = ["pc cm^-3"]

    if dm1 is not None:
        names += ["DM1"]; values += [dm1]
        components += ["DispersionDM"]; units += ["pc cm^-3/yr"]
    if dm2 is not None:
        names += ["DM2"]; values += [dm2]
        components += ["DispersionDM"]; units += ["pc cm^-3/yr"]

    names += ["DMEPOCH"]; values += [dmepoch_frac]
    components += ["DispersionDM"]; units += ["day"]

    return make_params(names, values, units=tuple(units),
                       components=tuple(components),
                       epoch_int_values={"DMEPOCH": dmepoch_int})


def make_dmx_params(dmx_values, dmxr1_mjds, dmxr2_mjds, frozen_dmx=None):
    """ParameterVector preset for DispersionDMX bin parameters."""
    n = len(dmx_values)
    names = []
    values = []
    units = []
    components = []
    epoch_int_values = {}
    frozen_mask = []

    for i in range(n):
        idx = f"{i + 1:04d}"
        names.append(f"DMX_{idx}")
        values.append(dmx_values[i])
        units.append("pc cm^-3")
        components.append("DispersionDMX")
        frozen_mask.append(False if frozen_dmx is None else frozen_dmx[i])

        names.append(f"DMXR1_{idx}")
        r1_int = float(int(dmxr1_mjds[i]))
        r1_frac = dmxr1_mjds[i] - r1_int
        values.append(r1_frac)
        units.append("day")
        components.append("DispersionDMX")
        epoch_int_values[f"DMXR1_{idx}"] = r1_int
        frozen_mask.append(True)

        names.append(f"DMXR2_{idx}")
        r2_int = float(int(dmxr2_mjds[i]))
        r2_frac = dmxr2_mjds[i] - r2_int
        values.append(r2_frac)
        units.append("day")
        components.append("DispersionDMX")
        epoch_int_values[f"DMXR2_{idx}"] = r2_int
        frozen_mask.append(True)

    return make_params(
        names, values,
        units=tuple(units),
        components=tuple(components),
        epoch_int_values=epoch_int_values,
        frozen_mask=tuple(frozen_mask),
    )


def make_noise_params(names, values, frozen=None):
    """ParameterVector preset for noise component tests (frozen by default)."""
    if frozen is None:
        frozen = [True] * len(names)
    return make_params(names, values, frozen_mask=tuple(frozen),
                       components="noise")


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
    components=None,
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
    components : str or tuple of str, optional
        Ignored (kept for backward compatibility of call sites).
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
