"""TOA simulation for JaxPINT.

Provides functions to adjust TOA timestamps so they encode a deterministic
timing model (zero residuals) and to apply arbitrary time delays to TOAs.

The ``zero_residuals`` function iteratively shifts each TOA by its time
residual until the model prediction matches the observation time, analogous
to :func:`pint.simulation.zero_residuals`.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent
from jaxpint.types.dual_float import DualFloat
from jaxpint.fitters import compute_time_residuals
from jaxpint.model import TimingModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.constants import SECS_PER_DAY


def apply_delay_to_toas(
    toa_data: TOAData,
    delays_seconds: Float[Array, " n_toas"],
) -> TOAData:
    """Return a new TOAData with *delays_seconds* added to MJD and TDB fields.

    Parameters
    ----------
    toa_data : TOAData
        Input TOA data (not modified).
    delays_seconds : (n_toas,)
        Time delays in seconds.  Positive values shift TOAs later.

    Returns
    -------
    TOAData
        Copy of *toa_data* with ``mjd_int/mjd_frac`` and
        ``tdb_int/tdb_frac`` updated.
    """
    delay_days = delays_seconds / SECS_PER_DAY

    new_mjd = DualFloat.from_days(toa_data.mjd_int, toa_data.mjd_frac + delay_days)
    new_tdb = DualFloat.from_days(toa_data.tdb_int, toa_data.tdb_frac + delay_days)

    return eqx.tree_at(
        lambda td: (td.mjd_int, td.mjd_frac, td.tdb_int, td.tdb_frac),
        toa_data,
        (new_mjd.int, new_mjd.frac, new_tdb.int, new_tdb.frac),
    )


def zero_residuals(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
    *,
    maxiter: int = 10,
    tolerance: float = 1e-9,
) -> TOAData:
    """Iteratively adjust TOA times until residuals are approximately zero.

    Each iteration computes time residuals and subtracts them from the
    TOA timestamps, converging in ~2-3 iterations.  After convergence
    the TOA timestamps encode the full deterministic timing model.

    Parameters
    ----------
    model : TimingModel
        JaxPINT timing model.
    toa_data : TOAData
        Input TOAs (not modified).
    params : ParameterVector
        Timing model parameters.
    maxiter : int
        Maximum number of iterations.
    tolerance : float
        Convergence threshold on ``max(|residual|)`` in seconds.
        Default is 1e-9 s (1 ns).

    Returns
    -------
    TOAData
        Adjusted TOAs with residuals < *tolerance*.

    Raises
    ------
    RuntimeError
        If convergence is not reached within *maxiter* iterations.
    """
    max_resid = float("inf")
    for i in range(maxiter):
        resids = compute_time_residuals(model, toa_data, params)
        max_resid = float(jnp.max(jnp.abs(resids)))
        if max_resid < tolerance:
            return toa_data
        toa_data = apply_delay_to_toas(toa_data, -resids)

    raise RuntimeError(
        f"zero_residuals did not converge after {maxiter} iterations "
        f"(max |residual| = {max_resid:.3e} s, tolerance = {tolerance:.3e} s)"
    )


def simulate_noise(
    toa_data: TOAData,
    params: ParameterVector,
    key: jax.Array,
    noise_components: Sequence[NoiseComponent],
) -> Float[Array, " n_toas"]:
    """Generate a combined noise realization from multiple noise sources.

    Parameters
    ----------
    toa_data : TOAData
        TOA data (used for uncertainties, flags, and array sizes).
    params : ParameterVector
        Timing model parameters (including noise parameter values).
    key : JAX PRNG key
        Random key; split internally for each component.
    noise_components : sequence of NoiseComponent
        Noise sources to sample from.

    Returns
    -------
    (n_toas,)
        Total noise delay in seconds.
    """
    delays = jnp.zeros(toa_data.n_toas)
    keys = jax.random.split(key, len(noise_components))
    for k, comp in zip(keys, noise_components):
        delays = delays + comp.generate(toa_data, params, k)
    return delays


def make_fake_toas(
    model: TimingModel,
    toa_data: TOAData,
    params: ParameterVector,
    key: jax.Array,
    noise_components: Sequence[NoiseComponent] = (),
) -> TOAData:
    """Create simulated TOAs: zero residuals, then optionally add noise.

    Parameters
    ----------
    model : TimingModel
        JaxPINT timing model.
    toa_data : TOAData
        Input TOAs (not modified).
    params : ParameterVector
        Timing model parameters.
    key : JAX PRNG key
        Random key for noise generation.
    noise_components : sequence of NoiseComponent
        Noise sources to add. If empty, returns noiseless TOAs.

    Returns
    -------
    TOAData
        Simulated TOAs with residuals encoding only noise.
    """
    toa_data = zero_residuals(model, toa_data, params)
    if noise_components:
        delays = simulate_noise(toa_data, params, key, noise_components)
        toa_data = apply_delay_to_toas(toa_data, delays)
    return toa_data


# ---------------------------------------------------------------------------
# Fake-TOA generation from scratch
# ---------------------------------------------------------------------------


def make_toa_data_from_mjds(
    mjds,
    par_result=None,
    *,
    obs: str = "gbt",
    freq_mhz=1400.0,
    error_us=1.0,
    ephem=None,
    include_bipm=None,
    bipm_version=None,
    planets=None,
):
    """Build a :class:`TOAData` from bare MJDs -- no ``.tim`` file involved.

    The counterpart of PINT's ``make_fake_toas_fromMJDs``: synthesizes
    :class:`~jaxpint.tim.RawTOA` records and runs them through the *identical*
    native pipeline a ``.tim`` file takes (clock chain, UTC->TDB, barycentric
    posvels, TZR, basis stamping) via
    :func:`jaxpint.loaders.native.toa_data_from_raw_toas`. The timestamps are
    raw grid points; they do not realize any timing model until
    :func:`make_fake_toas` / :func:`zero_residuals` adjusts them.

    Parameters
    ----------
    mjds : array_like
        Observation epochs, MJD (UTC at *obs*; TDB when *obs* is the
        barycentre ``"@"``, whose sites record TDB directly).
    par_result : ParResult, optional
        Parsed ``.par``; supplies EPHEM/CLK/PLANET_SHAPIRO defaults, TZR and
        the GP basis coordinate, exactly as in :func:`jaxpint.native.get_TOAs`.
    obs : str
        Observatory token, as it would appear in a ``.tim`` (``"gbt"``,
        ``"ao"``, ``"@"`` ...).
    freq_mhz : float or array_like
        Observing frequency. A scalar applies to every TOA; an array is
        cycled across the grid (PINT's multi-frequency convention). ``0`` or
        ``inf`` mean infinite frequency (dispersion-free), matching the
        ``.tim`` parser's 0 -> inf rule.
    error_us : float or array_like
        TOA uncertainty in microseconds, scalar or cycled like *freq_mhz*.
    ephem, include_bipm, bipm_version, planets
        Overrides forwarded to the native pipeline; ``None`` defers to the
        par (or the packaged defaults), as everywhere else.
    """
    from jaxpint.loaders.native import toa_data_from_raw_toas
    from jaxpint.tim import RawTOA

    mjds = np.asarray(mjds, dtype=np.float64)
    if mjds.ndim != 1 or mjds.size == 0:
        raise ValueError(f"mjds must be a non-empty 1-D array, got shape {mjds.shape}")
    n = mjds.size
    # np.resize cycles the values to length n -- the scalar case trivially so.
    freqs = np.resize(np.asarray(freq_mhz, dtype=np.float64), n)
    freqs = np.where(freqs == 0.0, np.inf, freqs)  # parser's 0 -> inf rule
    errors_s = np.resize(np.asarray(error_us, dtype=np.float64), n) * 1e-6

    raw = [
        RawTOA(
            mjd_int=float(np.floor(m)),
            mjd_frac=float(m - np.floor(m)),
            error_s=float(e),
            freq_mhz=float(f),
            obs=obs,
            flags={"fake": "1"},  # PINT marks synthetic TOAs; keep the parity
        )
        for m, f, e in zip(mjds, freqs, errors_s)
    ]
    return toa_data_from_raw_toas(
        raw,
        par_result,
        ephem=ephem,
        include_bipm=include_bipm,
        bipm_version=bipm_version,
        planets=planets,
    )


def make_uniform_toa_data(
    start_mjd: float,
    end_mjd: float,
    n_toas: int,
    par_result=None,
    **kwargs,
):
    """A uniform grid of *n_toas* epochs on [start, end], both ends included.

    ``linspace`` endpoints match PINT's ``make_fake_toas_uniform``. All other
    arguments as :func:`make_toa_data_from_mjds`.
    """
    if n_toas < 1:
        raise ValueError(f"n_toas must be >= 1, got {n_toas}")
    mjds = np.linspace(float(start_mjd), float(end_mjd), int(n_toas))
    return make_toa_data_from_mjds(mjds, par_result, **kwargs)


def make_fake_toas_uniform(
    start_mjd: float,
    end_mjd: float,
    n_toas: int,
    par_result,
    *,
    key: jax.Array | None = None,
    add_noise: bool = False,
    noise_components: Sequence[NoiseComponent] = (),
    **kwargs,
):
    """Uniform fake TOAs realizing the par's timing model -- PINT's namesake.

    Grid -> native pipeline -> :func:`zero_residuals`, so the returned TOAs
    encode the deterministic model exactly (residuals < 1 ns); *add_noise*
    then perturbs each TOA by ``N(0, error)`` (PINT's ``add_noise``), and
    *noise_components* layers correlated realizations on top via
    :func:`simulate_noise`.

    The timing model used for zeroing is built internally from *par_result*
    and the grid; build your own with ``build_model(par_result, toa_data)``
    afterwards -- it is deterministic, so it reproduces the same model.

    Returns the :class:`TOAData` only, mirroring PINT.
    """
    # Unlike the grid builders above, the par is NOT optional here: realizing
    # the model means building it. Without this guard, None surfaces as an
    # AttributeError from deep inside build_model.
    if par_result is None:
        raise ValueError(
            "make_fake_toas_uniform needs a ParResult: zeroing residuals "
            "requires building the timing model. To synthesize bare epochs "
            "with no model, use make_uniform_toa_data instead."
        )
    toa_data = make_uniform_toa_data(start_mjd, end_mjd, n_toas, par_result, **kwargs)
    return _realize(toa_data, par_result, key, add_noise, noise_components)


def make_fake_toas_from_mjds(
    mjds,
    par_result,
    *,
    key: jax.Array | None = None,
    add_noise: bool = False,
    noise_components: Sequence[NoiseComponent] = (),
    **kwargs,
):
    """Fake TOAs at the given epochs -- PINT's ``make_fake_toas_fromMJDs``.

    Identical to :func:`make_fake_toas_uniform` except the epochs are supplied
    instead of gridded; see there for the semantics of *add_noise* /
    *noise_components* and the remaining keyword arguments.
    """
    if par_result is None:
        raise ValueError(
            "make_fake_toas_from_mjds needs a ParResult: zeroing residuals "
            "requires building the timing model. To synthesize bare epochs "
            "with no model, use make_toa_data_from_mjds instead."
        )
    toa_data = make_toa_data_from_mjds(mjds, par_result, **kwargs)
    return _realize(toa_data, par_result, key, add_noise, noise_components)


def _realize(toa_data, par_result, key, add_noise, noise_components):
    """Zero residuals against the par's model, then layer noise on top."""
    from jaxpint.model_builder import build_model

    model, _noise = build_model(par_result, toa_data)
    toa_data = zero_residuals(model, toa_data, par_result.params)

    if add_noise or noise_components:
        if key is None:
            raise ValueError("add_noise / noise_components need a JAX PRNG key")
        rng = key  # narrowed: non-None past the guard (pyright)
        delays = jnp.zeros(toa_data.n_toas)
        if add_noise:
            rng, sub = jax.random.split(rng)
            delays = delays + toa_data.error * jax.random.normal(
                sub, (toa_data.n_toas,)
            )
        if noise_components:
            delays = delays + simulate_noise(
                toa_data, par_result.params, rng, noise_components
            )
        toa_data = apply_delay_to_toas(toa_data, delays)
    return toa_data
