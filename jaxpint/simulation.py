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
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent
from jaxpint.fitter import compute_time_residuals
from jaxpint.model import TimingModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.constants import SECS_PER_DAY

def apply_delay_to_toas(
    toa_data: TOAData,
    delays_seconds: Float[Array, " n_toas"],
) -> TOAData:
    """Return a new TOAData with time delays added to MJD and TDB fields.

    Converts *delays_seconds* to days, adds to the fractional part of
    each timestamp, and renormalises so ``frac`` stays in [0, 1).

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

    new_mjd_frac = toa_data.mjd_frac + delay_days
    mjd_overflow = jnp.floor(new_mjd_frac)
    new_mjd_int = toa_data.mjd_int + mjd_overflow
    new_mjd_frac = new_mjd_frac - mjd_overflow

    new_tdb_frac = toa_data.tdb_frac + delay_days
    tdb_overflow = jnp.floor(new_tdb_frac)
    new_tdb_int = toa_data.tdb_int + tdb_overflow
    new_tdb_frac = new_tdb_frac - tdb_overflow

    return eqx.tree_at(
        lambda td: (td.mjd_int, td.mjd_frac, td.tdb_int, td.tdb_frac),
        toa_data,
        (new_mjd_int, new_mjd_frac, new_tdb_int, new_tdb_frac),
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

    Each component receives an independent PRNG key derived from *key*.

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
