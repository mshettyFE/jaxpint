"""Multi-pulsar PTA log-likelihood.

Composes :func:`jaxpint.likelihood.single_pulsar_logL` across multiple
pulsars, with signal injections (CW, GWB, etc.) mediated by the
:class:`SignalInjector` abstract base class.

The per-pulsar likelihood uses the Woodbury matrix identity to evaluate
the Gaussian log-likelihood without forming the full covariance matrix;
see van Haasteren et al. (2009) [pta_vh09]_ Appendix A.

References
----------
.. [pta_vh09] van Haasteren et al. (2009), "On measuring the gravitational-wave
   background using pulsar timing arrays", MNRAS 395, 1005.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.likelihood import single_pulsar_logL
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector

from jaxpint.pta.params import GlobalParams


# ---------------------------------------------------------------------------
# Signal injector ABC
# ---------------------------------------------------------------------------


class SignalInjector(ABC):
    """Abstract base class for PTA signal components.

    Each injector:

    1. Registers its own parameters into :class:`GlobalParams` via
       :meth:`register_params` (**required** — abstract).
    2. Produces delay arrays and/or covariance ``(U, Phi)`` tuples per
       pulsar via :meth:`delay` / :meth:`covariance` (optional —
       default implementations return ``None``).

    Subclasses must implement :meth:`register_params`.  Override
    :meth:`delay` for deterministic signals (e.g. CW) and/or
    :meth:`covariance` for stochastic signals (e.g. GWB).

    :func:`pta_logL` is agnostic to the signal type.
    """

    @abstractmethod
    def register_params(self, global_params: GlobalParams) -> GlobalParams:
        """Append this signal's parameters to *global_params*.

        Parameters
        ----------
        global_params : GlobalParams
            Mutable accumulator of shared PTA parameters.

        Returns
        -------
        GlobalParams
            Updated copy with this signal's parameters appended.
        """
        ...

    def delay(
        self,
        p: int,
        toa_data: TOAData,
        pulsar_params: ParameterVector,
        global_params: GlobalParams,
    ) -> Optional[Float[Array, " n_toas"]]:
        """Return deterministic delay for pulsar *p*, or ``None``.

        Override for deterministic signals.  The default returns ``None``
        (no delay contribution).

        Parameters
        ----------
        p : int
            Pulsar index within the PTA.
        toa_data : TOAData
            Pulse time-of-arrival data for pulsar *p*.
        pulsar_params : ParameterVector
            Timing and noise parameters for pulsar *p*.
        global_params : GlobalParams
            Shared PTA parameters (CW source properties, GWB spectrum, etc.).

        Returns
        -------
        (n_toas,) array or None
            Deterministic timing delay in seconds, or ``None`` if this
            injector does not contribute a delay.
        """
        return None

    def covariance(
        self,
        p: int,
        toa_data: TOAData,
        pulsar_params: ParameterVector,
        global_params: GlobalParams,
    ) -> Optional[
        tuple[Float[Array, "n_toas n_basis"], Float[Array, " n_basis"]]
    ]:
        """Return ``(U, Phi)`` covariance contribution for pulsar *p*, or ``None``.

        Override for stochastic signals.  The default returns ``None``
        (no covariance contribution).

        Parameters
        ----------
        p : int
            Pulsar index within the PTA.
        toa_data : TOAData
            Pulse time-of-arrival data for pulsar *p*.
        pulsar_params : ParameterVector
            Timing and noise parameters for pulsar *p*.
        global_params : GlobalParams
            Shared PTA parameters (CW source properties, GWB spectrum, etc.).

        Returns
        -------
        tuple of ((n_toas, n_basis) array, (n_basis,) array) or None
            Design matrix ``U`` and diagonal PSD vector ``Phi``, or
            ``None`` if this injector does not contribute covariance.
        """
        return None


# ---------------------------------------------------------------------------
# PTA configuration
# ---------------------------------------------------------------------------


class PTAConfig(eqx.Module):
    """Configuration for per-pulsar PTA likelihood evaluation.

    Holds the per-pulsar TOA data, timing/noise models, and any
    :class:`SignalInjector` instances that contribute additional
    covariance terms (e.g. red, DM, or chromatic noise).

    ``toa_data_list`` and ``noise_models`` are *dynamic* (traced) fields;
    marking them static balloons jit memory because the per-pulsar arrays
    get baked into the compiled HLO. ``timing_models`` and
    ``signal_injectors`` are static structural metadata.

    Raises
    ------
    ValueError
        If ``toa_data_list``, ``timing_models``, and ``noise_models`` do
        not all have the same length.
    """

    toa_data_list: tuple[TOAData, ...]
    noise_models: tuple[NoiseModel, ...]
    timing_models: tuple[TimingModel, ...] = eqx.field(static=True)
    signal_injectors: tuple[SignalInjector, ...] = eqx.field(static=True)

    def __post_init__(self):
        n_toa = len(self.toa_data_list)
        n_tm = len(self.timing_models)
        n_nm = len(self.noise_models)
        if not (n_toa == n_tm == n_nm):
            raise ValueError(
                f"Mismatched pulsar counts: {n_toa} TOA datasets, "
                f"{n_tm} timing models, {n_nm} noise models. "
                f"All three must have the same length (one per pulsar)."
            )

    @property
    def n_pulsars(self) -> int:
        """Number of pulsars in this PTA configuration.

        Returns
        -------
        int
            Length of ``toa_data_list``.
        """
        return len(self.toa_data_list)


# ---------------------------------------------------------------------------
# PTA log-likelihood
# ---------------------------------------------------------------------------


def pta_logL(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: PTAConfig,
) -> Float[Array, ""]:
    """Multi-pulsar log-likelihood with signal injections.

    For each pulsar, collects delay and covariance contributions from every
    :class:`SignalInjector` in *config*, then delegates to
    :func:`jaxpint.likelihood.single_pulsar_logL`.

    Parameters
    ----------
    global_params : GlobalParams
        Shared parameters (CW source properties, GWB spectrum, etc.).
        This is the first differentiable argument.
    pulsar_params : tuple of ParameterVector
        Per-pulsar timing and noise parameters.
        This is the second differentiable argument.
    config : PTAConfig
        Static configuration (TOA data, models, injectors).

    Returns
    -------
    logL : scalar
        Sum of per-pulsar log-likelihoods.
    """
    total = jnp.float64(0.0)

    for p in range(len(pulsar_params)):
        # -- Collect delays from all injectors --
        delays = [
            inj.delay(
                p,
                config.toa_data_list[p],
                pulsar_params[p],
                global_params,
            )
            for inj in config.signal_injectors
        ]
        delays = [d for d in delays if d is not None]
        ext_delay = sum(delays) if delays else None

        # -- Collect covariances from all injectors --
        covs = [
            inj.covariance(
                p,
                config.toa_data_list[p],
                pulsar_params[p],
                global_params,
            )
            for inj in config.signal_injectors
        ]
        covs = [c for c in covs if c is not None]
        if covs:
            ext_cov = (
                jnp.concatenate([U for U, _ in covs], axis=1),
                jnp.concatenate([Phi for _, Phi in covs]),
            )
        else:
            ext_cov = None

        total += single_pulsar_logL(
            config.toa_data_list[p],
            config.timing_models[p],
            config.noise_models[p],
            pulsar_params[p],
            external_delay=ext_delay,
            external_cov=ext_cov,
        )

    return total
