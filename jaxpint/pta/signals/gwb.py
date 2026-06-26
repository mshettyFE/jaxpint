"""Gravitational wave background covariance injection.

Provides power-law PSD, Fourier design matrix, and the CURN (uncorrelated
common red noise) injector.  Ported from Discovery's ``signals.py``.

References
----------
.. [gwb_mod_p01] Phinney (2001), "A practical theorem on gravitational wave
   backgrounds", astro-ph/0108028.  Characteristic strain to PSD relation.
.. [gwb_mod_a16] Arzoumanian et al. (2016), "The NANOGrav Nine-year Data Set: Limits
   on the Isotropic Stochastic Gravitational Wave Background", ApJ 821, 13.
   Eq. 1 (NANOGrav power-law PSD parameterisation).
.. [gwb_mod_l13] Lentati et al. (2013), "Hyper-efficient model-independent Bayesian
   method for the analysis of pulsar timing data", PRD 87, 104021.
   Section II.A (Fourier basis for GP red noise modelling).
.. [gwb_mod_vh14] van Haasteren & Vallisneri (2014), "New advances in the
   Gaussian-process approach to pulsar-timing data analysis",
   PRD 90, 104012.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.types import TOAData
from jaxpint.pta.injectors import SignalInjector

# Year in seconds (NANOGrav convention)
FYR: float = 1.0 / (365.25 * 86400.0)


def powerlaw_psd(
    f: Float[Array, " n_freq"],
    log10_A: Float[Array, ""],
    gamma: Float[Array, ""],
) -> Float[Array, " n_freq"]:
    """Power-law power spectral density (NANOGrav convention).

    Follows the parameterisation of Arzoumanian et al. (2016) [gwb_a16]_ Eq. 1,
    derived from the characteristic-strain relation of Phinney (2001) [gwb_p01]_:
    ``S(f) = h_c^2(f) / (12 pi^2 f^3)``.

    .. math::
        S(f) = \\frac{A^2}{12\\pi^2}
               \\left(\\frac{f}{f_{\\rm yr}}\\right)^{-\\gamma}
               f_{\\rm yr}^{-3}

    Parameters
    ----------
    f : (n_freq,) array
        Frequencies in Hz.
    log10_A : scalar
        Log-10 of the dimensionless amplitude.
    gamma : scalar
        Spectral index (positive for red noise).

    Returns
    -------
    psd : (n_freq,) array
        Power spectral density in units of s^3.

    References
    ----------
    .. [gwb_a16] Arzoumanian et al. (2016), ApJ 821, 13.
    .. [gwb_p01] Phinney (2001), astro-ph/0108028.
    """
    return (
        (10.0 ** (2.0 * log10_A))
        / (12.0 * jnp.pi**2)
        * FYR ** (gamma - 3.0)
        * f ** (-gamma)
    )


def fourier_basis(
    toas_seconds: Float[Array, " n_toas"],
    n_components: int,
    T_span: float,
) -> tuple[Float[Array, "n_toas n_basis"], Float[Array, " n_freq"]]:
    """Fourier design matrix (sine/cosine pairs).

    Constructs the basis used for Gaussian-process red noise modelling
    as described in Lentati et al. (2013) [gwb_l13]_ Section II.A and
    van Haasteren & Vallisneri (2014) [gwb_vh14]_.

    Parameters
    ----------
    toas_seconds : (n_toas,) array
        TOA times in seconds.
    n_components : int
        Number of frequency components.
    T_span : float
        Observing time span in seconds.

    Returns
    -------
    F : (n_toas, 2 * n_components) array
        Design matrix with alternating sin/cos columns.
    freqs : (n_components,) array
        Frequencies in Hz.

    References
    ----------
    .. [gwb_l13] Lentati et al. (2013), PRD 87, 104021.
    .. [gwb_vh14] van Haasteren & Vallisneri (2014), PRD 90, 104012.
    """
    freqs = jnp.arange(1, n_components + 1) / T_span
    phase = 2.0 * jnp.pi * toas_seconds[:, None] * freqs[None, :]
    F = jnp.column_stack([jnp.sin(phase), jnp.cos(phase)])
    return F, freqs


def gwb_covariance(
    toa_data: TOAData,
    n_components: int,
    T_span: float,
    log10_A: Float[Array, ""],
    gamma: Float[Array, ""],
) -> tuple[Float[Array, "n_toas n_basis"], Float[Array, " n_basis"]]:
    """Compute (U, Phi) for CURN injection into ``single_pulsar_logL``.

    Parameters
    ----------
    toa_data : TOAData
        Pulse time-of-arrival data (uses TDB times).
    n_components : int
        Number of Fourier frequency components.
    T_span : float
        Observing time span in seconds.
    log10_A : scalar
        Log-10 GWB amplitude.
    gamma : scalar
        GWB spectral index.

    Returns
    -------
    U : (n_toas, 2 * n_components) array
        Fourier design matrix.
    Phi : (2 * n_components,) array
        PSD values for each basis function.
    """
    toas_seconds = toa_data.tdb_seconds
    F, freqs = fourier_basis(toas_seconds, n_components, T_span)
    df = 1.0 / T_span
    psd = powerlaw_psd(freqs, log10_A, gamma) * df
    Phi = jnp.repeat(psd, 2)  # same PSD for sin and cos
    return F, Phi


# ---------------------------------------------------------------------------
# CURN injector defaults
# ---------------------------------------------------------------------------

CURN_PARAM_DEFAULTS: dict[str, float] = {
    "log10_A": -15.0,  # log10 GWB amplitude
    "gamma": 4.33,  # GWB spectral index
}


class CURNInjector(SignalInjector):
    """Uncorrelated common red noise (CURN, Gamma = I) injector.

    Subclasses :class:`~jaxpint.pta.likelihood.SignalInjector`.
    Registers two global parameters (with *prefix*):
    ``{prefix}log10_A`` and ``{prefix}gamma``.

    Parameters
    ----------
    n_components : int
        Number of Fourier frequency components per pulsar.
    T_span : float
        Observing time span in seconds.
    prefix : str
        Naming prefix in :class:`GlobalParams`.
    initial_values : dict, optional
        Override default initial values (keys must be in
        ``CURN_PARAM_DEFAULTS``).
    """

    param_defaults = CURN_PARAM_DEFAULTS

    def __init__(
        self,
        n_components: int,
        T_span: float,
        prefix: str = "gwb_",
        initial_values: Optional[dict[str, float]] = None,
    ):
        self.n_components = n_components
        self.T_span = T_span
        self.prefix = prefix

        self.param_spec: dict[str, float] = dict(CURN_PARAM_DEFAULTS)
        if initial_values is not None:
            unknown = set(initial_values) - set(CURN_PARAM_DEFAULTS)
            if unknown:
                raise ValueError(
                    f"Unknown CURN parameters: {unknown}. "
                    f"Valid parameters: {list(CURN_PARAM_DEFAULTS.keys())}"
                )
            self.param_spec.update(initial_values)

    # -- SignalInjector ABC -----------------------------------------------------

    def register_params(self, global_params):
        """Register CURN amplitude and spectral index into *global_params*.

        Parameters
        ----------
        global_params : GlobalParams
            Mutable accumulator of shared PTA parameters.

        Returns
        -------
        GlobalParams
            Updated copy with ``{prefix}log10_A`` and ``{prefix}gamma``
            appended.
        """
        names = [f"{self.prefix}{n}" for n in self.param_spec]
        values = list(self.param_spec.values())
        return global_params.add_params(names, values)

    # delay() inherited from SignalInjector — returns None (CURN is stochastic)

    def covariance(self, p, toa_data, pulsar_params, global_params):
        """Compute ``(U, Phi)`` GWB covariance contribution for pulsar *p*.

        Parameters
        ----------
        p : int
            Pulsar index within the PTA (unused; CURN is identical for
            all pulsars).
        toa_data : TOAData
            Pulse time-of-arrival data for pulsar *p*.
        pulsar_params : ParameterVector
            Timing and noise parameters for pulsar *p* (unused).
        global_params : GlobalParams
            Shared PTA parameters containing GWB amplitude and spectral
            index.

        Returns
        -------
        tuple of ((n_toas, 2*n_components) array, (2*n_components,) array)
            Fourier design matrix ``U`` and diagonal PSD vector ``Phi``.
        """
        log10_A = global_params.param_value(f"{self.prefix}log10_A")
        gamma = global_params.param_value(f"{self.prefix}gamma")
        return gwb_covariance(toa_data, self.n_components, self.T_span, log10_A, gamma)
