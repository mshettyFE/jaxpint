"""HD-correlated gravitational wave background injector.

Implements :class:`~jaxpint.pta.likelihood.CorrelatedSignalInjector`
for a power-law GWB with Hellings-Downs (or other) inter-pulsar correlations.

References
----------
.. [cgwb_hd83] Hellings & Downs (1983), ApJL 265, L39.
.. [cgwb_vh09] van Haasteren et al. (2009), MNRAS 395, 1005.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.types import TOAData
from jaxpint.types import GlobalParams
from jaxpint.pta.likelihood import CorrelatedSignalInjector
from jaxpint.pta.signals.gwb import (
    CURN_PARAM_DEFAULTS,
    fourier_basis,
    powerlaw_psd,
)
from jaxpint.pta.signals.orf import hd_orf


class HDCorrelatedGWBInjector(CorrelatedSignalInjector):
    """Correlated GWB injector with configurable overlap reduction function.

    Registers two global parameters: ``{prefix}log10_A`` and
    ``{prefix}gamma``.  The ORF matrix is precomputed at construction time
    from the supplied pulsar positions and ORF function.

    Parameters
    ----------
    pulsar_positions : (n_psr, 3) array
        Unit vectors pointing to each pulsar (ICRS).
    n_components : int
        Number of Fourier frequency components.
    T_span : float
        Observing time span in seconds.
    orf_func : callable, optional
        Overlap reduction function ``(pos1, pos2) -> scalar``.
        Defaults to :func:`~jaxpint.pta.hd_orf`.
    prefix : str
        Naming prefix for parameters in :class:`GlobalParams`.
    initial_values : dict, optional
        Override default initial values for ``log10_A`` and ``gamma``.
    """

    def __init__(
        self,
        pulsar_positions: Float[Array, "n_psr 3"],
        n_components: int,
        T_span: float,
        orf_func: Callable = hd_orf,
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
                    f"Unknown GWB parameters: {unknown}. "
                    f"Valid: {list(CURN_PARAM_DEFAULTS.keys())}"
                )
            self.param_spec.update(initial_values)

        # Precompute ORF matrix
        n_psr = pulsar_positions.shape[0]
        Gamma = jnp.zeros((n_psr, n_psr))
        for a in range(n_psr):
            for b in range(a, n_psr):
                val = orf_func(pulsar_positions[a], pulsar_positions[b])
                Gamma = Gamma.at[a, b].set(val)
                Gamma = Gamma.at[b, a].set(val)
        self._orf_matrix = Gamma

    # -- CorrelatedSignalInjector ABC ------------------------------------------

    def register_params(self, global_params: GlobalParams) -> GlobalParams:
        names = [f"{self.prefix}{n}" for n in self.param_spec]
        values = list(self.param_spec.values())
        return global_params.add_params(names, values)

    def get_fourier_basis(
        self,
        toa_data: TOAData,
    ) -> Float[Array, "n_toas n_basis"]:
        toas_seconds = toa_data.tdb_seconds
        F, _ = fourier_basis(toas_seconds, self.n_components, self.T_span)
        return F

    def get_psd(
        self,
        global_params: GlobalParams,
    ) -> Float[Array, " n_basis"]:
        log10_A = global_params.param_value(f"{self.prefix}log10_A")
        gamma = global_params.param_value(f"{self.prefix}gamma")

        freqs = jnp.arange(1, self.n_components + 1) / self.T_span
        df = 1.0 / self.T_span
        psd = powerlaw_psd(freqs, log10_A, gamma) * df
        # Each frequency has sin and cos basis functions with the same PSD
        return jnp.repeat(psd, 2)

    def get_orf_matrix(self) -> Float[Array, "n_psr n_psr"]:
        return self._orf_matrix
