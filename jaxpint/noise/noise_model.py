"""NoiseModel: aggregates all noise sources into a single Woodbury interface.

::

    C = diag(Ndiag) + U · diag(Phidiag) · Uᵀ

where ``Ndiag`` comes from white noise and ``U`` / ``Phidiag`` are
horizontally concatenated from all correlated noise components.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent
from jaxpint.noise.dm_white import ScaleDmError
from jaxpint.noise.white import ScaleToaError
from jaxpint.types import TOAData, ParameterVector


class NoiseModel(eqx.Module):
    """Container that aggregates all noise sources into a single interface.

    Provides a unified Woodbury covariance decomposition::

        C = diag(Ndiag) + U · diag(Phidiag) · Uᵀ

    where ``Ndiag`` comes from white noise (EFAC/EQUAD-scaled TOA
    uncertainties) and ``U`` / ``Phidiag`` are horizontally concatenated
    from all correlated noise components (ECORR, red noise, etc.).

    Parameters
    ----------
    white_noise : ScaleToaError or None
        White noise model (EFAC/EQUAD).  When ``None``, raw TOA errors
        are used.
    correlated : tuple of NoiseComponent
        Correlated noise components whose basis matrices and weights
        are concatenated to form ``U`` and ``Phidiag``.
    """

    white_noise: Optional[ScaleToaError]
    correlated: tuple[NoiseComponent, ...]
    dm_white_noise: Optional[ScaleDmError] = None

    def scaled_sigma(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Return noise-scaled TOA uncertainties in seconds."""
        if self.white_noise is not None:
            return self.white_noise.scaled_sigma(toa_data, params)
        return toa_data.error

    def covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_basis"],
        Float[Array, " n_basis"],
    ]:
        """Return the combined Woodbury ``(Ndiag, U, Phidiag)`` triple.

        Returns
        -------
        Ndiag : (n_toas,)
            Diagonal variance (white noise contribution).
        U : (n_toas, n_basis)
            Concatenated basis matrices from all correlated components.
            Empty ``(n_toas, 0)`` when there are no correlated sources.
        Phidiag : (n_basis,)
            Concatenated basis weights.
        """
        Ndiag = self.scaled_sigma(toa_data, params) ** 2

        Us: list[Float[Array, "n_toas _"]] = []
        Phis: list[Float[Array, " _"]] = []
        for comp in self.correlated:
            _, U_i, Phi_i = comp.covariance(toa_data, params)
            if U_i is not None:
                Us.append(U_i)
                Phis.append(Phi_i)

        if Us:
            U = jnp.concatenate(Us, axis=1)
            Phidiag = jnp.concatenate(Phis)
        else:
            U = jnp.zeros((toa_data.n_toas, 0))
            Phidiag = jnp.zeros(0)

        return Ndiag, U, Phidiag

    def scaled_dm_sigma(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Return noise-scaled DM uncertainties in pc/cm³."""
        if self.dm_white_noise is not None:
            return self.dm_white_noise.scaled_dm_sigma(toa_data, params)
        return toa_data.dm_errors

    def wideband_covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_basis"],
        Float[Array, " n_basis"],
        Float[Array, " n_toas"],
    ]:
        """Return wideband noise decomposition.

        Returns
        -------
        Ndiag_toa : (n_toas,)
            TOA diagonal variance (white noise).
        U_toa : (n_toas, n_basis)
            TOA correlated noise basis.
        Phi_toa : (n_basis,)
            TOA correlated noise weights.
        Ndiag_dm : (n_toas,)
            DM diagonal variance (white noise only).
        """
        Ndiag_toa, U_toa, Phi_toa = self.covariance(toa_data, params)
        dm_sigma = self.scaled_dm_sigma(toa_data, params)
        Ndiag_dm = dm_sigma ** 2
        return Ndiag_toa, U_toa, Phi_toa, Ndiag_dm

    @property
    def has_correlated(self) -> bool:
        """True if any correlated noise components are present."""
        return len(self.correlated) > 0
