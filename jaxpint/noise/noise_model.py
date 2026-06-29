"""NoiseModel: aggregates all noise sources into a single Woodbury interface.

::

    C = diag(Ndiag) + U · diag(Phidiag) · Uᵀ

where ``Ndiag`` comes from white noise and ``U`` / ``Phidiag`` are
horizontally concatenated from all correlated noise components, each
recomputed per call by :meth:`NoiseModel.covariance`.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent, _make_component_names
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

        Column layout of the returned ``U`` is: pre-stacked static-basis
        columns first (in original ``correlated`` order, restricted to
        static components), followed by per-call dynamic-basis blocks
        (in original ``correlated`` order, restricted to dynamic
        components). ``Phidiag`` follows the same layout. The ordering
        of basis columns does not affect the value of ``C`` so long as
        ``U`` and ``Phidiag`` agree.

        Returns
        -------
        Ndiag : (n_toas,)
            Diagonal variance (white noise contribution).
        U : (n_toas, n_basis)
            Pre-stacked + dynamic basis matrix. Empty ``(n_toas, 0)``
            when there are no correlated sources.
        Phidiag : (n_basis,)
            Concatenated basis weights.
        """
        Ndiag = self.scaled_sigma(toa_data, params) ** 2

        if self.correlated:
            Us, Phis = zip(
                *(comp.covariance(toa_data, params)[1:] for comp in self.correlated)
            )
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
        assert toa_data.dm_errors is not None  # wideband data always carries DM
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
        Ndiag_dm = dm_sigma**2
        return Ndiag_toa, U_toa, Phi_toa, Ndiag_dm

    @property
    def has_correlated(self) -> bool:
        """True if any correlated noise components are present."""
        return len(self.correlated) > 0

    # ------------------------------------------------------------------
    # Component indexing
    # ------------------------------------------------------------------

    @property
    def components(self) -> tuple:
        """Return all non-None noise components in order (white, correlated, dm_white)."""
        result: list = []
        if self.white_noise is not None:
            result.append(self.white_noise)
        result.extend(self.correlated)
        if self.dm_white_noise is not None:
            result.append(self.dm_white_noise)
        return tuple(result)

    @property
    def component_names(self) -> tuple[str, ...]:
        """Unique names for all components, auto-disambiguated for duplicates."""
        return _make_component_names(self.components)

    def __getitem__(self, key):
        if isinstance(key, str):
            names = self.component_names
            comps = self.components
            for i, name in enumerate(names):
                if name == key:
                    return comps[i]
            raise KeyError(f"{key!r} not found. Available components: {names}")
        elif isinstance(key, (int, slice)):
            return self.components[key]
        raise TypeError(f"indices must be str, int, or slice, not {type(key).__name__}")
