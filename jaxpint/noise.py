"""Noise models for JaxPINT.

Implements white noise (EFAC/EQUAD), correlated noise (ECORR), and the
:class:`NoiseModel` container that aggregates all noise sources into a
single Woodbury covariance interface.

White noise — ``ScaleToaError``::

    σ_eff = EFAC × √(σ_raw² + EQUAD²)

Correlated noise — ``EcorrNoise``::

    C = diag(σ_eff²) + U · diag(ECORR²) · Uᵀ

where *U* is a quantization matrix mapping TOAs to observing epochs.

Each parameter applies to a subset of TOAs identified by a boolean
mask (pre-computed by the bridge layer and stored in ``TOAData.flag_masks``).
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent
from jaxpint.types import TOAData, ParameterVector


class ScaleToaError(NoiseComponent):
    """White noise model: EFAC/EQUAD scaling of TOA uncertainties.

    Stores the names of EFAC and EQUAD parameters (static metadata).
    The boolean masks selecting which TOAs each parameter applies to
    live in ``TOAData.flag_masks`` (extracted by the bridge).

    Parameters
    ----------
    efac_names : tuple of str
        Parameter names for EFAC instances (e.g. ``("EFAC1", "EFAC2")``).
    equad_names : tuple of str
        Parameter names for EQUAD instances (e.g. ``("EQUAD1", "EQUAD2")``).
        Values must be in **seconds** (the bridge converts from PINT's
        native microseconds).
    """

    efac_names: tuple[str, ...] = eqx.field(static=True)
    equad_names: tuple[str, ...] = eqx.field(static=True)

    def scaled_sigma(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Compute noise-scaled TOA uncertainties.

        Applies EQUAD in quadrature first, then multiplies by EFAC,
        matching PINT's ``ScaleToaError.scale_toa_sigma()`` convention.

        Parameters
        ----------
        toa_data : TOAData
            Must contain ``error`` (seconds) and ``flag_masks`` with
            entries for every name in ``efac_names`` and ``equad_names``.
        params : ParameterVector
            Must contain values for all EFAC/EQUAD parameters.

        Returns
        -------
        sigma_scaled : (n_toas,)
            Scaled uncertainties in seconds.
        """
        sigma_sq = toa_data.error ** 2

        # Step 1: add EQUAD in quadrature (per mask)
        for equad_name in self.equad_names:
            mask = toa_data.flag_masks[equad_name]
            equad_val = params.param_value(equad_name)
            sigma_sq = jnp.where(mask, sigma_sq + equad_val ** 2, sigma_sq)

        sigma = jnp.sqrt(sigma_sq)

        # Step 2: multiply by EFAC (per mask)
        for efac_name in self.efac_names:
            mask = toa_data.flag_masks[efac_name]
            efac_val = params.param_value(efac_name)
            sigma = jnp.where(mask, sigma * efac_val, sigma)

        return sigma

    def covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[Float[Array, " n_toas"], None, None]:
        sigma = self.scaled_sigma(toa_data, params)
        return sigma ** 2, None, None

    def generate(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        key: jax.Array,
    ) -> Float[Array, " n_toas"]:
        sigma = self.scaled_sigma(toa_data, params)
        return sigma * jax.random.normal(key, shape=(toa_data.n_toas,))


class EcorrNoise(NoiseComponent):
    """Epoch-correlated noise model (ECORR).

    ECORR adds a low-rank contribution to the TOA covariance matrix::

        C_ecorr = U · diag(ECORR²) · Uᵀ

    where *U* is a binary quantization matrix mapping TOAs to observing
    epochs (pre-computed by the bridge) and the weights are the squared
    ECORR values.

    Parameters
    ----------
    ecorr_names : tuple of str
        Parameter names for ECORR instances (e.g. ``("ECORR1", "ECORR2")``).
        Values must be in **seconds** (the bridge converts from PINT's
        native microseconds).
    quantization_matrix : array, shape (n_toas, n_epochs)
        Binary matrix mapping TOAs to epochs.  Pre-computed by the bridge
        because epoch identification is data-dependent and not JIT-compatible.
    ecorr_epoch_slices : tuple of (int, int)
        For each ECORR parameter, the ``(start_col, end_col)`` range in
        the quantization matrix's column dimension.
    """

    ecorr_names: tuple[str, ...] = eqx.field(static=True)
    quantization_matrix: Float[Array, "n_toas n_epochs"]
    ecorr_epoch_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def ecorr_weights(
        self,
        params: ParameterVector,
    ) -> Float[Array, " n_epochs"]:
        """Return ECORR² weight for each epoch column.

        Parameters
        ----------
        params : ParameterVector
            Must contain values for all ECORR parameters.

        Returns
        -------
        weights : (n_epochs,)
            Squared ECORR values (seconds²), one per epoch.
        """
        n_epochs = self.quantization_matrix.shape[1]
        weights = jnp.zeros(n_epochs)
        for name, (start, end) in zip(
            self.ecorr_names, self.ecorr_epoch_slices
        ):
            ecorr_val = params.param_value(name)
            weights = weights.at[start:end].set(ecorr_val ** 2)
        return weights

    def covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_epochs"],
        Float[Array, " n_epochs"],
    ]:
        U = self.quantization_matrix
        Phidiag = self.ecorr_weights(params)
        Ndiag = jnp.zeros(toa_data.n_toas)
        return Ndiag, U, Phidiag

    def generate(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        key: jax.Array,
    ) -> Float[Array, " n_toas"]:
        U = self.quantization_matrix
        weights = self.ecorr_weights(params)
        n_epochs = U.shape[1]
        a = jax.random.normal(key, shape=(n_epochs,))
        return U @ (jnp.sqrt(weights) * a)


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

    @property
    def has_correlated(self) -> bool:
        """True if any correlated noise components are present."""
        return len(self.correlated) > 0
