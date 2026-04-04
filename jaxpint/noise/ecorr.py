"""Epoch-correlated noise model (ECORR).

::

    C_ecorr = U · diag(ECORR²) · Uᵀ

where *U* is a quantization matrix mapping TOAs to observing epochs.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent
from jaxpint.types import TOAData, ParameterVector


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
