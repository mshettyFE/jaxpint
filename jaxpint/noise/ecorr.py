"""Epoch-correlated noise model (ECORR).

::

    C_ecorr = U · diag(ECORR²) · Uᵀ

where *U* is a quantization matrix mapping TOAs to observing epochs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl
from jaxpint.noise._basis_gp import _BasisGPNoise
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.types import ParameterVector

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext

log = logging.getLogger(__name__)


@register_component(component=Component.ECORR_NOISE, pint_names=("EcorrNoise",))
class EcorrNoise(_BasisGPNoise):
    """Epoch-correlated noise model (ECORR).

    ECORR adds a low-rank contribution to the TOA covariance matrix::

        C_ecorr = U · diag(ECORR²) · Uᵀ

    where *U* is a binary quantization matrix mapping TOAs to observing
    epochs (pre-computed by the bridge) and the weights are the squared
    ECORR values.  A non-Fourier (epoch-indicator) basis GP: the Woodbury
    covariance and realization drawing are inherited from
    :class:`~jaxpint.noise._basis_gp._BasisGPNoise`.

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

    PARAMS = (
        ParamDecl(
            "ECORR1",
            kind="mask",
            unit="us",
            prefix="ECORR",
            aliases=("ECORR", "TNECORR", "TNECORR1"),
            prefix_aliases=("TNECORR",),
        ),
    )

    ecorr_names: tuple[str, ...] = eqx.field(static=True)
    quantization_matrix: (
        Float[Array, "n_toas n_epochs"] | Float[np.ndarray, "n_toas n_epochs"]
    )
    ecorr_epoch_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[EcorrNoise]":
        """Construct from a parsed model (co-located with the physics it builds)."""
        import numpy as np
        import jax.numpy as jnp
        from jaxpint._build_context import basis_seconds
        from jaxpint.utils import build_quantization_matrix

        par = ctx.par
        toa_data = ctx.toa_data
        # Same prefix-discovery idiom as ScaleToaError / ScaleDmError.
        ecorr_names = par.params.names_with_prefix("ECORR")
        if toa_data is not None and len(ecorr_names) > 0:
            basis_s = basis_seconds(toa_data)
            # Missing mask -> all-False (this ECORR group selects no TOAs); the
            # build-time _validate_flag_masks check flags genuinely-absent masks.
            ecorr_masks = {
                ename: np.asarray(toa_data.flag_mask(ename, default=False))
                for ename in ecorr_names
            }

            U, eslices = build_quantization_matrix(basis_s, ecorr_masks)
            # Slices are looked up BY NAME, never by position.  ``ecorr_names``
            # is sorted lexicographically, so with >=10 parameters the column
            # blocks of ``U`` run ECORR1, ECORR10, ECORR11, ECORR2, ...  That is
            # harmless *here*: each parameter's weights land in its own named
            # slice, and ``U @ diag(Phi) @ U.T`` is invariant to column
            # permutation.  Do not "fix" the ordering -- correctness must not
            # depend on it.
            # TODO: Think of a consistent ordering in JaxPINT?

            ecorr_epoch_slices = tuple(eslices[n] for n in ecorr_names)
            return cls(
                ecorr_names=ecorr_names,
                quantization_matrix=jnp.asarray(U),
                ecorr_epoch_slices=ecorr_epoch_slices,
            )
        elif toa_data is None and len(ecorr_names) > 0:
            log.warning(
                "EcorrNoise found but no toa_data provided — ECORR not available"
            )
        return None

    def __post_init__(self):
        # Host numpy is the source of truth; the device view is built lazily
        # by _BasisGPNoise._columns_jax. See that module's docstring.
        self._coerce_host_field("quantization_matrix")

    def _host_columns(
        self,
    ) -> Float[np.ndarray, "n_toas n_epochs"] | Float[Array, "n_toas n_epochs"]:
        # Pass-through, never np.asarray: inside a jit trace of a
        # reconstructed instance this field is a tracer (see base docstring).
        return self.quantization_matrix

    def psd_weights(self, params: ParameterVector) -> Float[Array, " n_epochs"]:
        """Prior diagonal for the epoch basis: ECORR² per epoch column."""
        return self.ecorr_weights(params)

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
        for name, (start, end) in zip(self.ecorr_names, self.ecorr_epoch_slices):
            ecorr_val = params.param_value(name)
            weights = weights.at[start:end].set(ecorr_val**2)
        return weights

    def static_basis(
        self,
    ) -> Float[np.ndarray, "n_toas n_epochs"] | Float[Array, "n_toas n_epochs"]:
        # Fixed basis -> advertise it so NoiseModel can pre-stack it once.
        # Via _host_columns so this always advertises the same array
        # covariance() consumes.
        return self._host_columns()
