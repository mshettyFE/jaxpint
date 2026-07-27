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
from jaxtyping import Array, Float, Int

from jaxpint.components import ParamDecl
from jaxpint.noise._basis_gp import _BasisGPNoise
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.utils import NO_EPOCH
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
    epochs and the weights are the squared ECORR values.  A non-Fourier
    (epoch-indicator) basis GP: the Woodbury covariance and realization
    drawing are inherited from
    :class:`~jaxpint.noise._basis_gp._BasisGPNoise`.

    *U* has exactly one nonzero per assigned row, so the component stores only the per-TOA
    ``epoch_index`` (int32, ``-1`` = no epoch) and synthesizes the dense
    columns on demand; Drastically reduces memory usage with no precision loss.
    Use :meth:`from_dense` to construct from a pre-built dense matrix.

    Parameters
    ----------
    ecorr_names : tuple of str
        Parameter names for ECORR instances (e.g. ``("ECORR1", "ECORR2")``).
        Values must be in **seconds** (the bridge converts from PINT's
        native microseconds).
    epoch_index : array, shape (n_toas,), int32
        Epoch column for each TOA (:data:`jaxpint.utils.NO_EPOCH` = -1
        for TOAs in no kept epoch).
        Pre-computed by the bridge (:func:`jaxpint.utils.
        build_quantization_index`) because epoch identification is
        data-dependent and not JIT-compatible.
    n_epochs : int
        Total number of epoch columns.
    ecorr_epoch_slices : tuple of (int, int)
        For each ECORR parameter, the ``(start_col, end_col)`` range in
        the epoch-column dimension.
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
    epoch_index: Int[Array, " n_toas"] | Int[np.ndarray, " n_toas"]
    n_epochs: int = eqx.field(static=True)
    ecorr_epoch_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[EcorrNoise]":
        """Construct from a parsed model (co-located with the physics it builds)."""
        import numpy as np
        from jaxpint._build_context import basis_seconds
        from jaxpint.utils import build_quantization_index

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

            epoch_index, n_epochs, eslices = build_quantization_index(
                basis_s, ecorr_masks
            )
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
                epoch_index=epoch_index,
                n_epochs=n_epochs,
                ecorr_epoch_slices=ecorr_epoch_slices,
            )
        elif toa_data is None and len(ecorr_names) > 0:
            log.warning(
                "EcorrNoise found but no toa_data provided — ECORR not available"
            )
        return None

    @classmethod
    def from_dense(
        cls,
        *,
        ecorr_names: tuple[str, ...],
        quantization_matrix,
        ecorr_epoch_slices: tuple[tuple[int, int], ...],
    ) -> "EcorrNoise":
        """Construct from a dense binary quantization matrix.

        Back-compat path for callers holding the dense form (one nonzero
        per assigned row, value 1.0); converts to indexed storage.
        """
        U = np.asarray(quantization_matrix)
        nz = U != 0
        counts = nz.sum(axis=1)
        if np.any(counts > 1) or (nz.any() and not np.all(U[nz] == 1.0)):
            raise ValueError(
                "quantization_matrix must be binary with at most one "
                "nonzero per row (an epoch-indicator matrix)."
            )
        idx = np.where(counts == 1, nz.argmax(axis=1), NO_EPOCH).astype(np.int32)
        return cls(
            ecorr_names=ecorr_names,
            epoch_index=idx,
            n_epochs=int(U.shape[1]),
            ecorr_epoch_slices=ecorr_epoch_slices,
        )

    def __post_init__(self):
        # Host numpy is the source of truth; the device view is built lazily
        # by _BasisGPNoise._columns_jax. See that module's docstring.
        self._coerce_host_field("epoch_index")

    def _host_columns(
        self,
    ) -> Float[np.ndarray, "n_toas n_epochs"] | Float[Array, "n_toas n_epochs"]:
        idx = self.epoch_index
        if isinstance(idx, np.ndarray):
            from jaxpint.utils import quantization_matrix_from_index

            return quantization_matrix_from_index(idx, self.n_epochs)
        rows = jnp.arange(idx.shape[0])
        onehot = jnp.zeros((idx.shape[0], self.n_epochs), dtype=jnp.float64)
        # NO_EPOCH rows: clip keeps the scatter in range, the >= 0 weight
        # mask zeroes them .
        return onehot.at[rows, jnp.clip(idx, 0)].add((idx >= 0).astype(jnp.float64))

    def basis_width(self) -> int:
        # Cheap width: no dense synthesis just to read a shape.
        return self.n_epochs

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
        weights = jnp.zeros(self.n_epochs)
        for name, (start, end) in zip(self.ecorr_names, self.ecorr_epoch_slices):
            ecorr_val = params.param_value(name)
            weights = weights.at[start:end].set(ecorr_val**2)
        return weights

    def static_basis(
        self,
    ) -> Float[np.ndarray, "n_toas n_epochs"] | Float[Array, "n_toas n_epochs"]:
        # Fixed basis -> advertise it so NoiseModel can pre-stack it once.
        # Via _host_columns so this always advertises the same array
        # covariance() consumes.  NOTE: under indexed storage this call
        # SYNTHESIZES the dense matrix (O(n_toas x n_epochs) transient) —
        # cheap width queries must use basis_width() instead.
        return self._host_columns()
