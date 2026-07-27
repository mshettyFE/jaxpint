r"""Sherman–Morrison kernel ECORR: epoch jitter absorbed into the white noise.

Instead of contributing epoch-indicator *basis columns* (the
:class:`~jaxpint.noise.EcorrNoise` route, which adds ``n_epochs`` columns to
the Woodbury width), kernel ECORR treats the epoch jitter as part of the
white covariance::

    N = D + \sum_e j_e^2 u_e u_e^T,   D = diag(sigma^2),  u_e = epoch indicator

one **rank-1 update per epoch**, with epochs disjoint.  Everything the
likelihood needs then has a closed form per epoch block:

- ``s_e = sum_{i in e} 1/sigma_i^2``,  ``a_e = 1 + j_e^2 s_e``
- ``log|N| = sum_i log sigma_i^2 + sum_e log a_e``
- a whitener ``W`` with ``W N W^T = I`` (see below), applied to residuals
  and basis columns so the existing diagonal-``Ndiag`` Woodbury machinery
  runs unchanged with ``Ndiag = 1`` plus a log-det correction.

The whitener
------------
Per epoch, ``N = D^{1/2}(I + j^2 v v^T)D^{1/2}`` with ``v = D^{-1/2} 1``.
The symmetric inverse square root of the middle factor is closed-form for
rank one: ``(I + j^2 v v^T)^{-1/2} = I + gamma v v^T`` with

    ``gamma_e = (1/sqrt(a_e) - 1) / s_e``

so ``W = (I + gamma v v^T) D^{-1/2}`` satisfies ``W N W^T = I`` exactly —
including heteroscedastic within-epoch sigmas.  Applied to a vector::

    (W x)_i = x_i/sigma_i + gamma_e * (1/sigma_i) * sum_{k in e} x_k/sigma_k^2

i.e. one ``segment_sum`` over :data:`~jaxpint.utils.NO_EPOCH`-aware epoch
indices plus gathers — O(n_toas), fully vectorized, differentiable.  TOAs
with no epoch (``NO_EPOCH``) reduce to plain ``x_i/sigma_i``.

Why whitening instead of an N-operator through the engine: quantities in
*coefficient space* are invariant under consistent whitening
(``(WU)^T (WCW^T)^{-1} (Wr) = U^T C^{-1} r``), so pre-whitening ``r``, the
GP basis ``U``, and (in the PTA inner tier) the correlated basis
``F_corr`` makes every downstream solve correct without touching it; only
``log|C|`` needs the additive ``log|N|`` correction.

Use this form of ECORR if:
    - not dealing with conditional likelihoods
    - Can gaurentee that epochs are disjoint

"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from jaxpint.components import NoiseComponent
from jaxpint.noise.ecorr import EcorrNoise, epoch_jitter2
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import NO_EPOCH, SMWhitener


class EcorrKernelNoise(NoiseComponent):
    """ECORR applied as a Sherman–Morrison white-noise kernel.

    Carries the same epoch data as :class:`~jaxpint.noise.EcorrNoise`
    (``epoch_index`` / ``n_epochs`` / per-parameter slices — convert with
    :meth:`from_basis`) but is **not** a basis GP: it subclasses
    :class:`~jaxpint.components.NoiseComponent` directly, contributes no
    basis columns, and participates in the likelihood through
    :meth:`ops` (the whitener) instead of the ``(Ndiag, U, Phi)`` triple.
    Place it in ``NoiseModel.ecorr_kernel``, never in
    ``NoiseModel.correlated``. This is fine since you never actually need to
    construct the covariance most of the time, only quantities involving the
    covariance.

    Trade-off vs the basis form: identical covariance (pinned by tests),
    O(n_toas) instead of ``n_epochs`` Woodbury columns — the dominant
    width on NG15-scale pulsars.
    """

    # Constructed programmatically (from_basis / __init__): no PARAMS, no
    # par registration — same precedent as FreeSpectrumNoise/TimeNodeGPNoise.
    ecorr_names: tuple[str, ...] = eqx.field(static=True)
    epoch_index: Int[Array, " n_toas"] | Int[np.ndarray, " n_toas"]
    n_epochs: int = eqx.field(static=True)
    ecorr_epoch_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __post_init__(self):
        # Host numpy as source of truth (same convention as the basis
        # components; inlined because this class is not a _BasisGPNoise).
        if not isinstance(self.epoch_index, np.ndarray):
            object.__setattr__(self, "epoch_index", np.asarray(self.epoch_index))

    # -- NoiseComponent contract: covariance deliberately refused -----------

    def covariance(self, toa_data, params):  # noqa: D102 — loud redirect
        raise TypeError(
            "EcorrKernelNoise is applied as a whitening kernel; place it in "
            "NoiseModel.ecorr_kernel, not in NoiseModel.correlated."
        )

    # -- kernel surface ------------------------------------------------------

    def ecorr_weights(self, params: ParameterVector) -> Float[Array, " n_epochs"]:
        """Per-epoch j² (seconds²) — the rank-1 update amplitudes."""
        return epoch_jitter2(
            params, self.ecorr_names, self.ecorr_epoch_slices, self.n_epochs
        )

    @classmethod
    def from_basis(cls, ecorr: EcorrNoise) -> "EcorrKernelNoise":
        """Convert a basis-form ECORR component to the kernel form."""
        return cls(
            ecorr_names=ecorr.ecorr_names,
            epoch_index=np.asarray(ecorr.epoch_index),
            n_epochs=ecorr.n_epochs,
            ecorr_epoch_slices=ecorr.ecorr_epoch_slices,
        )

    def ops(
        self,
        Ndiag: Float[Array, " n_toas"],
        params: ParameterVector,
    ) -> SMWhitener:
        """Build the whitener for the current white diagonal and parameters.

        Parameters
        ----------
        Ndiag : (n_toas,)
            The *diagonal* white variance (post EFAC/EQUAD scaling) —
            exactly what ``NoiseModel.covariance`` returns.
        params : ParameterVector
            Read for the ECORR values (via ``ecorr_weights``: j_e^2 per
            epoch).
        """
        idx = jnp.asarray(self.epoch_index)
        valid = idx != NO_EPOCH
        valid_f = valid.astype(jnp.float64)
        idx_c = jnp.clip(idx, 0)

        inv_var = 1.0 / Ndiag
        inv_sigma = jnp.sqrt(inv_var)
        jitter2 = self.ecorr_weights(params)  # (n_epochs,) j_e^2

        s = jax.ops.segment_sum(inv_var * valid_f, idx_c, num_segments=self.n_epochs)
        a = 1.0 + jitter2 * s
        # Empty epochs (possible via from_dense round-trips): s = 0 -> a = 1,
        # log a = 0, and gamma is irrelevant; guard the 0/0.
        gamma = jnp.where(
            s > 0.0, (1.0 / jnp.sqrt(a) - 1.0) / jnp.where(s > 0.0, s, 1.0), 0.0
        )
        gamma_toa = jnp.where(valid, gamma[idx_c], 0.0)

        extra_logdet = jnp.sum(jnp.log(Ndiag)) + jnp.sum(jnp.log(a))
        return SMWhitener(
            inv_sigma=inv_sigma,
            inv_var=inv_var,
            gamma_toa=gamma_toa,
            idx_c=idx_c,
            valid_f=valid_f,
            extra_logdet=extra_logdet,
            n_epochs=self.n_epochs,
        )

    def generate(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        key: jax.Array,
    ) -> Float[Array, " n_toas"]:
        """Draw a realization: one Gaussian offset per epoch, gathered.

        Same distribution as the basis form's ``U @ (sqrt(w) z)`` (each
        epoch's TOAs share one offset of std ``j_e``), without
        synthesizing the one-hot matrix.
        """
        idx = jnp.asarray(self.epoch_index)
        valid = idx != NO_EPOCH
        z = jax.random.normal(key, shape=(self.n_epochs,))
        offsets = jnp.sqrt(self.ecorr_weights(params)) * z
        return jnp.where(valid, offsets[jnp.clip(idx, 0)], 0.0)


__all__ = ["EcorrKernelNoise", "SMWhitener"]
