r"""Time-node ("tent"/linear interpolation) GP noise:

The process is a GP on the linear-interpolation basis of
:func:`jaxpint.utils.build_linear_interp_basis`::

    C = U · diag(sigma²) · Uᵀ,   sigma = 10^{log10_sigma}

i.e. independent, identically-distributed node offsets (in seconds) linearly
interpolated to the TOAs.  With ``chrom_idx_name`` set, the basis rows are scaled by
``(f_ref / f_obs)^\alpha`` per TOA with a *sampled* chromatic index , turning the process chromatic
(DM-like at \alpha = 2, scattering-like at \alpha = 4).

TODO:Dense node priors (squared-exponential kernels between nodes) are a later
consumer of :meth:`~jaxpint.noise._basis_gp._BasisGPNoise._absorb_dense_prior`;
this component is deliberately the diagonal-prior first step .

Degeneracy with the timing model
--------------------------------
The tent basis spans (approximately) constant and linear — and, densely
noded, low-order polynomial — functions of time, which are degenerate with
phase offset / F0 / F1.
JaxPINT relies on analytic timing-model marginalization to break the degeneracy
(:func:`jaxpint.bayes.marginal.marginalize_single_pulsar`). The QR
Woodbury path is built for exactly such collinear blocks.  The degeneracy
is *expected*, not a bug.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jaxpint.noise._basis_gp import _BasisGPNoise, chromatic_row_scale
from jaxpint.types import TOAData, ParameterVector

# This component is constructed programmatically (via __init__ or
# ``from_times``) and declares no ``PARAMS``: there is no ``.par``
# convention for it yet (same precedent as FreeSpectrumNoise).


class TimeNodeGPNoise(_BasisGPNoise):
    """GP on a linear-interpolation (tent) time basis with an iid node prior.

    Parameters
    ----------
    interp_basis : (n_toas, n_nodes)
        Tent basis from :func:`jaxpint.utils.build_linear_interp_basis`
        (columns already pruned to supported nodes).
    node_times : (n_nodes,)
        Node times in seconds for the kept columns (metadata: off-grid
        evaluation and summaries; the likelihood never reads it).
    sigma_name : str
        Parameter name holding ``log10_sigma`` — the log10 RMS node
        offset in seconds.
    chrom_idx_name : str, optional
        Parameter name for a sampled chromatic index \alpha.  When set, the
        basis is scaled by ``(f_ref / f_obs)^\alpha`` per TOA at evaluation
        time and the component becomes dynamic (not pre-stackable).
        Default ``None`` — achromatic, static basis.
    fref : float
        Reference radio frequency in MHz (default 1400.0); only read
        when ``chrom_idx_name`` is set.
    """

    # Constructors may pass jax arrays; __post_init__ coerces to host
    # numpy (the source of truth) -- the unions reflect both stages.
    interp_basis: Float[Array, "n_toas n_nodes"] | Float[np.ndarray, "n_toas n_nodes"]
    node_times: Float[Array, " n_nodes"] | Float[np.ndarray, " n_nodes"]
    sigma_name: str = eqx.field(static=True)
    chrom_idx_name: Optional[str] = eqx.field(static=True, default=None)
    fref: float = eqx.field(static=True, default=1400.0)

    def __post_init__(self):
        # Host numpy is the source of truth; the device view is built lazily
        # by _BasisGPNoise._columns_jax. See that module's docstring.
        self._coerce_host_field("interp_basis")
        self._coerce_host_field("node_times")

    @classmethod
    def from_times(
        cls,
        times_s: np.ndarray,
        *,
        sigma_name: str,
        dt: float = 30.0 * 86400.0,
        node_times_s: Optional[np.ndarray] = None,
        chrom_idx_name: Optional[str] = None,
        fref: float = 1400.0,
    ) -> "TimeNodeGPNoise":
        """Build basis and component in one step from TOA times (seconds)."""
        from jaxpint.utils import build_linear_interp_basis

        U, nodes = build_linear_interp_basis(times_s, dt=dt, node_times_s=node_times_s)
        return cls(
            interp_basis=U,
            node_times=nodes,
            sigma_name=sigma_name,
            chrom_idx_name=chrom_idx_name,
            fref=fref,
        )

    # -- _BasisGPNoise hooks -----------------------------------------------

    def _host_columns(
        self,
    ) -> Float[np.ndarray, "n_toas n_nodes"] | Float[Array, "n_toas n_nodes"]:
        # Pass-through, never np.asarray: inside a jit trace of a
        # reconstructed instance this field is a tracer (see base docstring).
        return self.interp_basis

    def psd_weights(self, params: ParameterVector) -> Float[Array, " n_nodes"]:
        """iid node prior: ``sigma² = 10^(2·log10_sigma)`` per kept node."""
        sigma = 10.0 ** params.param_value(self.sigma_name)
        return jnp.ones(self.interp_basis.shape[1]) * sigma**2

    def _basis(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, "n_toas n_nodes"]:
        """Tent basis, chromatically scaled per TOA when α is sampled."""
        if self.chrom_idx_name is None:
            return self._columns_jax
        alpha = params.param_value(self.chrom_idx_name)
        D = chromatic_row_scale(toa_data.freq, alpha, self.fref)  # (n_toas,)
        return self._columns_jax * D[:, None]

    def static_basis(
        self,
    ) -> Optional[Float[np.ndarray, "n_toas n_nodes"] | Float[Array, "n_toas n_nodes"]]:
        # Achromatic tents are fixed at build time -> pre-stackable; a
        # sampled chromatic index makes the basis parameter-dependent.
        return None if self.chrom_idx_name is not None else self._host_columns()

    def basis_at(
        self,
        times_seconds: Float[Array, " n_times"],
        params: ParameterVector,
    ) -> Optional[Float[Array, "n_times n_nodes"]]:
        """Tent weights at arbitrary times (achromatic only for now).

        Evaluates on the *kept* node grid, treating consecutive kept nodes
        as intervals: on TOA times this agrees with ``interp_basis`` (a
        training TOA never lies in a pruned region), while times inside a
        pruned gap interpolate across it — the honest behavior for a basis
        with no support there.  Times outside the node span get zero
        weight.  Chromatic mode returns ``None``: off-grid times carry no
        radio frequency, so the per-TOA scaling is undefined (purely func signature problem).
        """
        if self.chrom_idx_name is not None:
            return None
        x = jnp.asarray(self.node_times)
        n_nodes = self.interp_basis.shape[1]
        t = jnp.asarray(times_seconds)
        i = jnp.clip(jnp.searchsorted(x, t, side="right") - 1, 0, n_nodes - 2)
        left, right = x[i], x[i + 1]
        w_right = (t - left) / (right - left)
        inside = (t >= x[0]) & (t <= x[-1])
        rows = jnp.arange(t.shape[0])
        U = jnp.zeros((t.shape[0], n_nodes))
        U = U.at[rows, i].add(jnp.where(inside, 1.0 - w_right, 0.0))
        U = U.at[rows, i + 1].add(jnp.where(inside, w_right, 0.0))
        return U


__all__ = ["TimeNodeGPNoise"]
