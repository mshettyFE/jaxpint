"""NoiseModel: aggregates all noise sources into a single Woodbury interface.

::

    C = diag(Ndiag) + U · diag(Phidiag) · Uᵀ

where ``Ndiag`` comes from white noise and ``U`` / ``Phidiag`` are
horizontally concatenated from all correlated noise components.

Parameter-independent bases (red noise, DM noise, ECORR quantization) are
pre-hstacked at construction time into ``_U_static`` so that the JIT
graph sees a single constant basis matrix, regardless of how many
correlated components were passed in. This mirrors the
``CompoundGP``/``WoodburyKernel`` pre-stacking pattern in
:mod:`discovery.matrix`, and prevents per-component basis ops from
multiplying the HLO graph size on every likelihood call.

``_U_static`` is stored as a *numpy* array (host RAM), not a
``jax.Array`` on device — that's the source of truth, used only at
construction time to derive the host-resident stacked basis. The hot
path uses a lazily-built JAX-converted view exposed via
:attr:`NoiseModel._U_static_jax`, a ``functools.cached_property`` that
mirrors :mod:`discovery.matrix.WoodburyKernel`'s ``jnparray()``-in-closure
pattern: the device buffer is created on first access and cached for
the lifetime of the ``NoiseModel`` instance, so MCMC-style repeated
calls amortize the host→device transfer to a one-time cost.
"""

from __future__ import annotations

import functools
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
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

    At construction the correlated components are partitioned into:

    - **static-basis** components (those with :meth:`NoiseComponent.static_basis`
      returning a non-``None`` array) — their bases are hstacked once into
      ``_U_static``.
    - **dynamic-basis** components — their bases are recomputed per call.

    The compiled likelihood thus sees a single pre-stacked constant
    basis (plus, optionally, a few small dynamic bases) instead of one
    constant per component.

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

    # ------------------------------------------------------------------
    # Computed in __post_init__ — see module docstring.
    # ------------------------------------------------------------------
    _U_static: Optional[Float[Array, "n_toas n_static_basis"]] = eqx.field(
        init=False, default=None
    )
    _static_indices: tuple[int, ...] = eqx.field(
        init=False, static=True, default=()
    )
    _dynamic_indices: tuple[int, ...] = eqx.field(
        init=False, static=True, default=()
    )

    def __post_init__(self):
        static_idx: list[int] = []
        dynamic_idx: list[int] = []
        # Bring each component's static basis to host RAM (numpy) before
        # stacking, so the concatenated U_static lives on the host. JAX
        # treats it as a pytree leaf and transfers to the JIT's device
        # on demand — meaning at full-PTA scale only one pulsar's
        # U_static is on GPU at a time, instead of all ``n_pulsars`` of
        # them sitting in device RAM forever.
        static_bases_np: list[np.ndarray] = []
        for i, comp in enumerate(self.correlated):
            sb = comp.static_basis()
            if sb is not None:
                static_idx.append(i)
                static_bases_np.append(np.asarray(sb))
            else:
                dynamic_idx.append(i)

        U_static = (
            np.concatenate(static_bases_np, axis=1)
            if static_bases_np else None
        )

        object.__setattr__(self, "_U_static", U_static)
        object.__setattr__(self, "_static_indices", tuple(static_idx))
        object.__setattr__(self, "_dynamic_indices", tuple(dynamic_idx))

    @functools.cached_property
    def _U_static_jax(self) -> Optional[Float[Array, "n_toas n_static_basis"]]:
        """Lazy device-converted view of ``_U_static``.

        Returns ``None`` if there are no static-basis components.
        Otherwise returns ``jnp.asarray(self._U_static)``, cached on
        ``self.__dict__`` for the lifetime of this ``NoiseModel``. The
        device buffer is created at first access (one host→device
        transfer per pulsar per session) and reused on every subsequent
        ``covariance(...)`` call — see module docstring.

        Note: the cached value lives in ``__dict__`` and is therefore
        invisible to JAX's pytree machinery. Operations that round-trip
        the module through :func:`jax.tree_util.tree_map` /
        :func:`equinox.tree_at` produce a new instance with an empty
        cache; the device buffer is rebuilt on first access of the new
        instance.
        """
        if self._U_static is None:
            return None
        return jnp.asarray(self._U_static)

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

        Phi_blocks: list[Float[Array, " _"]] = []
        for i in self._static_indices:
            _, _, Phi_i = self.correlated[i].covariance(toa_data, params)
            Phi_blocks.append(Phi_i)

        U_dyn: list[Float[Array, "n_toas _"]] = []
        for i in self._dynamic_indices:
            _, U_i, Phi_i = self.correlated[i].covariance(toa_data, params)
            if U_i is not None:
                U_dyn.append(U_i)
                Phi_blocks.append(Phi_i)

        # Hot path: use the cached jax.Array view so the device buffer
        # is reused across calls instead of re-transferred per call.
        if self._U_static is not None and U_dyn:
            U = jnp.concatenate([self._U_static_jax, *U_dyn], axis=1)
        elif self._U_static is not None:
            U = self._U_static_jax
        elif U_dyn:
            U = jnp.concatenate(U_dyn, axis=1) if len(U_dyn) > 1 else U_dyn[0]
        else:
            U = jnp.zeros((toa_data.n_toas, 0))

        Phidiag = (
            jnp.concatenate(Phi_blocks) if Phi_blocks else jnp.zeros(0)
        )

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

    # ------------------------------------------------------------------
    # Component indexing
    # ------------------------------------------------------------------

    @property
    def components(self) -> tuple:
        """All non-None noise components: white_noise, correlated, dm_white_noise."""
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
            raise KeyError(
                f"{key!r} not found. Available components: {names}"
            )
        elif isinstance(key, (int, slice)):
            return self.components[key]
        raise TypeError(
            f"indices must be str, int, or slice, not {type(key).__name__}"
        )
