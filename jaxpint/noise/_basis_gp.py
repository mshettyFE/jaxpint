"""Basis-neutral base for low-rank GP noise components.

Any component modeling noise as ``C = U · diag(w) · Uᵀ`` — Fourier red/DM/
chromatic/solar-wind noise, ECORR's epoch-quantization basis, future
time-node (tent) or user-supplied bases — shares the same machinery: a
precomputed host-side basis, a lazy device view, the Woodbury
``(Ndiag, U, Phidiag)`` covariance triple, and realization drawing.
:class:`_BasisGPNoise` captures that machinery and stays agnostic to what
the columns *are*; subclasses supply them via :meth:`_host_columns` and
map their hyperparameters to the per-column prior diagonal via
:meth:`psd_weights`.

Static vs dynamic basis
-----------------------
A component is "static" iff its basis is fixed at build time (red / DM —
the DM ``(1400/f)²`` factor is pre-baked by the bridge; ECORR's epoch
matrix). Such components override
:meth:`~jaxpint.components.NoiseComponent.static_basis` to return their
basis, so :class:`~jaxpint.noise.NoiseModel` can pre-stack it once.
Components whose basis depends on fitted parameters (chromatic
``(fref/f)^α``, solar-wind geometry) leave ``static_basis`` at the
default (``None`` → dynamic) and override :meth:`_BasisGPNoise._basis`
instead. Defaulting to dynamic fails safe: a dynamic basis is always
correct, just not pre-stackable.

Dense node priors
-----------------
The engine contract is a *diagonal* prior. Components whose natural prior
couples basis columns (squared-exponential / tf-quantization kernels on
time-node bases) absorb the prior's Cholesky factor into the basis
in-trace via :meth:`_BasisGPNoise._absorb_dense_prior` — an exact
reparametrization (``U K Uᵀ = (U L)(U L)ᵀ``) that never forms ``K⁻¹``, so
near-singular kernels stay well-conditioned.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent
from jaxpint.types import TOAData, ParameterVector


def chromatic_row_scale(freq_mhz, alpha, fref: float = 1400.0):
    """Per-TOA chromatic column weights ``(fref / f_obs)^alpha``."""
    return (fref / freq_mhz) ** alpha


class _BasisGPNoise(NoiseComponent):
    """Base for low-rank basis-GP noise (``C = U · diag(w) · Uᵀ``).

    Basis-neutral: subclasses supply the columns via :meth:`_host_columns`
    (host numpy, source of truth) and the prior diagonal via
    :meth:`psd_weights`; components with parameter-dependent per-TOA
    column scaling additionally override :meth:`_basis`.
    """

    # -- subclass hooks --------------------------------------------------

    def _host_columns(
        self,
    ) -> Float[np.ndarray, "n_toas n_basis"] | Float[Array, "n_toas n_basis"]:
        """Source-of-truth basis columns, shape (n_toas, n_basis).

        The mandatory basis hook (re: derived classes need to define this!).
        The point is to separate the decleration of the bases from the caching
        mechanism that prevents memory blowup (see :attr:`_columns_jax`).

        Host numpy on persistent instances (``__post_init__`` coercion).  On a
        tree-reconstructed instance inside a jit trace the field is a tracer
        and MUST be passed through untouched — implementations return the
        field directly, never ``np.asarray`` it (that raises
        ``TracerArrayConversionError``); :attr:`_columns_jax` handles both.
        """
        raise NotImplementedError

    def psd_weights(self, params: ParameterVector) -> Float[Array, " n_basis"]:
        """Prior weights, one per basis column (the diagonal of ``diag(w)``).

        The mandatory prior hook: each subclass maps its hyperparameters to
        the per-column weights (e.g. sin/cos pairs sharing a PSD value, or
        ECORR² per epoch).
        """
        raise NotImplementedError

    def _basis(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, "n_toas n_basis"]:
        """The basis used in :meth:`covariance` / :meth:`generate`.

        Defaults to the fixed (pre-computed) columns. Override for
        parameter-dependent per-TOA column scaling (chromatic
        ``(fref/f)^α``, solar-wind geometry).
        """
        return self._columns_jax

    def basis_width(self) -> int:
        """Number of basis columns, as cheaply as the subclass allows.

        Default reads the host columns' shape (free for stored bases);
        subclasses whose ``_host_columns`` *synthesizes* a matrix
        (indexed ECORR) override this so width queries — e.g. the
        summary's ``n_basis`` line — never materialize a dense basis.
        """
        return int(self._host_columns().shape[1])

    def basis_at(
        self,
        times_seconds: Float[Array, " n_times"],
        params: ParameterVector,
    ) -> Optional[Float[Array, "n_times n_basis"]]:
        """Basis evaluated at arbitrary times, or ``None`` if impossible.

        Capability hook for conditional-GP reconstruction off the TOA
        grid. Analytic bases (Fourier) and interpolating bases (time
        nodes) can override this; indicator bases (epoch quantization)
        cannot and keep the ``None`` default, which consumers must treat
        as "on-grid evaluation only". Not yet wired into the conditional
        machinery (phase 2 of ``Plans/basis_neutral_gp_plan.md``).
        """
        return None

    # -- shared machinery --------------------------------------------------

    def _coerce_host_field(self, name: str) -> None:
        """``__post_init__`` helper: coerce field *name* to host numpy in place.

        Keeps the (potentially large) basis off-device until first use;
        subclasses call this on their basis field during ``__post_init__``
        (equinox modules stay assignable until construction completes).
        """
        val = getattr(self, name)
        if not isinstance(val, np.ndarray):
            object.__setattr__(self, name, np.asarray(val))

    @property
    def _columns_jax(self) -> Float[Array, "n_toas n_basis"]:
        """Lazy device-converted view of :meth:`_host_columns` (cached per instance).

        Cached manually instead of via ``functools.cached_property``: inside
        a jit trace ``jnp.asarray`` returns a tracer, and caching a tracer on
        the (persistent) host instance leaks it into later traces. Only
        concrete arrays are cached; traced conversions are recomputed per
        trace.
        """
        cached = self.__dict__.get("_columns_jax_cache")
        if cached is None:
            cached = jnp.asarray(self._host_columns())
            if not isinstance(cached, jax.core.Tracer):  # pyright: ignore[reportAttributeAccessIssue]
                self.__dict__["_columns_jax_cache"] = cached  # pyright: ignore[reportIndexIssue]
        return cached

    def covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_basis"],
        Float[Array, " n_basis"],
    ]:
        """Woodbury ``(Ndiag, U, Phidiag)`` triple; purely low-rank (``Ndiag = 0``)."""
        Ndiag = jnp.zeros(toa_data.n_toas)
        return Ndiag, self._basis(toa_data, params), self.psd_weights(params)

    def generate(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        key: jax.Array,
    ) -> Float[Array, " n_toas"]:
        r"""Draw a realization: ``U \cdot (sqrt(w) \otimes z)`` with ``z ~ N(0, I)``."""
        weights = self.psd_weights(params)
        basis = self._basis(toa_data, params)
        a = jax.random.normal(key, shape=(basis.shape[1],))
        return basis @ (jnp.sqrt(weights) * a)

    # -- dense-prior helper (unused for now) ----------------------------

    @staticmethod
    def _absorb_dense_prior(
        U: Float[Array, "n_toas n_basis"],
        K_chol: Float[Array, "n_basis n_basis"],
    ) -> tuple[Float[Array, "n_toas n_basis"], Float[Array, " n_basis"]]:
        """Absorb a dense prior's Cholesky factor: ``(U, K) -> (U·L, 1)``.

        Exact: ``(U L)(U L)ᵀ = U K Uᵀ`` for ``K = L Lᵀ``. The square-root
        form never inverts ``K``, so near-singular kernels (long
        length-scale squared-exponential) stay well-conditioned — columns
        of ``U L`` shrink instead of ``K⁻¹`` blowing up. Callers factor
        ``K`` themselves (adding jitter as needed) because kernel
        construction is component-specific.
        """
        return U @ K_chol, jnp.ones(K_chol.shape[1])


__all__ = ["_BasisGPNoise", "chromatic_row_scale"]
