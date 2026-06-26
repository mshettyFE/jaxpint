"""NamedVector: shared base for flat, name-indexed parameter pytrees.

:class:`~jaxpint.types.ParameterVector` (per-pulsar timing parameters) and
:class:`~jaxpint.types.GlobalParams` (PTA-level shared parameters) are both
"named vectors": a single dynamic ``values`` leaf indexed by a static ``names``
tuple plus a ``_name_to_index`` lookup. This base captures that common
membership / value-access interface so it lives in one place.

It is a fields-less ``eqx.Module`` mixin (same pattern as
:class:`~jaxpint.components.NoiseComponent`): it declares no dataclass fields, so
subclasses keep full control of their own field order. Concrete subclasses must
provide the attributes the methods below rely on:

* ``values`` -- the single dynamic ``(n,)`` leaf,
* ``names`` -- static ``tuple[str, ...]`` of parameter names,
* ``_name_to_index`` -- static ``dict[str, int]`` name -> index lookup.
"""

from __future__ import annotations

from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float


class NamedVector(eqx.Module):
    """Mixin providing the name-indexed value-access interface (see module docs)."""

    # -- Membership / lookup --

    def __contains__(self, name: str) -> bool:
        """``name in vector`` -- whether the parameter is present."""
        return name in self._name_to_index

    def param_index(self, name: str) -> int:
        """Zero-based index of parameter ``name`` in ``values``."""
        return self._name_to_index[name]

    def param_value(self, name: str) -> Float[Array, ""]:
        """Value of a single parameter. JIT-safe when ``name`` is a static string."""
        return self.values[self._name_to_index[name]]

    def param_values(self, names) -> Float[Array, " k"]:
        """Values of several parameters as a 1-D array, in the given order.

        Plural companion to :meth:`param_value`, convenient for gathering a
        coefficient vector (e.g. ``[DM, DM1, DM2]``). JIT-compatible when
        ``names`` is a static sequence of strings.
        """
        return jnp.array([self.values[self._name_to_index[n]] for n in names])

    def param_value_or(self, name: str | None, default: float = 0.0):
        """Value of a parameter if *name* is not None, otherwise *default*.

        Convenient for optional parameters stored as ``Optional[str]``
        field names on components::

            pbdot = params.param_value_or(self.pbdot_name, 0.0)
        """
        if name is None:
            return default
        return self.values[self._name_to_index[name]]

    # -- Prefix-family queries --

    def names_with_prefix(self, prefix: str) -> tuple[str, ...]:
        """Sorted parameter names starting with ``prefix``.

        For flat prefix families, e.g. ``EFAC`` / ``JUMP`` / ``DMJUMP``.
        """
        return tuple(sorted(n for n in self.names if n.startswith(prefix)))

    def indexed_family(self, base: str, suffix: str = "") -> tuple[int, ...]:
        """Sorted unique integer indices ``i`` for names ``f"{base}{i}{suffix}"``.

        For integer-indexed families whose index is wrapped by a fixed prefix
        and (optionally) a fixed suffix, e.g. ``base="WAVE", suffix="_A"`` over
        ``WAVE1_A, WAVE2_A, ...`` -> ``(1, 2, ...)``. Names that don't carry a
        pure-integer index in that slot are ignored, so it cleanly skips longer
        siblings (e.g. ``base="FD"`` skips ``FDJUMP1``).
        """
        out: set[int] = set()
        end = -len(suffix) if suffix else None
        for name in self.names:
            if name.startswith(base) and name.endswith(suffix):
                try:
                    out.add(int(name[len(base):end]))
                except ValueError:
                    pass
        return tuple(sorted(out))

    def prefix_indices(self, prefix: str) -> tuple[int, ...]:
        """Sorted unique integer suffixes of names matching ``prefix``.

        The suffix-free special case of :meth:`indexed_family`, e.g.
        ``prefix="DMX_"`` over ``DMX_0001, DMX_0002, ...`` -> ``(1, 2, ...)``.
        Names whose suffix is not an integer are ignored.
        """
        return self.indexed_family(prefix)

    # -- Functional updates --

    def with_value(self, name: str, val: ArrayLike) -> Self:
        """Return a copy with one parameter replaced (all other fields preserved)."""
        new_values = self.values.at[self._name_to_index[name]].set(val)
        return eqx.tree_at(lambda v: v.values, self, new_values)

    def with_values(self, values: Float[Array, " n_params"]) -> Self:
        """Return a copy with the entire ``values`` leaf replaced.

        All static metadata (``names``, ``_name_to_index``, and any subclass
        fields) is preserved. Useful for un-flattening a parameter array back
        into a vector of the same shape (e.g. Fisher-matrix reconstruction).
        """
        return eqx.tree_at(lambda v: v.values, self, values)

    @property
    def n_params(self) -> int:
        """Total number of parameters."""
        return len(self.names)
