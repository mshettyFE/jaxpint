"""Global shared parameters for PTA likelihood evaluation.

GlobalParams holds parameters shared across all pulsars (e.g., CW source
properties, GWB amplitude/spectral index). Built up incrementally by
signal injectors via the add_params() method.
"""

from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float


class GlobalParams(eqx.Module):
    """Shared parameter container for PTA-level signals.

    An Equinox module where ``values`` is the only dynamic (traced) leaf.
    Static metadata (names, index mapping) is fixed at construction time.

    Build incrementally via :meth:`add_params` or the builder pattern::

        gp = GlobalParams.empty()
        for inj in signal_injectors:
            gp = inj.register_params(gp)
    """

    values: Float[Array, " n_global"]
    names: tuple[str, ...] = eqx.field(static=True)
    _name_to_index: dict[str, int] = eqx.field(static=True)

    @staticmethod
    def empty() -> GlobalParams:
        """Create an empty GlobalParams with no parameters."""
        return GlobalParams(jnp.array([]), (), {})

    def add_params(self, names: list[str], values: list[float]) -> GlobalParams:
        """Append new parameters, returning a new GlobalParams.

        Parameters
        ----------
        names : list[str]
            Parameter names to add.
        values : list[float]
            Initial values for each parameter (same length as *names*).

        Returns
        -------
        GlobalParams
            New instance with the appended parameters.

        Raises
        ------
        ValueError
            If *names* and *values* have different lengths, or if any name
            is already present (prevents silent overwrites from duplicate
            injectors or prefix collisions).
        """
        if len(names) != len(values):
            raise ValueError(
                f"names and values must have the same length, "
                f"got {len(names)} names and {len(values)} values."
            )
        duplicates = set(names) & set(self.names)
        if duplicates:
            raise ValueError(
                f"Parameters already registered: {duplicates}. "
                f"Use distinct prefixes for each signal source."
            )
        new_names = self.names + tuple(names)
        offset = len(self.names)
        new_index = {
            **self._name_to_index,
            **{n: offset + i for i, n in enumerate(names)},
        }
        new_values = jnp.concatenate([self.values, jnp.array(values)])
        return GlobalParams(new_values, new_names, new_index)

    def param_value(self, name: str) -> Float[Array, ""]:
        """Look up a single parameter value by name."""
        return self.values[self._name_to_index[name]]

    def with_value(self, name: str, val: float) -> GlobalParams:
        """Return a copy with one parameter replaced."""
        idx = self._name_to_index[name]
        new_values = self.values.at[idx].set(val)
        return GlobalParams(new_values, self.names, self._name_to_index)

    @property
    def n_params(self) -> int:
        """Total number of parameters."""
        return len(self.names)
