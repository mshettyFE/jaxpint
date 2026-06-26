"""Global shared parameters for PTA likelihood evaluation.

GlobalParams holds parameters shared across all pulsars (e.g., CW source
properties, GWB amplitude/spectral index), so they are not duplicated on every
pulsar's :class:`~jaxpint.types.ParameterVector`. Built up incrementally by
signal injectors via the :meth:`GlobalParams.add_params` method.

It is a foundational named-vector pytree -- a sibling of
:class:`~jaxpint.types.ParameterVector` -- hence its home in ``types``.
"""

from __future__ import annotations

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.types.named_vector import NamedVector


class GlobalParams(NamedVector):
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

    # param_value / param_values / param_index / with_value / n_params and the
    # ``name in gp`` membership check are inherited from NamedVector.
