"""Base component types for JaxPINT timing model modules.

Component fields that store parameter names must follow the naming
convention: end with ``_name`` for a single parameter name, or
``_names`` for a tuple of parameter names.  This enables
:meth:`required_params` to discover them automatically.
"""

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array, Float

from jaxpint.types import TOAData, ParameterVector
from jaxpint.phase_result import PhaseResult


# ---------------------------------------------------------------------------
# Shared introspection helper
# ---------------------------------------------------------------------------

def _collect_param_names(module) -> tuple[str, ...]:
    """Collect parameter names from fields following the naming convention.

    Fields ending in ``_name`` holding a ``str`` value, and fields ending
    in ``_names`` holding a ``tuple`` of strings, are treated as parameter
    name references.  ``None`` values (optional parameters not in use) are
    skipped.
    """
    names = []
    for field_name, val in vars(module).items():
        if field_name.endswith("_name") and isinstance(val, str):
            names.append(val)
        elif field_name.endswith("_names") and isinstance(val, tuple):
            names.extend(v for v in val if isinstance(v, str))
    return tuple(sorted(set(names)))


class PhaseComponent(eqx.Module):
    """Base class for components that contribute to pulse phase.

    Subclasses implement ``__call__(self, toa_data, params, delay) -> PhaseResult``.

    In the timing model, all PhaseComponents see the same total delay
    and their phase contributions are summed.

    Fields that store parameter names must end with ``_name`` (single)
    or ``_names`` (tuple).  This enables :meth:`required_params`.
    """

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> PhaseResult:
        """Compute this component's phase contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times, frequencies, positions, etc.).
        params : ParameterVector
            Timing model parameters.
        delay : (n_toas,)
            Accumulated signal delay in seconds from all delay components.

        Returns
        -------
        PhaseResult
            Phase contribution in cycles (int + frac split).
        """
        raise NotImplementedError

    def required_params(self) -> tuple[str, ...]:
        """Parameter names this component reads from the ParameterVector.

        Discovered by convention: fields ending in ``_name`` (single
        parameter) or ``_names`` (tuple of parameters).  New component
        fields that hold parameter names **must** follow this convention.
        """
        return _collect_param_names(self)


class NoiseComponent(eqx.Module):
    """Base class for stochastic noise sources.

    Every noise source decomposes its covariance as::

        C = diag(Ndiag) + U @ diag(Phidiag) @ Uᵀ

    Subclasses must implement:

    - ``covariance`` — returns the ``(Ndiag, U, Phidiag)`` triple.
    - ``generate``   — draws a random noise realization.

    The fitter combines multiple ``NoiseComponent`` instances by summing
    their diagonal contributions and horizontally concatenating their
    basis matrices and weight vectors.

    Fields that store parameter names must end with ``_name`` (single)
    or ``_names`` (tuple).  This enables :meth:`required_params`.
    """

    def covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_basis"] | None,
        Float[Array, " n_basis"] | None,
    ]:
        """Return the Woodbury decomposition of this component's covariance.

        Returns ``(Ndiag, U, Phidiag)`` such that::

            C = diag(Ndiag) + U @ diag(Phidiag) @ Uᵀ

        For purely diagonal noise, return ``(Ndiag, None, None)``.
        For purely low-rank noise, return ``(zeros, U, Phidiag)``.

        Returns
        -------
        Ndiag : (n_toas,)
            Diagonal variance contribution.
        U : (n_toas, n_basis) or None
            Basis matrix for low-rank contribution.
        Phidiag : (n_basis,) or None
            Basis weights for low-rank contribution.
        """
        raise NotImplementedError

    def generate(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        key: jax.Array,
    ) -> Float[Array, " n_toas"]:
        """Draw a random noise realization.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data.
        params : ParameterVector
            Timing model parameters (including noise parameter values).
        key : JAX PRNG key
            Random key for reproducible sampling.

        Returns
        -------
        (n_toas,)
            Noise delays in seconds.
        """
        raise NotImplementedError

    def required_params(self) -> tuple[str, ...]:
        """Parameter names this component reads from the ParameterVector.

        Discovered by convention: fields ending in ``_name`` (single
        parameter) or ``_names`` (tuple of parameters).  New component
        fields that hold parameter names **must** follow this convention.
        """
        return _collect_param_names(self)


class DelayComponent(eqx.Module):
    """Base class for components that contribute to signal delay.

    Subclasses implement ``__call__(self, toa_data, params, delay) -> Array``.

    In the timing model, DelayComponents are applied sequentially:
    each component sees the accumulated delay from prior components.

    Fields that store parameter names must end with ``_name`` (single)
    or ``_names`` (tuple).  This enables :meth:`required_params`.
    """

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute this component's delay contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times, frequencies, positions, etc.).
        params : ParameterVector
            Timing model parameters.
        delay : (n_toas,)
            Accumulated signal delay in seconds from prior delay components.

        Returns
        -------
        (n_toas,)
            Delay contribution in seconds.
        """
        raise NotImplementedError

    def required_params(self) -> tuple[str, ...]:
        """Parameter names this component reads from the ParameterVector.

        Discovered by convention: fields ending in ``_name`` (single
        parameter) or ``_names`` (tuple of parameters).  New component
        fields that hold parameter names **must** follow this convention.
        """
        return _collect_param_names(self)
