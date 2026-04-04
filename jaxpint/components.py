"""Base component types for JaxPINT timing model modules."""

from __future__ import annotations

import equinox as eqx
import jax
from jaxtyping import Array, Float

from jaxpint.types import TOAData, ParameterVector
from jaxpint.phase_result import PhaseResult


class PhaseComponent(eqx.Module):
    """Base class for components that contribute to pulse phase.

    Subclasses implement ``__call__(self, toa_data, params, delay) -> PhaseResult``.

    In the timing model, all PhaseComponents see the same total delay
    and their phase contributions are summed.
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


class DelayComponent(eqx.Module):
    """Base class for components that contribute to signal delay.

    Subclasses implement ``__call__(self, toa_data, params, delay) -> Array``.

    In the timing model, DelayComponents are applied sequentially:
    each component sees the accumulated delay from prior components.
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
