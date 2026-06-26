"""Injector ABCs: the contract between the PTA likelihood and signal models.

A *signal injector* contributes shared PTA parameters and either a deterministic
delay, a per-pulsar stochastic covariance, or -- for correlated signals -- the
ingredients of a cross-pulsar covariance.  These abstract base classes are the
interface that :func:`jaxpint.pta.pta_logL` consumes and that the concrete
implementations in :mod:`jaxpint.pta.signals` provide.

They live in their own leaf module (depending only on the core data types) so
both the likelihood engine and the signal implementations can import the
contract without either side depending on the other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from jaxtyping import Array, Float

from jaxpint.types import GlobalParams, ParameterVector, TOAData


# ---------------------------------------------------------------------------
# Signal injector ABC
# ---------------------------------------------------------------------------


class SignalInjector(ABC):
    """Abstract base class for PTA signal components.

    Each injector:

    1. Registers its own parameters into :class:`GlobalParams` via
       :meth:`register_params` (**required** — abstract).
    2. Produces delay arrays and/or covariance ``(U, Phi)`` tuples per
       pulsar via :meth:`delay` / :meth:`covariance` (optional —
       default implementations return ``None``).

    Subclasses must implement :meth:`register_params`.  Override
    :meth:`delay` for deterministic signals (e.g. CW) and/or
    :meth:`covariance` for stochastic signals (e.g. GWB).

    :func:`pta_logL` is agnostic to the signal type.
    """

    @abstractmethod
    def register_params(self, global_params: GlobalParams) -> GlobalParams:
        """Append this signal's parameters to *global_params*.

        Parameters
        ----------
        global_params : GlobalParams
            Mutable accumulator of shared PTA parameters.

        Returns
        -------
        GlobalParams
            Updated copy with this signal's parameters appended.
        """
        ...

    def delay(
        self,
        p: int,
        toa_data: TOAData,
        pulsar_params: ParameterVector,
        global_params: GlobalParams,
    ) -> Optional[Float[Array, " n_toas"]]:
        """Return deterministic delay for pulsar *p*, or ``None``.

        Override for deterministic signals.  The default returns ``None``
        (no delay contribution).

        Parameters
        ----------
        p : int
            Pulsar index within the PTA.
        toa_data : TOAData
            Pulse time-of-arrival data for pulsar *p*.
        pulsar_params : ParameterVector
            Timing and noise parameters for pulsar *p*.
        global_params : GlobalParams
            Shared PTA parameters (CW source properties, GWB spectrum, etc.).

        Returns
        -------
        (n_toas,) array or None
            Deterministic timing delay in seconds, or ``None`` if this
            injector does not contribute a delay.
        """
        return None

    def covariance(
        self,
        p: int,
        toa_data: TOAData,
        pulsar_params: ParameterVector,
        global_params: GlobalParams,
    ) -> Optional[tuple[Float[Array, "n_toas n_basis"], Float[Array, " n_basis"]]]:
        """Return ``(U, Phi)`` covariance contribution for pulsar *p*, or ``None``.

        Override for stochastic signals.  The default returns ``None``
        (no covariance contribution).

        Parameters
        ----------
        p : int
            Pulsar index within the PTA.
        toa_data : TOAData
            Pulse time-of-arrival data for pulsar *p*.
        pulsar_params : ParameterVector
            Timing and noise parameters for pulsar *p*.
        global_params : GlobalParams
            Shared PTA parameters (CW source properties, GWB spectrum, etc.).

        Returns
        -------
        tuple of ((n_toas, n_basis) array, (n_basis,) array) or None
            Design matrix ``U`` and diagonal PSD vector ``Phi``, or
            ``None`` if this injector does not contribute covariance.
        """
        return None


# ---------------------------------------------------------------------------
# Correlated signal injector ABC
# ---------------------------------------------------------------------------


class CorrelatedSignalInjector(ABC):
    """Abstract base class for cross-pulsar correlated signal components.

    Unlike :class:`SignalInjector`, which produces per-pulsar covariance
    contributions, a ``CorrelatedSignalInjector`` provides the ingredients
    to build a PTA-wide covariance with inter-pulsar correlations: a
    per-pulsar Fourier basis, a global PSD vector, and an overlap reduction
    function (ORF) matrix coupling pulsar pairs.
    """

    @abstractmethod
    def register_params(self, global_params: GlobalParams) -> GlobalParams:
        """Append this signal's parameters to *global_params*.

        Parameters
        ----------
        global_params : GlobalParams
            Accumulator of shared PTA parameters.

        Returns
        -------
        GlobalParams
            Updated copy with this signal's parameters appended.
        """
        ...

    @abstractmethod
    def get_fourier_basis(
        self,
        toa_data: TOAData,
    ) -> Float[Array, "n_toas n_basis"]:
        """Return the Fourier design matrix for a single pulsar.

        Parameters
        ----------
        toa_data : TOAData
            Pulse time-of-arrival data for one pulsar.

        Returns
        -------
        F : (n_toas, n_basis) array
            Fourier design matrix (sin/cos columns).
        """
        ...

    @abstractmethod
    def get_psd(
        self,
        global_params: GlobalParams,
    ) -> Float[Array, " n_basis"]:
        """Return the GWB power spectral density vector.

        Parameters
        ----------
        global_params : GlobalParams
            Shared PTA parameters (amplitude, spectral index, etc.).

        Returns
        -------
        S : (n_basis,) array
            PSD values for each Fourier basis function (sin and cos each
            get the same value for their frequency).
        """
        ...

    @abstractmethod
    def get_orf_matrix(self) -> Float[Array, "n_psr n_psr"]:
        """Return the overlap reduction function matrix.

        The matrix must be invertible (full rank). Rank-deficient ORFs
        such as the monopole (all ones) are not supported by the
        two-tier Woodbury scheme.

        Returns
        -------
        Gamma : (n_psr, n_psr) array
            Symmetric, positive-definite ORF matrix. ``Gamma[a, b]``
            is the correlation coefficient between pulsars *a* and *b*.
        """
        ...


__all__ = ["SignalInjector", "CorrelatedSignalInjector"]
