"""Base component types for JaxPINT timing model modules.

Each timing-model component describes its parameters through **two related but
distinct conventions**, used at two different stages of a component's life:

1. **``PARAMS`` — the static schema (parse time).**
   Every concrete component declares a class-level
   ``PARAMS: ClassVar[tuple[ParamDecl, ...]]`` listing the parameters it models,
   with their type/unit/prefix/aliases (see :class:`ParamDecl`).  This exists
   *before any instance* and is the vocabulary the native ``.par`` parser uses:
   :mod:`jaxpint.par.spec` aggregates every component's ``PARAMS`` to know how to
   parse each parameter and which component a parameter activates (detection).
   It is a ``ClassVar`` (equinox excludes it from the pytree), so it is pure
   static metadata, never a JIT leaf.  Declaring ``PARAMS`` is **required** for a
   concrete component — the aggregator raises if it is missing/empty.

2. **``*_name`` / ``*_names`` fields — the runtime config (post-parse).**
   Instance fields whose names end in ``_name`` (a single parameter name) or
   ``_names`` (a tuple of names) hold the **concrete** parameter names *this*
   configured instance reads from the :class:`~jaxpint.types.ParameterVector` at
   runtime — e.g. ``raj_name="RAJ"`` or ``spin_param_names=("F0","F1","F2")``.
   The model builder fills these in from a parsed model, and they are static
   ``eqx.field`` values so they stay constant inside JIT.  The naming convention
   lets :meth:`PhaseComponent.required_params` (etc.) discover them via
   ``_collect_param_names``.

**How they relate.**
``PARAMS`` is the *template* a component owns (e.g. Spindown owns the ``F``
prefix family + ``PEPOCH``); the ``*_name``/``*_names`` fields are the *concrete,
file-specific expansion* for one model (e.g. this pulsar's ``("F0","F1")``).
The parser turns ``.par`` text into a ``ParameterVector`` using ``PARAMS``; the
builder then sets each component's ``*_name`` fields from that vector.  They are
the same parameters seen at two stages — declaration vs. configured instance —
not redundant copies.

Example::

    class Spindown(PhaseComponent):
        PARAMS = (ParamDecl("F0", unit="Hz", prefix="F"),   # static schema
                  ParamDecl("PEPOCH", kind="mjd"))
        spin_param_names: tuple[str, ...] = eqx.field(static=True)  # runtime config
        pepoch_name: str = eqx.field(static=True, default="PEPOCH")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import equinox as eqx
import jax
from jaxtyping import Array, Float

from jaxpint.types import TOAData, ParameterVector
from jaxpint.types.dual_float import DualFloat


# ---------------------------------------------------------------------------
# Parameter declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamDecl:
    """Static declaration of a parameter a component models.

    Each component class carries a class-level ``PARAMS: tuple[ParamDecl, ...]``
    listing the parameters it consumes.  :mod:`jaxpint.par.spec` aggregates these
    across all components to drive the native ``.par`` parser, so the parameter
    vocabulary is owned by JaxPINT's own model rather than mined from PINT.

    ``PARAMS`` is a plain class attribute (NOT an ``eqx.field``), so it stays
    static metadata and never enters the pytree / JIT trace.  It is the static
    *vocabulary*; the instance ``*_name``/``*_names`` fields are the per-fit
    *config* the builder fills from a parsed model.

    Fields
    ------
    name
        Canonical (PINT) template name, e.g. ``RAJ``, ``F0``, ``DMX_0001``,
        ``EQUAD1``, ``PB``.
    kind
        One of ``angle|mjd|mask|pair|str|bool|int|float``.  Drives parsing.
    unit
        Native unit string; used for angle parsing and the deg->rad / us->s
        coercions.  Documentation-only (and may be ``""``) for plain floats.
    aliases
        True alternate spellings (e.g. ``("RA",)`` for ``RAJ``).
    prefix
        For an indexed/repeatable family, the prefix string (``"F"``, ``"DMX_"``,
        ``"EQUAD"``, ``"JUMP"``); other indices reuse this declaration.
    prefix_aliases
        Alternate prefixes for the family (e.g. ``("T2EQUAD",)`` for ``EQUAD``).
    scale, scale_threshold
        PINT ``unit_scale``: a value above ``scale_threshold`` is multiplied by
        ``scale`` (e.g. ``PBDOT 1.59`` -> ``1.59e-12``).
    frozen_default
        Frozen state when the par line has no fit flag (``False`` for families
        like ``DMX_``/WaveX that default to free).

    Detection note: a parameter activates ("triggers") its component when it is
    *uniquely owned* by exactly one non-binary component; this is derived in
    :mod:`jaxpint.par.spec`, not declared here (binary models are selected by
    the ``BINARY`` line, not by parameter presence).
    """

    name: str
    kind: str = "float"
    unit: str = ""
    aliases: tuple[str, ...] = ()
    prefix: Optional[str] = None
    prefix_aliases: tuple[str, ...] = ()
    scale: Optional[float] = None
    scale_threshold: Optional[float] = None
    frozen_default: bool = True


# ---------------------------------------------------------------------------
# Shared introspection helper
# ---------------------------------------------------------------------------


def _make_component_names(components: tuple) -> tuple[str, ...]:
    """Generate unique names for components from their class names.

    When multiple components share the same class name, they are
    disambiguated with ``_0``, ``_1``, … suffixes.  Components with
    unique class names are left unsuffixed.
    """
    from collections import Counter

    raw = [type(c).__name__ for c in components]
    counts = Counter(raw)
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in raw:
        if counts[name] > 1:
            idx = seen.get(name, 0)
            seen[name] = idx + 1
            result.append(f"{name}_{idx}")
        else:
            result.append(name)
    return tuple(result)


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

    Subclasses implement ``__call__(self, toa_data, params, delay) -> DualFloat``.

    In the timing model, all PhaseComponents see the same total delay
    and their phase contributions are summed.

    Fields that store parameter names must end with ``_name`` (single)
    or ``_names`` (tuple).  This enables :meth:`required_params`.

    Concrete subclasses must declare ``PARAMS`` (the parameters they model);
    the native ``.par`` parser aggregates these (see :mod:`jaxpint.par.spec`).
    """

    PARAMS: ClassVar[tuple[ParamDecl, ...]] = ()

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> DualFloat:
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
        DualFloat
            Phase contribution in cycles (int + frac split).

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
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

    Concrete subclasses must declare ``PARAMS`` (the parameters they model);
    the native ``.par`` parser aggregates these (see :mod:`jaxpint.par.spec`).
    """

    PARAMS: ClassVar[tuple[ParamDecl, ...]] = ()

    def covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_basis"],
        Float[Array, " n_basis"],
    ]:
        """Return the Woodbury decomposition of this component's covariance.

        Returns ``(Ndiag, U, Phidiag)`` such that::

            C = diag(Ndiag) + U @ diag(Phidiag) @ Uᵀ

        Both ``U`` and ``Phidiag`` are always arrays.  Components without
        a low-rank contribution return zero-width arrays of shape
        ``(n_toas, 0)`` and ``(0,)`` respectively.  Components without a
        diagonal contribution return ``jnp.zeros(n_toas)`` for ``Ndiag``.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times, frequencies, positions, etc.).
        params : ParameterVector
            Timing model parameters.

        Returns
        -------
        Ndiag : (n_toas,)
            Diagonal variance contribution.
        U : (n_toas, n_basis)
            Basis matrix for low-rank contribution; ``n_basis`` may be 0.
        Phidiag : (n_basis,)
            Basis weights for low-rank contribution; ``n_basis`` may be 0.

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
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

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
        """
        raise NotImplementedError

    def required_params(self) -> tuple[str, ...]:
        """Parameter names this component reads from the ParameterVector.

        Discovered by convention: fields ending in ``_name`` (single
        parameter) or ``_names`` (tuple of parameters).  New component
        fields that hold parameter names **must** follow this convention.
        """
        return _collect_param_names(self)

    # ------------------------------------------------------------------
    # Optional pre-stacking hooks
    #
    # Components whose basis ``U`` does not depend on any traced
    # parameter override :meth:`static_basis` so that
    # :class:`~jaxpint.noise.NoiseModel` can hstack the bases at
    # construction time. This avoids tracing per-component basis
    # operations on every likelihood call (the discovery-style
    # ``CompoundGP`` pattern).
    # ------------------------------------------------------------------

    def static_basis(self) -> Optional[Float[Array, "n_toas n_basis"]]:
        """Return ``U`` if it is parameter-independent, else ``None``."""
        return None


class DelayComponent(eqx.Module):
    """Base class for components that contribute to signal delay.

    Subclasses implement ``__call__(self, toa_data, params, delay) -> Array``.

    In the timing model, DelayComponents are applied sequentially:
    each component sees the accumulated delay from prior components.

    Fields that store parameter names must end with ``_name`` (single)
    or ``_names`` (tuple).  This enables :meth:`required_params`.

    Concrete subclasses must declare ``PARAMS`` (the parameters they model);
    the native ``.par`` parser aggregates these (see :mod:`jaxpint.par.spec`).
    """

    PARAMS: ClassVar[tuple[ParamDecl, ...]] = ()

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

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
        """
        raise NotImplementedError

    def required_params(self) -> tuple[str, ...]:
        """Parameter names this component reads from the ParameterVector.

        Discovered by convention: fields ending in ``_name`` (single
        parameter) or ``_names`` (tuple of parameters).  New component
        fields that hold parameter names **must** follow this convention.
        """
        return _collect_param_names(self)


class DispersionDelayComponent(DelayComponent):
    """Base class for delay components that contribute to dispersion measure.

    Subclasses must implement :meth:`compute_dm` returning the DM
    contribution in pc/cm³, in addition to ``__call__`` (inherited from
    :class:`DelayComponent`) which returns delay in seconds.  The timing
    model uses ``compute_dm`` to evaluate the total model DM for wideband
    fitting.
    """

    def compute_dm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Return this component's DM contribution in pc/cm³.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data.
        params : ParameterVector
            Timing model parameters.
        delay : (n_toas,)
            Accumulated signal delay in seconds from prior delay components.

        Returns
        -------
        (n_toas,)
            DM contribution in pc/cm³.

        Raises
        ------
        NotImplementedError
            Must be overridden by subclasses.
        """
        raise NotImplementedError


class BinaryDelayComponent(DelayComponent):
    """Marker base for binary orbital-delay components.

    Identifies the components whose ``__call__`` is the binary orbital delay
    so :meth:`jaxpint.model.TimingModel.compute_delay_to_binary` knows where to
    stop (mirroring PINT's ``delay(cutoff=<binary>)``).  Membership is by *type*
    — inheriting this is the single source of truth, so a new binary model is
    recognized automatically instead of having to be added to a hand-maintained
    roster.  Pure marker: it adds nothing to :class:`DelayComponent`.
    """
