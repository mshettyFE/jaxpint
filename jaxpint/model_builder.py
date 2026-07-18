"""Model builder: ParResult → TimingModel + NoiseModel.

Constructs JaxPINT timing and noise components from a
:class:`~jaxpint.par.result.ParResult`.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from jaxpint.par.registry import Component
from jaxpint.par.registry_table import PRIORITY

from jaxpint.par.result import ParResult
from jaxpint.types import TOAData

# BuildContext and the parse-result helpers live in a neutral module so component
# ``build`` methods can reference them without importing this module (a cycle).
# The model builder still assembles the context and resolves astrometry names, so
# it imports the two helpers ``_resolve_astrometry`` needs (aliased to their
# historical private names).
from jaxpint._build_context import (
    BuildContext,
    epoch_or_pepoch as _epoch_or_pepoch,
    param_is_set as _param_is_set,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_astrometry(par: ParResult):
    """Resolve astrometry parameter names once, before the build loop.

    Returns ``(raj, decj, pmra, pmdec, posepoch, obliquity_arcsec)``.  The frame
    (equatorial vs ecliptic) is chosen from the detected component set.  These
    names are read by the astrometry, Shapiro, solar-wind and binary builders,
    so resolving them up front makes the result independent of build order
    (previously the astrometry arms mutated shared state later arms relied on).
    """
    cs = par.component_set
    raj, decj = "RAJ", "DECJ"
    pmra = pmdec = posepoch = None
    obliquity_arcsec = None

    if Component.ASTROMETRY_ECLIPTIC in cs:
        from jaxpint.constants import OBLIQUITY_ARCSEC

        raj, decj = "ELONG", "ELAT"
        if _param_is_set(par, "PMELONG"):
            pmra = "PMELONG"
        if _param_is_set(par, "PMELAT"):
            pmdec = "PMELAT"
        ecl_name = par.metadata.get("ECL", "IERS2010")
        obliquity_arcsec = OBLIQUITY_ARCSEC[ecl_name]
    elif Component.ASTROMETRY_EQUATORIAL in cs:
        if _param_is_set(par, "PMRA"):
            pmra = "PMRA"
        if _param_is_set(par, "PMDEC"):
            pmdec = "PMDEC"

    if pmra is not None or pmdec is not None:
        posepoch = _epoch_or_pepoch(par, "POSEPOCH")

    return raj, decj, pmra, pmdec, posepoch, obliquity_arcsec


# Component -> builder.  Every component is now self-registered, so the whole
# table is derived from the registry: each component module supplies its own
# ``build`` (see the component classes + jaxpint/binary/_build.py for the family).
# A component absent here that is nonetheless active raises NotImplementedError in
# build_model.
_BUILDERS: dict[Component, Callable[[BuildContext], object]] = {}

from jaxpint.par._component_registry import registered as _registered  # noqa: E402

_BUILDERS.update({rc.component: rc.build for rc in _registered().values()})


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _active_components(par: ParResult) -> set[Component]:
    """The components to build, in addition to those detected in ``par``.

    Components enabled only by special conditions (not membership in
    ``par.component_set`` alone) are added here: Shapiro rides with astrometry,
    troposphere with ``CORRECT_TROPOSPHERE``, and a binary with any
    ``BINARY`` line.
    """
    _auto_detect_only = {
        Component.TROPOSPHERE_DELAY,
        Component.SOLAR_SYSTEM_SHAPIRO,
    }
    active = {comp for comp in par.component_set if comp not in _auto_detect_only}

    has_astrometry = (
        Component.ASTROMETRY_EQUATORIAL in active
        or Component.ASTROMETRY_ECLIPTIC in active
    )
    if has_astrometry:
        active.add(Component.SOLAR_SYSTEM_SHAPIRO)
    if par.bool_params.get("CORRECT_TROPOSPHERE", False):
        active.add(Component.TROPOSPHERE_DELAY)
    if par.binary_model is not None and Component.BINARY not in active:
        active.add(Component.BINARY)

    return active


def _validate_referenced_params(timing_model, noise_model, params) -> None:
    """Verify every parameter name a component references exists in ``params``.

    Components bind to parameters by name (static ``*_name`` / ``*_names`` fields)
    and resolve them lazily inside ``__call__`` via ``params.param_value(name)``.
    A name that is absent from the ``ParameterVector`` would otherwise only fail
    as a ``KeyError`` deep in a (possibly jitted) evaluation. This check runs once
    at build time and fails early with a message naming every offending component
    and parameter.

    Raises
    ------
    ValueError
        If any component (timing or noise) -- or the model's ``PHOFF`` binding --
        references a name not present in ``params``.
    """
    # `name in params` uses NamedVector.__contains__; epoch params (PEPOCH/T0/...)
    # are present in the vector too.
    missing: list[tuple[str, str]] = []  # (component label, parameter name)

    def _check(components, labels):
        for comp, label in zip(components, labels):
            for pname in comp.required_params():  # public API; skips unset Optionals
                if pname not in params:
                    missing.append((label, pname))

    _check(timing_model.components, timing_model.component_names)
    _check(noise_model.components, noise_model.component_names)

    # PHOFF is the one name-bearing field on TimingModel itself, not a sub-component.
    if timing_model.phoff_name is not None and timing_model.phoff_name not in params:
        missing.append(("TimingModel", timing_model.phoff_name))

    if missing:
        lines = "\n".join(
            f"  - {label} references unknown parameter {pname!r}"
            for label, pname in missing
        )
        raise ValueError(
            "build_model: component(s) reference parameter names not present in "
            f"the ParameterVector:\n{lines}\n"
            "This usually means a component's *_name field, its PARAMS schema, and "
            "the builder wiring disagree. Available parameters: "
            f"{tuple(params.names)}"
        )


def _validate_flag_masks(par: ParResult, toa_data: TOAData) -> None:
    """Verify the TOAData carries a flag mask for every masked parameter.

    The native loader builds ``toa_data.flag_masks`` with exactly the keys in
    ``par.mask_info``, so for native-loaded data this always holds. It can break
    when the TOAData comes from a different source than the par (the bridge path,
    or a hand-built TOAData): a masked parameter (EFAC/EQUAD/JUMP/...) would then
    either raise a ``KeyError`` deep in a jitted evaluation (the no-default
    ``flag_mask(name)`` consumers in white/dm_white/dispersion_jump) or silently
    contribute nothing (the ``default=False`` consumers). This check fails early
    with a clear message instead.

    Raises
    ------
    ValueError
        If any masked parameter declared in ``par`` lacks a mask in ``toa_data``.
    """
    missing = sorted(set(par.mask_info) - set(toa_data.flag_masks))
    if missing:
        raise ValueError(
            "build_model: TOAData is missing flag masks for masked parameter(s) "
            f"{missing} declared in the par. This usually means the TOAData was "
            "built from a different source than the par (e.g. the bridge path or "
            f"a hand-built TOAData). Masks present: {sorted(toa_data.flag_masks)}"
        )


def build_model(
    par: ParResult,
    toa_data: Optional[TOAData] = None,
):
    """Build JaxPINT TimingModel + NoiseModel from a parsed .par result.

    Parameters
    ----------
    par : ~jaxpint.par.result.ParResult
        Output of :func:`jaxpint.bridge.pint_model_to_params`.
    toa_data : TOAData, optional
        If provided, TOA-dependent components (ECORR, red noise, etc.)
        will be constructed.

    Returns
    -------
    (TimingModel, NoiseModel)
    """
    from jaxpint.model import TimingModel
    from jaxpint.components import (
        DelayComponent,
        DispersionDelayComponent,
        NoiseComponent,
        PhaseComponent,
    )
    from jaxpint.noise.noise_model import NoiseModel
    from jaxpint.noise.white import ScaleToaError
    from jaxpint.noise.dm_white import ScaleDmError

    # Astrometry names, resolved once up front (read by astrometry / Shapiro /
    # solar wind / binary builders -- so the result is independent of build order).
    raj, decj, pmra, pmdec, posepoch, obliquity_arcsec = _resolve_astrometry(par)
    ctx = BuildContext(
        par=par,
        toa_data=toa_data,
        raj=raj,
        decj=decj,
        pmra=pmra,
        pmdec=pmdec,
        posepoch=posepoch,
        obliquity_arcsec=obliquity_arcsec,
    )

    phoff_name = "PHOFF" if _param_is_set(par, "PHOFF") else None

    # Process the active components in PINT execution order and route each
    # result to its bucket by base class.  Iteration order is the priority
    # order; ``comp.value`` breaks ties identically to the old priority heap.
    active = _active_components(par)
    delay_components = []
    phase_components = []
    noise_components = []  # in priority order: white, dm_white, then correlated

    for comp in sorted(active, key=lambda c: (PRIORITY.get(c, len(PRIORITY)), c.value)):
        builder = _BUILDERS.get(comp)
        if builder is None:
            raise NotImplementedError(
                f"Component {comp!r} is present in the par file "
                f"but is not yet implemented in JaxPINT"
            )
        obj = builder(ctx)
        if obj is None:
            continue
        if isinstance(obj, DelayComponent):
            delay_components.append(obj)
        elif isinstance(obj, PhaseComponent):
            phase_components.append(obj)
        elif isinstance(obj, NoiseComponent):
            noise_components.append(obj)
        else:
            raise TypeError(
                f"Builder for {comp!r} returned an unroutable {type(obj).__name__}"
            )

    # ---- Assemble ----
    dispersion_components = tuple(
        c for c in delay_components if isinstance(c, DispersionDelayComponent)
    )

    timing_model = TimingModel(
        delay_components=tuple(delay_components),
        phase_components=tuple(phase_components),
        dispersion_components=dispersion_components,
        phoff_name=phoff_name,
    )

    # Partition noise by type.  ScaleToaError -> white slot, ScaleDmError -> DM
    # white slot, the rest -> correlated (kept in priority order, which equals
    # the historical ECORR, PLRed, PLDM, PLChrom, PLSW append sequence).
    white_noise = next(
        (c for c in noise_components if isinstance(c, ScaleToaError)), None
    )
    dm_white_noise = next(
        (c for c in noise_components if isinstance(c, ScaleDmError)), None
    )
    correlated = tuple(
        c for c in noise_components if not isinstance(c, (ScaleToaError, ScaleDmError))
    )

    combined_noise = NoiseModel(
        white_noise=white_noise,
        correlated=correlated,
        dm_white_noise=dm_white_noise,
    )

    _validate_referenced_params(timing_model, combined_noise, par.params)
    if toa_data is not None:
        _validate_flag_masks(par, toa_data)
    return timing_model, combined_noise
