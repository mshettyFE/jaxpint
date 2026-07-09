"""Prior specification and bulk-assignment for the NumPyro sampler layer.

Usage::

    from jaxpint.bayes.samplers.priors import (
        noise_priors_simple, distance_priors, cw_priors, resolve_priors,
    )
    import numpyro.distributions as dist

    spec = (
        noise_priors_simple(psrs)                       # per-pulsar white + red
        | distance_priors(psrs)                         # par-file Gaussian on PX
        | cw_priors()                                   # CW source
        | {"crn_log10_A": dist.Uniform(-18, -11)}       # one-off override
    )
    priors = resolve_priors(free_fqns, spec)            # loud on missing/conflict

Naming convention: a per-pulsar timing/noise parameter ``param`` of pulsar
``psr`` is keyed ``f"{psr_name}_{param}"``; global/shared parameters use the
names already carried by ``GlobalParams.names`` (which include their own
prefixes, e.g. ``cw_log10_h``).

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
import numpyro.distributions as dist

if TYPE_CHECKING:
    from jaxpint.types import GlobalParams, ParameterVector


__all__ = [
    "PriorSpec",
    "PulsarBundle",
    "PRIOR_DEFAULTS",
    "noise_priors_simple",
    "distance_priors",
    "from_par_file",
    "cw_priors",
    "cw_phi_psr_priors",
    "timing_marg_set",
    "resolve_priors",
    "collect_free_fqns",
]


# ---------------------------------------------------------------------------
# PriorSpec container (independent per-name priors)
# ---------------------------------------------------------------------------


# A dist factory so each assignment gets a fresh distribution object.
_DistFactory = Callable[[], dist.Distribution]

# What ``|`` accepts on either side: a PriorSpec or a bare {fqn: dist} mapping.
PriorSpecLike = Union["PriorSpec", Mapping[str, dist.Distribution]]


@dataclass(frozen=True)
class PriorSpec:
    """A composed prior specification: independent per-name priors.

    ``flat`` maps a (fully-qualified) parameter name to an independent
    :class:`~numpyro.distributions.Distribution`.  Compose with ``|`` (last
    assignment wins per name); a bare ``{name: dist}`` mapping composes as a
    spec.

    """

    flat: dict[str, dist.Distribution] = field(default_factory=dict)

    def owned_names(self) -> set[str]:
        """Every parameter name this spec claims."""
        return set(self.flat)

    def __or__(self, other: PriorSpecLike) -> "PriorSpec":
        # Last assignment wins per name: `other` overrides `self`.
        other = _coerce(other)
        return PriorSpec({**self.flat, **other.flat})

    def __ror__(self, other: PriorSpecLike) -> "PriorSpec":
        # Supports ``{name: dist} | priorspec`` (dict.__or__ returns NotImplemented).
        return _coerce(other) | self


def _coerce(obj: PriorSpecLike) -> PriorSpec:
    """Coerce a PriorSpec or a bare ``{fqn: dist}`` mapping to a PriorSpec."""
    if isinstance(obj, PriorSpec):
        return obj
    if isinstance(obj, Mapping):
        return PriorSpec(dict(obj))
    raise TypeError(
        f"Cannot compose {type(obj).__name__} with a PriorSpec; expected a "
        f"PriorSpec or a mapping of {{name: numpyro Distribution}}."
    )


# ---------------------------------------------------------------------------
# Pulsar-bundle protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PulsarBundle(Protocol):
    """Structural type for a per-pulsar bundle of names + parameter vectors.

    Any object exposing these two parallel attributes is accepted by the
    assembly helpers -- in practice a :class:`~jaxpint.loaders.NanogravPTA`, or a
    lightweight ``namedtuple``.  Declared as a :class:`typing.Protocol` so this
    module stays decoupled from ``loaders`` / ``notebook_utils``.
    """

    pulsar_names: tuple[str, ...]
    pulsar_params_list: tuple["ParameterVector", ...]


def _resolve_pulsars(
    psrs: PulsarBundle,
) -> tuple[tuple[str, ...], tuple["ParameterVector", ...]]:
    return tuple(psrs.pulsar_names), tuple(psrs.pulsar_params_list)


# ---------------------------------------------------------------------------
# Standard NANOGrav prior bounds (dist factories; values match 12.5/15-yr
# practice and mirror discovery's priordict_standard for the per-class entries).
# ---------------------------------------------------------------------------


PRIOR_DEFAULTS: dict[str, _DistFactory] = {
    # White-noise scaling (per-backend in real analyses; see noise_priors_simple).
    "efac": lambda: dist.Uniform(0.1, 10.0),
    "t2equad": lambda: dist.Uniform(-8.5, -5.0),
    "log10_ecorr": lambda: dist.Uniform(-8.5, -5.0),
    # Per-pulsar power-law red noise.
    "rednoise_log10_A": lambda: dist.Uniform(-20.0, -11.0),
    "rednoise_gamma": lambda: dist.Uniform(0.0, 7.0),
    # Common (uncorrelated) red noise across the array.
    "crn_log10_A": lambda: dist.Uniform(-18.0, -11.0),
    "crn_gamma": lambda: dist.Uniform(0.0, 7.0),
    # Hellings-Downs gravitational-wave background.
    "gw_log10_A": lambda: dist.Uniform(-18.0, -11.0),
    "gw_gamma": lambda: dist.Uniform(0.0, 7.0),
}


# ---------------------------------------------------------------------------
# Bulk helpers (all return PriorSpec, flat-only in this phase)
# ---------------------------------------------------------------------------


def noise_priors_simple(
    psrs: PulsarBundle,
    *,
    include_red_noise: bool = True,
    defaults: Mapping[str, _DistFactory] = PRIOR_DEFAULTS,
    suffixes: Sequence[str] = ("efac", "t2equad", "log10_ecorr"),
) -> PriorSpec:
    """One white-noise (and optionally red-noise) prior set per pulsar.

    A first-pass helper that assigns one EFAC / one EQUAD / one ECORR prior per
    pulsar (rather than per backend, which NANOGrav production analyses use).
    Adequate for synthetic-data tests and single-backend pulsars.
    """
    names, _ = _resolve_pulsars(psrs)
    flat: dict[str, dist.Distribution] = {}
    for psr_name in names:
        for suffix in suffixes:
            if suffix in defaults:
                flat[f"{psr_name}_{suffix}"] = defaults[suffix]()
        if include_red_noise:
            for key in ("rednoise_log10_A", "rednoise_gamma"):
                if key in defaults:
                    flat[f"{psr_name}_{key}"] = defaults[key]()
    return PriorSpec(flat)


# Polymorphic prior argument for distance_priors.
_DistanceArg = Union[
    dist.Distribution, Callable[["ParameterVector"], dist.Distribution], None
]


def distance_priors(
    psrs: PulsarBundle,
    prior: _DistanceArg = None,
    *,
    n_sigma: float = 1.0,
    param_name: str = "PX",
) -> PriorSpec:
    """Bulk-assign a prior to ``PX`` (or ``param_name``) for every pulsar.

    Three usage patterns:

    1. ``prior=None`` (default): build ``dist.Normal`` from each pulsar's
       par-file value and uncertainty, with width ``n_sigma * par_uncert`` --
       the informative-prior default that preserves the par-file constraint.
    2. ``prior=<Distribution>``: apply the same dist to every pulsar.
    3. ``prior=<callable>``: per-pulsar customisation, called ``prior(pp)``.

    Pulsars lacking ``param_name`` are silently skipped.

    .. note::

       The par-file ``(value, uncertainty)`` was derived from a fit to the same
       TOAs you may now analyze; using it as a prior double-counts the parallax
       information (posterior tighter by ~sqrt(2)).  For ``PX`` as a science
       output, prefer an independent distance prior or widen with ``n_sigma>1``.
    """
    names, params = _resolve_pulsars(psrs)
    flat: dict[str, dist.Distribution] = {}
    for psr_name, pp in zip(names, params, strict=True):
        if param_name not in pp.names:
            continue
        key = f"{psr_name}_{param_name}"
        if prior is None:
            mu, sigma = _par_file_gaussian_args(pp, param_name, n_sigma)
            flat[key] = dist.Normal(mu, sigma)
        elif isinstance(prior, dist.Distribution):
            flat[key] = prior
        elif callable(prior):
            flat[key] = prior(pp)
        else:
            raise TypeError(
                f"distance_priors: `prior` must be a numpyro Distribution, a "
                f"callable(pp)->Distribution, or None; got {type(prior).__name__}."
            )
    return PriorSpec(flat)


def from_par_file(
    psrs: PulsarBundle,
    parameter_values: Mapping[str, Mapping[str, tuple[float, float]]],
    *,
    n_sigma: float = 1.0,
) -> PriorSpec:
    """Build ``dist.Normal`` priors from an explicit per-pulsar (mu, sigma) map.

    ``parameter_values`` is ``{psr_name: {param_name: (mu, sigma), ...}, ...}``.
    Use when par-file fit values/uncertainties are available externally rather
    than stored on the :class:`ParameterVector`.
    """
    valid_pulsars = set(_resolve_pulsars(psrs)[0])
    flat: dict[str, dist.Distribution] = {}
    for psr_name, params in parameter_values.items():
        if psr_name not in valid_pulsars:
            raise KeyError(f"from_par_file: pulsar name {psr_name!r} not in `psrs`.")
        for param_name, (mu, sigma) in params.items():
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError(
                    f"from_par_file: bad sigma {sigma} for {psr_name!r}.{param_name!r}"
                )
            flat[f"{psr_name}_{param_name}"] = dist.Normal(
                float(mu), n_sigma * float(sigma)
            )
    return PriorSpec(flat)


def _par_file_gaussian_args(
    pp: "ParameterVector", name: str, n_sigma: float
) -> tuple[float, float]:
    """Extract (mu, n_sigma * sigma) from a ParameterVector for parameter *name*."""
    mu = float(pp.param_value(name))
    sigma = _maybe_par_uncert(pp, name)
    if sigma is None:
        raise ValueError(
            f"distance_priors: pulsar parameter {name!r} has no recorded par-file "
            f"uncertainty.  Pass an explicit `prior=` (Distribution or callable), "
            f"or use from_par_file() with a precomputed (mu, sigma) mapping."
        )
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            f"distance_priors: pulsar parameter {name!r} has unusable par-file "
            f"uncertainty {sigma}; cannot build a Normal prior. Pass `prior=`."
        )
    return mu, n_sigma * sigma


def _maybe_par_uncert(pp: "ParameterVector", name: str) -> Optional[float]:
    """Per-parameter 1-sigma uncertainty from a ParameterVector.

    Prefers the canonical :meth:`ParameterVector.param_uncertainty` accessor
    (returns ``nan`` when the par file reported none); falls back to a few
    legacy accessor names for non-``ParameterVector`` bundles.  Returns ``None``
    when no usable source is found.
    """
    accessor = getattr(pp, "param_uncertainty", None)
    if callable(accessor):
        try:
            return float(cast(float, accessor(name)))
        except (KeyError, ValueError, IndexError, TypeError):
            pass
    for attr in ("param_uncert", "param_sigma", "uncertainties", "sigmas"):
        meth = getattr(pp, attr, None)
        if meth is None:
            continue
        try:
            if callable(meth):
                return float(cast(float, meth(name)))
            if isinstance(meth, dict):
                if name in meth:
                    return float(meth[name])
            elif hasattr(meth, "__getitem__"):
                return float(meth[pp.names.index(name)])
        except (KeyError, ValueError, IndexError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# CW priors
# ---------------------------------------------------------------------------


def cw_priors(prefix: str = "cw_") -> PriorSpec:
    """Standard continuous-wave source priors (no ``phi_psr`` nuisances).

    The seven canonical CW parameters used by :class:`~jaxpint.pta.CWInjector`.
    No per-pulsar ``phi_psr`` is included: JaxPINT parameterises the pulsar-term
    phase via the physical distance ``PX`` (see :func:`distance_priors`), so a
    free phase per pulsar is redundant.  For the discovery-style
    distance-marginalised parameterisation, use :func:`cw_phi_psr_priors`.
    """
    return PriorSpec(
        {
            f"{prefix}log10_h": dist.Uniform(-18.0, -11.0),
            f"{prefix}log10_fgw": dist.Uniform(-9.0, -7.0),
            f"{prefix}cos_gwtheta": dist.Uniform(-1.0, 1.0),
            f"{prefix}gwphi": dist.Uniform(0.0, 2.0 * np.pi),
            f"{prefix}cos_inc": dist.Uniform(-1.0, 1.0),
            f"{prefix}psi": dist.Uniform(0.0, np.pi),
            f"{prefix}phase0": dist.Uniform(0.0, 2.0 * np.pi),
        }
    )


def cw_phi_psr_priors(psrs: PulsarBundle, *, prefix: str = "cw_") -> PriorSpec:
    """Per-pulsar CW pulsar-term phase nuisances (discovery-style).

    Use only when the pulsar-term phase is *not* parameterised via ``PX``.
    Mutually exclusive with the JaxPINT default of using ``PX`` in the CW model.
    """
    names, _ = _resolve_pulsars(psrs)
    return PriorSpec(
        {f"{psr}_{prefix}phi_psr": dist.Uniform(0.0, 2.0 * np.pi) for psr in names}
    )


# ---------------------------------------------------------------------------
# Marginalization set
# ---------------------------------------------------------------------------


def timing_marg_set(
    psrs: PulsarBundle, *, only: Optional[Iterable[str]] = None
) -> set[str]:
    """names of the free timing-model parameters to marginalize analytically.

    Parameters
    ----------
    psrs
        A :class:`PulsarBundle`.
    only
        If given, restrict to these bare parameter names (e.g. ``{"F0", "F1"}``);
        otherwise every free per-pulsar timing parameter is included.

    Returns
    -------
    set of str
        ``{f"{psr}_{param}"}`` for each free timing parameter (subject to
        ``only``).
    """
    names, params = _resolve_pulsars(psrs)
    only_set = set(only) if only is not None else None
    out: set[str] = set()
    for psr_name, pp in zip(names, params, strict=True):
        for bare in pp.free_names():
            if only_set is None or bare in only_set:
                out.add(f"{psr_name}_{bare}")
    return out


# ---------------------------------------------------------------------------
# Resolution + completeness check
# ---------------------------------------------------------------------------


def collect_free_fqns(
    pulsar_names: Iterable[str],
    reduced_pulsar_params: Iterable["ParameterVector"],
    global_params: Optional["GlobalParams"] = None,
) -> list[str]:
    """names of the parameters that will be *sampled* (need a prior).

    Per-pulsar: each reduced skeleton's ``free_names()`` (marginalized params are
    already excluded) becomes ``f"{psr}_{name}"``.  Globals contribute their
    names verbatim.  Feed the result to :func:`resolve_priors`.
    """
    out: list[str] = []
    for psr_name, pp in zip(pulsar_names, reduced_pulsar_params, strict=True):
        out.extend(f"{psr_name}_{n}" for n in pp.free_names())
    if global_params is not None:
        out.extend(global_params.names)
    return out


class PriorResolutionError(ValueError):
    """A sampled parameter has no prior assigned in a PriorSpec."""


def resolve_priors(
    free_fqns: Iterable[str],
    spec: PriorSpecLike,
) -> dict[str, dist.Distribution]:
    """Map every sampled name to its distribution, failing loud on gaps.

    Every name in ``free_fqns`` must have an entry in ``spec`` -- a sampled
    parameter with no prior is a :class:`PriorResolutionError` (the §3 "no
    silent prior" contract).  Returns the ``{name: Distribution}`` mapping in
    ``free_fqns`` order.
    """
    spec = _coerce(spec)
    free_list = list(free_fqns)

    missing = sorted(f for f in set(free_list) if f not in spec.flat)
    if missing:
        raise PriorResolutionError(
            f"resolve_priors: no prior assigned for sampled parameter(s): {missing}."
        )

    return {name: spec.flat[name] for name in free_list}
