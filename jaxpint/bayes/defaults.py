"""Bulk prior-dict factories and NANOGrav-conventional defaults.

The user-facing prior dictionary is assembled by composing dicts
returned from these helpers:

.. code-block:: python

    priors = (
        timing_priors(psrs, prior=ImproperPrior())
        | noise_priors_simple(psrs)
        | distance_priors(psrs)                 # par-file Gaussian on PX
        | cw_priors()
        | {"crn_log10_A": Uniform(-18, -11)}    # one-off override
    )
    validate_priors(priors, collect_param_names(...))

Naming convention used throughout:

- Per-pulsar timing parameter ``param`` of pulsar ``psr`` becomes
  ``f"{psr_name}_{param}"``.
- Global / shared parameters (CW source, GWB hyperparameters, etc.)
  use the names already carried by ``GlobalParams.names`` (which
  themselves typically include their own prefixes such as ``cw_``).

This first-pass module covers the "easy" defaults — timing-model
parameters, CW source, single-pulsar PX overrides, par-file Gaussian
factories.  Per-backend noise priors require parsing the noise model's
backend structure (EFAC/EQUAD/ECORR per backend); a general
:func:`noise_priors` that walks an arbitrary ``NoiseModel`` is left as
a future extension.  :func:`noise_priors_simple` provides a flat
"one EFAC per pulsar" assignment that suffices for many test setups.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

import jax.numpy as jnp

from jaxpint.bayes.priors import Gaussian, ImproperPrior, Prior, Uniform

if TYPE_CHECKING:
    # Imports used only for type-hint resolution.  Kept under TYPE_CHECKING
    # to avoid pulling in the full ``jaxpint.types`` / ``jaxpint.pta``
    # modules at import time and to keep ``jaxpint.bayes`` cleanly
    # decoupled from the rest of the package's import graph.
    from jaxpint.pta.params import GlobalParams
    from jaxpint.types import ParameterVector


__all__ = [
    "NANOGRAV_NOISE_DEFAULTS",
    "timing_priors",
    "distance_priors",
    "from_par_file",
    "cw_priors",
    "cw_phi_psr_priors",
    "noise_priors_simple",
    "collect_param_names",
]


# ---------------------------------------------------------------------------
# Standard NANOGrav prior bounds (mirrors discovery's priordict_standard for
# the per-class entries; values match published 12.5/15-yr practice).
# ---------------------------------------------------------------------------


NANOGRAV_NOISE_DEFAULTS: dict[str, Prior] = {
    # White-noise scaling (per-backend in real analyses; see noise_priors_simple).
    "efac":             Uniform(0.1, 10.0),
    "t2equad":          Uniform(-8.5, -5.0),
    "log10_ecorr":      Uniform(-8.5, -5.0),
    # Per-pulsar power-law red noise.
    "rednoise_log10_A": Uniform(-20.0, -11.0),
    "rednoise_gamma":   Uniform(0.0, 7.0),
    # Common (uncorrelated) red noise across the array.
    "crn_log10_A":      Uniform(-18.0, -11.0),
    "crn_gamma":        Uniform(0.0, 7.0),
    # Hellings-Downs gravitational-wave background.
    "gw_log10_A":       Uniform(-18.0, -11.0),
    "gw_gamma":         Uniform(0.0, 7.0),
}


# ---------------------------------------------------------------------------
# Bulk helpers
# ---------------------------------------------------------------------------


def collect_param_names(
    pulsar_names: Iterable[str],
    pulsar_params_list: Iterable["ParameterVector"],
    global_params: Optional["GlobalParams"] = None,
) -> list[str]:
    """Build the canonical list of fully-qualified parameter names.

    Use it to feed
    :func:`~jaxpint.bayes.validate_priors` the canonical
    ``expected_params`` argument so completeness checks line up with what
    the helpers would produce.

    Parameters
    ----------
    pulsar_names
        Per-pulsar string names, in the same order as ``pulsar_params_list``.
    pulsar_params_list
        Per-pulsar :class:`~jaxpint.types.ParameterVector` objects.  Each
        contributes ``f"{psr}_{name}"`` for every name in ``pp.names``.
    global_params
        Optional :class:`~jaxpint.pta.params.GlobalParams`.  Each name in
        ``global_params.names`` is added verbatim (these names already
        carry their own prefixes, e.g. ``cw_log10_h0``).

    Returns
    -------
    list of str
        Fully-qualified parameter names, in pulsar-then-global order.
    """
    out: list[str] = []
    for psr_name, pp in zip(pulsar_names, pulsar_params_list, strict=True):
        for n in pp.names:
            out.append(f"{psr_name}_{n}")
    if global_params is not None:
        out.extend(global_params.names)
    return out


def _resolve_pulsars(psrs) -> tuple[tuple[str, ...], tuple[ParameterVector, ...]]:
    """Return ``(names, params_list)`` tuples from a NanogravPTA-like input.

    Accepts either:

    - An object with ``.pulsar_names`` and ``.pulsar_params_list`` (e.g.
      :class:`~jaxpint.loaders.NanogravPTA`,
      :class:`~jaxpint.notebook_utils.SyntheticPTA`).
    - An iterable of ``(name, ParameterVector)`` pairs.
    """
    if hasattr(psrs, "pulsar_names") and hasattr(psrs, "pulsar_params_list"):
        return tuple(psrs.pulsar_names), tuple(psrs.pulsar_params_list)
    pairs = tuple(psrs)
    if not pairs:
        return (), ()
    names, params = zip(*pairs)
    return tuple(names), tuple(params)


def timing_priors(
    psrs,
    prior: Optional[Prior] = None,
) -> dict[str, Prior]:
    """Assign ``prior`` to every timing-model parameter of every pulsar.

    Parameters
    ----------
    psrs
        Either a :class:`~jaxpint.loaders.NanogravPTA`-like
        container with ``.pulsar_names`` / ``.pulsar_params_list``, or
        an iterable of ``(name, ParameterVector)`` pairs.
    prior
        :class:`Prior` to assign to *every* parameter of every pulsar.
        Defaults to :class:`~jaxpint.bayes.priors.ImproperPrior`, matching
        discovery's analytic-marginalization convention for timing-model
        parameters.

    Returns
    -------
    dict
        ``{f"{psr_name}_{param_name}": prior}`` for every per-pulsar
        timing parameter.

    Notes
    -----
    The same ``prior`` instance is shared across every entry — this is
    fine because :class:`Prior` instances are frozen :class:`equinox.Module` objects.
    Override individual parameters by composing later via dict union::

        priors = (
            timing_priors(psrs, prior=ImproperPrior())   # default
            | distance_priors(psrs)                      # PX overrides
        )
    """
    if prior is None:
        prior = ImproperPrior()
    names, params = _resolve_pulsars(psrs)
    out: dict[str, Prior] = {}
    for psr_name, pp in zip(names, params, strict=True):
        for param_name in pp.names:
            out[f"{psr_name}_{param_name}"] = prior
    return out


# Type alias for distance_priors' polymorphic prior argument.
_DistancePriorArg = Union[Prior, Callable[[object], Prior], None]


def distance_priors(
    psrs,
    prior: _DistancePriorArg = None,
    *,
    n_sigma: float = 1.0,
    param_name: str = "PX",
) -> dict[str, Prior]:
    """Bulk-assign a prior to ``PX`` for every pulsar.

    Three usage patterns:

    1. ``prior=None`` (default): build :class:`Gaussian` from each pulsar's
       par-file PX value and uncertainty, with width
       ``n_sigma * par_uncert``.  This is the recommended informative-prior
       default; it preserves the parallax-distance constraint that the
       par-file fit measured.

    2. ``prior=<Prior instance>``: apply the same prior to every pulsar
       (e.g. ``ImproperPrior()`` to match discovery's behaviour, or
       ``Uniform(0.1, 100.0)`` for an uninformative bounded scan).

    3. ``prior=<callable>``: per-pulsar customisation.  The callable is
       called as ``prior(pp)`` for each pulsar's
       :class:`~jaxpint.types.ParameterVector`; it may return any
       :class:`Prior`.  Use this for heterogeneous priors (e.g. some
       pulsars get VLBI Gaussians, others get par-file Gaussians, others
       get loose uniforms).

    Parameters
    ----------
    psrs
        :class:`~jaxpint.loaders.NanogravPTA`-like container or
        iterable of ``(name, ParameterVector)`` pairs.
    prior
        See above.
    n_sigma
        Width multiplier for the par-file-Gaussian default (used only
        when ``prior=None``).  Defaults to ``1.0`` — i.e., the resulting
        Gaussian uses the par-file uncertainty as-is.  See note below
        on double-counting if the par file was fit to the same TOAs
        used in the analysis.
    param_name
        Name of the parallax parameter in each :class:`ParameterVector`.
        Defaults to ``"PX"``.

    Returns
    -------
    dict
        ``{f"{psr_name}_{param_name}": Prior}`` for every pulsar that
        has ``param_name`` in its parameter list.  Pulsars without
        ``param_name`` are silently skipped.

    Notes
    -----
    For the par-file-Gaussian default, requires that each
    ``ParameterVector`` has been built with a usable uncertainty value.
    JaxPINT's :class:`ParameterVector` does not currently store
    per-parameter uncertainties on the object itself; in that case use
    :func:`from_par_file` (which accepts an external mapping) or the
    callable form here.

    .. note::

       The par-file ``(value, uncertainty)`` was derived from a timing
       fit to the same TOAs you may now be analyzing.  Using that
       ``(mu, sigma)`` as a prior and then multiplying by a likelihood
       over the same TOAs uses the annual-parallax modulation
       information twice — the resulting PX posterior will be tighter
       by roughly :math:`\\sqrt{2}` than the par-file uncertainty alone
       would imply.  This effect is small in absolute terms but real:
       for analyses where ``PX`` is a science output rather than a
       nuisance, consider either (a) using an *independent* distance
       prior (e.g. VLBI), or (b) widening with ``n_sigma>1`` to
       deliberately weaken the par-file prior.
    """
    names, params = _resolve_pulsars(psrs)
    out: dict[str, Prior] = {}
    for psr_name, pp in zip(names, params, strict=True):
        if param_name not in pp.names:
            continue
        if prior is None:
            mu, sigma = _par_file_gaussian_args(pp, param_name, n_sigma)
            out[f"{psr_name}_{param_name}"] = Gaussian(mu=mu, sigma=sigma)
        elif callable(prior) and not isinstance(prior, Prior):
            out[f"{psr_name}_{param_name}"] = prior(pp)
        else:
            out[f"{psr_name}_{param_name}"] = prior
    return out


def _par_file_gaussian_args(
    pp: "ParameterVector", name: str, n_sigma: float
) -> tuple[float, float]:
    """Extract (mu, n_sigma * sigma) from a ParameterVector for parameter *name*."""
    mu = float(pp.param_value(name))
    sigma = _maybe_par_uncert(pp, name)
    if sigma is None:
        raise ValueError(
            f"distance_priors: pulsar parameter {name!r} has no recorded "
            f"par-file uncertainty.  Pass an explicit `prior=` argument "
            f"(Prior instance or callable), or use from_par_file() with "
            f"a precomputed mapping of (mu, sigma) values."
        )
    if not bool(jnp.isfinite(sigma)) or sigma <= 0:
        raise ValueError(
            f"distance_priors: pulsar parameter {name!r} has unusable "
            f"par-file uncertainty {sigma}; cannot build a Gaussian prior. "
            f"Pass an explicit `prior=` argument."
        )
    return mu, n_sigma * sigma

def _maybe_par_uncert(pp: "ParameterVector", name: str) -> Optional[float]:
    """Best-effort per-parameter uncertainty extraction from a ParameterVector.

    Looks for one of the following accessors, returning the first that
    succeeds; falls back to ``None`` if no uncertainty source is available.
    """
    for attr in ("param_uncert", "param_sigma", "uncertainties", "sigmas"):
        meth = getattr(pp, attr, None)
        if meth is None:
            continue
        try:
            if callable(meth):
                return float(cast(float, meth(name)))
            # Attribute access (mapping or array)
            if hasattr(meth, "__getitem__"):
                if isinstance(meth, dict):
                    if name in meth:
                        return float(meth[name])
                else:
                    # Assume same indexing as `pp.names`
                    idx = pp.names.index(name)
                    return float(meth[idx])
        except (KeyError, ValueError, IndexError, TypeError):
            continue
    return None


def from_par_file(
    psrs,
    parameter_values: Mapping[str, Mapping[str, tuple[float, float]]],
    *,
    n_sigma: float = 1.0,
) -> dict[str, Prior]:
    """Build Gaussian priors from an explicit per-pulsar (mu, sigma) mapping.

    Use this when the par-file fit values and uncertainties are
    available externally (e.g., extracted from the .par file before
    JaxPINT loaded it) and not stored on the
    :class:`~jaxpint.types.ParameterVector` object itself.

    Parameters
    ----------
    psrs
        :class:`~jaxpint.loaders.NanogravPTA`-like container or
        iterable of ``(name, ParameterVector)`` pairs.  Used only to
        validate that the keys in ``parameter_values`` correspond to
        existing pulsars.
    parameter_values
        Nested mapping ``{psr_name: {param_name: (mu, sigma), ...}, ...}``.
        Only entries present here generate priors.
    n_sigma
        Width multiplier applied to ``sigma``.  Defaults to 1 (use the
        par-file uncertainty as-is).

    Returns
    -------
    dict
        ``{f"{psr_name}_{param_name}": Gaussian(mu, n_sigma*sigma)}``.
    """
    valid_pulsars = set(_resolve_pulsars(psrs)[0])
    out: dict[str, Prior] = {}
    for psr_name, params in parameter_values.items():
        if psr_name not in valid_pulsars:
            raise KeyError(
                f"from_par_file: pulsar name {psr_name!r} not in `psrs`."
            )
        for param_name, (mu, sigma) in params.items():
            if not bool(jnp.isfinite(sigma)) or sigma <= 0:
                raise ValueError(
                    f"from_par_file: bad sigma {sigma} for "
                    f"{psr_name!r}.{param_name!r}"
                )
            out[f"{psr_name}_{param_name}"] = Gaussian(
                mu=float(mu), sigma=n_sigma * float(sigma)
            )
    return out


# ---------------------------------------------------------------------------
# CW priors
# ---------------------------------------------------------------------------


def cw_priors(prefix: str = "cw_") -> dict[str, Prior]:
    """Standard continuous-wave source priors (no ``phi_psr`` nuisances).

    Returns priors for the seven canonical CW source parameters used by
    :class:`~jaxpint.pta.CWInjector`:

    - ``log10_h``       — strain amplitude (uniform in log10).
    - ``log10_fgw``     — GW frequency (uniform in log10).
    - ``cos_gwtheta``   — sky position polar cosine.
    - ``gwphi``         — sky position azimuth.
    - ``cos_inc``       — inclination cosine.
    - ``psi``           — polarization angle.
    - ``phase0``        — Earth-term GW phase at reference time.

    No per-pulsar ``cw_phi_psr`` nuisance parameters are included by
    default — JaxPINT parameterises the pulsar-term phase via the
    physical pulsar distance ``PX`` (treat the parallax distance as a
    science parameter via :func:`distance_priors`), so a separate
    ``phi_psr`` per pulsar is redundant.  If you do want the
    discovery-style distance-marginalised parameterisation, use
    :func:`cw_phi_psr_priors` explicitly.

    Parameters
    ----------
    prefix
        Name prefix used by the CW injector when registering its
        parameters into :class:`~jaxpint.pta.params.GlobalParams`.
        Defaults to ``"cw_"``, matching :class:`~jaxpint.pta.CWInjector`'s default.

    Returns
    -------
    dict
        ``{prefix + name: Prior}`` for the seven canonical CW parameters.
    """
    return {
        f"{prefix}log10_h":     Uniform(-18.0, -11.0),
        f"{prefix}log10_fgw":   Uniform(-9.0, -7.0),
        f"{prefix}cos_gwtheta": Uniform(-1.0, 1.0),
        f"{prefix}gwphi":       Uniform(0.0, 2.0 * jnp.pi),
        f"{prefix}cos_inc":     Uniform(-1.0, 1.0),
        f"{prefix}psi":         Uniform(0.0, jnp.pi),
        f"{prefix}phase0":      Uniform(0.0, 2.0* jnp.pi),
    }


def cw_phi_psr_priors(psrs, *, prefix: str = "cw_") -> dict[str, Prior]:
    """Per-pulsar CW pulsar-term phase nuisances (discovery-style).

    Use this only when you have *not* parameterised the pulsar-term
    phase via ``PX`` (i.e., when distance is *not* a science parameter
    and you instead want a free phase per pulsar that gets sampled
    over).  Mutually exclusive with the JaxPINT default of using ``PX``
    in the CW model.

    Parameters
    ----------
    psrs
        :class:`~jaxpint.loaders.NanogravPTA`-like container or
        iterable of ``(name, ParameterVector)`` pairs.
    prefix
        See :func:`cw_priors`.

    Returns
    -------
    dict
        ``{f"{psr_name}_{prefix}phi_psr": Uniform(0, 2π)}`` for every
        pulsar.
    """
    names, _ = _resolve_pulsars(psrs)
    return {
        f"{psr_name}_{prefix}phi_psr": Uniform(0.0, 2.0 * jnp.pi)
        for psr_name in names
    }


# ---------------------------------------------------------------------------
# Noise priors (simple version; full backend-aware version is future work)
# ---------------------------------------------------------------------------


def noise_priors_simple(
    psrs,
    *,
    include_red_noise: bool = True,
    defaults: Mapping[str, Prior] = NANOGRAV_NOISE_DEFAULTS,
    suffixes: Sequence[str] = ("efac", "t2equad", "log10_ecorr"),
) -> dict[str, Prior]:
    """Assign one white-noise prior set per pulsar (no backend split).

    A first-pass helper that assigns one EFAC / one EQUAD / one ECORR
    prior per pulsar (rather than per backend, which is what NANOGrav
    production analyses use).  Adequate for synthetic-data tests and for
    pulsars with a single backend.

    For the full per-backend assignment used in production NANOGrav
    analyses, a backend-aware ``noise_priors`` helper is planned; it would walk a
    :class:`~jaxpint.noise.NoiseModel` to discover the backend structure.

    Parameters
    ----------
    psrs
        :class:`~jaxpint.loaders.NanogravPTA`-like container or
        iterable of ``(name, ParameterVector)`` pairs.
    include_red_noise
        If ``True`` (default), also assign per-pulsar red-noise
        amplitude and spectral-index priors using
        ``defaults["rednoise_log10_A"]`` and ``defaults["rednoise_gamma"]``.
    defaults
        Mapping from suffix → :class:`Prior`.  Defaults to
        ``NANOGRAV_NOISE_DEFAULTS``.
    suffixes
        Which white-noise parameter suffixes to assign per pulsar.
        Defaults to ``("efac", "t2equad", "log10_ecorr")``.

    Returns
    -------
    dict
        Per-pulsar noise prior assignments under canonical names.
    """
    names, _ = _resolve_pulsars(psrs)
    out: dict[str, Prior] = {}
    for psr_name in names:
        for suffix in suffixes:
            if suffix in defaults:
                out[f"{psr_name}_{suffix}"] = defaults[suffix]
        if include_red_noise:
            for key in ("rednoise_log10_A", "rednoise_gamma"):
                if key in defaults:
                    out[f"{psr_name}_{key}"] = defaults[key]
    return out
