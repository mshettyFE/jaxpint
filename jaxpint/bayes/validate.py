"""Validation helpers for prior dictionaries.

The intended usage pattern is that the
user assembles a ``dict[str, Prior]`` covering every parameter in their
model, and then runs :func:`validate_priors` to confirm completeness
*before* the dict reaches the sampler.  Loud failure here is the design's
defence against the "I added a parameter and forgot its prior" bug.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from jaxpint.bayes.priors import Prior


__all__ = ["validate_priors", "PriorValidationError"]


class PriorValidationError(ValueError):
    """Raised when a prior dictionary doesn't match the model's parameter set."""


def validate_priors(
    priors: Mapping[str, Prior],
    expected_params: Iterable[str],
    *,
    allow_extras: bool = False,
    extras_warn: bool = True,
) -> None:
    """Check that ``priors`` covers every name in ``expected_params``.

    Parameters
    ----------
    priors
        Mapping from parameter names to :class:`Prior` instances.
    expected_params
        Iterable of parameter names the model expects priors for.  Any
        name in ``expected_params`` not present in ``priors.keys()``
        triggers a :class:`PriorValidationError`.

        The canonical way to construct this list is via
        :func:`jaxpint.bayes.collect_param_names`, which
        produces the same fully-qualified names
        (``f"{psr_name}_{param_name}"`` per pulsar plus global parameter
        names) that the bulk-prior helpers (:func:`timing_priors`,
        :func:`distance_priors`, :func:`cw_priors`, etc.) generate as
        keys.  See ``Plans/priors_design.md`` for the full naming
        convention.
    allow_extras
        If ``True``, entries in ``priors`` that don't appear in
        ``expected_params`` are tolerated (silently, or with a warning if
        ``extras_warn=True``).  If ``False`` (default), extras raise.
    extras_warn
        If ``True`` (default) and ``allow_extras=True``, emit a warning
        listing extras (useful for catching stale entries during
        development).

    Raises
    ------
    PriorValidationError
        If any expected parameter has no prior assigned, or if extras
        are present and ``allow_extras=False``.
    TypeError
        If any value in ``priors`` is not a :class:`Prior` instance.

    See Also
    --------
    jaxpint.bayes.collect_param_names :
        Build the ``expected_params`` list from a pulsar collection plus
        an optional :class:`~jaxpint.pta.params.GlobalParams`.
    """
    expected = list(expected_params)

    # Type check the values
    bad_types = [
        (name, type(p).__name__)
        for name, p in priors.items()
        if not isinstance(p, Prior)
    ]
    if bad_types:
        raise TypeError(
            "validate_priors: prior dict values must be Prior instances. "
            f"Got non-Prior types: {bad_types[:5]}"
            + (" ..." if len(bad_types) > 5 else "")
        )

    expected_set = set(expected)
    have_set = set(priors.keys())

    # Sort errors for reproducibility
    missing = sorted(expected_set - have_set)
    extras = sorted(have_set - expected_set)

    parts = []
    if missing:
        head = missing[:10]
        parts.append(
            f"missing priors for {len(missing)} parameter(s): "
            f"{head}{' ...' if len(missing) > 10 else ''}"
        )
    if extras and not allow_extras:
        head = extras[:10]
        parts.append(
            f"unexpected priors for {len(extras)} name(s) not in the model: "
            f"{head}{' ...' if len(extras) > 10 else ''}"
        )
    if parts:
        raise PriorValidationError("validate_priors failed: " + "; ".join(parts))

    if extras and extras_warn:
        import warnings

        head = extras[:10]
        warnings.warn(
            f"validate_priors: {len(extras)} extra prior name(s) not in "
            f"the model (will be ignored): {head}"
            + (" ..." if len(extras) > 10 else ""),
            stacklevel=2,
        )
