"""Human-readable summaries of assembled models: what actually feeds the logL.

A JaxPINT likelihood is assembled from parts -- ``TimingModel``, ``NoiseModel``,
``ParameterVector``, ``GlobalParams``, per-pulsar ``TOAData``, PTA-level injectors, and a prior
spec -- and you want a human readable way of seeing the whole structure.
:func:`summarize_model` (single pulsar) and :func:`summarize_pta` (PTA) render
exactly what a likelihood evaluation will consume, so a forgotten component, a
mistyped parameter name, or a prior that covers the wrong site is visible in a
text dump instead of a silently wrong logL.

Beyond listing the parts, the summaries run cheap cross-checks:

- parameters in the vector that **no component reads** (a mistyped name in a
  hand-built vector lands here);
- flag masks that match **zero TOAs** (a mistyped selector flag lands here);
- global parameters vs. what the config's injectors actually register;
- prior sites vs. the site names the numpyro model builders will request
  (``f"{pulsar}_{param}"`` per-pulsar + bare global names).

Known blind spot: parameter-reader discovery walks only ``TimingModel`` and
``NoiseModel`` components, whose ``*_name``/``*_names`` field convention makes
``required_params()`` work.  PTA-level injectors receive ``pulsar_params`` in
``delay()``/``covariance()`` and may read per-pulsar parameters (e.g. the CW
injector's PX-based pulsar-term distance) without declaring them, so the
"read by" column cannot credit an injector, and a parameter read *only* by an
injector is wrongly listed as read-by-nothing.  Fixing this properly means
giving injectors a declared-parameter convention.

Everything returns a plain ``str``; write it to a file or ``print`` it.  All
functions are host-side (they may trigger eager JAX evaluation for basis
shapes) and must not be called inside ``jax.jit``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from jaxpint.components import _make_component_names
from jaxpint.types import GlobalParams, ParameterVector, TOAData

__all__ = ["summarize_model", "summarize_pta"]


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _fmt_float(v: float) -> str:
    return f"{v:.12g}"


# Display abbreviations for ParameterVector.param_status (the status
# semantics live on the datatype; only the rendering is decided here).
_STATUS_ABBREV = {"marginalized": "marg"}


def _param_status(params: ParameterVector, name: str) -> str:
    status = params.param_status(name)
    return _STATUS_ABBREV.get(status, status)


def _fmt_param_value(params: ParameterVector, name: str) -> str:
    """Render a parameter value; epoch params keep their int/frac split visible."""
    i = params.param_index(name)
    frac = float(params.values[i])
    if name in params.epoch_int_values:
        return f"{params.epoch_int_values[name]:g} {frac:+.12g}"
    return _fmt_float(frac)


def _fmt_uncertainty(params: ParameterVector, name: str) -> str:
    sigma = params.param_uncertainty(name)
    return "--" if math.isnan(sigma) else f"{sigma:.3g}"


def _fmt_config_value(val: Any) -> Optional[str]:
    """Render one component config field, or None to skip it."""
    if val is None:
        return None
    if isinstance(val, (bool, int, str)):
        return repr(val)
    if isinstance(val, float):
        return _fmt_float(val)
    if isinstance(val, (tuple, list)):
        if len(val) <= 6 and all(isinstance(x, (bool, int, float, str)) for x in val):
            return repr(tuple(val))
        return f"({len(val)} entries)"
    if isinstance(val, dict):
        return f"{{{len(val)} entries}}"
    if hasattr(val, "shape") and hasattr(val, "dtype"):
        return f"array{tuple(val.shape)}"
    return type(val).__name__


def _component_config(comp) -> str:
    """Non-parameter-name config of a component, as ``k=v`` pairs.

    Parameter bindings (``*_name`` / ``*_names`` fields) are rendered
    separately with their live values, so they are skipped here.  Works for
    both equinox modules (dataclass fields) and plain classes such as the
    signal injectors (public instance attributes).
    """
    if dataclasses.is_dataclass(comp):
        # eqx.Module components: fields() gives the declared schema in
        # declaration order and excludes __check_init__-style cached attrs.
        items = [
            (f.name, getattr(comp, f.name, None)) for f in dataclasses.fields(comp)
        ]
    else:
        # Plain-class signal injectors (CURNInjector etc.): attributes are set
        # ad hoc in _setup_spectrum, so vars() is the only introspection. If
        # the injectors are ever converted to eqx.Module, this branch becomes
        # dead and the function collapses to the dataclass path.
        items = [(k, v) for k, v in vars(comp).items() if not k.startswith("_")]
    pairs = []
    for attr, val in items:
        if attr.endswith("_name") or attr.endswith("_names"):
            continue
        rendered = _fmt_config_value(val)
        if rendered is not None:
            pairs.append(f"{attr}={rendered}")
    return ", ".join(pairs)


def _component_param_line(comp, params: ParameterVector) -> str:
    """``NAME=value [status]`` for every parameter this component reads."""
    entries = []
    for pname in comp.required_params():
        if pname not in params:
            entries.append(f"{pname}=<MISSING FROM VECTOR>")
            continue
        entries.append(
            f"{pname}={_fmt_param_value(params, pname)} [{_param_status(params, pname)}]"
        )
    return ", ".join(entries) if entries else "(none)"


def _table(
    rows: Sequence[Sequence[str]], header: Sequence[str], indent: str = "  "
) -> str:
    widths = []
    for c in range(len(header)):
        width = len(str(header[c]))
        for row in rows:
            width = max(width, len(str(row[c])))
        widths.append(width)

    def fmt(row):
        return indent + "  ".join(str(v).ljust(w) for v, w in zip(row, widths)).rstrip()

    lines = [fmt(header), indent + "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in rows)
    return "\n".join(lines)


def _rule(title: str) -> str:
    bar = "=" * 72
    return f"{bar}\n{title}\n{bar}"


def _dist_repr(d) -> str:
    """Compact numpyro distribution repr: ``Uniform(low=-18, high=-11)``."""
    args = []
    for k in getattr(d, "arg_constraints", {}):
        v = getattr(d, k, None)
        if v is None:
            continue
        try:
            args.append(f"{k}={float(v):.6g}")
        except (TypeError, ValueError):
            args.append(f"{k}={v}")
    return f"{type(d).__name__}({', '.join(args)})"


# ---------------------------------------------------------------------------
# Single-pulsar summary
# ---------------------------------------------------------------------------


def _toa_data_section(toa_data: TOAData) -> list[str]:
    # Display-only recombination: collapsing the int/frac split costs ~1 us at
    # MJD ~5e4, fine for a 0.1-day header but never acceptable in a computation.
    mjd = np.asarray(toa_data.mjd_int) + np.asarray(toa_data.mjd_frac)
    err_us = np.asarray(toa_data.error) * 1e6
    freq = np.asarray(toa_data.freq)
    wideband = toa_data.dm_values is not None
    lines = [
        f"Data: {toa_data.n_toas} TOAs, MJD {mjd.min():.1f}-{mjd.max():.1f} "
        f"(span {mjd.max() - mjd.min():.1f} d), "
        f"freq {freq.min():.0f}-{freq.max():.0f} MHz, "
        f"error {err_us.min():.3g}-{err_us.max():.3g} us, "
        f"{'wideband' if wideband else 'narrowband'}"
    ]
    coord = getattr(toa_data, "basis_coord", None)
    if coord is not None:
        lines.append(f"GP basis time coordinate: {coord}")
    if toa_data.flag_masks:
        entries, warnings = [], []
        for pname in sorted(toa_data.flag_masks):
            n = int(np.asarray(toa_data.flag_masks[pname]).sum())
            entries.append(f"{pname}->{n}")
            if n == 0:
                warnings.append(pname)
        lines.append(f"Flag masks (param -> n_toas matched): {', '.join(entries)}")
        if warnings:
            lines.append(
                f"  WARNING: mask(s) matching 0 TOAs (mistyped selector flag?): "
                f"{', '.join(warnings)}"
            )
    return lines


def _basis_width(comp, toa_data: TOAData, params: ParameterVector) -> str:
    try:
        _, U, _ = comp.covariance(toa_data, params)
        return f"n_basis={U.shape[1]}"
    except Exception as e:  # summary must never fail on a component quirk
        return f"n_basis=<error: {e}>"


def summarize_model(
    timing_model,
    noise_model,
    params: ParameterVector,
    toa_data: Optional[TOAData] = None,
    name: Optional[str] = None,
) -> str:
    """Render one pulsar's assembled model as a text report.

    Sections: TOA data (if given), the timing model's delay chain **in
    execution order**, phase components, the noise model with per-component
    basis widths (if ``toa_data`` is given), the full parameter table, and a
    list of parameters no component reads.

    Parameters
    ----------
    timing_model : ~jaxpint.model.TimingModel
    noise_model : ~jaxpint.noise.NoiseModel
    params : ParameterVector
        The exact vector that will be passed to the likelihood.
    toa_data : TOAData, optional
        Enables the data section, flag-mask counts, and noise basis widths.
    name : str, optional
        Pulsar label for the header.

    Returns
    -------
    str
    """
    out: list[str] = [_rule(f"JaxPINT model summary: {name or '(unnamed pulsar)'}")]

    if toa_data is not None:
        out.extend(_toa_data_section(toa_data))
        out.append("")

    # --- timing model, in evaluation order -------------------------------
    delay_names = _make_component_names(timing_model.delay_components)
    out.append("Timing model -- delay chain (applied sequentially, in order):")
    if not timing_model.delay_components:
        out.append("  (no delay components)")
    for i, (label, comp) in enumerate(
        zip(delay_names, timing_model.delay_components), 1
    ):
        cfg = _component_config(comp)
        out.append(f"  {i}. {label}" + (f"  [{cfg}]" if cfg else ""))
        out.append(f"       params: {_component_param_line(comp, params)}")

    phase_names = _make_component_names(timing_model.phase_components)
    out.append("Phase components (summed):")
    if not timing_model.phase_components:
        out.append("  (no phase components)")
    for i, (label, comp) in enumerate(
        zip(phase_names, timing_model.phase_components), 1
    ):
        cfg = _component_config(comp)
        out.append(f"  {i}. {label}" + (f"  [{cfg}]" if cfg else ""))
        out.append(f"       params: {_component_param_line(comp, params)}")

    if timing_model.phoff_name is not None:
        out.append(
            f"Phase offset: {timing_model.phoff_name}="
            f"{_fmt_param_value(params, timing_model.phoff_name)}"
        )
    out.append("")

    # --- noise model ------------------------------------------------------
    out.append("Noise model  (C = diag(Ndiag) + U diag(Phi) U^T):")

    def _noise_lines(slot: str, comp) -> list[str]:
        if comp is None:
            return [f"  {slot}: (none)"]
        cfg = _component_config(comp)
        is_correlated = any(comp is c for c in noise_model.correlated)
        width = (
            "  " + _basis_width(comp, toa_data, params)
            if toa_data is not None and is_correlated
            else ""
        )
        return [
            f"  {slot}: {type(comp).__name__}" + (f"  [{cfg}]" if cfg else "") + width,
            f"       params: {_component_param_line(comp, params)}",
        ]

    out.extend(_noise_lines("white (Ndiag)", noise_model.white_noise))
    if noise_model.correlated:
        for comp in noise_model.correlated:
            out.extend(_noise_lines("correlated   ", comp))
    else:
        out.append("  correlated   : (none)")
    out.extend(_noise_lines("dm_white     ", noise_model.dm_white_noise))
    if toa_data is not None:
        try:
            Ndiag, U, Phi = noise_model.covariance(toa_data, params)
            out.append(
                f"  total: Ndiag ({Ndiag.shape[0]},), "
                f"U {tuple(U.shape)}, Phi ({Phi.shape[0]},)"
            )
        except Exception as e:
            out.append(f"  total: <covariance evaluation failed: {e}>")
    out.append("")

    # --- parameter table --------------------------------------------------
    referenced: dict[str, list[str]] = {}
    all_comps = list(zip(timing_model.component_names, timing_model.components)) + list(
        zip(noise_model.component_names, noise_model.components)
    )
    for label, comp in all_comps:
        for pname in comp.required_params():
            referenced.setdefault(pname, []).append(label)
    if timing_model.phoff_name is not None:
        referenced.setdefault(timing_model.phoff_name, []).append("TimingModel")

    n = params.n_params
    n_free = params.n_free
    n_marg = sum(params.marginalized_mask)
    n_frozen = n - n_free - n_marg
    out.append(
        f"Parameters (n={n}: {n_free} free, {n_frozen} frozen, {n_marg} marginalized):"
    )
    rows = []
    for i, pname in enumerate(params.names):
        readers = referenced.get(pname, [])
        read_by = ", ".join(readers)
        if len(read_by) > 60:
            read_by = f"{readers[0]} +{len(readers) - 1} more"
        rows.append(
            (
                pname,
                _fmt_param_value(params, pname),
                params.units[i],
                _param_status(params, pname),
                _fmt_uncertainty(params, pname),
                read_by or "--",
            )
        )
    out.append(_table(rows, ("name", "value", "unit", "status", "sigma", "read by")))

    orphans = [p for p in params.names if p not in referenced]
    if orphans:
        out.append("")
        out.append(
            "Parameters not read by any component (admin/diagnostic entries are "
            "expected here;\nanything else may be a mistyped name):"
        )
        out.append("  " + ", ".join(orphans))

    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# PTA summary
# ---------------------------------------------------------------------------


def _registered_names(injector) -> Optional[tuple[str, ...]]:
    """Global-parameter names an injector registers (probed side-effect-free)."""
    try:
        return injector.register_params(GlobalParams.empty()).names
    except Exception:
        return None


def summarize_pta(
    config,
    pulsar_params: Optional[Sequence[ParameterVector]] = None,
    global_params: Optional[GlobalParams] = None,
    priors: Optional[Mapping] = None,
    pulsar_names: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> str:
    """Render a PTA configuration as a text report.

    Sections: per-pulsar overview table, per-pulsar and correlated signal
    injectors (with the global parameters each registers), the global
    parameter table cross-checked against injector registration, and -- when
    ``priors`` is given -- prior coverage of every site the numpyro model
    builders will request (``f"{pulsar}_{param}"`` per free parameter plus
    bare global names, matching
    ``jaxpint.bayes.samplers.numpyro.build_pta_model`` -- documented
    in-source, not in the API reference, hence no cross-link).

    Parameters
    ----------
    config : ~jaxpint.pta.likelihood.PTAConfig
    pulsar_params : sequence of ParameterVector, optional
        Per-pulsar vectors (same order as ``config``); enables free-parameter
        counts, prior-site checks, and ``verbose`` per-pulsar dumps.
    global_params : GlobalParams, optional
        The vector to be passed to ``pta_logL``; cross-checked against what
        the config's injectors register.
    priors : dict, optional
        A ``PriorSpec`` or bare ``{site: numpyro Distribution}`` mapping.
    pulsar_names : sequence of str, optional
        Labels (and numpyro site prefixes). Defaults to ``psr0, psr1, ...`` --
        pass the real names if you plan to use the prior-coverage check.
    verbose : bool
        Append a full :func:`summarize_model` dump per pulsar.

    Returns
    -------
    str
    """
    n_psr = config.n_pulsars
    names = (
        list(pulsar_names)
        if pulsar_names is not None
        else [f"psr{p}" for p in range(n_psr)]
    )
    out: list[str] = [_rule(f"JaxPINT PTA summary: {n_psr} pulsar(s)")]

    # --- per-pulsar overview ---------------------------------------------
    rows = []
    for p in range(n_psr):
        td = config.toa_data_list[p]
        tm = config.timing_models[p]
        nm = config.noise_models[p]
        # Display-only recombination (see _toa_data_section).
        mjd = np.asarray(td.mjd_int) + np.asarray(td.mjd_frac)
        rows.append(
            (
                p,
                names[p],
                td.n_toas,
                f"{mjd.max() - mjd.min():.0f}",
                len(tm.delay_components),
                len(tm.phase_components),
                len(nm.components),
                pulsar_params[p].n_free if pulsar_params is not None else "?",
            )
        )
    out.append(
        _table(
            rows,
            ("idx", "name", "n_toas", "span(d)", "delay", "phase", "noise", "n_free"),
        )
    )
    out.append("")

    # --- injectors --------------------------------------------------------

    def _injector_lines(kind: str, injectors) -> tuple[list[str], dict[str, str]]:
        """Section lines plus the {global param -> injector class} it registers."""
        lines = [f"{kind}:"]
        registered: dict[str, str] = {}
        if not injectors:
            lines.append("  (none)")
            return lines, registered
        for i, inj in enumerate(injectors, 1):
            cfg = _component_config(inj)
            lines.append(f"  {i}. {type(inj).__name__}" + (f"  [{cfg}]" if cfg else ""))
            reg = _registered_names(inj)
            if reg is None:
                lines.append("       registers: <register_params probe failed>")
            else:
                lines.append(f"       registers: {', '.join(reg) if reg else '(none)'}")
                for r in reg:
                    registered.setdefault(r, type(inj).__name__)
        return lines, registered

    # First registrant wins on duplicate names, so merge in section order.
    registered_by: dict[str, str] = {}
    for kind, injectors in (
        ("Signal injectors (per-pulsar delay/covariance)", config.signal_injectors),
        (
            "Correlated injectors (cross-pulsar, outer tier)",
            config.correlated_injectors,
        ),
    ):
        lines, registered = _injector_lines(kind, injectors)
        out.extend(lines)
        for r, cls in registered.items():
            registered_by.setdefault(r, cls)
    for inj in config.correlated_injectors:
        try:
            F = inj.get_fourier_basis(config.toa_data_list[0])
            Gamma = inj.get_orf_matrix()
            out.append(
                f"       {type(inj).__name__}: n_basis={F.shape[1]} "
                f"(pulsar 0), ORF {tuple(Gamma.shape)}"
            )
        except Exception:
            pass
    out.append("")

    # --- global parameters ------------------------------------------------
    if global_params is not None:
        out.append(f"Global parameters (n={global_params.n_params}):")
        rows = [
            (
                gname,
                _fmt_float(float(global_params.values[i])),
                registered_by.get(gname, "<no injector in this config>"),
            )
            for i, gname in enumerate(global_params.names)
        ]
        out.append(_table(rows, ("name", "value", "registered by")))
        missing = [r for r in registered_by if r not in global_params.names]
        if missing:
            out.append(
                "  WARNING: injector-registered name(s) absent from GlobalParams: "
                + ", ".join(missing)
            )
        unregistered = [g for g in global_params.names if g not in registered_by]
        if unregistered and (config.signal_injectors or config.correlated_injectors):
            out.append(
                "  WARNING: GlobalParams entries no injector in this config "
                "registers: " + ", ".join(unregistered)
            )
        out.append("")

    # --- prior coverage ---------------------------------------------------
    if priors is not None:
        flat: Mapping = getattr(priors, "flat", priors)
        out.append(f"Priors ({len(flat)} sites provided):")
        expected: list[str] = []
        if pulsar_params is not None:
            for prefix, pv in zip(names, pulsar_params):
                expected.extend(f"{prefix}_{b}" for b in pv.free_names())
        if global_params is not None:
            expected.extend(global_params.names)
        if expected:
            rows = [
                (site, _dist_repr(flat[site]) if site in flat else "<< MISSING >>")
                for site in expected
            ]
            out.append(_table(rows, ("site", "prior")))
            extra = sorted(set(flat) - set(expected))
            if extra:
                out.append(
                    "  NOTE: prior site(s) matching no expected site (unused, "
                    "or the pulsar_names prefixes differ): " + ", ".join(extra)
                )
            n_missing = sum(1 for s in expected if s not in flat)
            if n_missing:
                out.append(
                    f"  WARNING: {n_missing} expected site(s) have no prior -- "
                    "build_pta_model will raise."
                )
        else:
            rows = [(site, _dist_repr(d)) for site, d in flat.items()]
            out.append(_table(rows, ("site", "prior")))
            out.append(
                "  (pass pulsar_params and global_params to cross-check "
                "coverage against expected sites)"
            )
        out.append("")

    # --- verbose per-pulsar dumps ----------------------------------------
    if verbose:
        if pulsar_params is None:
            out.append("(verbose=True requires pulsar_params; skipping full dumps)")
        else:
            for p in range(n_psr):
                out.append("")
                out.append(
                    summarize_model(
                        config.timing_models[p],
                        config.noise_models[p],
                        pulsar_params[p],
                        toa_data=config.toa_data_list[p],
                        name=names[p],
                    ).rstrip()
                )

    return "\n".join(out) + "\n"
