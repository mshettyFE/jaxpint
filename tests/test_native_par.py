"""Tests for the native ``.par`` parser (``jaxpint.par.get_model``).

Parity is checked against the PINT bridge: native ``par.get_model`` must produce
a ``ParResult`` that builds the *same* model as PINT's
``pint_model_to_params``.  Because PINT injects defaulted parameters not present
in the file (and ``build_model`` reads those via ``.get(name, default)``), the
native ``ParResult`` is a subset; we assert exact agreement on detected
components, binary model, mask info, and every parameter present in the file,
plus end-to-end residual agreement on pulsars with TOAs.

Also: a PINT-free unit suite for the tokenizer/adapter, and a consistency check
on the spec aggregated from each component's ``PARAMS``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from jaxpint.par.parfile import tokenize_lines
from jaxpint.par.raw_params import ParamKind
from jaxpint.par.text_adapter import to_raw_params
import jaxpint.par as par

_REPO = pathlib.Path(__file__).resolve().parents[1]

_CORPUS = [
    "NGC6440E.par",
    "B1855+09_NANOGrav_9yv1.gls.par",
    "B1855+09_NANOGrav_dfg+12_TAI.par",
    "PSRJ0030+0451_psrcat.par",
    "J0613-sim.par",
    "J1614-2230_NANOGrav_12yv3.wb.gls.par",
    "J0740+6620.FCP+21.wb.DMX3.0.par",
    "J1028-5819-example.par",
    "waves.par",
]


# ---------------------------------------------------------------------------
# Parity vs the PINT bridge
# ---------------------------------------------------------------------------


def _index(par_result):
    return {n: i for i, n in enumerate(par_result.params.names)}


@pytest.mark.slow
@pytest.mark.parametrize("parname", _CORPUS)
def test_parresult_parity_vs_pint(parname):
    import pint.models as pm
    from pint.config import examplefile
    from jaxpint.bridge import pint_model_to_params

    try:
        path = examplefile(parname)
        ref = pint_model_to_params(pm.get_model(path))
    except Exception as exc:
        pytest.skip(f"PINT could not load {parname}: {exc}")
    nat = par.get_model(path)

    assert nat.component_set == ref.component_set
    assert nat.binary_model == ref.binary_model
    assert nat.mask_info == ref.mask_info

    # Native names are a subset of PINT's (PINT injects defaulted params).
    nat_only = set(nat.params.names) - set(ref.params.names)
    assert not nat_only, f"native produced params PINT did not: {sorted(nat_only)}"

    ri, ni = _index(ref), _index(nat)
    epochs = set(ref.params.epoch_int_values) | set(nat.params.epoch_int_values)
    for name in set(nat.params.names) & set(ref.params.names):
        # Note: unit strings are deliberately NOT compared -- they are
        # documentation only (the JAX value convention is fixed), and PINT's
        # prefix-family units are index-dependent (F0=Hz, F1=Hz/s, ...) while
        # the native parser carries the template unit.  Values + the built
        # model are what must agree.
        assert nat.params.frozen_mask[ni[name]] == ref.params.frozen_mask[ri[name]], name
        rv = float(ref.params.values[ri[name]])
        nv = float(nat.params.values[ni[name]])
        if name in epochs:
            # Compare the full MJD; the bridge stores prefix-MJD fracs via a
            # single float64 (~1e-11 day noise), so allow a generous day-level
            # tolerance -- far below timing relevance for an epoch/boundary.
            rfull = ref.params.epoch_int_values.get(name, 0.0) + rv
            nfull = nat.params.epoch_int_values.get(name, 0.0) + nv
            assert abs(rfull - nfull) < 1e-7, (name, rfull, nfull)
        else:
            assert rv == nv or np.isclose(rv, nv, rtol=1e-9, atol=0.0), (name, rv, nv)


@pytest.mark.slow
@pytest.mark.parametrize("parname,timname", [
    ("NGC6440E.par", "NGC6440E.tim"),
    ("B1855+09_NANOGrav_9yv1.gls.par", "B1855+09_NANOGrav_9yv1.tim"),
])
def test_residual_parity_vs_pint(parname, timname):
    """End-to-end: native-par and PINT-par build models with identical residuals.

    Uses the same PINT-derived TOAs for both, so this isolates the parser.
    """
    import pint.models as pm
    import pint.toa as pt
    from pint.config import examplefile
    from jaxpint.bridge import pint_model_to_params, pint_toas_to_jax
    from jaxpint.bridge._model_builder import build_model
    from jaxpint.fitters import compute_time_residuals

    parf = examplefile(parname)
    timf = examplefile(timname)
    pmodel = pm.get_model(parf)
    toas = pt.get_TOAs(timf, model=pmodel, ephem="DE421")
    toa_data = pint_toas_to_jax(toas, model=pmodel)

    ref = pint_model_to_params(pmodel)
    nat = par.get_model(parf)
    tm_ref, _ = build_model(ref, toa_data)
    tm_nat, _ = build_model(nat, toa_data)

    r_ref = np.asarray(compute_time_residuals(tm_ref, toa_data, ref.params))
    r_nat = np.asarray(compute_time_residuals(tm_nat, toa_data, nat.params))
    assert np.allclose(r_ref, r_nat, atol=1e-9, rtol=0.0), float(np.max(np.abs(r_ref - r_nat)))


# ---------------------------------------------------------------------------
# Spec aggregated from each component's PARAMS (PINT-free)
# ---------------------------------------------------------------------------


def test_param_spec_consistency():
    from jaxpint.par import spec as S

    # every prefix / prefix-alias resolves to a known template
    for pfx, tmpl in S.PREFIX_MAP.items():
        assert tmpl in S.KNOWN_PARAMS, (pfx, tmpl)
    # every alias points at a known canonical name
    for alias, canon in S.ALIAS_MAP.items():
        assert canon in S.KNOWN_PARAMS, (alias, canon)
    # every trigger param is known and maps to a real component
    for name, comp in S.TRIGGER_MAP.items():
        assert name in S.KNOWN_PARAMS and comp is not None, name
    # canonical-prefix is the inverse of (the prefix entries of) PREFIX_MAP
    for tmpl, pfx in S.CANONICAL_PREFIX.items():
        assert S.PREFIX_MAP.get(pfx) == tmpl, (tmpl, pfx)
    # core sanity: known physics triggers and an unknown lookup
    assert S.spec_for("ZZZ_NOT_A_PARAM") is None
    assert {"F0", "RAJ", "DM", "DMEPOCH", "EQUAD1", "ECORR1"} <= set(S.TRIGGER_MAP)


def test_params_is_required_on_components(monkeypatch):
    """PARAMS is part of the base-class contract; a component without it fails loudly."""
    from jaxpint.components import DelayComponent
    from jaxpint.par import spec as S

    # base classes default PARAMS to () (a ClassVar, not an eqx field)
    class _Bare(DelayComponent):
        def __call__(self, toa_data, params, delay):  # pragma: no cover
            return delay
    assert _Bare.PARAMS == ()

    # a component reaching the aggregator without PARAMS is a hard, named error
    monkeypatch.setattr(S, "_component_classes", lambda: [(_Bare, None)])
    S._tables.cache_clear()
    try:
        with pytest.raises(TypeError, match="_Bare declares no PARAMS"):
            S._tables()
    finally:
        S._tables.cache_clear()   # restore the real tables for other tests


def _parse_one(line: str):
    parsed = to_raw_params(tokenize_lines([line]))
    return parsed.raw_params[0] if parsed.raw_params else None


def test_tokenizer_skips_comments_and_blanks():
    lines = ["# comment", "C also comment", "", "   ", "F0 100 1"]
    toks = tokenize_lines(lines)
    assert len(toks) == 1 and toks[0].name == "F0"


def test_fit_flag_vs_uncertainty():
    assert _parse_one("F0 100 1").frozen is False     # fit flag 1 -> free
    assert _parse_one("F0 100 0").frozen is True       # fit flag 0 -> frozen
    assert _parse_one("F0 100").frozen is True         # no flag -> frozen
    assert _parse_one("F0 100 5e-9").frozen is True    # uncertainty, no flag


def test_uncertainty_extraction():
    # sigma after a fit flag
    assert np.isclose(_parse_one("PX 0.5 1 0.12").uncertainty, 0.12)
    # sigma with no fit flag (the "anything else" branch)
    assert np.isclose(_parse_one("PX 0.5 0.12").uncertainty, 0.12)
    # fit flag but no sigma -> None
    assert _parse_one("PX 0.5 1").uncertainty is None
    # value only -> None
    assert _parse_one("PX 0.5").uncertainty is None
    # fortran-D exponent in the sigma token
    assert np.isclose(_parse_one("PX 0.5 1 1.2D-2").uncertainty, 0.012)
    # PINT auto-scaled param: the sigma rides the same scale as the value
    rp = _parse_one("PBDOT 1.59 1 0.05")
    assert np.isclose(rp.value, 1.59e-12) and np.isclose(rp.uncertainty, 0.05e-12)
    # frozen flag (0) with an explicit sigma still keeps the sigma
    assert np.isclose(_parse_one("PX 0.5 0 0.12").uncertainty, 0.12)
    # PINT override rule: with two trailing tokens the sigma is the 2nd, even if
    # the first is a non-0/1 integer (pathological, but matches PINT exactly)
    assert np.isclose(_parse_one("F0 100 2 0.5").uncertainty, 0.5)


def test_param_uncertainty_accessor(tmp_path):
    import astropy.units as u
    par_text = (
        "PSR J0000+0000\n"
        "RAJ 12:00:00 1 0.001\n"     # HMS angle: sigma in sec-of-time -> rad
        "DECJ -30:00:00\n"           # value only -> NaN
        "F0 100 1 5e-9\n"            # fitted float with sigma
        "PX 0.5 1 0.12\n"           # fitted float with sigma
        "DM 15 0\n"                  # frozen, no sigma -> NaN
    )
    p = tmp_path / "u.par"
    p.write_text(par_text)
    pv = par.get_model(str(p)).params
    assert np.isclose(pv.param_uncertainty("F0"), 5e-9)
    assert np.isclose(pv.param_uncertainty("PX"), 0.12)
    # RAJ sigma 0.001 sec-of-time converted to radians
    assert np.isclose(pv.param_uncertainty("RAJ"),
                      float((0.001 * u.hourangle / 3600).to(u.rad).value))
    assert np.isnan(pv.param_uncertainty("DM"))      # frozen / no sigma
    assert np.isnan(pv.param_uncertainty("DECJ"))    # value only
    # aligned with values, JIT-traceable (static metadata)
    assert len(pv.uncertainties) == pv.values.shape[0]


def test_uncertainty_all_kinds():
    import astropy.units as u
    # ANGLE: HMS RA -> sec-of-time; DMS DEC -> arcsec; decimal -> deg
    assert np.isclose(_parse_one("RAJ 17:48:52.75 1 0.05").uncertainty,
                      float((0.05 * u.hourangle / 3600).to(u.rad).value))
    assert np.isclose(_parse_one("DECJ -20:21:29.0 1 0.4").uncertainty,
                      float((0.4 * u.arcsec).to(u.rad).value))
    assert np.isclose(_parse_one("ELONG 286.8634 1 8.4e-9").uncertainty,
                      float((8.4e-9 * u.deg).to(u.rad).value))
    # MJD: sigma in days, stored as-is
    t0 = _parse_one("T0 53113.95509 1 0.00266858")
    assert t0.kind is ParamKind.MJD and np.isclose(t0.uncertainty, 0.00266858)
    # MASK: JUMP carries a sigma (seconds)
    jp = _parse_one("JUMP -fe L-wide -0.000009449 1 0.000009439")
    assert jp.kind is ParamKind.MASK and np.isclose(jp.uncertainty, 9.439e-6)
    # PAIR (WAVE): no sigma in the format -> None
    assert _parse_one("WAVE1 -1.2e-7 3.4e-8").uncertainty is None


def test_fortran_float():
    rp = _parse_one("PX 1.5D0")
    assert rp.kind is ParamKind.FLOAT and np.isclose(rp.value, 1.5)


def test_sexagesimal_angles():
    raj = _parse_one("RAJ 12:00:00")    # 12h = 180 deg = pi rad
    dec = _parse_one("DECJ -30:00:00")  # -30 deg
    assert raj.kind is ParamKind.ANGLE and np.isclose(raj.value, np.pi)
    assert np.isclose(dec.value, np.deg2rad(-30.0))


def test_alias_resolution():
    assert _parse_one("RA 12:00:00").name == "RAJ"
    assert _parse_one("LAMBDA 30:00:00").name == "ELONG"


def test_mask_flag_key():
    rp = _parse_one("JUMP -fe 430 0.0 1")
    assert rp.kind is ParamKind.MASK and rp.name == "JUMP1"
    assert (rp.mask_key, rp.mask_key_value, rp.mask_key_value2) == ("-fe", "430", None)
    assert rp.frozen is False


def test_mask_range_key():
    rp = _parse_one("JUMP mjd 50000 55000 0.0")
    assert rp.mask_key == "mjd"
    assert (rp.mask_key_value, rp.mask_key_value2) == ("50000.0", "55000.0")


def test_repeatable_mask_indexing():
    parsed = to_raw_params(tokenize_lines([
        "EQUAD -fe 430 0.1",
        "EQUAD -fe 820 0.2",
    ]))
    names = [rp.name for rp in parsed.raw_params]
    assert names == ["EQUAD1", "EQUAD2"]


def test_prefix_family():
    rp = _parse_one("DMX_0001 0.001")
    assert rp.kind is ParamKind.FLOAT and rp.name == "DMX_0001"


def test_mjd_split():
    rp = _parse_one("PEPOCH 55000.5")
    assert rp.kind is ParamKind.MJD and rp.mjd_split == (55000.0, 0.5)


def test_unit_scale_pbdot():
    rp = _parse_one("PBDOT 1.59")     # PINT scales values above threshold
    assert np.isclose(rp.value, 1.59e-12)
    rp2 = _parse_one("PBDOT 1.59e-12")  # already small -> unchanged
    assert np.isclose(rp2.value, 1.59e-12)


def test_tzrfrq_is_metadata():
    rp = _parse_one("TZRFRQ inf")
    assert rp.kind is ParamKind.STR and rp.str_value == "inf"


def test_bool_and_str():
    assert _parse_one("PLANET_SHAPIRO Y").bool_value is True
    assert _parse_one("PLANET_SHAPIRO N").bool_value is False
    assert _parse_one("UNITS TDB").str_value == "TDB"
