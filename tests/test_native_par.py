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
        pmod = pm.get_model(path)
    except Exception as exc:
        pytest.skip(f"PINT could not load {parname}: {exc}")
    # Outside the try: a bug in the JaxPINT bridge/parser under test must fail
    # the test, not be swallowed as a skip.
    ref = pint_model_to_params(pmod)
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
    from jaxpint.model_builder import build_model
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
    from jaxpint.par import registry_table

    monkeypatch.setattr(
        registry_table, "derive_component_classes", lambda: [(_Bare, None)]
    )
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


# ---------------------------------------------------------------------------
# tempo2's "SINI KIN" idiom
#
# A literal ``SINI KIN`` line means SINI is *derived* from the DDK inclination
# angle, so the value is a sentinel naming another parameter rather than a
# number.  Real IPTA DR1/DR2 pars ship it (J1713+0747 line 13), and parsing it
# as a float raised ``could not convert string to float: 'KIN'``.  PINT drops
# SINI when KIN is present (models/model_builder.py:986).
# ---------------------------------------------------------------------------


def test_sini_kin_sentinel_drops_sini():
    parsed = to_raw_params(tokenize_lines(["SINI KIN", "KIN 71.9"]))
    names = [rp.name for rp in parsed.raw_params]
    assert "KIN" in names
    assert "SINI" not in names  # derived, not fitted


def test_sini_kin_sentinel_is_case_insensitive():
    parsed = to_raw_params(tokenize_lines(["SINI kin", "KIN 71.9"]))
    assert "SINI" not in [rp.name for rp in parsed.raw_params]


def test_sini_kin_without_kin_raises():
    """The sentinel is only meaningful with something to derive from.

    SINI can precede KIN in the file (it does in J1713+0747: line 13 vs 30),
    so this is checked after the parse loop, not inline.
    """
    with pytest.raises(ValueError, match="no KIN parameter"):
        to_raw_params(tokenize_lines(["SINI KIN"]))


def test_numeric_sini_still_parsed():
    parsed = to_raw_params(tokenize_lines(["SINI 0.9656", "KIN 71.9"]))
    rp = next(rp for rp in parsed.raw_params if rp.name == "SINI")
    assert np.isclose(rp.value, 0.9656)


def test_unrecognized_sini_sentinel_still_raises():
    """Only 'KIN' is a known sentinel; anything else is a corrupt file."""
    with pytest.raises(ValueError, match="could not convert string to float"):
        to_raw_params(tokenize_lines(["SINI GARBAGE", "KIN 71.9"]))


# ---------------------------------------------------------------------------
# tempo2 "BINARY T2" resolution
#
# T2 is a generic superset parameterisation with no PINT/JaxPINT equivalent --
# tempo2 resolves it at runtime from which parameters are set. IPTA DR1 (93
# pars) and EPTA DR2 (53) ship it, so it has to be recovered the same way:
# the simplest model whose parameter set covers everything the file specifies.
# Mirrors PINT's guess_binary_model (models/model_builder.py:969); verified
# to agree with PINT on the cases below.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "present,expected",
    [
        # Real IPTA DR1 J1713+0747 -- KIN/KOM force DDK (PINT suggests DDK too).
        ({"PB", "T0", "A1", "OM", "ECC", "M2", "KOM", "KIN"}, "DDK"),
        # Real EPTA DR2 J1909-3744 -- TASC/EPS1/EPS2 put it in the ELL1 family.
        (
            {"PB", "A1", "PBDOT", "A1DOT", "TASC", "EPS1", "EPS2", "M2", "SINI"},
            "ELL1",
        ),
        ({"PB", "A1", "T0", "OM", "ECC"}, "BT"),  # simplest that fits
        ({"PB", "A1", "T0", "OM", "ECC", "M2", "SINI"}, "DD"),
        ({"PB", "A1", "TASC", "EPS1", "EPS2"}, "ELL1"),
        ({"PB", "A1", "TASC", "EPS1", "EPS2", "H3", "H4"}, "ELL1H"),
        ({"PB", "A1", "T0", "OM", "ECC", "SHAPMAX"}, "DDS"),
        ({"PB", "A1", "T0", "OM", "ECC", "MTOT"}, "DDGR"),
    ],
)
def test_guess_binary_model(present, expected):
    from jaxpint.par import spec as S

    assert S.guess_binary_model(present) == expected


def test_guess_binary_model_incompatible_returns_none():
    """T0 and TASC are different parameterisations; no model holds both."""
    from jaxpint.par import spec as S

    assert S.guess_binary_model({"PB", "A1", "T0", "TASC", "EPS1"}) is None


def test_binary_t2_resolves_to_concrete_model():
    from jaxpint.par.components import detect_components
    from jaxpint.par.registry import BinaryModel

    _, model = detect_components(
        {"PB", "A1", "TASC", "EPS1", "EPS2", "M2", "SINI"}, "T2"
    )
    assert model is BinaryModel.ELL1


def test_binary_t2_incompatible_params_raise():
    from jaxpint.par.components import detect_components

    with pytest.raises(ValueError, match="BINARY T2 could not be resolved"):
        detect_components({"PB", "A1", "T0", "TASC", "EPS1"}, "T2")


# -- drift guards for the hand-maintained table ------------------------------


def test_binary_model_params_are_known():
    """Every name in BINARY_MODEL_PARAMS must be a real declared parameter."""
    from jaxpint.par import spec as S

    known = S.KNOWN_PARAMS
    unknown = {
        name
        for names in S.BINARY_MODEL_PARAMS.values()
        for name in names
        if name not in known
    }
    assert not unknown, f"BINARY_MODEL_PARAMS names not declared anywhere: {unknown}"


def test_binary_core_names_match():
    """BINARY_CORE_NAMES must stay in step with BINARY_CORE itself."""
    from jaxpint.binary._param_decls import BINARY_CORE, BINARY_CORE_NAMES

    assert BINARY_CORE_NAMES == {d.name for d in BINARY_CORE}


def test_binary_priority_covers_every_model():
    """A model missing from the priority list can never be guessed."""
    from jaxpint.par import spec as S

    assert set(S.BINARY_MODEL_PARAMS) == set(S.BINARY_PRIORITY)


def test_model_params_cover_every_binary_model():
    """The per-model table is keyed by BinaryModel *values*; pin them."""
    from jaxpint.binary._param_decls import MODEL_EXTRA_PARAMS
    from jaxpint.par.registry import BinaryModel

    assert set(MODEL_EXTRA_PARAMS) == {m.value for m in BinaryModel}


def test_model_extras_belong_to_the_implementing_class():
    """Each model's extras must actually be declared by the class that builds it.

    Ten models share six classes, so a class's PARAMS is the union over its
    variants and cannot separate DDS's SHAPMAX from DDH's H3.  It *can* catch a
    parameter filed under the wrong family entirely -- e.g. KIN (DDK) placed
    under ELL1H -- which the name/priority guards do not.
    """
    from jaxpint.binary._build import _MODEL_CLASSES
    from jaxpint.binary._param_decls import MODEL_EXTRA_PARAMS

    for model, cls in _MODEL_CLASSES.items():
        declared = {d.name for d in cls.PARAMS}
        stray = MODEL_EXTRA_PARAMS[model] - declared
        assert not stray, f"{model} extras not declared by {cls.__name__}: {stray}"


# ---------------------------------------------------------------------------
# TCB -> TDB conversion
#
# Verified bit-exact against PINT's convert_tcb_tdb on F0/F1/F2/DM/PX/PM*/PB/
# A1/M2/ECC/OM/SINI/PEPOCH/T0. Refuses on TZRMJD (see below) and on any numeric
# parameter with no known dimensionality.
# ---------------------------------------------------------------------------

_TCB_BASE = (
    "PSRJ J1\nRAJ 12:34:56.0\nDECJ 56:12:00.0\n"
    "F0 100.0\nF1 -1e-15\nPEPOCH 55000\nDM 10.0\n"
)


def _load_par(tmp_path, text, name="t.par"):
    p = tmp_path / name
    p.write_text(text)
    return par.get_model(str(p))


def test_tcb_scales_by_effective_dimensionality(tmp_path):
    from jaxpint.par._tcb_tables import IFTE_K

    r = _load_par(tmp_path, _TCB_BASE + "UNITS TCB\n")
    i = {n: k for k, n in enumerate(r.params.names)}
    v = np.asarray(r.params.values)
    # F0 is s^-1 (n=-1) -> K^1 ; F1 is s^-2 (n=-2) -> K^2 ; DM (n=-1) -> K^1
    assert np.isclose(v[i["F0"]], 100.0 * float(IFTE_K), rtol=0, atol=1e-12)
    assert np.isclose(v[i["F1"]], -1e-15 * float(IFTE_K) ** 2, rtol=1e-15)
    assert np.isclose(v[i["DM"]], 10.0 * float(IFTE_K), rtol=0, atol=1e-12)
    # ...and the file is TDB afterwards, so nothing downstream re-converts it.
    assert r.metadata["UNITS"] == "TDB"


def test_tcb_transforms_epochs(tmp_path):
    """~15.9 s at MJD 55000; must survive the int/frac split."""
    r = _load_par(tmp_path, _TCB_BASE + "UNITS TCB\n")
    i = {n: k for k, n in enumerate(r.params.names)}
    mjd = r.params.epoch_int_values["PEPOCH"] + float(np.asarray(r.params.values)[i["PEPOCH"]])
    assert np.isclose(mjd, 54999.99981617038, atol=1e-9)


def test_tdb_par_is_untouched(tmp_path):
    r = _load_par(tmp_path, _TCB_BASE + "UNITS TDB\n")
    i = {n: k for k, n in enumerate(r.params.names)}
    assert float(np.asarray(r.params.values)[i["F0"]]) == 100.0


def test_tcb_refuses_tzrmjd(tmp_path):
    """TZRMJD is the phase anchor and PINT cannot convert it; converting the
    other epochs while it stays put corrupts absolute phase by ~2760 turns."""
    with pytest.raises(NotImplementedError, match="TZRMJD"):
        _load_par(tmp_path, _TCB_BASE + "UNITS TCB\nTZRMJD 55000\nTZRFRQ 1400\n")


def test_tcb_refuses_unknown_dimensionality(tmp_path):
    """Scaling by the wrong power of IFTE_K is worse than refusing."""
    from jaxpint.par import _tcb_tables as T

    saved = T.SCALE_DIMENSIONALITY.pop("PX", None)
    try:
        with pytest.raises(NotImplementedError, match="no TCB scaling is known"):
            _load_par(tmp_path, _TCB_BASE + "PX 0.5\nUNITS TCB\n")
    finally:
        if saved is not None:
            T.SCALE_DIMENSIONALITY["PX"] = saved


def test_tcb_noise_params_pass_through(tmp_path):
    """EFAC/EQUAD are left alone (PINT does too): dimensionless or ~8 fs."""
    r = _load_par(
        tmp_path, _TCB_BASE + "UNITS TCB\nEFAC -f L 1.2\nEQUAD -f L 0.5\n"
    )
    i = {n: k for k, n in enumerate(r.params.names)}
    v = np.asarray(r.params.values)
    assert float(v[i["EFAC1"]]) == 1.2
    assert np.isclose(float(v[i["EQUAD1"]]), 0.5e-6, rtol=1e-12)  # us -> s only


@pytest.mark.parametrize(
    "name,expected",
    [("F0", -1), ("F1", -2), ("F2", -3), ("DM", -1), ("DM1", -2),
     ("NE_SW", -2), ("NE_SW1", -3), ("DMX_0042", -1), ("JUMP3", 1),
     ("PEPOCH", "mjd"), ("GLEP_2", "mjd"), ("TZRMJD", None), ("EQUAD1", None)],
)
def test_dimensionality_resolution(name, expected):
    """Derivative families vary with index; instance families collapse."""
    from jaxpint.par._tcb_tables import dimensionality_for

    assert dimensionality_for(name) == expected


def test_tcb_tables_are_up_to_date():
    """The committed TCB tables must match a fresh extraction from PINT.

    They encode physics (each parameter's effective dimensionality) that we
    deliberately do not re-derive by hand, so the only guard against drift --
    PINT gaining parameters, or JaxPINT declaring new ones -- is re-running the
    extractor and comparing. See tools/regen_tcb_tables.py.
    """
    pytest.importorskip("pint")
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, str(_REPO / "tools" / "regen_tcb_tables.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr or r.stdout


# ---------------------------------------------------------------------------
# Declared-but-unconsumed invariant
#
# Every serious bug found while auditing the parser had one shape: a parameter
# declared in the schema, parsed into the model, and then read by nothing.
# `UNITS TCB` loaded as TDB; `CLK` was overridden by a hardcoded BIPM2023;
# `FB2+` was dropped from the orbital phase. Each produced wrong numbers with no
# error, and each was found by accident rather than by a check.
#
# The unrecognized-parameter warning in text_adapter does NOT cover this: these
# names ARE recognized, so they resolve fine and never reach that path.
#
# This test makes the class visible. Adding a ParamDecl that nothing consumes
# now fails until it is either implemented or explicitly classified below.
# ---------------------------------------------------------------------------

# Files that *name* parameters without acting on them -- declaration sites,
# generated lookup tables, ignore-lists. Counting these as "consumed" would make
# the test vacuous: _tcb_generated.py alone lists ~163 names as dict keys.
_INERT_MODULES = {
    "model.py",
    "_tcb_generated.py",
    "_tcb_tables.py",
    "spec.py",
    "registry_table.py",
}

# Metadata and fit-result bookkeeping. Ignoring these is correct, not a gap:
# PINT writes most of them back out after a fit and consumes none of them in
# the timing model (fitter.py update_model; timing_model.py:2839 groups them).
_INFORMATIONAL = {
    "PSR",     # pulsar name
    "NTOA",    # fit outputs, written back by a fitter
    "CHI2",
    "CHI2R",
    "TRES",
    "DMRES",
    "START",   # fit range; PINT force-freezes these and never filters TOAs by them
    "FINISH",
    "RM",      # rotation measure: affects polarization, not timing
}

# Recognized physics we do not implement. Tracked debt, NOT permission -- each
# is a par file asking for something JaxPINT silently does not do.
#   TIMEEPH   -- TT->TDB series. Both JaxPINT and PINT always compute FB90, so
#                the numbers agree; only the notification differs (PINT warns).
#   T2CMETHOD -- terrestrial->celestial method; PINT coerces to IAU2000B + warns.
#   DILATEFREQ-- PINT warns it does not support 'DILATEFREQ Y'.
_UNIMPLEMENTED = {"TIMEEPH", "T2CMETHOD", "DILATEFREQ"}


def _declared_and_consumed():
    import re

    root = _REPO / "jaxpint"
    declared = re.findall(
        r'ParamDecl\("([A-Z0-9_]+)"', (root / "model.py").read_text()
    )
    body = "".join(
        p.read_text() for p in root.rglob("*.py") if p.name not in _INERT_MODULES
    )
    unconsumed = {
        d for d in declared if f'"{d}"' not in body and f"'{d}'" not in body
    }
    return set(declared), unconsumed


def test_no_undeclared_silently_unconsumed_params():
    """Every TimingModel param is consumed, informational, or tracked debt."""
    _declared, unconsumed = _declared_and_consumed()
    unclassified = unconsumed - _INFORMATIONAL - _UNIMPLEMENTED
    assert not unclassified, (
        f"parameter(s) parsed but read by nothing: {sorted(unclassified)}. "
        "A par file setting these gets a silently incomplete model. Either "
        "consume them, or add them to _INFORMATIONAL / _UNIMPLEMENTED with a "
        "reason."
    )


def test_classification_lists_are_not_stale():
    """A param that becomes consumed must leave the exemption lists.

    Without this the lists rot into permanent excuses -- CLOCK sat in this
    category until the CLK-derivation fix, and should not still be exempt.
    """
    declared, unconsumed = _declared_and_consumed()
    stale = (_INFORMATIONAL | _UNIMPLEMENTED) - unconsumed
    assert not stale, (
        f"{sorted(stale)} are now consumed (or no longer declared); remove them "
        "from _INFORMATIONAL / _UNIMPLEMENTED."
    )
