"""Tests for the ``.par`` and ``.tim`` writers.

Three fidelity levels, each pinned separately:

1. **Self round-trip** -- write then re-read through the native parser and get
   the same object back (bit-level for ``.tim`` records; <= 1 ulp for par
   values, exact for names/frozen/epochs/components).
2. **PINT cross-read** -- PINT parses our output and agrees on the values.
3. **Workflow closure** -- a JaxPINT *fit* written with ``as_parfile`` gives
   PINT the identical post-fit rms, and natively *generated* TOAs written with
   ``write_tim`` re-read (by us and by PINT) to the same corrected times.

Known, deliberate tolerances:

* ``toa_data_to_raw`` (jaxpint.loaders.native -- loader-stage physics, kept
  out of the text-layer tim package) un-applies clock corrections from the realization
  stamped on the TOAData; PINT re-applies its own clock *data*, whose vintage
  differs from the vendored IPTA set at the ~0.3 us level.
* At finite frequency PINT dedisperses these files at the topocentric
  frequency (it warns so itself) where JaxPINT used barycentric -- a ~1e-4
  relative frequency difference that maps to ~1e-4 s at DM 224. Cross-checks
  therefore use infinite frequency, where the convention is moot.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np
import pytest

import jaxpint.par as jpar
from jaxpint.par.writer import as_parfile
from jaxpint.loaders.native import toa_data_to_raw
from jaxpint.tim import read_tim, write_tim
from jaxpint.tim.writer import format_toa_line

_DATA = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"

# Every vendored par the native parser can load. As of 2026-07-22 that is all
# 25 of them -- the try/except is future-proofing for a vendored par that the
# parser rejects, not a description of the current corpus. Enumerated at import
# so a new vendored par joins the sweep automatically.
_PARSEABLE_PARS = []
for _p in sorted(_DATA.glob("*.par")):
    try:
        jpar.get_model(str(_p))
    except Exception:
        continue
    _PARSEABLE_PARS.append(_p.name)


# ---------------------------------------------------------------------------
# .tim writer
# ---------------------------------------------------------------------------


def test_format_toa_line_fields():
    from jaxpint.tim import RawTOA

    line = format_toa_line(
        RawTOA(
            mjd_int=55000.0,
            mjd_frac=0.5,
            error_s=1.5e-6,
            freq_mhz=1400.0,
            obs="gbt",
            flags={"fe": "Rcvr_800"},
        ),
        name="t0",
    )
    # Shortest-round-trip formatting: no fixed-width padding (see the writer's
    # _shortest docstring for the corpus failures fixed-width caused).
    assert line.split() == [
        "t0",
        "1400.0",
        "55000.5",
        "1.5",
        "gbt",
        "-fe",
        "Rcvr_800",
    ]


def test_infinite_frequency_writes_as_zero():
    """The parser's 0 -> inf convention, inverted on the way out."""
    from jaxpint.tim import RawTOA

    line = format_toa_line(
        RawTOA(
            mjd_int=55000.0, mjd_frac=0.5, error_s=1e-6, freq_mhz=float("inf"), obs="@"
        )
    )
    assert line.split()[1] == "0.0"


def test_bad_frac_rejected():
    from jaxpint.tim import RawTOA

    with pytest.raises(ValueError, match="mjd_frac"):
        format_toa_line(
            RawTOA(
                mjd_int=55000.0, mjd_frac=1.5, error_s=1e-6, freq_mhz=1400.0, obs="gbt"
            )
        )


_ALL_TIMS = sorted(
    p.name for p in list(_DATA.glob("*.tim")) + list(_DATA.glob("*.toa"))
)


@pytest.mark.parametrize("tim", _ALL_TIMS)
def test_tim_round_trip_is_exact(tmp_path, tim):
    """read -> write -> read reproduces every record bit-for-bit, corpus-wide.

    Bit-level, not toleranced: the writer prints shortest-round-trip digits
    from the stored values, so nothing is recomputed and nothing can drift.
    The corpus is what found the three formatting traps recorded in the writer
    docstrings (%.6f frequency truncation, the 16-vs-17 significant-digit MJD
    fraction, the us <-> s ulp), so the whole corpus stays in the sweep.

    The one semantic mapping: delta_pulse_number is serialized as -padd, so
    the flag dicts are compared modulo that key while dpn itself is compared
    exactly.
    """
    first = read_tim(str(_DATA / tim))
    out = tmp_path / "rt.tim"
    write_tim(first, out)
    second = read_tim(str(out))
    assert len(first.toas) == len(second.toas)
    for a, b in zip(first.toas, second.toas):
        assert (a.mjd_int, a.mjd_frac) == (b.mjd_int, b.mjd_frac)
        assert (a.freq_mhz, a.error_s, a.obs) == (b.freq_mhz, b.error_s, b.obs)
        assert a.delta_pulse_number == b.delta_pulse_number
        fa = {k: v for k, v in a.flags.items() if k != "padd"}
        fb = {k: v for k, v in b.flags.items() if k != "padd"}
        assert fa == fb


def test_parkes_corpus_converts_to_tempo2(tmp_path):
    """Reading a Parkes file and writing it emits valid FORMAT 1 -- a working
    dialect converter, and the only way to hand 0437's TOAs to PINT (which
    cannot read the Parkes original at all)."""
    first = read_tim(str(_DATA / "0437.tim"))
    out = tmp_path / "0437_t2.tim"
    write_tim(first, out)
    assert out.read_text().startswith("FORMAT 1")
    second = read_tim(str(out))
    assert len(second.toas) == len(first.toas) == 5163
    worst = max(
        abs((a.mjd_int - b.mjd_int) + (a.mjd_frac - b.mjd_frac)) * 86400.0
        for a, b in zip(first.toas, second.toas)
    )
    assert worst == 0.0


@pytest.mark.slow
def test_written_fake_toas_reread_to_same_times(tmp_path):
    """generate -> un-correct -> write -> native re-read: TDB round-trips.

    The un-correction (fixed point over the clock chain) is the part that can
    silently be wrong by a whole clock correction (~us), so the tolerance is
    set well below that scale.
    """
    from jaxpint import native
    from jaxpint.simulation import make_fake_toas_uniform

    par = jpar.get_model(str(_DATA / "NGC6440E.par"))
    td = make_fake_toas_uniform(
        53000.0, 54000.0, 40, par, obs="gbt", freq_mhz=1400.0, error_us=1.0
    )
    out = tmp_path / "fake.tim"
    write_tim(toa_data_to_raw(td), out)
    td2 = native.get_TOAs(str(out), par)

    tdb1 = np.asarray(td.tdb_int) + np.asarray(td.tdb_frac)
    tdb2 = np.asarray(td2.tdb_int) + np.asarray(td2.tdb_frac)
    assert np.abs(tdb2 - tdb1).max() * 86400.0 < 1e-8  # measured ~1e-10


@pytest.mark.slow
def test_written_fake_toas_evaluate_clean_in_pint(tmp_path):
    """PINT reads our written fake TOAs and sees ~zero residuals.

    Infinite frequency, so PINT's topocentric-dedispersion convention (see
    module docstring) cannot enter; what remains is clock-data vintage at the
    ~0.3 us level, hence the 2 us ceiling.
    """
    pytest.importorskip("pint")
    import pint.models
    import pint.residuals
    import pint.toa

    from jaxpint.simulation import make_fake_toas_uniform

    par = jpar.get_model(str(_DATA / "NGC6440E.par"))
    td = make_fake_toas_uniform(
        53000.0, 54000.0, 40, par, obs="gbt", freq_mhz=0.0, error_us=1.0
    )
    out = tmp_path / "fake0.tim"
    write_tim(toa_data_to_raw(td), out)

    m = pint.models.get_model(str(_DATA / "NGC6440E.par"))
    t = pint.toa.get_TOAs(str(out), model=m)
    r = pint.residuals.Residuals(t, m).time_resids.to("s").value
    assert np.abs(r).max() < 2e-6  # measured 7.4e-07


# ---------------------------------------------------------------------------
# .par writer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("par", _PARSEABLE_PARS)
def test_par_round_trip(par):
    """write -> re-parse reproduces the full ParResult, corpus-wide.

    Values to <= 1e-12 relative (the only loss is one ulp where a unit
    conversion inverts, e.g. EQUAD us <-> s); everything discrete -- names,
    frozen flags, epoch integer days, components, binary model, int/bool
    params, mask selectors -- must be exact.
    """
    a = jpar.get_model(str(_DATA / par))
    b = jpar.get_model(io.StringIO(as_parfile(a)))

    na, nb = list(a.params.names), list(b.params.names)
    assert set(na) == set(nb)
    idx = [nb.index(n) for n in na]
    va = np.asarray(a.params.values)
    vb = np.asarray(b.params.values)[idx]
    rel = np.abs(vb - va) / np.maximum(np.abs(va), 1e-30)
    assert rel.max() < 1e-12, f"{na[int(np.argmax(rel))]}: rel {rel.max():.2e}"
    assert tuple(b.params.frozen_mask[i] for i in idx) == a.params.frozen_mask
    assert b.params.epoch_int_values == a.params.epoch_int_values
    assert b.component_set == a.component_set
    assert b.binary_model == a.binary_model
    assert b.int_params == a.int_params
    assert b.bool_params == a.bool_params
    assert set(b.mask_info) == set(a.mask_info)
    for name, info in a.mask_info.items():
        got = b.mask_info[name]
        assert (got.key.lstrip("-"), got.key_value, got.key_value2) == (
            info.key.lstrip("-"),
            info.key_value,
            info.key_value2,
        ), name


def test_fitted_values_are_what_gets_written():
    """params= overrides the stored values -- the persistence entry point."""
    a = jpar.get_model(str(_DATA / "NGC6440E.par"))
    names = list(a.params.names)
    import jax.numpy as jnp

    shifted = a.params.with_values(
        jnp.asarray(np.asarray(a.params.values)).at[names.index("F0")].add(1e-9)
    )
    b = jpar.get_model(io.StringIO(as_parfile(a, params=shifted)))
    f0_b = float(np.asarray(b.params.values)[list(b.params.names).index("F0")])
    f0_a = float(np.asarray(a.params.values)[names.index("F0")])
    assert f0_b == pytest.approx(f0_a + 1e-9, abs=1e-15)


@pytest.mark.slow
def test_pint_parses_our_par():
    """PINT reads the serialized B1855 model and agrees on 13 key params."""
    pytest.importorskip("pint")
    import pint.models

    a = jpar.get_model(str(_DATA / "B1855+09_NANOGrav_9yv1.gls.par"))
    ours = pint.models.get_model(io.StringIO(as_parfile(a)))
    orig = pint.models.get_model(str(_DATA / "B1855+09_NANOGrav_9yv1.gls.par"))
    for pn in (
        "F0",
        "F1",
        "DM",
        "PB",
        "A1",
        "T0",
        "ECC",
        "OM",
        "SINI",
        "M2",
        "ELAT",
        "ELONG",
        "PX",
    ):
        va, vo = getattr(ours, pn).value, getattr(orig, pn).value
        assert abs(va - vo) / max(abs(vo), 1e-30) < 1e-11, pn


@pytest.mark.slow
def test_pint_reproduces_our_fit_from_written_par():
    """The full persistence loop: fit natively, write the par, PINT evaluates.

    PINT's rms under our fitted par must equal JaxPINT's own post-fit rms --
    this is the number that says a fitted model survives serialization.
    """
    pytest.importorskip("pint")
    import pint.models
    import pint.residuals
    import pint.toa

    from jaxpint import WLSFitter, build_model, native

    parsed = jpar.get_model(str(_DATA / "NGC6440E.par"))
    td = native.get_TOAs(str(_DATA / "NGC6440E.tim"), parsed)
    tm, nm = build_model(parsed, td)
    res = WLSFitter(tm, td, parsed.params, noise_model=nm).fit_toas()

    m_fit = pint.models.get_model(io.StringIO(as_parfile(parsed, params=res.params)))
    t = pint.toa.get_TOAs(str(_DATA / "NGC6440E.tim"), model=m_fit)
    r = pint.residuals.Residuals(t, m_fit).time_resids.to("s").value
    # JaxPINT's post-fit rms on this dataset is 3.3334e-05 s (chi2 59.5747).
    assert r.std() == pytest.approx(3.3334e-5, rel=1e-3)
