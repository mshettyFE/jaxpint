"""Tests for ``jaxpint.loaders.nanograv.load_nanograv_pta``.

Uses PINT's bundled ``B1855+09_NANOGrav_9yv1`` par/tim pair as a stand-in for a
single-pulsar "PTA". Stages the pair into a tmpdir under both Zenodo layouts
(``par/`` + ``tim/`` siblings, and ``<PSR>/<files>``) to exercise the
discovery branch.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("pint")  # optional dependency; skip module if absent
from pint.config import examplefile

from jaxpint import NanogravPTA, load_nanograv_pta
from jaxpint.pta.likelihood import PTAConfig


def _example_par_tim() -> tuple[Path, Path]:
    return (
        Path(examplefile("B1855+09_NANOGrav_9yv1.gls.par")),
        Path(examplefile("B1855+09_NANOGrav_9yv1.tim")),
    )


def _stage_par_tim_layout(root: Path) -> None:
    """``<root>/par/<stem>.par`` + ``<root>/tim/<stem>.tim``."""
    par_src, tim_src = _example_par_tim()
    (root / "par").mkdir(parents=True)
    (root / "tim").mkdir(parents=True)
    shutil.copy2(par_src, root / "par" / par_src.name)
    shutil.copy2(tim_src, root / "tim" / tim_src.name)


def _stage_per_pulsar_layout(root: Path) -> None:
    """``<root>/<PSR>/<stem>.par`` + sibling ``.tim``."""
    par_src, tim_src = _example_par_tim()
    psr_dir = root / "B1855+09"
    psr_dir.mkdir(parents=True)
    shutil.copy2(par_src, psr_dir / par_src.name)
    shutil.copy2(tim_src, psr_dir / tim_src.name)


@pytest.mark.parametrize("stager", [_stage_par_tim_layout, _stage_per_pulsar_layout])
def test_load_nanograv_pta_layouts(tmp_path, stager):
    stager(tmp_path)

    psrs = load_nanograv_pta(tmp_path, planets=False)

    assert isinstance(psrs, NanogravPTA)
    assert psrs.pulsar_names == ("B1855+09",)
    assert (
        len(psrs.toa_data_list)
        == len(psrs.pulsar_params_list)
        == len(psrs.timing_models)
        == len(psrs.noise_models)
        == 1
    )
    # Real TOAs (not synthetic) → at least a few hundred entries.
    assert psrs.toa_data_list[0].mjd_int.shape[0] > 100

    # Result must drop straight into PTAConfig without further massaging.
    cfg = PTAConfig(
        toa_data_list=psrs.toa_data_list,
        timing_models=psrs.timing_models,
        noise_models=psrs.noise_models,
        signal_injectors=(),
    )
    assert cfg.n_pulsars == 1


def test_load_nanograv_pta_pulsar_names_and_exclude(tmp_path):
    _stage_per_pulsar_layout(tmp_path)

    # Explicit selection of a known pulsar works.
    psrs = load_nanograv_pta(tmp_path, pulsar_names=["B1855+09"], planets=False)
    assert psrs.pulsar_names == ("B1855+09",)

    # Unknown pulsar → KeyError.
    with pytest.raises(KeyError):
        load_nanograv_pta(tmp_path, pulsar_names=["J9999+9999"], planets=False)

    # Excluding the only pulsar empties the set → ValueError.
    with pytest.raises(ValueError):
        load_nanograv_pta(tmp_path, exclude=["B1855+09"], planets=False)


def test_load_nanograv_pta_planet_shapiro(tmp_path):
    """Real par files often set PLANET_SHAPIRO Y; the bridge must populate
    planet positions so SolarSystemShapiroDelay's runtime check doesn't trip."""
    import jax.numpy as jnp
    from jaxpint import single_pulsar_logL

    par_src, tim_src = _example_par_tim()
    psr_dir = tmp_path / "B1855+09"
    psr_dir.mkdir(parents=True)
    par_dst = psr_dir / par_src.name
    shutil.copy2(tim_src, psr_dir / tim_src.name)

    # Force PLANET_SHAPIRO Y. The bundled par file has "PLANET_SHAPIRO N";
    # rewrite that line, or append the directive if absent.
    par_text = par_src.read_text()
    if "PLANET_SHAPIRO" in par_text:
        par_text = (
            "\n".join(
                "PLANET_SHAPIRO Y"
                if line.strip().startswith("PLANET_SHAPIRO")
                else line
                for line in par_text.splitlines()
            )
            + "\n"
        )
    else:
        par_text += "PLANET_SHAPIRO Y\n"
    par_dst.write_text(par_text)

    psrs = load_nanograv_pta(tmp_path)

    assert psrs.toa_data_list[0].planet_positions is not None
    # Likelihood evaluation should now succeed instead of raising
    # "planet_shapiro=True but toa_data.planet_positions is None".
    logL = single_pulsar_logL(
        psrs.toa_data_list[0],
        psrs.timing_models[0],
        psrs.noise_models[0],
        psrs.pulsar_params_list[0],
    )
    assert jnp.isfinite(logL)


def test_load_nanograv_pta_synthesizes_tnredamp_from_rnamp(tmp_path):
    """NANOGrav 15-yr par files specify red noise via tempo2-style RNAMP/RNIDX
    only; the bridge must synthesize TNREDAMP/TNREDGAM (the names PLRedNoise
    reads) using PINT's conversion. Mirror that situation by stripping the
    bundled par file's TNRedAmp/TNRedGam/TNRedC lines."""
    import math
    import jax.numpy as jnp
    from jaxpint import single_pulsar_logL

    par_src, tim_src = _example_par_tim()
    psr_dir = tmp_path / "B1855+09"
    psr_dir.mkdir(parents=True)
    shutil.copy2(tim_src, psr_dir / tim_src.name)

    par_text = par_src.read_text()
    stripped = (
        "\n".join(
            line
            for line in par_text.splitlines()
            if not line.strip().startswith(("TNRedAmp", "TNRedGam", "TNRedC"))
        )
        + "\n"
    )
    assert "RNAMP" in stripped and "RNIDX" in stripped
    assert "TNRedAmp" not in stripped
    (psr_dir / par_src.name).write_text(stripped)

    psrs = load_nanograv_pta(tmp_path, planets=False)
    params = psrs.pulsar_params_list[0]

    assert "TNREDAMP" in params.names
    assert "TNREDGAM" in params.names
    assert params.names.count("TNREDAMP") == 1
    assert params.names.count("TNREDGAM") == 1

    # Bundled par values: RNAMP=0.017173, RNIDX=-4.91353.
    fac = (86400.0 * 365.24 * 1e6) / (2.0 * math.pi * math.sqrt(3.0))
    expected_tnredamp = math.log10(0.017173 / fac)
    expected_tnredgam = 4.91353
    got_tnredamp = float(params.values[params.names.index("TNREDAMP")])
    got_tnredgam = float(params.values[params.names.index("TNREDGAM")])
    assert abs(got_tnredamp - expected_tnredamp) < 1e-5
    assert abs(got_tnredgam - expected_tnredgam) < 1e-10

    # Smoke test: prior bug raised KeyError("TNREDAMP") here.
    logL = single_pulsar_logL(
        psrs.toa_data_list[0],
        psrs.timing_models[0],
        psrs.noise_models[0],
        params,
    )
    assert jnp.isfinite(logL)


def test_load_nanograv_pta_does_not_synthesize_when_tnredamp_set(tmp_path):
    """The unmodified bundled par file has TNRedAmp populated. Synthesis must
    not fire — TNREDAMP appears once (from the normal extraction path), not
    twice."""
    _stage_per_pulsar_layout(tmp_path)

    psrs = load_nanograv_pta(tmp_path, planets=False)
    params = psrs.pulsar_params_list[0]

    assert params.names.count("TNREDAMP") == 1
    assert params.names.count("TNREDGAM") == 1


def test_synthesize_pb_from_fb0():
    """FB0 only → PB synthesized as 1 / (FB0 * 86400) days, appended to the list."""
    from jaxpint.par import ParamKind, RawParam
    from jaxpint.par.aliases import synthesize_pb_from_fb

    fb0_hz = 8.3387216e-5  # J0023+0923 value
    raw = [RawParam("FB0", ParamKind.FLOAT, value=fb0_hz, unit="Hz", frozen=False)]
    synthesize_pb_from_fb(raw)
    synth = raw[1:]

    assert [r.name for r in synth] == ["PB"]
    assert abs(synth[0].value - 1.0 / (fb0_hz * 86400.0)) < 1e-15
    assert synth[0].unit == "d"
    assert synth[0].frozen is False


def test_synthesize_pbdot_from_fb1():
    """FB0 and FB1 set → both PB and PBDOT synthesized."""
    from jaxpint.par import ParamKind, RawParam
    from jaxpint.par.aliases import synthesize_pb_from_fb

    fb0_hz = 8.3387216e-5
    fb1 = 3.6553667e-20
    raw = [
        RawParam("FB0", ParamKind.FLOAT, value=fb0_hz, unit="Hz", frozen=False),
        RawParam("FB1", ParamKind.FLOAT, value=fb1, unit="Hz / s", frozen=True),
    ]
    synthesize_pb_from_fb(raw)
    synth = raw[2:]

    assert [r.name for r in synth] == ["PB", "PBDOT"]
    assert abs(synth[0].value - 1.0 / (fb0_hz * 86400.0)) < 1e-15
    assert abs(synth[1].value - (-fb1 / (fb0_hz * fb0_hz))) < 1e-25
    assert [r.unit for r in synth] == ["d", "s / s"]
    assert [r.frozen for r in synth] == [False, True]


def test_synthesize_pb_skips_when_pb_set():
    """If PB is already present, synthesis must not fire (would otherwise
    duplicate PB in the parameter vector)."""
    from jaxpint.par import ParamKind, RawParam
    from jaxpint.par.aliases import synthesize_pb_from_fb

    raw = [
        RawParam("FB0", ParamKind.FLOAT, value=8.3387216e-5, unit="Hz"),
        RawParam("PB", ParamKind.FLOAT, value=0.139, unit="d"),  # days
    ]
    synthesize_pb_from_fb(raw)

    assert [r.name for r in raw] == ["FB0", "PB"]  # nothing appended


def test_synthesize_pbdot_skips_when_pbdot_set():
    """PB synthesized, but PBDOT already present → don't overwrite."""
    from jaxpint.par import ParamKind, RawParam
    from jaxpint.par.aliases import synthesize_pb_from_fb

    raw = [
        RawParam("FB0", ParamKind.FLOAT, value=8.3387216e-5, unit="Hz"),
        RawParam("FB1", ParamKind.FLOAT, value=3.6553667e-20, unit="Hz / s"),
        RawParam("PBDOT", ParamKind.FLOAT, value=1e-12, unit="s / s"),
    ]
    synthesize_pb_from_fb(raw)
    synth = raw[3:]

    assert [r.name for r in synth] == ["PB"]


def test_synthesize_pb_noop_without_fb0():
    """No FB0 → no synthesis, regardless of other state."""
    from jaxpint.par.aliases import synthesize_pb_from_fb

    raw = []
    synthesize_pb_from_fb(raw)

    assert raw == []


def _ell1h_par_result(*, h3=False, h4=False, stigma=False, nharms=None):
    """Minimal ParResult for the ELL1H branch of ``_build_binary``. Includes
    only the orbital params the ELL1H builder reads — H3/H4/STIGMA are
    appended only when the matching kwarg is set."""
    import jax.numpy as jnp
    from jaxpint.par import BinaryModel, ParResult
    from jaxpint.types import ParameterVector

    names = ["PB", "TASC", "A1", "EPS1", "EPS2"]
    values = [0.7, 58314.0, 3.7, 2.6e-6, 2.1e-6]
    if h3:
        names.append("H3")
        values.append(1.0e-7)
    if h4:
        names.append("H4")
        values.append(0.5e-7)
    if stigma:
        names.append("STIGMA")
        values.append(0.3)
    int_params = {"NHARMS": nharms} if nharms is not None else {}
    return ParResult(
        params=ParameterVector(
            values=jnp.asarray(values),
            frozen_mask=tuple(False for _ in names),
            names=tuple(names),
            units=tuple("" for _ in names),
            epoch_int_values={},
        ),
        binary_model=BinaryModel.ELL1H,
        int_params=int_params,
    )


def _ell1h_ctx(**kwargs):
    """Wrap an ELL1H ParResult in a BuildContext for ``_build_binary``.

    ``_build_binary`` takes a single ``BuildContext``; the ELL1H branch only
    reads ``ctx.par``, so the astrometry fields are placeholders.
    """
    from jaxpint.model_builder import BuildContext

    return BuildContext(
        par=_ell1h_par_result(**kwargs),
        toa_data=None,
        raj="RAJ",
        decj="DECJ",
        pmra=None,
        pmdec=None,
        posepoch=None,
        obliquity_arcsec=None,
    )


def test_ell1h_no_shapiro_params_uses_none_mode():
    """ELL1H par with no H3/H4/STIGMA must produce shapiro_mode='none' so
    the binary model never tries to read an absent H3 at logL time. Mirrors
    NANOGrav 15-yr J1802-2124 (BINARY ELL1H, NHARMS only)."""
    from jaxpint.model_builder import _build_binary

    binary = _build_binary(_ell1h_ctx())
    assert binary.shapiro_mode == "none"
    assert binary.h3_name is None
    assert binary.stigma_name is None
    assert binary.h4_name is None


def test_ell1h_with_h3_only_uses_h3nharms():
    """H3 set but H4/STIGMA absent → Freire-Wex 2010 H3-only Fourier mode.
    Mirrors NANOGrav 15-yr J2145-0750 and J2317+1439 (BINARY ELL1H, H3 set,
    no STIGMA/H4)."""
    from jaxpint.model_builder import _build_binary

    binary = _build_binary(_ell1h_ctx(h3=True, nharms=3))
    assert binary.shapiro_mode == "h3nharms"
    assert binary.h3_name == "H3"
    assert binary.nharms == 3


def test_ell1h_h3_only_default_nharms_is_seven():
    """Without NHARMS in int_params, the bridge falls back to 7 (PINT default)."""
    from jaxpint.model_builder import _build_binary

    binary = _build_binary(_ell1h_ctx(h3=True))
    assert binary.nharms == 7


def test_ell1h_with_h3_h4_uses_h3h4():
    from jaxpint.model_builder import _build_binary

    binary = _build_binary(_ell1h_ctx(h3=True, h4=True))
    assert binary.shapiro_mode == "h3h4"
    assert binary.h4_name == "H4"


def test_ell1h_with_stigma_uses_h3stigma():
    from jaxpint.model_builder import _build_binary

    binary = _build_binary(_ell1h_ctx(h3=True, stigma=True))
    assert binary.shapiro_mode == "h3stigma"
    assert binary.stigma_name == "STIGMA"


@pytest.mark.parametrize("stigma", [0.0, 0.3])
@pytest.mark.parametrize("nharms", [3, 7])
def test_ell1h_fourier_shapiro_matches_pint(stigma, nharms):
    """JaxPINT's ``ell1h_fourier_shapiro`` must agree with PINT's
    ``ELL1H_shapiro_delay_fourier_harms`` (Freire & Wex 2010 Eq. 19) at
    machine precision. stigma=0 collapses the series to a single k=3
    term; stigma=0.3 exercises every harmonic up through ``nharms``."""
    import numpy as np
    import jax.numpy as jnp
    from pint.models.stand_alone_psr_binaries.ELL1H_model import ELL1Hmodel

    from jaxpint.binary.common import ell1h_fourier_shapiro

    h3 = 1.0e-7
    phi = np.linspace(0.0, 2.0 * np.pi, 50, endpoint=False)

    pint_model = ELL1Hmodel()
    selected_harms = np.arange(3, nharms + 1)
    pint_sum = pint_model.ELL1H_shapiro_delay_fourier_harms(
        selected_harms, phi, stigma, factor_out_power=3
    )
    pint_delay = -2.0 * h3 * pint_sum

    jax_delay = ell1h_fourier_shapiro(h3, stigma, jnp.asarray(phi), nharms)
    np.testing.assert_allclose(
        np.asarray(jax_delay), pint_delay, atol=1e-20, rtol=1e-12
    )


def test_ell1h_fourier_shapiro_h3_only_collapses_to_k3_term():
    """For stigma=0 every k>3 term carries stigma**(k-3) = 0, so the sum
    must reduce to ``-(4/3) * H3 * sin(3*Φ)`` regardless of NHARMS."""
    import numpy as np
    import jax.numpy as jnp

    from jaxpint.binary.common import ell1h_fourier_shapiro

    h3 = 1.0e-7
    phi = jnp.linspace(0.0, 2.0 * np.pi, 25, endpoint=False)
    expected = -(4.0 / 3.0) * h3 * jnp.sin(3.0 * phi)
    for nharms in (3, 5, 7, 12):
        got = ell1h_fourier_shapiro(h3, 0.0, phi, nharms)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(expected), atol=1e-20, rtol=1e-12
        )


def test_load_nanograv_pta_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_nanograv_pta(tmp_path / "does-not-exist")


def test_load_nanograv_pta_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_nanograv_pta(tmp_path)


# ---------------------------------------------------------------------------
# iter_nanograv_pta (streaming loader)
# ---------------------------------------------------------------------------


def _stage_two_pulsars(root: Path) -> None:
    """par/+tim/ layout with a second pulsar (same data, different name)."""
    _stage_par_tim_layout(root)
    par_src, tim_src = _example_par_tim()
    shutil.copy2(par_src, root / "par" / "B1899+09_copy.gls.par")
    shutil.copy2(tim_src, root / "tim" / "B1899+09_copy.tim")


def test_iter_matches_load(tmp_path):
    """iter_nanograv_pta yields exactly what load_nanograv_pta materializes."""
    import numpy as np

    from jaxpint import iter_nanograv_pta
    from jaxpint.fitters import compute_time_residuals

    _stage_two_pulsars(tmp_path)
    loaded = load_nanograv_pta(tmp_path, planets=False)
    streamed = list(iter_nanograv_pta(tmp_path, planets=False))

    assert tuple(r.name for r in streamed) == loaded.pulsar_names
    for i, rec in enumerate(streamed):
        assert rec.toa_data.n_toas == loaded.toa_data_list[i].n_toas
        np.testing.assert_array_equal(
            np.asarray(rec.params.values),
            np.asarray(loaded.pulsar_params_list[i].values),
        )
        r_stream = compute_time_residuals(rec.timing_model, rec.toa_data, rec.params)
        r_load = compute_time_residuals(
            loaded.timing_models[i],
            loaded.toa_data_list[i],
            loaded.pulsar_params_list[i],
        )
        np.testing.assert_allclose(
            np.asarray(r_stream), np.asarray(r_load), rtol=0, atol=0
        )


def test_iter_releases_references(tmp_path):
    """Nothing in the load path retains a record once the consumer drops it.
    """
    import gc
    import weakref

    from jaxpint import iter_nanograv_pta

    _stage_two_pulsars(tmp_path)
    gen = iter_nanograv_pta(tmp_path, planets=False)
    rec = next(gen)
    refs = [weakref.ref(rec.toa_data), weakref.ref(rec.noise_model)]
    del rec
    _ = next(gen)  # advance: generator frame must not still hold record 1
    gc.collect()
    assert all(r() is None for r in refs), "dropped record still referenced"
    gen.close()


def test_iter_selection_order_and_exclude(tmp_path):
    from jaxpint import iter_nanograv_pta

    _stage_two_pulsars(tmp_path)
    # Explicit ordering is honored (reversed vs discovery order).
    names = [
        r.name
        for r in iter_nanograv_pta(
            tmp_path, pulsar_names=["B1899+09", "B1855+09"], planets=False
        )
    ]
    assert names == ["B1899+09", "B1855+09"]
    # Exclude drops after discovery.
    names = [
        r.name
        for r in iter_nanograv_pta(tmp_path, exclude=("B1899+09",), planets=False)
    ]
    assert names == ["B1855+09"]
    # Unknown selection raises on first next() (generator semantics).
    gen = iter_nanograv_pta(tmp_path, pulsar_names=["J0000+0000"], planets=False)
    with pytest.raises(KeyError):
        next(gen)


def test_iter_early_break_loads_nothing_further(tmp_path):
    """Lazy loading: breaking after the first record never builds the second."""
    from unittest.mock import patch

    import jaxpint.loaders.nanograv as nanograv_mod
    from jaxpint import iter_nanograv_pta

    _stage_two_pulsars(tmp_path)
    with patch.object(
        nanograv_mod, "_load_one", wraps=nanograv_mod._load_one
    ) as load_one:
        for rec in iter_nanograv_pta(tmp_path, planets=False):
            assert rec.toa_data.n_toas > 100
            break
    assert load_one.call_count == 1  # second pulsar was never loaded
