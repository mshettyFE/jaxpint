"""Backend-selected white noise: JaxPINT par masks vs enterprise ``by_backend``.

The two stacks reach per-backend white noise from opposite directions:

- **JaxPINT** reads explicit ``EFAC -f <backend>`` / ``T2EQUAD -f <backend>``
  lines from the par file, resolving each into a per-TOA boolean mask.
- **enterprise** is handed one templated ``MeasurementNoise`` plus
  ``Selection(selections.by_backend)``, and *derives* the split at runtime from
  the TOAs' backend flags.

They must agree, and the agreement is only meaningful if both split the TOAs
the same way -- so ``test_backend_partition_matches`` pins the partition before
any numerical comparison runs.

Flag choice is load-bearing: enterprise's ``Pulsar.backend_flags`` ranks
``group`` > ``g`` > ``sys`` > ``i`` > ``f`` > ``fe``+``be``, so a ``.tim``
carrying any higher-ranked flag would make ``by_backend`` split on a different
key than the par's ``-f`` masks, and the two stacks would silently disagree.
These fixtures write ``-f`` only.

Layering follows the rest of the package: a kernel-level comparison that feeds
enterprise's own residuals into a dense logL built from JaxPINT's covariance
(tight), then an end-to-end comparison whose tolerance is set by residual
parity.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.enterprise_checks._ent_helpers import (
    build_pulsar,
    clustered_mjds_by_backend,
    dense_logL,
    make_par,
    )

BACKENDS = ("be_a", "be_b")
EFACS = {"be_a": 1.1, "be_b": 1.4}
EQUADS_US = {"be_a": 0.3, "be_b": 0.6}


@pytest.fixture(scope="module")
def mb_bundle(tmp_path_factory):
    """One pulsar whose TOAs are split across two backends.

    Session/module-scoped because enterprise ``Pulsar`` construction dominates
    this package's runtime.
    """
    par = make_par(efac=EFACS, equad_us=EQUADS_US)
    mjds, labels = clustered_mjds_by_backend(BACKENDS, n_epochs=25, per_backend=2)
    tmp = tmp_path_factory.mktemp("ent_multibackend")
    return build_pulsar(tmp, par, mjds, seed=11, backend_labels=labels)


def _ent_pta(bundle):
    """Enterprise white-noise PTA using by_backend selection (Uniform priors).

    ``Uniform`` rather than ``Constant`` so the per-backend parameters appear in
    ``pta.param_names`` and can be set explicitly per backend; ``Constant``
    parameters are filtered out of ``param_names`` and would have to be supplied
    via ``set_default_params``.
    """
    from enterprise.signals import parameter, selections, signal_base, white_signals

    mn = white_signals.MeasurementNoise(
        efac=parameter.Uniform(0.1, 10.0),
        log10_t2equad=parameter.Uniform(-9.0, -5.0),
        selection=selections.Selection(selections.by_backend),
    )
    return signal_base.PTA([signal_base.SignalCollection([mn])(bundle.psr)])


def _backend_of(param_name: str, psr_name: str, suffix: str) -> str:
    """Extract the backend from ``{psr}_{backend}_{suffix}``.

    Split by strip-both-ends rather than ``str.split('_')``: backend labels
    themselves contain underscores (``be_a``), so positional splitting is wrong.
    """
    assert param_name.startswith(f"{psr_name}_"), param_name
    assert param_name.endswith(f"_{suffix}"), param_name
    return param_name[len(psr_name) + 1 : -(len(suffix) + 1)]


def _ent_param_values(pta, psr_name: str) -> dict[str, float]:
    """Map each enterprise per-backend parameter to the par file's value."""
    values = {}
    for name in pta.param_names:
        if name.endswith("_efac"):
            values[name] = EFACS[_backend_of(name, psr_name, "efac")]
        elif name.endswith("_log10_t2equad"):
            be = _backend_of(name, psr_name, "log10_t2equad")
            values[name] = float(np.log10(EQUADS_US[be] * 1e-6))
        else:  # pragma: no cover - guards against a silent convention change
            raise AssertionError(f"unexpected enterprise parameter {name!r}")
    return values


# --------------------------------------------------------------- partition


def test_backend_partition_matches(mb_bundle):
    """Both stacks group the TOAs into identical backend sets.

    Everything downstream is meaningless if this fails: enterprise derives the
    split from ``backend_flags`` while JaxPINT resolves the par's ``-f`` masks,
    and the two could disagree (different flag key, dropped flags on re-read).
    """
    b = mb_bundle
    ent_flags = np.asarray(b.psr.backend_flags)
    assert set(np.unique(ent_flags)) == set(BACKENDS)

    import jaxpint.par as jpar

    mask_info = jpar.get_model(b.par_path).mask_info
    for name, info in mask_info.items():
        if not name.startswith("EFAC"):
            continue
        jax_mask = np.asarray(b.toa_data.flag_mask(name))
        ent_mask = ent_flags == info.key_value
        assert jax_mask.sum() > 0, f"{name} selects no TOAs"
        npt.assert_array_equal(
            jax_mask,
            ent_mask,
            err_msg=(
                f"{name} (par selector {info.key} {info.key_value}) selects a "
                f"different TOA set than enterprise's by_backend group"
            ),
        )


def test_enterprise_backend_param_naming(mb_bundle):
    """Pins enterprise's ``{psr}_{backend}_{param}`` naming.

    This is the convention a NANOGrav noisedict is keyed by, so it is the
    property that makes noise values portable between the stacks; a silent
    change upstream would break that mapping.
    """
    b = mb_bundle
    pta = _ent_pta(b)
    expected = {
        f"{b.psr.name}_{be}_{suffix}"
        for be in BACKENDS
        for suffix in ("efac", "log10_t2equad")
    }
    assert set(pta.param_names) == expected


# ------------------------------------------------------------ scaled sigma


def test_per_backend_scaled_variance(mb_bundle):
    """Ndiag follows EFAC_b^2 * (err^2 + EQUAD_b^2) with per-backend values.

    Pure-JaxPINT check of the tempo2/T2EQUAD convention, evaluated separately
    on each backend's TOAs -- a global EFAC/EQUAD would pass a whole-array
    comparison against the wrong constants only if the backends happened to
    share values, so this asserts per group.
    """
    b = mb_bundle
    Ndiag = np.asarray(b.noise_model.scaled_sigma(b.toa_data, b.params)) ** 2
    err = np.asarray(b.toa_data.error)
    ent_flags = np.asarray(b.psr.backend_flags)

    for be in BACKENDS:
        m = ent_flags == be
        expected = EFACS[be] ** 2 * (err[m] ** 2 + (EQUADS_US[be] * 1e-6) ** 2)
        npt.assert_allclose(
            Ndiag[m],
            expected,
            rtol=1e-12,
            err_msg=f"scaled variance wrong for backend {be!r}",
        )

    # The two backends must actually differ, or the whole comparison is vacuous.
    # atol=0 is required: Ndiag ~ 1e-12, so np.isclose's default atol=1e-8 would
    # call *any* two values "close" and the guard would never fire.
    a, c = ent_flags == BACKENDS[0], ent_flags == BACKENDS[1]
    assert not np.isclose(
        Ndiag[a].mean(), Ndiag[c].mean(), rtol=1e-6, atol=0.0
    ), "backends have indistinguishable Ndiag; the per-backend test is vacuous"


# ------------------------------------------------------------------- logL


def test_backend_selected_white_logL_kernel(mb_bundle):
    """Enterprise by_backend logL == dense logL from JaxPINT's Ndiag.

    Kernel-level: enterprise's own residuals on both sides, so a failure is a
    noise-convention disagreement rather than residual drift.
    """
    b = mb_bundle
    pta = _ent_pta(b)
    logL_ent = pta.get_lnlikelihood(_ent_param_values(pta, b.psr.name))

    Ndiag, U, Phi = b.noise_model.covariance(b.toa_data, b.params)
    logL_jax = dense_logL(b.psr.residuals, Ndiag, U, Phi)

    npt.assert_allclose(
        logL_jax,
        logL_ent,
        rtol=1e-12,
        err_msg="per-backend white-noise covariance or logL normalization mismatch",
    )


def test_backend_selected_white_logL_end_to_end(mb_bundle):
    """Full JaxPINT single_pulsar_logL vs enterprise PTA (same normalization).

    Tolerance is set by residual parity (~1e-9 s, see
    test_ent_building_blocks.py), not by the noise model.
    """
    from jaxpint.likelihood import single_pulsar_logL

    b = mb_bundle
    pta = _ent_pta(b)
    logL_ent = pta.get_lnlikelihood(_ent_param_values(pta, b.psr.name))
    logL_jax = float(
        single_pulsar_logL(b.toa_data, b.timing_model, b.noise_model, b.params)
    )
    npt.assert_allclose(
        logL_jax,
        logL_ent,
        atol=5e-3,
        rtol=0,
        err_msg="end-to-end per-backend white-noise logL mismatch",
    )
