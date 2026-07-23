"""Shared builders for the enterprise cross-validation suite.

Both stacks (enterprise and JaxPINT) are fed the *same* par/tim files: fake
TOAs are simulated with PINT, written to disk, and re-read once with
``get_model_and_toas`` — the re-read PINT ``(model, toas)`` pair is then handed
to enterprise's ``Pulsar`` (PintPulsar object form) and to JaxPINT's bridge, so
every comparison consumes identical bytes.

All ``pint`` / ``enterprise`` imports live inside function bodies: the root
conftest skips modules on a source-regex for those imports, and this helper
must stay importable when either dependency is missing.

Convention notes that shape every test in this package:

- BOTH stacks include the Gaussian normalization constant ``-n/2 log(2*pi)``
  in their logL (enterprise's ``get_lnlikelihood`` and JaxPINT's
  ``single_pulsar_logL``/``pta_logL`` alike), as does :func:`dense_logL`
  below — logLs are directly comparable, no offset correction.  
- BOTH stacks build Fourier bases and ECORR quantization at *barycentered*
  TOAs (enterprise via ``model.get_barycentric_toas``, JaxPINT via
  ``TOAData.basis_seconds``), and the time arrays agree exactly — see
  ``test_ent_building_blocks.test_gp_basis_frame_matches_enterprise``.
  GP logL tolerances are therefore residual-parity-limited (~1e-4), not
  Roemer-budgeted.
- ``PHOFF`` must be non-zero in the par files: JaxPINT's model builder treats
  a 0.0-valued parameter as unset (``model_builder._param_is_set``) and would
  drop the phase-offset binding that the marginalization tests rely on.
"""

from __future__ import annotations

import io
from typing import NamedTuple

import numpy as np

TWO_PI = 2.0 * np.pi

# White-noise values shared by conftest's ``white_bundle`` par and the
# enterprise Constant parameters in test_ent_white_noise.py.  Single-sourced
# here so the par file and the enterprise model can never silently drift
# apart (they used to be duplicated in both places).
WHITE_EFAC = 1.3
WHITE_EQUAD_US = 0.5
WHITE_ECORR_US = 0.4


def dense_logL(r, Ndiag, U, Phi) -> float:
    """Dense-numpy Gaussian logL, ``-n/2 log(2*pi)`` normalization included.

    Both stacks include the constant (enterprise's ``get_lnlikelihood`` and
    JaxPINT's ``single_pulsar_logL`` alike), so this reference is directly
    comparable to either.  ``C = diag(Ndiag) + U @ diag(Phi) @ U.T``;
    independent of both stacks' solvers, so it localizes disagreements to the
    covariance ingredients.
    """
    r = np.asarray(r, dtype=float)
    C = np.diag(np.asarray(Ndiag, dtype=float))
    U = np.asarray(U, dtype=float)
    Phi = np.asarray(Phi, dtype=float)
    if U.shape[1] > 0:
        C = C + (U * Phi) @ U.T
    sign, logdet = np.linalg.slogdet(C)
    assert sign > 0, "covariance not positive definite"
    return float(
        -0.5 * r @ np.linalg.solve(C, r)
        - 0.5 * logdet
        - 0.5 * len(r) * np.log(TWO_PI)
    )


PAR_TEMPLATE = """\
PSR           {name}
RAJ           {raj}      {fit_pos}
DECJ          {decj}     {fit_pos}
PEPOCH        54500
F0            {f0}       {fit_spin}
F1            {f1}       {fit_spin}
DM            {dm}       {fit_spin}
PHOFF         0.01       {fit_spin}
{noise_lines}TZRMJD        54500
TZRFRQ        1400
TZRSITE       @
EPHEM         DE440
CLOCK         TT(BIPM2019)
UNITS         TDB
"""


def make_par(
    name="J0123+4500",
    raj="01:23:45.0",
    decj="45:00:00.0",
    f0=123.456789,
    f1=-1.5e-15,
    dm=12.0,
    efac=None,
    equad_us=None,
    ecorr_us=None,
    red=None,
    fit_spin=False,
) -> str:
    """Build an inline par string (same style as tests/test_pl_noise_vs_pint.py).

    ``red`` is an optional ``(log10_amp, gamma, n_components)`` tuple mapped to
    TNREDAMP/TNREDGAM/TNREDC.
    Sky position is always frozen so enterprise's TimingModel design matrix
    stays within the parameter set JaxPINT marginalizes analytically.

    ``efac`` / ``equad_us`` / ``ecorr_us`` each accept either

    * a **scalar** -- one line selecting the uniform ``-f fake_be`` flag that
      :func:`build_pulsar` stamps on every TOA (the single-backend default), or
    * a **mapping** ``{backend: value}`` -- one line per backend, e.g.
      ``efac={"be_a": 1.1, "be_b": 1.4}`` -> ``EFAC -f be_a 1.1`` and
      ``EFAC -f be_b 1.4``.  Pair this with
      :func:`clustered_mjds_by_backend` so the TOAs actually carry those flags.

    The ``-f`` flag is deliberate on both sides: enterprise's
    ``Pulsar.backend_flags`` ranks ``group`` > ``g`` > ``sys`` > ``i`` > ``f``
    > ``fe``+``be``, so a ``.tim`` carrying any higher-ranked flag would make
    ``selections.by_backend`` split on a *different* key than the par's masks.
    These fixtures write ``-f`` only.
    """

    def _mask_lines(keyword, value):
        if value is None:
            return []
        if isinstance(value, dict):
            return [f"{keyword} -f {be} {v}" for be, v in value.items()]
        return [f"{keyword} -f fake_be {value}"]

    lines = []
    lines += _mask_lines("EFAC", efac)
    lines += _mask_lines("T2EQUAD", equad_us)
    lines += _mask_lines("ECORR", ecorr_us)
    if red is not None:
        log10_amp, gamma, n_comp = red
        lines.append(f"TNREDAMP      {log10_amp}")
        lines.append(f"TNREDGAM      {gamma}")
        lines.append(f"TNREDC        {n_comp}")
    noise_lines = "".join(line + "\n" for line in lines)
    return PAR_TEMPLATE.format(
        name=name,
        raj=raj,
        decj=decj,
        f0=f0,
        f1=f1,
        dm=dm,
        noise_lines=noise_lines,
        fit_pos=0,
        fit_spin=1 if fit_spin else 0,
    )


def clustered_mjds(n_epochs=40, per_epoch=3, start=53000.0, end=56000.0):
    """Epoch-clustered MJDs: ``per_epoch`` TOAs 0.3 s apart at each epoch.

    The 0.3 s intra-epoch spacing keeps each epoch inside one dt=1 s
    quantization bucket (both stacks' ECORR default), and nmin=2 keeps every
    epoch — uniformly spaced single TOAs would quantize to an empty ECORR
    basis in both stacks (verified) and the comparison would be vacuous.
    """
    epochs = np.linspace(start, end, n_epochs)
    return np.concatenate([e + 0.3 / 86400.0 * np.arange(per_epoch) for e in epochs])


def clustered_mjds_by_backend(
    backends, n_epochs=30, per_backend=2, start=53000.0, end=56000.0, spacing_s=0.15
):
    """Epoch-clustered MJDs where *every* backend observes *every* epoch.

    Returns ``(mjds, labels)``, both length ``n_epochs * len(backends) *
    per_backend`` and already time-sorted.  Within an epoch the backends are
    interleaved round-robin, which mirrors real data (several receivers /
    backends recording the same observing session).

    The critical constraint is ECORR.  ``build_quantization_matrix`` groups a
    parameter's *own* selected TOAs into ``dt=1 s`` epochs and drops any epoch
    with fewer than ``nmin=2`` TOAs.  So splitting a fixed per-epoch TOA budget
    *between* backends would leave each backend one TOA per epoch, every epoch
    would be dropped, and the per-backend ECORR basis would come out empty --
    a silently vacuous comparison.  Giving each backend ``per_backend >= 2``
    TOAs per epoch keeps one basis column per backend per epoch.

    ``spacing_s`` must keep the whole epoch inside the 1 s bucket; asserted.
    """
    backends = tuple(backends)
    n_per_epoch = len(backends) * per_backend
    span_s = spacing_s * (n_per_epoch - 1)
    assert span_s < 1.0, (
        f"epoch spans {span_s:.2f} s with {n_per_epoch} TOAs at {spacing_s} s "
        f"spacing; it must stay inside the dt=1 s ECORR quantization bucket"
    )
    assert per_backend >= 2, "per_backend < 2 makes every ECORR epoch fail nmin=2"

    mjds, labels = [], []
    for epoch in np.linspace(start, end, n_epochs):
        for k in range(n_per_epoch):
            mjds.append(epoch + spacing_s / 86400.0 * k)
            labels.append(backends[k % len(backends)])
    return np.asarray(mjds), np.asarray(labels)


class PulsarBundle(NamedTuple):
    """Everything both stacks need for one pulsar, from one par/tim pair."""

    par_path: str
    tim_path: str
    model: object  # pint.models.TimingModel
    toas: object  # pint.toa.TOAs
    psr: object  # enterprise.pulsar.PintPulsar
    toa_data: object  # jaxpint TOAData
    timing_model: object  # jaxpint TimingModel
    noise_model: object  # jaxpint NoiseModel
    params: object  # jaxpint ParameterVector


def build_pulsar(
    tmp_path, par_str, mjds, seed=42, obs="gbt", backend_labels=None
) -> PulsarBundle:
    """Simulate, write, re-read, and convert one pulsar for both stacks.

    ``backend_labels``, when given, is a per-TOA sequence of ``-f`` flag values
    parallel to ``mjds`` (see :func:`clustered_mjds_by_backend`).  It is sorted
    jointly with ``mjds`` so the pairing survives.  Default ``None`` stamps the
    uniform ``fake_be`` flag, i.e. the single-backend behaviour.
    """
    import astropy.time as at
    import astropy.units as u
    from pint.models import get_model, get_model_and_toas
    from pint.simulation import make_fake_toas_fromMJDs

    from enterprise.pulsar import Pulsar

    from jaxpint.bridge import (
        build_timing_model,
        pint_model_to_params,
        pint_toas_to_jax,
    )

    m0 = get_model(io.StringIO(par_str))
    np.random.seed(seed)
    # Sort MJDs and any backend labels *together* so the pairing survives.
    order = np.argsort(np.asarray(mjds))
    mjds_sorted = np.asarray(mjds)[order]
    labels = None if backend_labels is None else np.asarray(backend_labels)[order]

    toas0 = make_fake_toas_fromMJDs(
        at.Time(mjds_sorted, format="mjd"),
        model=m0,
        obs=obs,
        freq=1400.0 * u.MHz,
        error=1.0 * u.us,
        add_noise=True,
    )
    if labels is None:
        # Uniform backend flag: the par's "-f fake_be" EFAC/EQUAD/ECORR masks
        # then select every TOA, matching enterprise's default (no-selection)
        # signals.
        for flags in toas0.table["flags"]:
            flags["f"] = "fake_be"
    else:
        for flags, be in zip(toas0.table["flags"], labels):
            flags["f"] = str(be)

    name = m0.PSR.value
    par_path = str(tmp_path / f"{name}.par")
    tim_path = str(tmp_path / f"{name}.tim")
    with open(par_path, "w") as fh:
        fh.write(m0.as_parfile())
    toas0.write_TOA_file(tim_path)

    model, toas = get_model_and_toas(
        par_path, tim_path, ephem="DE440", bipm_version="BIPM2019", planets=True
    )
    psr = Pulsar(toas, model, planets=True, drop_pintpsr=False)
    toa_data = pint_toas_to_jax(toas, model=model)
    timing_model, noise_model = build_timing_model(model, toas)
    params = pint_model_to_params(model).params
    return PulsarBundle(
        par_path, tim_path, model, toas, psr, toa_data, timing_model, noise_model, params
    )


# Three well-separated sky positions for the multi-pulsar (CURN/HD) fixtures.
PTA_PULSARS = (
    dict(name="J0100+1500", raj="01:00:00.0", decj="15:00:00.0",
         f0=200.0, f1=-1e-15, dm=10.0, efac=1.1),
    dict(name="J0900-3000", raj="09:00:00.0", decj="-30:00:00.0",
         f0=310.0, f1=-2e-15, dm=20.0, efac=1.3),
    dict(name="J1700+6000", raj="17:00:00.0", decj="60:00:00.0",
         f0=150.0, f1=-0.5e-15, dm=30.0, efac=0.9),
)


def build_pta_bundles(tmp_path) -> list[PulsarBundle]:
    """Three pulsars with white noise only and frozen timing parameters.

    Frozen timing (fit flags 0) keeps residuals fully determined by the par
    values, so neither stack needs a timing-model signal / marginalization and
    the CURN/HD comparisons isolate the common-red-noise machinery.
    """
    bundles = []
    for i, spec in enumerate(PTA_PULSARS):
        par = make_par(
            name=spec["name"],
            raj=spec["raj"],
            decj=spec["decj"],
            f0=spec["f0"],
            f1=spec["f1"],
            dm=spec["dm"],
            efac=spec["efac"],
            fit_spin=False,
        )
        mjds = clustered_mjds(n_epochs=25, per_epoch=2)
        bundles.append(build_pulsar(tmp_path, par, mjds, seed=10 + i))
    return bundles


def shared_tspan(bundles) -> float:
    """Common Tspan (seconds) across enterprise pulsars (barycentered toas).

    Passed explicitly to *both* stacks' common-signal builders so their
    Fourier frequency grids coincide exactly (enterprise's default is the
    per-pulsar span, JaxPINT's injectors take T_span as a constructor arg).
    """
    tmin = min(b.psr.toas.min() for b in bundles)
    tmax = max(b.psr.toas.max() for b in bundles)
    return float(tmax - tmin)
