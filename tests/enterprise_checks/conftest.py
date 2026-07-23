"""Shared fixtures for the enterprise cross-validation suite.

Session-scoped because enterprise ``Pulsar`` construction (PINT ephemeris +
clock work per pulsar) dominates this package's runtime; the underlying
par/tim files live in a pytest-managed temp dir and are consumed by both
stacks (see ``_ent_helpers.build_pulsar``).
"""

from __future__ import annotations

import pytest

from tests.enterprise_checks._ent_helpers import (
    WHITE_ECORR_US,
    WHITE_EFAC,
    WHITE_EQUAD_US,
    build_pta_bundles,
    build_pulsar,
    clustered_mjds,
    make_par,
)


@pytest.fixture(scope="session")
def white_bundle(tmp_path_factory):
    """Single pulsar with EFAC + T2EQUAD + ECORR white noise (120 TOAs).

    Noise values come from the shared ``WHITE_*`` constants so the par file
    and test_ent_white_noise.py's enterprise Constants stay in lockstep.
    """
    par = make_par(efac=WHITE_EFAC, equad_us=WHITE_EQUAD_US, ecorr_us=WHITE_ECORR_US)
    tmp = tmp_path_factory.mktemp("ent_white")
    return build_pulsar(tmp, par, clustered_mjds(n_epochs=40, per_epoch=3), seed=42)


@pytest.fixture(scope="session")
def pta_bundles(tmp_path_factory):
    """Three white-noise pulsars with frozen timing params, for CURN/HD."""
    tmp = tmp_path_factory.mktemp("ent_pta")
    return build_pta_bundles(tmp)
