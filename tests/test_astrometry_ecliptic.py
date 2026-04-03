"""Tests for the AstrometryEcliptic delay component."""

import copy
from io import StringIO

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pint.models import get_model
from pint.fitter import WLSFitter as PINTWLSFitter
from pint.simulation import make_fake_toas_uniform

from jaxpint.astrometry import AstrometryEcliptic, _geometric_delay
from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model
from jaxpint.fitter import WLSFitter, compute_time_residuals
from jaxpint.constants import OBLIQUITY_ARCSEC
from jaxpint.utils import ecl_to_icrs_rotation


# ---------------------------------------------------------------------------
# Par file templates
# ---------------------------------------------------------------------------

_PAR_ECL_SIMPLE = """\
PSR           J0000+0000
ELONG         120.5
ELAT          -30.2
F0            100.0
F1            -1.0e-15
PEPOCH        55000
DM            10.0
EPHEM         DE421
CLK           UTC(NIST)
UNITS         TDB
ECL           IERS2010
CORRECT_TROPOSPHERE  N
"""

_PAR_ECL_PM = """\
PSR           J0000+0000
ELONG         120.5
ELAT          -30.2
PMELONG       3.5
PMELAT        -1.2
F0            100.0
F1            -1.0e-15
PEPOCH        55000
POSEPOCH      55000
DM            10.0
EPHEM         DE421
CLK           UTC(NIST)
UNITS         TDB
ECL           IERS2010
CORRECT_TROPOSPHERE  N
"""

_PAR_ECL_PX = """\
PSR           J0000+0000
ELONG         120.5
ELAT          -30.2
PMELONG       3.5
PMELAT        -1.2
PX            2.0
F0            100.0
F1            -1.0e-15
PEPOCH        55000
POSEPOCH      55000
DM            10.0
EPHEM         DE421
CLK           UTC(NIST)
UNITS         TDB
ECL           IERS2010
CORRECT_TROPOSPHERE  N
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_setup(par_str, ntoas=50):
    """Create PINT model + TOAs and extract astrometry delay."""
    model = get_model(StringIO(par_str))
    toas = make_fake_toas_uniform(
        startMJD=54500, endMJD=55500,
        ntoas=ntoas, model=model, freq=1400.0,
        add_noise=False,
    )
    toas.compute_TDBs()
    toas.compute_posvels()

    astro_comp = model.components["AstrometryEcliptic"]
    pint_delay = np.array(
        astro_comp.solar_system_geometric_delay(toas).to("s").value,
        dtype=np.float64,
    )

    toa_data = pint_toas_to_jax(toas, model)
    params = pint_model_to_params(model)

    return toa_data, params, pint_delay, model


@pytest.fixture
def ecl_simple():
    """Ecliptic astrometry without proper motion or parallax."""
    return _make_setup(_PAR_ECL_SIMPLE)


@pytest.fixture
def ecl_pm():
    """Ecliptic astrometry with proper motion."""
    return _make_setup(_PAR_ECL_PM)


@pytest.fixture
def ecl_px():
    """Ecliptic astrometry with proper motion and parallax."""
    return _make_setup(_PAR_ECL_PX)


# ---------------------------------------------------------------------------
# Rotation matrix tests
# ---------------------------------------------------------------------------


class TestRotationMatrix:
    """Tests for ecl_to_icrs_rotation."""

    def test_ecliptic_pole_maps_correctly(self):
        """Ecliptic north pole (0,0,1) should map to (0, -sin(obl), cos(obl))."""
        obl = OBLIQUITY_ARCSEC["IERS2010"]
        R = ecl_to_icrs_rotation(obl)
        ecl_pole = jnp.array([[0.0, 0.0, 1.0]])
        result = ecl_pole @ R

        obl_rad = obl * jnp.pi / (180.0 * 3600.0)
        expected = jnp.array([[0.0, -jnp.sin(obl_rad), jnp.cos(obl_rad)]])
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-15)

    def test_vernal_equinox_unchanged(self):
        """Vernal equinox (1,0,0) is the same in both frames."""
        obl = OBLIQUITY_ARCSEC["IERS2010"]
        R = ecl_to_icrs_rotation(obl)
        ve = jnp.array([[1.0, 0.0, 0.0]])
        result = ve @ R
        np.testing.assert_allclose(np.array(result), np.array(ve), atol=1e-15)

    def test_rotation_is_orthogonal(self):
        """Rotation matrix should be orthogonal (R @ R.T = I)."""
        obl = OBLIQUITY_ARCSEC["IERS2010"]
        R = ecl_to_icrs_rotation(obl)
        np.testing.assert_allclose(
            np.array(R @ R.T), np.eye(3), atol=1e-15,
        )

    def test_different_obliquities_differ(self):
        """Different ECL standards should produce different rotations."""
        R_2010 = ecl_to_icrs_rotation(OBLIQUITY_ARCSEC["IERS2010"])
        R_1976 = ecl_to_icrs_rotation(OBLIQUITY_ARCSEC["IAU1976"])
        assert not jnp.allclose(R_2010, R_1976, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# Delay matches PINT
# ---------------------------------------------------------------------------


class TestDelayMatchesPINT:
    """AstrometryEcliptic delay matches PINT's solar_system_geometric_delay."""

    def test_simple_no_pm(self, ecl_simple):
        """No proper motion, no parallax: delay matches PINT."""
        toa_data, params, pint_delay, _ = ecl_simple

        comp = AstrometryEcliptic(
            obliquity_arcsec=OBLIQUITY_ARCSEC["IERS2010"],
        )
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    def test_with_proper_motion(self, ecl_pm):
        """With proper motion: delay matches PINT."""
        toa_data, params, pint_delay, _ = ecl_pm

        comp = AstrometryEcliptic(
            pmelong_name="PMELONG",
            pmelat_name="PMELAT",
            posepoch_name="POSEPOCH",
            obliquity_arcsec=OBLIQUITY_ARCSEC["IERS2010"],
        )
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    def test_with_parallax(self, ecl_px):
        """With proper motion and parallax: delay matches PINT.

        Tolerance is looser because PINT uses ERFA's pmsafe for proper
        motion while JaxPINT uses a linear approximation.
        """
        toa_data, params, pint_delay, _ = ecl_px

        comp = AstrometryEcliptic(
            pmelong_name="PMELONG",
            pmelat_name="PMELAT",
            px_name="PX",
            posepoch_name="POSEPOCH",
            obliquity_arcsec=OBLIQUITY_ARCSEC["IERS2010"],
        )
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=5e-7, atol=1e-10,
        )

    def test_delay_is_nonzero(self, ecl_simple):
        """Delay should be non-trivially nonzero."""
        toa_data, params, _, _ = ecl_simple

        comp = AstrometryEcliptic(
            obliquity_arcsec=OBLIQUITY_ARCSEC["IERS2010"],
        )
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        assert jnp.all(jnp.isfinite(jax_delay))
        assert jnp.max(jnp.abs(jax_delay)) > 1e-5


# ---------------------------------------------------------------------------
# Bridge integration
# ---------------------------------------------------------------------------


class TestBridge:
    """build_timing_model correctly creates AstrometryEcliptic."""

    def test_bridge_creates_ecliptic_component(self, ecl_px):
        _, _, _, pint_model = ecl_px
        jax_model, _, _ecorr = build_timing_model(pint_model)

        ecl_comps = [
            c for c in jax_model.delay_components
            if isinstance(c, AstrometryEcliptic)
        ]
        assert len(ecl_comps) == 1
        assert ecl_comps[0].obliquity_arcsec == OBLIQUITY_ARCSEC["IERS2010"]

    def test_bridge_full_phase_finite(self, ecl_px):
        """Full model phase is finite with ecliptic astrometry."""
        toa_data, params, _, pint_model = ecl_px
        jax_model, _, _ecorr = build_timing_model(pint_model)

        phase = jax_model.compute_phase(toa_data, params)
        total = phase.int + phase.frac

        assert jnp.all(jnp.isfinite(total))
        assert total.shape == (toa_data.n_toas,)


# ---------------------------------------------------------------------------
# Autodiff
# ---------------------------------------------------------------------------


class TestAutodiff:
    """AstrometryEcliptic is differentiable w.r.t. sky position."""

    def test_grad_elong_finite(self, ecl_simple):
        toa_data, params, _, _ = ecl_simple
        comp = AstrometryEcliptic(
            obliquity_arcsec=OBLIQUITY_ARCSEC["IERS2010"],
        )

        def total_delay(p):
            return jnp.sum(comp(toa_data, p, jnp.zeros(toa_data.n_toas)))

        grad = jax.grad(total_delay)(params)

        elong_idx = params.param_index("ELONG")
        assert jnp.isfinite(grad.values[elong_idx])
        assert grad.values[elong_idx] != 0.0

    def test_grad_elat_finite(self, ecl_simple):
        toa_data, params, _, _ = ecl_simple
        comp = AstrometryEcliptic(
            obliquity_arcsec=OBLIQUITY_ARCSEC["IERS2010"],
        )

        def total_delay(p):
            return jnp.sum(comp(toa_data, p, jnp.zeros(toa_data.n_toas)))

        grad = jax.grad(total_delay)(params)

        elat_idx = params.param_index("ELAT")
        assert jnp.isfinite(grad.values[elat_idx])
        assert grad.values[elat_idx] != 0.0

    def test_jit_compatible(self, ecl_simple):
        """AstrometryEcliptic runs under jax.jit."""
        toa_data, params, _, _ = ecl_simple
        comp = AstrometryEcliptic(
            obliquity_arcsec=OBLIQUITY_ARCSEC["IERS2010"],
        )

        delay_eager = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        delay_jit = jax.jit(comp)(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(delay_jit), np.array(delay_eager), rtol=1e-14,
        )


# ---------------------------------------------------------------------------
# Fitting integration: synthetic ecliptic pulsar, JaxPINT vs PINT
# ---------------------------------------------------------------------------

# Synthetic ecliptic pulsar for fitting (no PM to avoid linear-vs-pmsafe
# discrepancy; that is tested separately in TestDelayMatchesPINT).
_PAR_FIT = """\
PSR           J0000+0000
ELONG         120.5  1
ELAT          -30.2  1
F0            100.0  1
F1            -1e-15  1
PEPOCH        55000
DM            10.0  1
EPHEM         DE421
CLK           TT(BIPM2019)
UNITS         TDB
ECL           IERS2010
CORRECT_TROPOSPHERE  N
TZRMJD        55000
TZRFRQ        1400
TZRSITE       @
"""


@pytest.fixture(scope="module")
def fit_data():
    """Generate synthetic multi-frequency TOAs for an ecliptic pulsar."""
    m_true = get_model(StringIO(_PAR_FIT))
    toas_lo = make_fake_toas_uniform(
        54500, 55500, 30, m_true,
        error=1 * u.us, add_noise=True, freq=1400 * u.MHz,
    )
    toas_hi = make_fake_toas_uniform(
        54500, 55500, 30, m_true,
        error=1 * u.us, add_noise=True, freq=2000 * u.MHz,
    )
    toas_lo.merge(toas_hi)
    return m_true, toas_lo


@pytest.fixture(scope="module")
def pint_fit_result(fit_data):
    """Run PINT's WLS fitter."""
    m_true, toas = fit_data
    mc = copy.deepcopy(m_true)
    f = PINTWLSFitter(toas, mc)
    f.fit_toas(maxiter=3)
    return f


@pytest.fixture(scope="module")
def jax_fit_result(fit_data):
    """Run JaxPINT's WLS fitter."""
    m_true, toas = fit_data
    toa_data = pint_toas_to_jax(toas, model=m_true)
    params = pint_model_to_params(m_true)
    jax_model, _noise, _ecorr = build_timing_model(m_true)
    fitter = WLSFitter(jax_model, toa_data, params)
    fitter.fit_toas(maxiter=3)
    return fitter


class TestFitMatchesPINT:
    """JaxPINT WLS fit of an ecliptic pulsar matches PINT."""

    def test_chi2_matches(self, pint_fit_result, jax_fit_result):
        """Post-fit chi2 should agree between PINT and JaxPINT."""
        pint_chi2 = pint_fit_result.resids.chi2
        jax_chi2 = jax_fit_result.result.chi2
        np.testing.assert_allclose(jax_chi2, pint_chi2, rtol=0.01)

    def test_reduced_chi2_reasonable(self, jax_fit_result):
        """Reduced chi2 should be close to 1 for synthetic data."""
        assert 0.1 < jax_fit_result.result.reduced_chi2 < 5.0

    def test_f0_matches(self, pint_fit_result, jax_fit_result):
        pint_val = float(pint_fit_result.model.F0.value)
        jax_val = float(jax_fit_result.result.params.param_value("F0"))
        pint_err = float(pint_fit_result.model.F0.uncertainty_value)
        assert abs(jax_val - pint_val) < 3 * pint_err

    def test_elong_matches(self, pint_fit_result, jax_fit_result):
        pint_val = float(pint_fit_result.model.ELONG.quantity.to(u.rad).value)
        jax_val = float(jax_fit_result.result.params.param_value("ELONG"))
        np.testing.assert_allclose(jax_val, pint_val, atol=5e-9)

    def test_elat_matches(self, pint_fit_result, jax_fit_result):
        pint_val = float(pint_fit_result.model.ELAT.quantity.to(u.rad).value)
        jax_val = float(jax_fit_result.result.params.param_value("ELAT"))
        np.testing.assert_allclose(jax_val, pint_val, atol=5e-9)

    def test_dm_matches(self, pint_fit_result, jax_fit_result):
        pint_val = float(pint_fit_result.model.DM.value)
        jax_val = float(jax_fit_result.result.params.param_value("DM"))
        pint_err = float(pint_fit_result.model.DM.uncertainty_value)
        assert abs(jax_val - pint_val) < 3 * pint_err

    def test_uncertainties_positive(self, jax_fit_result):
        assert jnp.all(jax_fit_result.result.parameter_uncertainties > 0)


# ---------------------------------------------------------------------------
# Real pulsar: B1855+09 astrometry delay comparison
# ---------------------------------------------------------------------------


class TestB1855Delay:
    """Compare AstrometryEcliptic delay against PINT for the real B1855+09."""

    @pytest.fixture
    def b1855(self):
        """Load B1855+09 from PINT examples."""
        import pint.toa as toa
        from pint.config import examplefile

        model = get_model(examplefile("B1855+09_NANOGrav_9yv1.gls.par"))
        toas = toa.get_TOAs(
            examplefile("B1855+09_NANOGrav_9yv1.tim"), ephem="DE421",
        )
        return model, toas

    def test_geometric_delay_matches_pint(self, b1855):
        """Roemer+parallax delay matches PINT for a real ecliptic pulsar.

        Tolerance is loose because PINT uses ERFA's pmsafe for proper
        motion while JaxPINT uses a linear approximation. Over the
        ~3000-day B1855+09 baseline this gives ~1e-6 relative error.
        """
        pint_model, toas = b1855

        # PINT delay
        astro = pint_model.components["AstrometryEcliptic"]
        pint_delay = np.array(
            astro.solar_system_geometric_delay(toas).to("s").value,
            dtype=np.float64,
        )

        # JaxPINT delay — extract only the astrometry component
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)
        jax_model, _, _ecorr = build_timing_model(pint_model)

        ecl_comps = [
            c for c in jax_model.delay_components
            if isinstance(c, AstrometryEcliptic)
        ]
        assert len(ecl_comps) == 1, "Expected one AstrometryEcliptic component"

        jax_delay = ecl_comps[0](toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=5e-6, atol=1e-10,
        )

    def test_delay_nonzero_and_finite(self, b1855):
        """Delay values should be finite and non-trivially nonzero."""
        pint_model, toas = b1855
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)
        jax_model, _, _ecorr = build_timing_model(pint_model)

        ecl_comp = [
            c for c in jax_model.delay_components
            if isinstance(c, AstrometryEcliptic)
        ][0]
        jax_delay = ecl_comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        assert jnp.all(jnp.isfinite(jax_delay))
        assert jnp.max(jnp.abs(jax_delay)) > 1e-3  # Roemer delay is ~500s
