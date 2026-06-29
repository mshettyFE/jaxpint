"""Integration tests: compare JaxPINT power-law noise models against PINT.

Each test creates a PINT model with the relevant noise parameters,
calls PINT's ``get_noise_basis()`` and ``get_noise_weights()`` methods,
then converts via the bridge and checks that JaxPINT produces matching
basis matrices and PSD weights.
"""

from __future__ import annotations

import io

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

# ---------------------------------------------------------------------------
# Fixtures — synthetic PINT models with noise parameters
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pldm_pint_model():
    """PINT model with PLDMNoise parameters and multi-frequency TOAs."""
    import astropy.units as u
    import pint.models as models
    import pint.toa as toa
    from pint.simulation import make_fake_toas_uniform

    par = """\
PSR           J0000+0000
RAJ           05:00:00   1
DECJ          +20:00:00  1
PEPOCH        55000
F0            100        1
F1            -1e-15     1
DM            15         1
TNDMAMP       -13
TNDMGAM       3.5
TNDMC         10
TZRMJD        55000
TZRFRQ        1400
TZRSITE       @
EPHEM         DE421
CLOCK         TT(BIPM2019)
UNITS         TDB
"""
    m = models.get_model(io.StringIO(par))
    # Two frequency bands to exercise DM scaling
    t1 = make_fake_toas_uniform(
        54500, 55500, 50, model=m, obs="gbt", freq=820.0,
        error=1.0 * u.us, add_noise=False,
    )
    t2 = make_fake_toas_uniform(
        54500, 55500, 50, model=m, obs="gbt", freq=1400.0,
        error=1.0 * u.us, add_noise=False,
    )
    t = toa.merge_TOAs([t1, t2])
    t.compute_TDBs()
    return m, t


@pytest.fixture(scope="module")
def plred_pint_model():
    """PINT model with PLRedNoise parameters."""
    import astropy.units as u
    import pint.models as models
    from pint.simulation import make_fake_toas_uniform

    par = """\
PSR           J0000+0000
RAJ           05:00:00   1
DECJ          +20:00:00  1
PEPOCH        55000
F0            100        1
F1            -1e-15     1
DM            15         1
TNREDAMP      -13
TNREDGAM      3.5
TNREDC        10
TZRMJD        55000
TZRFRQ        1400
TZRSITE       @
EPHEM         DE421
CLOCK         TT(BIPM2019)
UNITS         TDB
"""
    m = models.get_model(io.StringIO(par))
    t = make_fake_toas_uniform(
        54500, 55500, 100, model=m, obs="gbt", freq=1400.0,
        error=1.0 * u.us, add_noise=False,
    )
    t.compute_TDBs()
    return m, t


@pytest.fixture(scope="module")
def plchrom_pint_model():
    """PINT model with PLChromNoise parameters and ChromaticCM."""
    import astropy.units as u
    import pint.models as models
    import pint.toa as toa
    from pint.simulation import make_fake_toas_uniform

    par = """\
PSR           J0000+0000
RAJ           05:00:00   1
DECJ          +20:00:00  1
PEPOCH        55000
F0            100        1
F1            -1e-15     1
DM            15         1
TNCHROMAMP    -13
TNCHROMGAM    3.5
TNCHROMC      10
TNCHROMIDX    4.0
TZRMJD        55000
TZRFRQ        1400
TZRSITE       @
EPHEM         DE421
CLOCK         TT(BIPM2019)
UNITS         TDB
"""
    m = models.get_model(io.StringIO(par))
    # Two frequency bands to exercise chromatic scaling
    t1 = make_fake_toas_uniform(
        54500, 55500, 50, model=m, obs="gbt", freq=820.0,
        error=1.0 * u.us, add_noise=False,
    )
    t2 = make_fake_toas_uniform(
        54500, 55500, 50, model=m, obs="gbt", freq=1400.0,
        error=1.0 * u.us, add_noise=False,
    )
    t = toa.merge_TOAs([t1, t2])
    t.compute_TDBs()
    return m, t


@pytest.fixture(scope="module")
def plsw_pint_model():
    """PINT model with PLSWNoise and SolarWindDispersion."""
    import astropy.units as u
    import pint.models as models
    import pint.toa as toa
    from pint.simulation import make_fake_toas_uniform

    par = """\
PSR           J0000+0000
RAJ           05:00:00   1
DECJ          +20:00:00  1
PEPOCH        55000
F0            100        1
F1            -1e-15     1
DM            15         1
NE_SW         4.0
TNSWAMP       -13
TNSWGAM       3.5
TNSWC         10
TZRMJD        55000
TZRFRQ        1400
TZRSITE       @
EPHEM         DE421
CLOCK         TT(BIPM2019)
UNITS         TDB
"""
    m = models.get_model(io.StringIO(par))
    t1 = make_fake_toas_uniform(
        54500, 55500, 50, model=m, obs="gbt", freq=820.0,
        error=1.0 * u.us, add_noise=False,
    )
    t2 = make_fake_toas_uniform(
        54500, 55500, 50, model=m, obs="gbt", freq=1400.0,
        error=1.0 * u.us, add_noise=False,
    )
    t = toa.merge_TOAs([t1, t2])
    t.compute_TDBs()
    return m, t


# ---------------------------------------------------------------------------
# PLRedNoise — baseline comparison (already implemented)
# ---------------------------------------------------------------------------


class TestPLRedNoiseVsPINT:
    """Verify PLRedNoise basis and weights match PINT's PLRedNoise."""

    @pytest.mark.slow
    def test_red_noise_basis_matches_pint(self, plred_pint_model):
        """JaxPINT Fourier basis matches PINT's get_noise_basis().

        PINT uses long-double TOA times for the Fourier basis while
        JaxPINT uses float64, so we allow ~1e-5 relative tolerance.
        """
        from jaxpint.bridge import build_timing_model

        pint_model, toas = plred_pint_model

        # PINT reference
        plred_comp = pint_model.components.get("PLRedNoise")
        if plred_comp is None:
            pytest.skip("No PLRedNoise in test model")
        pint_basis = plred_comp.get_noise_basis(toas)

        # JaxPINT via bridge
        _tm, noise_model = build_timing_model(pint_model, toas)
        assert noise_model.has_correlated

        # Find PLRedNoise component
        from jaxpint.noise.red_noise import PLRedNoise
        plred_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLRedNoise):
                plred_jax = comp
                break
        assert plred_jax is not None, "PLRedNoise not found in JaxPINT noise model"

        jax_basis = np.array(plred_jax.fourier_basis)

        npt.assert_allclose(
            jax_basis, pint_basis,
            rtol=1e-5, atol=1e-15,
            err_msg="PLRedNoise Fourier basis mismatch",
        )

    @pytest.mark.slow
    def test_red_noise_weights_match_pint(self, plred_pint_model):
        """JaxPINT PSD weights match PINT's get_noise_weights()."""
        from jaxpint.bridge import build_timing_model, pint_model_to_params

        pint_model, toas = plred_pint_model

        plred_comp = pint_model.components.get("PLRedNoise")
        pint_weights = plred_comp.get_noise_weights(toas)

        _tm, noise_model = build_timing_model(pint_model, toas)
        params = pint_model_to_params(pint_model).params

        from jaxpint.noise.red_noise import PLRedNoise
        plred_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLRedNoise):
                plred_jax = comp
                break
        assert plred_jax is not None

        jax_weights = np.array(plred_jax.psd_weights(params))

        npt.assert_allclose(
            jax_weights, pint_weights,
            rtol=1e-10,
            err_msg="PLRedNoise PSD weights mismatch",
        )

    @pytest.mark.slow
    def test_red_noise_covariance_matches_pint(self, plred_pint_model):
        """JaxPINT full covariance F @ diag(w) @ F.T matches PINT."""
        from jaxpint.bridge import build_timing_model, pint_toas_to_jax, pint_model_to_params

        pint_model, toas = plred_pint_model

        plred_comp = pint_model.components.get("PLRedNoise")
        pint_basis, pint_weights = plred_comp.pl_rn_basis_weight_pair(toas)
        pint_cov = pint_basis * pint_weights[None, :] @ pint_basis.T

        _tm, noise_model = build_timing_model(pint_model, toas)
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params

        from jaxpint.noise.red_noise import PLRedNoise
        plred_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLRedNoise):
                plred_jax = comp
                break

        _, U, Phi = plred_jax.covariance(toa_data, params)
        jax_cov = np.array(U @ jnp.diag(Phi) @ U.T)

        npt.assert_allclose(
            jax_cov, pint_cov,
            rtol=1e-8, atol=1e-25,
            err_msg="PLRedNoise covariance matrix mismatch",
        )


# ---------------------------------------------------------------------------
# PLDMNoise
# ---------------------------------------------------------------------------


class TestPLDMNoiseVsPINT:
    """Verify PLDMNoise basis and weights match PINT's PLDMNoise."""

    @pytest.mark.slow
    def test_dm_noise_basis_matches_pint(self, pldm_pint_model):
        """JaxPINT DM-scaled Fourier basis matches PINT's get_noise_basis().

        Allows ~1e-5 rtol due to long-double vs float64 time precision.
        """
        from jaxpint.bridge import build_timing_model

        pint_model, toas = pldm_pint_model

        pldm_comp = pint_model.components.get("PLDMNoise")
        if pldm_comp is None:
            pytest.skip("No PLDMNoise in test model")
        pint_basis = pldm_comp.get_noise_basis(toas)

        _tm, noise_model = build_timing_model(pint_model, toas)

        from jaxpint.noise.dm_noise import PLDMNoise
        pldm_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLDMNoise):
                pldm_jax = comp
                break
        assert pldm_jax is not None, "PLDMNoise not found in JaxPINT noise model"

        jax_basis = np.array(pldm_jax.fourier_basis)

        npt.assert_allclose(
            jax_basis, pint_basis,
            rtol=1e-5, atol=1e-15,
            err_msg="PLDMNoise basis mismatch",
        )

    @pytest.mark.slow
    def test_dm_noise_weights_match_pint(self, pldm_pint_model):
        """JaxPINT PSD weights match PINT's get_noise_weights()."""
        from jaxpint.bridge import build_timing_model, pint_model_to_params

        pint_model, toas = pldm_pint_model

        pldm_comp = pint_model.components.get("PLDMNoise")
        pint_weights = pldm_comp.get_noise_weights(toas)

        _tm, noise_model = build_timing_model(pint_model, toas)
        params = pint_model_to_params(pint_model).params

        from jaxpint.noise.dm_noise import PLDMNoise
        pldm_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLDMNoise):
                pldm_jax = comp
                break
        assert pldm_jax is not None

        jax_weights = np.array(pldm_jax.psd_weights(params))

        npt.assert_allclose(
            jax_weights, pint_weights,
            rtol=1e-10,
            err_msg="PLDMNoise PSD weights mismatch",
        )

    @pytest.mark.slow
    def test_dm_noise_covariance_matches_pint(self, pldm_pint_model):
        """Full DM noise covariance matches PINT's pl_dm_cov_matrix()."""
        from jaxpint.bridge import build_timing_model, pint_toas_to_jax, pint_model_to_params

        pint_model, toas = pldm_pint_model

        pldm_comp = pint_model.components.get("PLDMNoise")
        pint_cov = pldm_comp.pl_dm_cov_matrix(toas)

        _tm, noise_model = build_timing_model(pint_model, toas)
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params

        from jaxpint.noise.dm_noise import PLDMNoise
        pldm_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLDMNoise):
                pldm_jax = comp
                break

        _, U, Phi = pldm_jax.covariance(toa_data, params)
        jax_cov = np.array(U @ jnp.diag(Phi) @ U.T)

        npt.assert_allclose(
            jax_cov, pint_cov,
            rtol=1e-8, atol=1e-25,
            err_msg="PLDMNoise covariance matrix mismatch",
        )

    @pytest.mark.slow
    def test_dm_noise_frequency_scaling(self, pldm_pint_model):
        """820 MHz TOAs should have ~2.9x larger basis columns than 1400 MHz.

        (1400/820)^2 ≈ 2.91, so the ratio of basis column norms should
        reflect this DM frequency scaling.
        """
        from jaxpint.bridge import build_timing_model

        pint_model, toas = pldm_pint_model
        _tm, noise_model = build_timing_model(pint_model, toas)

        from jaxpint.noise.dm_noise import PLDMNoise
        pldm_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLDMNoise):
                pldm_jax = comp
                break

        basis = np.array(pldm_jax.fourier_basis)

        # Get barycentric frequencies from PINT
        import astropy.units as u
        bary_freqs = pint_model.barycentric_radio_freq(toas).to(u.MHz).value

        # Find a pair of TOAs at different frequencies
        low_mask = bary_freqs < 1000.0
        high_mask = bary_freqs > 1200.0
        assert np.any(low_mask) and np.any(high_mask)

        # Column norms at low freq vs high freq
        low_norm = np.mean(np.abs(basis[low_mask]))
        high_norm = np.mean(np.abs(basis[high_mask]))

        # Ratio should be approximately (1400/820)^2 / (1400/1400)^2 ≈ 2.91
        ratio = low_norm / high_norm
        assert ratio > 2.0, f"DM frequency scaling ratio = {ratio:.2f}, expected > 2.0"


# ---------------------------------------------------------------------------
# PLChromNoise
# ---------------------------------------------------------------------------


class TestPLChromNoiseVsPINT:
    """Verify PLChromNoise basis and weights match PINT's PLChromNoise."""

    @pytest.mark.slow
    def test_chrom_noise_basis_matches_pint(self, plchrom_pint_model):
        """JaxPINT chromatic basis matches PINT's get_noise_basis().

        Allows ~1e-5 rtol due to long-double vs float64 time precision.
        """
        from jaxpint.bridge import build_timing_model, pint_toas_to_jax, pint_model_to_params

        pint_model, toas = plchrom_pint_model

        plchrom_comp = pint_model.components.get("PLChromNoise")
        if plchrom_comp is None:
            pytest.skip("No PLChromNoise in test model")
        pint_basis = plchrom_comp.get_noise_basis(toas)

        _tm, noise_model = build_timing_model(pint_model, toas)
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params

        from jaxpint.noise.chrom_noise import PLChromNoise
        plchrom_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLChromNoise):
                plchrom_jax = comp
                break
        assert plchrom_jax is not None, "PLChromNoise not found in JaxPINT noise model"

        # PLChromNoise computes scaled basis at runtime
        _, U, _ = plchrom_jax.covariance(toa_data, params)
        jax_basis = np.array(U)

        npt.assert_allclose(
            jax_basis, pint_basis,
            rtol=1e-5, atol=1e-15,
            err_msg="PLChromNoise basis mismatch",
        )

    @pytest.mark.slow
    def test_chrom_noise_weights_match_pint(self, plchrom_pint_model):
        """JaxPINT PSD weights match PINT's get_noise_weights()."""
        from jaxpint.bridge import build_timing_model, pint_model_to_params

        pint_model, toas = plchrom_pint_model

        plchrom_comp = pint_model.components.get("PLChromNoise")
        pint_weights = plchrom_comp.get_noise_weights(toas)

        _tm, noise_model = build_timing_model(pint_model, toas)
        params = pint_model_to_params(pint_model).params

        from jaxpint.noise.chrom_noise import PLChromNoise
        plchrom_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLChromNoise):
                plchrom_jax = comp
                break
        assert plchrom_jax is not None

        jax_weights = np.array(plchrom_jax.psd_weights(params))

        npt.assert_allclose(
            jax_weights, pint_weights,
            rtol=1e-10,
            err_msg="PLChromNoise PSD weights mismatch",
        )

    @pytest.mark.slow
    def test_chrom_noise_covariance_matches_pint(self, plchrom_pint_model):
        """Full chromatic noise covariance matches PINT's pl_chrom_cov_matrix()."""
        from jaxpint.bridge import build_timing_model, pint_toas_to_jax, pint_model_to_params

        pint_model, toas = plchrom_pint_model

        plchrom_comp = pint_model.components.get("PLChromNoise")
        pint_cov = plchrom_comp.pl_chrom_cov_matrix(toas)

        _tm, noise_model = build_timing_model(pint_model, toas)
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params

        from jaxpint.noise.chrom_noise import PLChromNoise
        plchrom_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLChromNoise):
                plchrom_jax = comp
                break

        _, U, Phi = plchrom_jax.covariance(toa_data, params)
        jax_cov = np.array(U @ jnp.diag(Phi) @ U.T)

        npt.assert_allclose(
            jax_cov, pint_cov,
            rtol=1e-8, atol=1e-25,
            err_msg="PLChromNoise covariance matrix mismatch",
        )


# ---------------------------------------------------------------------------
# PLSWNoise
# ---------------------------------------------------------------------------


class TestPLSWNoiseVsPINT:
    """Verify PLSWNoise basis and weights match PINT's PLSWNoise."""

    @pytest.mark.slow
    def test_sw_noise_basis_matches_pint(self, plsw_pint_model):
        """JaxPINT SW-scaled Fourier basis matches PINT's get_noise_basis().

        Allows ~1e-5 rtol due to long-double vs float64 time precision
        in the Fourier basis, and minor geometry precision differences.
        """
        from jaxpint.bridge import build_timing_model, pint_toas_to_jax, pint_model_to_params

        pint_model, toas = plsw_pint_model

        plsw_comp = pint_model.components.get("PLSWNoise")
        if plsw_comp is None:
            pytest.skip("No PLSWNoise in test model")
        pint_basis = plsw_comp.get_noise_basis(toas)

        _tm, noise_model = build_timing_model(pint_model, toas)
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params

        from jaxpint.noise.sw_noise import PLSWNoise
        plsw_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLSWNoise):
                plsw_jax = comp
                break
        assert plsw_jax is not None, "PLSWNoise not found in JaxPINT noise model"

        # PLSWNoise computes scaled basis at runtime
        _, U, _ = plsw_jax.covariance(toa_data, params)
        jax_basis = np.array(U)

        npt.assert_allclose(
            jax_basis, pint_basis,
            rtol=1e-5, atol=1e-20,
            err_msg="PLSWNoise basis mismatch",
        )

    @pytest.mark.slow
    def test_sw_noise_weights_match_pint(self, plsw_pint_model):
        """JaxPINT PSD weights match PINT's get_noise_weights()."""
        from jaxpint.bridge import build_timing_model, pint_model_to_params

        pint_model, toas = plsw_pint_model

        plsw_comp = pint_model.components.get("PLSWNoise")
        pint_weights = plsw_comp.get_noise_weights(toas)

        _tm, noise_model = build_timing_model(pint_model, toas)
        params = pint_model_to_params(pint_model).params

        from jaxpint.noise.sw_noise import PLSWNoise
        plsw_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLSWNoise):
                plsw_jax = comp
                break
        assert plsw_jax is not None

        jax_weights = np.array(plsw_jax.psd_weights(params))

        npt.assert_allclose(
            jax_weights, pint_weights,
            rtol=1e-10,
            err_msg="PLSWNoise PSD weights mismatch",
        )

    @pytest.mark.slow
    def test_sw_noise_covariance_matches_pint(self, plsw_pint_model):
        """Full SW noise covariance matches PINT's pl_sw_cov_matrix()."""
        from jaxpint.bridge import build_timing_model, pint_toas_to_jax, pint_model_to_params

        pint_model, toas = plsw_pint_model

        plsw_comp = pint_model.components.get("PLSWNoise")
        pint_cov = plsw_comp.pl_sw_cov_matrix(toas)

        _tm, noise_model = build_timing_model(pint_model, toas)
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params

        from jaxpint.noise.sw_noise import PLSWNoise
        plsw_jax = None
        for comp in noise_model.correlated:
            if isinstance(comp, PLSWNoise):
                plsw_jax = comp
                break

        _, U, Phi = plsw_jax.covariance(toa_data, params)
        jax_cov = np.array(U @ jnp.diag(Phi) @ U.T)

        npt.assert_allclose(
            jax_cov, pint_cov,
            rtol=1e-8, atol=1e-25,
            err_msg="PLSWNoise covariance matrix mismatch",
        )
