"""Per-pulsar conditional noise reconstruction: basis_at + delay consumers.

Pins the analytic Fourier ``basis_at`` against the stored grid basis, and
the :func:`conditional_noise_delays` / :func:`conditional_noise_delay_bands`
consumers against the stacked-basis identities and sampled draws.

Reuses the red + two-backend-ECORR pulsar from ``test_ecorr_kernel`` — a
model with one off-grid-capable component (Fourier red) and one
incapable one (epoch-indicator ECORR), which is exactly the mixed case
the consumers must handle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from typing import cast

from jaxpint.noise import (
    FreeSpectrumNoise,
    NoiseModel,
    PLChromNoise,
    PLDMNoise,
    PLRedNoise,
    PLSWNoise,
)
from jaxpint.noise._basis_gp import _BasisGPNoise
from jaxpint.pta import (
    conditional_noise_delay_bands,
    conditional_noise_delays,
    conditional_single_pulsar,
    sample_conditional,
)
from tests.helpers import make_params
from tests.test_ecorr_kernel import _setup


# ---------------------------------------------------------------------------
# Analytic Fourier basis_at
# ---------------------------------------------------------------------------


class TestFourierBasisAt:
    def test_matches_stored_basis_on_grid(self):
        toa_data, _, nm_b, _, params, _ = _setup()
        red = cast(PLRedNoise, nm_b.correlated[0])
        t = jnp.asarray(toa_data.tdb_seconds)
        F_at = red.basis_at(t, params)
        # numpy sin/cos (build) vs XLA sin/cos (basis_at): equal to ulps.
        npt.assert_allclose(
            np.asarray(F_at), np.asarray(red.fourier_basis), atol=1e-12
        )
        # Achromatic: freq_mhz is accepted and ignored.
        npt.assert_array_equal(
            np.asarray(red.basis_at(t, params, freq_mhz=700.0)),
            np.asarray(F_at),
        )

    def test_free_spectrum_overrides_basis_at(self):
        toa_data, _, nm_b, _, params, _ = _setup()
        red = cast(PLRedNoise, nm_b.correlated[0])
        fs = FreeSpectrumNoise(
            fourier_basis=red.fourier_basis,
            freqs=red.freqs,
            freq_bin_widths=red.freq_bin_widths,
            rho_names=("RHO_1", "RHO_2", "RHO_3"),
        )
        t = jnp.asarray(toa_data.tdb_seconds)
        # The substantive assertion: FreeSpectrumNoise has the override at
        # all (does not inherit the base's None default).
        fs_at = fs.basis_at(t, params)
        assert fs_at is not None
        # Tripwire, not an independent check: both classes delegate to the
        # same _fourier_basis_at helper, so equality can only fail if the
        # implementations are ever forked.
        npt.assert_array_equal(
            np.asarray(fs_at), np.asarray(red.basis_at(t, params))
        )

    def test_chromatic_requires_freq_mhz_and_matches_grid(self):
        toa_data, _, nm_b, _, _, _ = _setup()
        red = cast(PLRedNoise, nm_b.correlated[0])
        chrom = PLChromNoise(
            fourier_basis=red.fourier_basis,
            freqs=red.freqs,
            freq_bin_widths=red.freq_bin_widths,
            tnchromamp_name="TNCHROMAMP",
            tnchromgam_name="TNCHROMGAM",
            tnchromidx_name="TNCHROMIDX",
        )
        params = make_params(
            ("TNCHROMAMP", "TNCHROMGAM", "TNCHROMIDX"), (-14.0, 3.0, 4.0)
        )
        t = jnp.asarray(toa_data.tdb_seconds)
        # Without a frequency there is no defensible scaling: None, loudly.
        assert chrom.basis_at(t, params) is None
        # Grid consistency: per-time frequencies == the on-grid basis.
        F_at = chrom.basis_at(t, params, freq_mhz=toa_data.freq)
        npt.assert_allclose(
            np.asarray(F_at),
            np.asarray(chrom._basis(toa_data, params)),
            rtol=1e-12,
            atol=1e-15,
        )
        # Scalar reference frequency at fref: pure time basis (D = 1).
        npt.assert_allclose(
            np.asarray(chrom.basis_at(t, params, freq_mhz=1400.0)),
            np.asarray(red.basis_at(t, params)),
            rtol=1e-12,
        )

    def test_dm_basis_at_reference_frequency(self):
        toa_data, _, nm_b, _, params, _ = _setup()
        red = cast(PLRedNoise, nm_b.correlated[0])
        # The stored (pre-baked) basis is irrelevant to basis_at — DM
        # rebuilds analytically; hand it the raw basis for construction.
        dm = PLDMNoise(
            fourier_basis=red.fourier_basis,
            freqs=red.freqs,
            freq_bin_widths=red.freq_bin_widths,
            tndmamp_name="TNDMAMP",
            tndmgam_name="TNDMGAM",
        )
        t = jnp.asarray(toa_data.tdb_seconds)
        assert dm.basis_at(t, params) is None
        # At the 1400 MHz convention the (1400/f)^2 factor is exactly 1.
        npt.assert_allclose(
            np.asarray(dm.basis_at(t, params, freq_mhz=1400.0)),
            np.asarray(red.basis_at(t, params)),
            rtol=1e-12,
        )
        # alpha = 2 exactly: 700 MHz -> factor 4.
        npt.assert_allclose(
            np.asarray(dm.basis_at(t, params, freq_mhz=700.0)),
            4.0 * np.asarray(red.basis_at(t, params)),
            rtol=1e-12,
        )

    def test_solar_wind_keeps_none_default(self):
        # SW's row scaling is Earth-Sun geometry, derivable from time — a
        # covariate parameter would create two sources of truth. It stays
        # None until the component can recompute its geometry host-side.
        assert PLSWNoise.basis_at is _BasisGPNoise.basis_at


# ---------------------------------------------------------------------------
# conditional_noise_delays / _bands
# ---------------------------------------------------------------------------


class TestConditionalNoiseDelays:
    def test_on_grid_components_sum_to_total(self):
        toa_data, tm, nm_b, _, params, _ = _setup()
        cond = conditional_single_pulsar(toa_data, tm, nm_b, params)
        delays = conditional_noise_delays(toa_data, nm_b, params, cond.mean)

        assert set(delays) == {"PLRedNoise", "EcorrNoise"}
        assert all(d is not None for d in delays.values())
        # The documented total realization is U @ mean with the stacked U;
        # the consumer must reproduce it from per-component slices.
        _, U, _ = nm_b.covariance(toa_data, params)
        total = np.asarray(U @ cond.mean)
        npt.assert_allclose(
            np.asarray(delays["PLRedNoise"] + delays["EcorrNoise"]),
            total,
            rtol=1e-12,
            atol=1e-18,
        )

    def test_off_grid_raises_by_default(self):
        # Loud by default (house rule, and consistent with the PTA-tier
        # conditional_gwb_delays): an off-grid-incapable component is an
        # error unless the caller acknowledges it with skip_ungridded.
        toa_data, tm, nm_b, _, params, _ = _setup()
        cond = conditional_single_pulsar(toa_data, tm, nm_b, params)
        with pytest.raises(ValueError, match="EcorrNoise.*skip_ungridded"):
            conditional_noise_delays(
                toa_data, nm_b, params, cond.mean,
                times_seconds=toa_data.tdb_seconds,
            )
        with pytest.raises(ValueError, match="EcorrNoise.*skip_ungridded"):
            conditional_noise_delay_bands(
                toa_data, nm_b, params, cond,
                times_seconds=toa_data.tdb_seconds,
            )

    def test_off_grid_red_matches_on_grid_at_toa_times(self):
        toa_data, tm, nm_b, _, params, _ = _setup()
        cond = conditional_single_pulsar(toa_data, tm, nm_b, params)
        on_grid = conditional_noise_delays(toa_data, nm_b, params, cond.mean)
        off_grid = conditional_noise_delays(
            toa_data, nm_b, params, cond.mean,
            times_seconds=toa_data.tdb_seconds,
            skip_ungridded=True,
        )
        # Same times passed explicitly: analytic red must agree with the
        # stored basis; acknowledged-skipped ECORR maps to explicit None.
        npt.assert_allclose(
            np.asarray(off_grid["PLRedNoise"]),
            np.asarray(on_grid["PLRedNoise"]),
            rtol=1e-9,
            atol=1e-18,
        )
        assert off_grid["EcorrNoise"] is None

    def test_skipped_component_width_still_advances_offsets(self):
        # The layout must count a skipped (None-basis) component's width so
        # LATER components slice the right coefficients. Red comes first in
        # _setup's model, so swap the order: ECORR's n_epochs coefficients
        # now precede red's, and a broken offset advance would hand red the
        # wrong slice.
        toa_data, tm, nm_b, _, params, _ = _setup()
        red, ecorr = nm_b.correlated
        nm_swapped = NoiseModel(
            white_noise=nm_b.white_noise, correlated=(ecorr, red)
        )
        cond = conditional_single_pulsar(toa_data, tm, nm_swapped, params)
        on_grid = conditional_noise_delays(
            toa_data, nm_swapped, params, cond.mean
        )
        off_grid = conditional_noise_delays(
            toa_data, nm_swapped, params, cond.mean,
            times_seconds=toa_data.tdb_seconds,
            skip_ungridded=True,
        )
        assert off_grid["EcorrNoise"] is None
        npt.assert_allclose(
            np.asarray(off_grid["PLRedNoise"]),
            np.asarray(on_grid["PLRedNoise"]),
            rtol=1e-9,
            atol=1e-18,
        )

    def test_dense_grid_shapes_and_finite(self):
        # Smoke by design for values (sin/cos are always finite); the real
        # content is the shape assertion on an n_times != n_toas grid,
        # which catches accidental broadcasting against the TOA axis.
        toa_data, tm, nm_b, _, params, _ = _setup()
        cond = conditional_single_pulsar(toa_data, tm, nm_b, params)
        t = np.asarray(toa_data.tdb_seconds)
        grid = jnp.linspace(t.min(), t.max(), 400)
        delays = conditional_noise_delays(
            toa_data, nm_b, params, cond.mean, times_seconds=grid,
            skip_ungridded=True,
        )
        assert delays["PLRedNoise"].shape == (400,)
        assert bool(jnp.all(jnp.isfinite(delays["PLRedNoise"])))

    def test_bands_mean_matches_delays_and_std_matches_draws(self):
        toa_data, tm, nm_b, _, params, _ = _setup()
        cond = conditional_single_pulsar(toa_data, tm, nm_b, params)
        bands = conditional_noise_delay_bands(toa_data, nm_b, params, cond)
        delays = conditional_noise_delays(toa_data, nm_b, params, cond.mean)
        # Mean equality is a cross-consistency tripwire, not an independent
        # check: both consumers share _noise_component_layout and compute
        # basis @ mean[slice]. The draw-based std below is the substance.
        for name in ("PLRedNoise", "EcorrNoise"):
            npt.assert_allclose(
                np.asarray(bands[name].mean), np.asarray(delays[name]),
                rtol=1e-12,
            )
        # Draw-based verification of the band std (same style as the GWB
        # band test): per-draw component delays' sample std ~ band std.
        draws = sample_conditional(jax.random.PRNGKey(3), cond, n_draws=4000)
        _, U, _ = nm_b.covariance(toa_data, params)
        n_red = 6
        red_delays = draws[:, :n_red] @ np.asarray(U)[:, :n_red].T
        sample_std = np.std(np.asarray(red_delays), axis=0)
        npt.assert_allclose(
            np.asarray(bands["PLRedNoise"].std), sample_std, rtol=0.08
        )

    def test_coefficient_length_validation(self):
        toa_data, tm, nm_b, _, params, _ = _setup()
        cond = conditional_single_pulsar(toa_data, tm, nm_b, params)
        with pytest.raises(ValueError, match="correlated blocks span"):
            conditional_noise_delays(toa_data, nm_b, params, cond.mean[:-1])
        # Trailing coefficients (an external_cov block) are ignored.
        padded = jnp.concatenate([cond.mean, jnp.ones(4)])
        base = conditional_noise_delays(toa_data, nm_b, params, cond.mean)
        with_ext = conditional_noise_delays(toa_data, nm_b, params, padded)
        npt.assert_array_equal(
            np.asarray(with_ext["PLRedNoise"]), np.asarray(base["PLRedNoise"])
        )

    def test_chromatic_consumer_freq_mhz(self):
        # A red + chromatic model: off-grid reconstruction must demand the
        # evaluation frequency, and with per-time frequencies at the TOA
        # times must reproduce the on-grid chromatic delay.
        toa_data, tm, nm_b, _, _, _ = _setup()
        red = cast(PLRedNoise, nm_b.correlated[0])
        chrom = PLChromNoise(
            fourier_basis=red.fourier_basis,
            freqs=red.freqs,
            freq_bin_widths=red.freq_bin_widths,
            tnchromamp_name="TNCHROMAMP",
            tnchromgam_name="TNCHROMGAM",
            tnchromidx_name="TNCHROMIDX",
        )
        nm = NoiseModel(white_noise=nm_b.white_noise, correlated=(red, chrom))
        params = make_params(
            names=(
                "F0", "F1", "PEPOCH", "EFAC1", "EQUAD1",
                "TNREDAMP", "TNREDGAM",
                "TNCHROMAMP", "TNCHROMGAM", "TNCHROMIDX",
            ),
            values=(200.0, -1e-15, 0.0, 1.1, 3e-7, -13.5, 3.2, -14.0, 3.0, 4.0),
            frozen_mask=(False, False, True, True, True,
                         False, False, True, True, True),
            epoch_int_values={"PEPOCH": 59000.0},
        )
        cond = conditional_single_pulsar(toa_data, tm, nm, params)
        with pytest.raises(ValueError, match="freq_mhz"):
            conditional_noise_delays(
                toa_data, nm, params, cond.mean,
                times_seconds=toa_data.tdb_seconds,
            )
        on_grid = conditional_noise_delays(toa_data, nm, params, cond.mean)
        off_grid = conditional_noise_delays(
            toa_data, nm, params, cond.mean,
            times_seconds=toa_data.tdb_seconds,
            freq_mhz=toa_data.freq,
        )
        npt.assert_allclose(
            np.asarray(off_grid["PLChromNoise"]),
            np.asarray(on_grid["PLChromNoise"]),
            rtol=1e-9,
            atol=1e-18,
        )

    def test_kernel_model_reconstructs_red_identically(self):
        # The non-redundant content is the set assertion: the consumer
        # accepts a kernel NoiseModel (correlated=(red,) only, no kernel
        # entry). The delay equality is transitively implied by the kernel
        # suite's conditional red-block pin (cond_k.mean == cond_b.mean[:6])
        # plus delays being basis @ mean — kept as a cheap corollary.
        toa_data, tm, nm_b, nm_k, params, _ = _setup()
        cond_b = conditional_single_pulsar(toa_data, tm, nm_b, params)
        cond_k = conditional_single_pulsar(toa_data, tm, nm_k, params)
        d_b = conditional_noise_delays(toa_data, nm_b, params, cond_b.mean)
        d_k = conditional_noise_delays(toa_data, nm_k, params, cond_k.mean)
        assert set(d_k) == {"PLRedNoise"}
        npt.assert_allclose(
            np.asarray(d_k["PLRedNoise"]), np.asarray(d_b["PLRedNoise"]),
            rtol=1e-8,
        )
