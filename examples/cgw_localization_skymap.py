"""CGW sky-localization map vs. anchor-pulsar count.

Reproduces, approximately, the anchor-pulsar scaling result of Wen et al. 2026
(arXiv:2603.28897, "From Detection to Host Galaxy Identification: Precision
CGW Localization with a Few Anchor Pulsars"): a small subset of pulsars with
sub-wavelength PX precision is enough to phase-lock the array and dramatically
shrink the 90% credible sky-localization area.

Method (Fisher matrix at the truth point; see ``jaxpint/pta/cw_localization.py``):

1. Each pixel of a HEALPix grid is a candidate true sky position. At that pixel
   the unit-strain signal power ``Y = (s_hat | s_hat)`` is computed via
   :func:`jaxpint.pta.cw_upper_limit.quadratic_coeffs`; ``h0`` is calibrated per
   pixel so the optimal matched-filter SNR equals ``SNR_TARGET`` (default 20).
2. With ``h0`` fixed at that calibration the timing-marginalized log-likelihood is
   approximately quadratic in the sky parameters ``(cos_gwtheta, gwphi)`` near
   the truth, so the 2-D Fisher information is just ``F = -Hessian_sky(logL)``.
3. The 90% credible area is ``pi * 4.605 * sqrt(det F^{-1})`` steradians, with
   no Jacobian correction since ``(cos_gwtheta, gwphi)`` is the area-preserving
   sky parameterization. Convert to deg^2 via ``(180/pi)^2``.
4. "Anchor pulsars" are encoded via a per-pulsar ``pulsar_term_mask`` on
   ``CWInjector``: ``True`` for an anchor (pulsar term included, PX pegged),
   ``False`` for a non-anchor (Earth-term-only for that pulsar — the Fisher-level
   approximation of the ``Phi_p``-uniform / Bessel-``I_0`` marginalization).

Source parameters match Wen et al.: ``M_c = 5e8 M_sun``, ``f_GW = 10^-8.4 Hz``,
SNR target 20 by default.

Usage
-----
    python examples/cgw_localization_skymap.py generate [--output PATH] \
        [--nside N] [--anchor-pulsars NAME ...] [--snr SNR]
    python examples/cgw_localization_skymap.py plot     [--input PATH]
    python examples/cgw_localization_skymap.py sweep    [--out-dir DIR]

The ``sweep`` mode runs a small built-in anchor-count sweep and produces both the
per-config sky maps and a 1-D scaling plot (90%-area median + min/max across the
sky vs. number of anchors).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Reuse loader / drop list / marginalization parameter set / pulsar-vector helper
# from the distance-skymap example — same PTA setup either way.
from examples.cgw_distance_skymap import (
    DATA_DIR, DROP_PULSARS, SMOKE_SUBSET, MARG_PARAMS,
    pulsar_unit_vector_icrs, _import_healpy, _log,
)


# Wen et al. 2026 fiducial source.
LOG10_MC = float(np.log10(5.0e8))  # chirp mass 5e8 Msun
LOG10_FGW = -8.4                   # f_GW = 10^-8.4 Hz ≈ 4 nHz
SNR_TARGET_DEFAULT = 20.0

# CGW orientation held at face-on/optimal (matches the distance-skymap default).
FIXED_ORIENTATION = (1.0, 0.0, 0.0)  # cos_inc, psi, phase0

DEFAULT_OUTPUT = Path("cgw_localization_skymap.npz")

# ---------------------------------------------------------------------------
# Wen et al. 2026 (arXiv:2603.28897) configurations.
#
# Wen uses 25 pulsars from PPTA DR3 + EPTA DR2 + NG15 + MeerKAT-4.5yr.  Of those
# 25, 18 are also in ocarina; the other 7 (J1857+0943, J0636-3044, J1946-5403,
# J2222-0137, J2241-5236, J0125-2327, J1017-7156) are missing.  Two of Wen's six
# anchor pulsars (J2222-0137 in the 25-3 set, J0636-3044 in the 25-6 set) are
# among the missing 7; we substitute the canonical sub-λ-PX MSPs that *are* in
# ocarina (J1713+0747 and J1640+2224 respectively) — Wen's anchor-selection
# intent is preserved (sub-wavelength-PX precision on well-timed MSPs).
# ---------------------------------------------------------------------------
WEN_OCARINA_18: tuple[str, ...] = (
    # EPTA+InPTA in ocarina (5 of 6)
    "J0613-0200", "J1024-0719", "J1600-3053", "J1730-2304", "J1843-1113",
    # MPTA in ocarina (1 of 5)
    "J0437-4715",
    # NANOGrav in ocarina (10 of 10)
    "J0030+0451", "J1640+2224", "J1741+1351", "J1909-3744", "J1911+1347",
    "J2017+0603", "J2043+1711", "J2234+0611", "J2234+0944", "J2317+1439",
    # PPTA in ocarina (2 of 4)
    "J1713+0747", "J1744-1134",
)

# Wen's three discrete array configurations (analogs).  Anchor pulsars get the
# pulsar term included (PX pegged); non-anchors get Earth-term only.
WEN_CONFIGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("standard", ()),
    # 25-3 analog: Wen's = {J0030+0451, J0437-4715, J2222-0137}.
    # J2222-0137 missing from ocarina → substitute J1713+0747.
    ("25-3", ("J0437-4715", "J0030+0451", "J1713+0747")),
    # 25-6 analog: Wen's = above + {J0636-3044, J1744-1134, J1909-3744}.
    # J0636-3044 missing → substitute J1640+2224.
    ("25-6", ("J0437-4715", "J0030+0451", "J1713+0747",
              "J1640+2224", "J1744-1134", "J1909-3744")),
)


def compute_localization_skymap(
    *,
    pulsar_subset=SMOKE_SUBSET,
    anchor_pulsars: tuple[str, ...] = (),
    nside: int = 8,
    snr_target: float = SNR_TARGET_DEFAULT,
    orientation=FIXED_ORIENTATION,
    pixel_chunk: int = 32,
    validate_linearity: bool = False,
):
    """Compute the per-pixel 90% credible localization area in deg^2.

    Parameters
    ----------
    pulsar_subset
        Names of pulsars to include from the ocarina dataset.
    anchor_pulsars
        Names of pulsars whose pulsar term is included (PX pegged to the par
        value). Non-anchor pulsars get Earth-term-only treatment.
    nside
        HEALPix nside; npix = 12*nside^2.
    snr_target
        Per-pixel optimal SNR set by calibrating ``h0``. The Fisher (and hence
        the credible area) scales as ``1/SNR^2``.
    orientation
        ``(cos_inc, psi, phase0)`` held fixed at the truth point.
    pixel_chunk
        Pixels vmapped together per chunk; trades memory for speed (same
        pattern as ``cgw_distance_skymap.compute_skymap``).
    validate_linearity
        Forwarded to ``marginalize``.

    Returns
    -------
    results : dict
        Keys: ``loc_area_90_deg2`` and ``loc_area_50_deg2`` (HEALPix maps,
        RING ordering), ``nside``, ``snr_target``, ``pulsar_names``,
        ``anchor_pulsars``, ``pulsar_term_mask``, ``log10_mc``, ``log10_fgw``,
        orientation, ``pulsar_pos`` (ICRS unit vectors for overlay plotting).

    Notes
    -----
    **Two-injector cross-derivative Gram extraction.** Uses TWO
    :class:`CWInjector` in ``PTAConfig.signal_injectors`` (prefixes ``cwt_``
    and ``cwd_``).  Both are *templates* (no data injection); the second
    exists only to give the autodiff machinery two independent sky parameters
    to differentiate against.

    Per-pixel Fisher comes from :func:`jaxpint.pta.cw_localization.gram_at_pixel`:
    the bilinear structure of the Gaussian likelihood in
    ``(h_a, h_b)`` means the mixed amplitude derivative
    ``-d²logL/dh_a dh_b`` isolates the cross-inner-product
    ``Z(sky_a, sky_b) = (s_hat(sky_a) | s_hat(sky_b))_N`` exactly.  Taking the
    mixed sky-Hessian ``d²Z/dsky_a dsky_b`` at ``sky_a = sky_b = pixel`` gives
    the Gram matrix ``(d_i s_hat | d_j s_hat)_N`` — PSD by construction.

    ``F = h0_target² * Gram`` is the proper Wen-style sensitivity Fisher.
    Unlike the data-injection ``-Hessian(logL)`` approach (which equals
    ``h0² * Gram`` only in expectation over noise), this is exact per
    realization and noise-independent, so every pixel produces a finite,
    PSD Fisher and a finite credible area.
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from loguru import logger

    hp = _import_healpy()
    logger.disable("pint")

    from jaxpint import load_nanograv_pta
    from jaxpint.pta.likelihood import PTAConfig, pta_logL
    from jaxpint.pta.params import GlobalParams
    from jaxpint.bayes import ImproperPrior, marginalize
    from jaxpint.pta.signals.cw import CWInjector
    from jaxpint.pta.cw_upper_limit import quadratic_coeffs
    from jaxpint.pta.cw_localization import (
        h0_for_snr, credible_area_deg2, gram_at_pixel,
    )

    # ---- 1. Load + filter --------------------------------------------------
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"DATA_DIR {DATA_DIR} not found.")
    psrs = load_nanograv_pta(DATA_DIR, pulsar_names=pulsar_subset)
    keep = [i for i, n in enumerate(psrs.pulsar_names) if n not in DROP_PULSARS]
    names = [psrs.pulsar_names[i] for i in keep]
    toa_list = tuple(psrs.toa_data_list[i] for i in keep)
    pp_list = tuple(psrs.pulsar_params_list[i] for i in keep)
    tm_list = tuple(psrs.timing_models[i] for i in keep)
    nm_list = tuple(psrs.noise_models[i] for i in keep)
    n_toa_total = int(sum(int(td.n_toas) for td in toa_list))
    _log(f"Loaded {len(names)} pulsars, {n_toa_total} TOAs total.")

    # ---- 2. Per-pulsar anchor mask -----------------------------------------
    anchor_set = set(anchor_pulsars)
    unknown = anchor_set - set(names)
    if unknown:
        raise ValueError(
            f"Requested anchor pulsars not in pulsar subset: {sorted(unknown)}. "
            f"Available: {names}"
        )
    pulsar_term_mask = tuple(name in anchor_set for name in names)
    n_anchors = sum(pulsar_term_mask)
    _log(f"Anchor pulsars ({n_anchors}/{len(names)}): "
         f"{sorted(anchor_set) if anchor_set else '(none — all Earth-term-only)'}")

    positions = jnp.asarray(np.stack([pulsar_unit_vector_icrs(pp) for pp in pp_list]))

    # ---- 3. Two injectors + config + global params ------------------------
    # Template: model template the inference would fit. Variable (h_t, sky_t).
    template_injector = CWInjector(
        positions, prefix="cwt_",
        earth_term_only=False, linear_amplitude=True,
        pulsar_term_mask=pulsar_term_mask,
        initial_values={"log10_fgw": LOG10_FGW},
    )
    # Data injection: signal with NEGATIVE amplitude at the pixel adds to the
    # effective residual via (data − signal). h_d and sky_d are bound at
    # evaluation time so this acts as data, not a second model source.
    data_injector = CWInjector(
        positions, prefix="cwd_",
        earth_term_only=False, linear_amplitude=True,
        pulsar_term_mask=pulsar_term_mask,
        initial_values={"log10_fgw": LOG10_FGW},
    )
    gp = template_injector.register_params(GlobalParams.empty())
    gp = data_injector.register_params(gp)
    config = PTAConfig(
        toa_data_list=toa_list, timing_models=tm_list,
        noise_models=nm_list,
        signal_injectors=(template_injector, data_injector),
    )

    # ---- 4. Timing-model marginalization (improper priors) -----------------
    over, priors = set(), {}
    for pn, pp in zip(names, pp_list):
        for nm in pp.free_names():
            if nm in MARG_PARAMS:
                fqn = f"{pn}_{nm}"
                over.add(fqn)
                priors[fqn] = ImproperPrior()
    _log(f"Marginalizing {len(over)} timing params across {len(names)} pulsars...")
    g, _, reduced_pp = marginalize(
        pta_logL, over=over, priors=priors, config=config,
        pulsar_names=tuple(names),
        fiducial_pulsar_params=pp_list, fiducial_global_params=gp,
        validate_linearity=validate_linearity, allow_nonlinear=True,
    )

    # ---- 5. logL closure: full (h_t, h_d, sky_t, sky_d) variation ----------
    # Both injectors carry the same fixed (cos_inc, psi, phase0, log10_fgw).
    cos_inc_fix, psi_fix, phase0_fix = (float(x) for x in orientation)
    idx_t = {k: gp._name_to_index[f"cwt_{k}"] for k in
             ("h0", "cos_gwtheta", "gwphi", "cos_inc", "psi", "phase0")}
    idx_d = {k: gp._name_to_index[f"cwd_{k}"] for k in
             ("h0", "cos_gwtheta", "gwphi", "cos_inc", "psi", "phase0")}
    base_vals = (gp.values
                 .at[idx_t["cos_inc"]].set(cos_inc_fix)
                 .at[idx_t["psi"]].set(psi_fix)
                 .at[idx_t["phase0"]].set(phase0_fix)
                 .at[idx_d["cos_inc"]].set(cos_inc_fix)
                 .at[idx_d["psi"]].set(psi_fix)
                 .at[idx_d["phase0"]].set(phase0_fix))

    def logL_full(h_t, h_d, sky_t, sky_d):
        v = (base_vals
             .at[idx_t["h0"]].set(h_t)
             .at[idx_t["cos_gwtheta"]].set(sky_t[0])
             .at[idx_t["gwphi"]].set(sky_t[1])
             .at[idx_d["h0"]].set(h_d)
             .at[idx_d["cos_gwtheta"]].set(sky_d[0])
             .at[idx_d["gwphi"]].set(sky_d[1]))
        gp_new = eqx.tree_at(lambda gg: gg.values, gp, v)
        return g(gp_new, reduced_pp)

    # ---- 6. Pixel loop: Y → h0(SNR=target) → -Hessian → area --------------
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    sky = jnp.stack([jnp.cos(jnp.asarray(theta)), jnp.asarray(phi)], axis=1)  # (npix, 2)

    def area_for_pixel(sky_row):
        # Y(pixel) for SNR calibration: with h_d=0 (and h_t free), only the
        # template injector contributes — quadratic_coeffs on h_t returns the
        # same Y(sky_pixel) as the single-injector path would.
        amp_logL = lambda h: logL_full(h, jnp.float64(0.0), sky_row, sky_row)
        _X, Y = quadratic_coeffs(amp_logL)
        h0 = h0_for_snr(jnp.float64(snr_target), Y)
        # Proper Fisher via cross-derivative Gram extraction (noise-free, exact
        # per realization, PSD by construction).  The likelihood is bilinear in
        # (h_t, h_d), so the mixed h-derivative isolates Z(sky_t, sky_d); the
        # mixed sky-Hessian of Z then gives Gram_ij at sky_t = sky_d = pixel.
        # F = h0**2 * Gram is the proper Wen-style sensitivity Fisher.
        Gram = gram_at_pixel(logL_full, sky_row)
        F = h0**2 * Gram
        return (credible_area_deg2(F, level=0.9),
                credible_area_deg2(F, level=0.5))

    @jax.jit
    def all_areas(sky_arr):
        return jax.lax.map(area_for_pixel, sky_arr, batch_size=pixel_chunk)

    # Warm up PLRedNoise._fourier_basis_jax (a @cached_property) by calling g
    # once in eager mode. Without this, the FIRST autodiff call inside
    # area_for_pixel (quadratic_coeffs) caches a *tracer* in the property, which
    # then leaks into the SECOND autodiff call (jax.hessian) → UnexpectedTracerError.
    _ = g(gp, reduced_pp)

    _log(f"nside={nside} -> {npix} HEALPix pixels, SNR={snr_target}, "
         f"chunked vmap (batch={pixel_chunk}), compiling...")
    area_90, area_50 = all_areas(sky)
    loc_area_90 = np.asarray(area_90)
    loc_area_50 = np.asarray(area_50)
    finite_90 = loc_area_90[np.isfinite(loc_area_90)]
    finite_50 = loc_area_50[np.isfinite(loc_area_50)]
    _log(f"Done. 90% area min/median/max = "
         f"{finite_90.min():.3e}/{np.median(finite_90):.3e}/{finite_90.max():.3e} deg^2 "
         f"({len(finite_90)}/{npix} finite); n_anchors={n_anchors}")
    _log(f"      50% area min/median/max = "
         f"{finite_50.min():.3e}/{np.median(finite_50):.3e}/{finite_50.max():.3e} deg^2 "
         f"({len(finite_50)}/{npix} finite)")

    return {
        "loc_area_90_deg2": loc_area_90,
        "loc_area_50_deg2": loc_area_50,
        "nside": np.int64(nside),
        "snr_target": np.float64(snr_target),
        "log10_mc": np.float64(LOG10_MC),
        "log10_fgw": np.float64(LOG10_FGW),
        "cos_inc": np.float64(cos_inc_fix),
        "psi": np.float64(psi_fix),
        "phase0": np.float64(phase0_fix),
        "pulsar_names": np.array(names),
        "anchor_pulsars": np.array(sorted(anchor_set)) if anchor_set else np.array([], dtype="<U1"),
        "pulsar_term_mask": np.array(pulsar_term_mask, dtype=bool),
        "n_anchors": np.int64(n_anchors),
        "n_pulsars": np.int64(len(names)),
        "pulsar_pos": np.asarray(positions),
    }


def save_results(path: Path, results: dict) -> None:
    path = Path(path)
    np.savez_compressed(path, **results)
    print(f"Saved {path} ({path.stat().st_size / 1e3:.1f} kB).")


def load_results(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {k: (v.item() if v.ndim == 0 else v) for k, v in data.items()}


def plot_results(results: dict, output: str = "cgw_localization_skymap.png",
                 level: str = "90") -> None:
    """Mollview of the credible area at the given level ("90" or "50")."""
    hp = _import_healpy()
    import matplotlib.pyplot as plt

    # Back-compat with older single-area outputs.
    if level == "90" and "loc_area_90_deg2" in results:
        area = results["loc_area_90_deg2"]
    elif level == "50" and "loc_area_50_deg2" in results:
        area = results["loc_area_50_deg2"]
    elif "loc_area_deg2" in results:
        area = results["loc_area_deg2"]   # legacy
    else:
        raise KeyError(f"No loc_area_{level}_deg2 in results.")
    n_anchors = int(results["n_anchors"])
    n_psr = int(results["n_pulsars"])
    snr = float(results["snr_target"])

    # Log scale — area ranges over orders of magnitude across anchor configs.
    log_area = np.log10(np.where(np.isfinite(area) & (area > 0), area, np.nan))
    hp.mollview(
        log_area,
        title=(f"$\\log_{{10}}$ {level}% credible area [deg$^2$]  "
               f"($\\mathcal{{M}}=5\\times 10^8 M_\\odot$, $f=10^{{-8.4}}$ Hz, "
               f"SNR={snr:.0f})\n"
               f"{n_anchors}/{n_psr} anchor pulsars"),
        unit=f"$\\log_{{10}}$ {level}% area [deg$^2$]",
        cmap="magma_r",
        rot=[180, 0],
    )
    hp.graticule()
    if "pulsar_pos" in results:
        pos = np.atleast_2d(np.asarray(results["pulsar_pos"]))
        mask = np.atleast_1d(np.asarray(results["pulsar_term_mask"]))
        theta = np.arccos(np.clip(pos[:, 2], -1.0, 1.0))
        phi = np.arctan2(pos[:, 1], pos[:, 0])
        # Anchors: bright red stars; non-anchors: dimmer grey circles.
        if np.any(mask):
            hp.projscatter(theta[mask], phi[mask], marker="*", s=180, color="red",
                           edgecolors="black", linewidths=0.5, zorder=5,
                           label="anchor")
        if np.any(~mask):
            hp.projscatter(theta[~mask], phi[~mask], marker="o", s=40, color="0.5",
                           edgecolors="black", linewidths=0.4, zorder=4,
                           label="non-anchor")
    plt.savefig(output, dpi=130, bbox_inches="tight")
    print(f"Wrote {output}")
    plt.close()


def run_sweep(out_dir: Path, *, nside: int = 4,
              pulsar_subset: tuple[str, ...] | None = None,
              configs: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
              snr_target: float = SNR_TARGET_DEFAULT,
              pixel_chunk: int = 32) -> None:
    """Wen et al. 2026 Table 1 analog: per-config 90% / 50% credible areas.

    Defaults: ``WEN_OCARINA_18`` array, ``WEN_CONFIGS`` (Standard / 25-3 / 25-6).
    For each config: per-pixel Fisher → 90% and 50% area maps, summary stats,
    Mollview plots. The whole sweep emits ``wen_table1_analog.csv`` and a
    matching ``.txt`` with the per-config min/median/max for both levels.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = list(pulsar_subset) if pulsar_subset is not None else list(WEN_OCARINA_18)
    cfgs = configs if configs is not None else WEN_CONFIGS

    rows: list[dict] = []
    for name, anchors in cfgs:
        anchors_in_subset = tuple(a for a in anchors if a in subset)
        missing = tuple(a for a in anchors if a not in subset)
        if missing:
            _log(f"[warn] config {name!r}: dropping anchors not in subset: {missing}")
        out_path = out_dir / f"cgw_loc_{name}.npz"
        _log(f"\n=== Config: {name!r}, {len(anchors_in_subset)} anchors: "
             f"{list(anchors_in_subset)} ===")
        results = compute_localization_skymap(
            pulsar_subset=tuple(subset),
            anchor_pulsars=anchors_in_subset,
            nside=nside, snr_target=snr_target, pixel_chunk=pixel_chunk,
        )
        save_results(out_path, results)
        # Per-config Mollviews at both levels.
        plot_results(results, output=str(out_dir / f"cgw_loc_{name}_90pct.png"),
                     level="90")
        plot_results(results, output=str(out_dir / f"cgw_loc_{name}_50pct.png"),
                     level="50")

        a90 = results["loc_area_90_deg2"]
        a50 = results["loc_area_50_deg2"]
        f90 = a90[np.isfinite(a90)]
        f50 = a50[np.isfinite(a50)]
        rows.append({
            "config": name,
            "n_pulsars": int(results["n_pulsars"]),
            "n_anchors": len(anchors_in_subset),
            "anchors": " ".join(anchors_in_subset),
            "missing_anchors": " ".join(missing),
            "area_90_median": float(np.median(f90)),
            "area_90_min": float(np.min(f90)),
            "area_90_max": float(np.max(f90)),
            "area_90_p10": float(np.percentile(f90, 10)),
            "area_90_p90": float(np.percentile(f90, 90)),
            "n_pixels_finite_90": int(len(f90)),
            "area_50_median": float(np.median(f50)),
            "area_50_min": float(np.min(f50)),
            "area_50_max": float(np.max(f50)),
            "area_50_p10": float(np.percentile(f50, 10)),
            "area_50_p90": float(np.percentile(f50, 90)),
            "n_pixels_finite_50": int(len(f50)),
        })

    # ---- Table 1 analog: CSV + human-readable .txt ------------------------
    import csv
    csv_path = out_dir / "wen_table1_analog.csv"
    columns = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    txt_path = out_dir / "wen_table1_analog.txt"
    with open(txt_path, "w") as f:
        f.write("Wen et al. 2026 Table 1 analog\n")
        f.write(f"  Source: M_c = 5e8 Msun, f_GW = 10^-8.4 Hz, SNR = {snr_target}\n")
        f.write(f"  Array: {len(subset)}-pulsar ocarina subset\n")
        f.write(f"  Method: data-injection Fisher via two CWInjectors\n\n")
        f.write(f"{'config':>10} {'n_psr':>6} {'n_anc':>6} "
                f"{'90 median':>11} {'90 min':>11} {'90 max':>11} "
                f"{'50 median':>11} {'50 min':>11} {'50 max':>11}  anchors\n")
        f.write("-" * 110 + "\n")
        for r in rows:
            f.write(f"{r['config']:>10} {r['n_pulsars']:>6} {r['n_anchors']:>6} "
                    f"{r['area_90_median']:>11.3e} {r['area_90_min']:>11.3e} {r['area_90_max']:>11.3e} "
                    f"{r['area_50_median']:>11.3e} {r['area_50_min']:>11.3e} {r['area_50_max']:>11.3e}"
                    f"  {r['anchors']}\n")
            if r["missing_anchors"]:
                f.write(f"           [missing/substituted: {r['missing_anchors']}]\n")
    print(f"Wrote {txt_path}")
    # Print to stdout too.
    with open(txt_path) as f:
        print(f.read())

    # Scaling plot: 90% and 50% median areas vs. config.
    import matplotlib.pyplot as plt
    ks = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7, 4))
    for level, marker, color in [("90", "o", "C0"), ("50", "s", "C1")]:
        med = np.array([r[f"area_{level}_median"] for r in rows])
        lo = np.array([r[f"area_{level}_p10"] for r in rows])
        hi = np.array([r[f"area_{level}_p90"] for r in rows])
        ax.fill_between(ks, lo, hi, alpha=0.2, color=color)
        ax.plot(ks, med, marker=marker, lw=2, color=color,
                label=f"{level}% area (median)")
    ax.set_xticks(ks)
    ax.set_xticklabels([r["config"] for r in rows])
    ax.set_yscale("log")
    ax.set_xlabel("array configuration")
    ax.set_ylabel("credible localization area [deg$^2$]")
    ax.set_title("CGW localization area, Wen et al. 2026 Table 1 analog\n"
                 f"($\\mathcal{{M}}=5\\times 10^8 M_\\odot$, SNR={snr_target:.0f}, "
                 f"{len(subset)}-pulsar ocarina subset)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "wen_table1_scaling.png", dpi=130)
    plt.close(fig)
    print(f"Wrote {out_dir / 'wen_table1_scaling.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(mode="sweep")
    sub = p.add_subparsers(dest="mode")

    sp = sub.add_parser("generate")
    sp.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sp.add_argument("--nside", type=int, default=8)
    sp.add_argument("--pixel-chunk", type=int, default=32)
    sp.add_argument("--snr", type=float, default=SNR_TARGET_DEFAULT)
    sp.add_argument("--anchor-pulsars", nargs="*", default=[])
    sp.add_argument("--full", action="store_true",
                    help="Use the Wen-ocarina 18-pulsar subset (--full) instead "
                         "of the 4-pulsar SMOKE_SUBSET.")
    sp.add_argument("--validate-linearity", action="store_true")

    sp = sub.add_parser("plot")
    sp.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    sp.add_argument("--output", type=str, default="cgw_localization_skymap.png")
    sp.add_argument("--level", choices=("90", "50"), default="90",
                    help="Which credible level to plot from the npz.")

    sp = sub.add_parser("sweep")
    sp.add_argument("--out-dir", type=Path, default=Path("cgw_loc_sweep"))
    sp.add_argument("--nside", type=int, default=4)
    sp.add_argument("--pixel-chunk", type=int, default=32)
    sp.add_argument("--snr", type=float, default=SNR_TARGET_DEFAULT)

    args = p.parse_args()
    if args.mode == "plot":
        results = load_results(args.input)
        plot_results(results, output=args.output, level=args.level)
        return

    if args.mode == "sweep":
        run_sweep(args.out_dir, nside=args.nside, snr_target=args.snr,
                  pixel_chunk=args.pixel_chunk)
        return

    # generate: --full → 18-pulsar Wen-ocarina subset; else → 4-pulsar SMOKE_SUBSET.
    subset = WEN_OCARINA_18 if getattr(args, "full", False) else SMOKE_SUBSET
    results = compute_localization_skymap(
        pulsar_subset=subset, nside=args.nside,
        anchor_pulsars=tuple(args.anchor_pulsars),
        snr_target=args.snr, pixel_chunk=args.pixel_chunk,
        validate_linearity=args.validate_linearity,
    )
    save_results(args.output, results)


if __name__ == "__main__":
    main()
