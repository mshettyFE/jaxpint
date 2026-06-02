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
        Keys: ``loc_area_deg2`` (HEALPix map, RING ordering), ``nside``,
        ``snr_target``, ``pulsar_names``, ``anchor_pulsars``,
        ``pulsar_term_mask``, ``log10_mc``, ``log10_fgw``, orientation,
        ``pulsar_pos`` (ICRS unit vectors for overlay plotting).
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
    from jaxpint.pta.cw_localization import h0_for_snr, credible_area_deg2

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

    # ---- 3. Injector + config + global params ------------------------------
    # Linear-amplitude template so logL is quadratic in h0 and quadratic_coeffs
    # gives us Y for SNR calibration cheaply. earth_term_only=False at the
    # injector level lets each pulsar decide via pulsar_term_mask.
    injector = CWInjector(
        positions, prefix="cw_",
        earth_term_only=False, linear_amplitude=True,
        pulsar_term_mask=pulsar_term_mask,
        initial_values={"log10_fgw": LOG10_FGW},
    )
    gp = injector.register_params(GlobalParams.empty())
    config = PTAConfig(
        toa_data_list=toa_list, timing_models=tm_list,
        noise_models=nm_list, signal_injectors=(injector,),
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

    # ---- 5. logL closures over (h0, sky), orientation fixed at truth -------
    cos_inc_fix, psi_fix, phase0_fix = (float(x) for x in orientation)
    idx = {k: gp._name_to_index[f"cw_{k}"] for k in
           ("h0", "cos_gwtheta", "gwphi", "cos_inc", "psi", "phase0")}
    base_vals = gp.values.at[idx["cos_inc"]].set(cos_inc_fix)\
                         .at[idx["psi"]].set(psi_fix)\
                         .at[idx["phase0"]].set(phase0_fix)

    def logL_at_h0_sky(h0, cos_gwtheta, gwphi):
        v = (base_vals
             .at[idx["h0"]].set(h0)
             .at[idx["cos_gwtheta"]].set(cos_gwtheta)
             .at[idx["gwphi"]].set(gwphi))
        gp_new = eqx.tree_at(lambda gg: gg.values, gp, v)
        return g(gp_new, reduced_pp)

    # ---- 6. Pixel loop: Y → h0(SNR=target) → Fisher → area -----------------
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    sky = jnp.stack([jnp.cos(jnp.asarray(theta)), jnp.asarray(phi)], axis=1)  # (npix, 2)

    # Expected Fisher in noise-only ocarina data: F = 0.5 * h0² * Hessian_sky(Y).
    # Y(sky) is the noise-weighted unit-strain signal power — data-INDEPENDENT.
    # The observed -Hessian(logL) on null data also contains a noise term
    # h0*∂²X/∂sky² (zero in expectation but non-zero per realization), which
    # can flip eigenvalue signs at SNR-20 magnitudes. Computing Hessian of Y
    # directly bypasses that noise.
    def Y_of_sky(sky_vec):
        amp_logL = lambda amp: logL_at_h0_sky(amp, sky_vec[0], sky_vec[1])
        _X, Y = quadratic_coeffs(amp_logL)
        return Y

    def area_for_pixel(sky_row):
        Y_pix = Y_of_sky(sky_row)
        h0 = h0_for_snr(jnp.float64(snr_target), Y_pix)
        H_Y = jax.hessian(Y_of_sky)(sky_row)
        F = 0.5 * h0**2 * H_Y
        return credible_area_deg2(F)

    @jax.jit
    def all_areas(sky_arr):
        return jax.lax.map(area_for_pixel, sky_arr, batch_size=pixel_chunk)

    # Warm up PLRedNoise._fourier_basis_jax (a @cached_property) by calling g
    # once in eager mode. Without this, the FIRST autodiff call inside
    # area_for_pixel (quadratic_coeffs) caches a *tracer* in the property, which
    # then leaks into the SECOND autodiff call (jax.hessian) → UnexpectedTracerError.
    # Eager call here stores a concrete jnp.ndarray in the cache instead.
    _ = g(gp, reduced_pp)

    _log(f"nside={nside} -> {npix} HEALPix pixels, SNR={snr_target}, "
         f"chunked vmap (batch={pixel_chunk}), compiling...")
    loc_area = np.asarray(all_areas(sky))
    finite = loc_area[np.isfinite(loc_area)]
    _log(f"Done. 90% area min/median/max = "
         f"{finite.min():.3e}/{np.median(finite):.3e}/{finite.max():.3e} deg^2 "
         f"({len(finite)}/{npix} finite); n_anchors={n_anchors}")

    return {
        "loc_area_deg2": loc_area,
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


def plot_results(results: dict, output: str = "cgw_localization_skymap.png") -> None:
    hp = _import_healpy()
    import matplotlib.pyplot as plt

    area = results["loc_area_deg2"]
    n_anchors = int(results["n_anchors"])
    n_psr = int(results["n_pulsars"])
    snr = float(results["snr_target"])

    # Log scale — area ranges over orders of magnitude across anchor configs.
    log_area = np.log10(np.where(np.isfinite(area) & (area > 0), area, np.nan))
    hp.mollview(
        log_area,
        title=(f"$\\log_{{10}}$ 90% credible area [deg$^2$]  "
               f"($\\mathcal{{M}}=5\\times 10^8 M_\\odot$, $f=10^{{-8.4}}$ Hz, "
               f"SNR={snr:.0f})\n"
               f"{n_anchors}/{n_psr} anchor pulsars"),
        unit="$\\log_{10}$ 90% area [deg$^2$]",
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
              snr_target: float = SNR_TARGET_DEFAULT,
              pixel_chunk: int = 32) -> None:
    """Built-in anchor-count sweep + 1-D scaling plot.

    On the 4-pulsar SMOKE_SUBSET, sweeps over: 0 anchors, then anchors picked
    in this order: J1909-3744, J1713+0747, J0613-0200, J1744-1134. Produces per-
    configuration skymaps and a final scaling plot of 90% area (median + min/max
    across the sky) vs. anchor count.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = list(pulsar_subset) if pulsar_subset is not None else list(SMOKE_SUBSET)
    # Anchor-add ordering: highest-SNR PTA workhorses first.
    add_order = ["J1909-3744", "J1713+0747", "J0613-0200", "J1744-1134"]
    add_order = [p for p in add_order if p in subset]

    sweep = []
    cumulative_anchors: list[str] = []
    for k in range(len(add_order) + 1):
        cumulative_anchors = list(add_order[:k])
        tag = f"{k:02d}anchors"
        out_path = out_dir / f"cgw_loc_{tag}.npz"
        _log(f"\n=== Config: {k} anchors {cumulative_anchors!r} ===")
        results = compute_localization_skymap(
            pulsar_subset=tuple(subset),
            anchor_pulsars=tuple(cumulative_anchors),
            nside=nside, snr_target=snr_target, pixel_chunk=pixel_chunk,
        )
        save_results(out_path, results)
        plot_results(results, output=str(out_dir / f"cgw_loc_{tag}.png"))
        finite = results["loc_area_deg2"][np.isfinite(results["loc_area_deg2"])]
        sweep.append({
            "n_anchors": k,
            "anchors": list(cumulative_anchors),
            "median_deg2": float(np.median(finite)),
            "min_deg2": float(np.min(finite)),
            "max_deg2": float(np.max(finite)),
            "p10_deg2": float(np.percentile(finite, 10)),
            "p90_deg2": float(np.percentile(finite, 90)),
        })

    # 1-D scaling plot
    import matplotlib.pyplot as plt
    ks = np.array([s["n_anchors"] for s in sweep])
    med = np.array([s["median_deg2"] for s in sweep])
    p10 = np.array([s["p10_deg2"] for s in sweep])
    p90 = np.array([s["p90_deg2"] for s in sweep])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(ks, p10, p90, alpha=0.25, label="10–90% across sky")
    ax.plot(ks, med, marker="o", lw=2, label="median across sky")
    ax.set_yscale("log")
    ax.set_xlabel("# anchor pulsars")
    ax.set_ylabel("90% credible localization area [deg$^2$]")
    ax.set_title(f"CGW localization vs. anchor count "
                 f"($\\mathcal{{M}}=5\\times 10^8 M_\\odot$, SNR={snr_target:.0f}, "
                 f"{len(subset)}-pulsar ocarina subset)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "anchor_scaling.png", dpi=130)
    plt.close(fig)
    print(f"Wrote {out_dir / 'anchor_scaling.png'}")

    # Save sweep summary
    np.savez_compressed(
        out_dir / "sweep_summary.npz",
        n_anchors=np.array([s["n_anchors"] for s in sweep]),
        median_deg2=np.array([s["median_deg2"] for s in sweep]),
        min_deg2=np.array([s["min_deg2"] for s in sweep]),
        max_deg2=np.array([s["max_deg2"] for s in sweep]),
        p10_deg2=np.array([s["p10_deg2"] for s in sweep]),
        p90_deg2=np.array([s["p90_deg2"] for s in sweep]),
        anchor_lists=np.array([" ".join(s["anchors"]) for s in sweep]),
    )
    print(f"Wrote {out_dir / 'sweep_summary.npz'}")


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
                    help="Use all pulsars instead of the smoke subset.")
    sp.add_argument("--validate-linearity", action="store_true")

    sp = sub.add_parser("plot")
    sp.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    sp.add_argument("--output", type=str, default="cgw_localization_skymap.png")

    sp = sub.add_parser("sweep")
    sp.add_argument("--out-dir", type=Path, default=Path("cgw_loc_sweep"))
    sp.add_argument("--nside", type=int, default=4)
    sp.add_argument("--pixel-chunk", type=int, default=32)
    sp.add_argument("--snr", type=float, default=SNR_TARGET_DEFAULT)

    args = p.parse_args()
    if args.mode == "plot":
        results = load_results(args.input)
        plot_results(results, output=args.output)
        return

    if args.mode == "sweep":
        run_sweep(args.out_dir, nside=args.nside, snr_target=args.snr,
                  pixel_chunk=args.pixel_chunk)
        return

    # generate
    subset = None if getattr(args, "full", False) else SMOKE_SUBSET
    results = compute_localization_skymap(
        pulsar_subset=subset, nside=args.nside,
        anchor_pulsars=tuple(args.anchor_pulsars),
        snr_target=args.snr, pixel_chunk=args.pixel_chunk,
        validate_linearity=args.validate_linearity,
    )
    save_results(args.output, results)


if __name__ == "__main__":
    main()
