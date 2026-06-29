"""K-source CGW sky-localization map vs. anchor-pulsar count.

Direct extension of :mod:`examples.cgw_localization_skymap` to K simultaneous
SMBHB sources.  Source 0 is scanned over a HEALPix grid; sources 1..K-1 are
held at canonical galaxy-cluster sky positions.  Per pixel, per source, the
output is the 90% and 50% credible localization area in deg^2.

Math (see ``/home/hector/.claude/plans/phase2-multi-source-localization-math.md``):
the joint Fisher is the ``2K x 2K`` matrix with structure

    F = [[h_a*h_b * Gram_ab]_{a,b=0}^{K-1}],

with ``K(K+1)/2`` unique Gram blocks extracted via the cross-derivative trick
from Level 1 (:func:`jaxpint.pta.cw_localization.gram_block_at_pair`).
Per-source 90% credible area falls out of inverting F, slicing the marginal
2x2 covariance per source, and applying the analytic Delta_chi^2 area formula.

This file uses **2K CWInjectors**: a template + data injector per source
(prefixes ``cw{k}t_`` and ``cw{k}d_``).  The two injectors for source ``a``
share parameters except for amplitude and sky, so the cross-derivative
extracts the proper Gram block at source ``a``'s position.  For off-diagonal
blocks the cross-derivative is between the templates of two different sources.

Usage
-----
    python examples/cgw_multi_source_localization.py generate \
        [--K K] [--nside N] [--output PATH] [--anchor-pulsars NAME ...]
    python examples/cgw_multi_source_localization.py sweep \
        [--K K] [--out-dir DIR] [--nside N]
    python examples/cgw_multi_source_localization.py plot \
        [--input PATH] [--source IDX] [--level 90|50]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Reuse loader / drop list / Wen subset + configs / fixed-source constants from Level 1.
from examples.cgw_localization_skymap import (
    DATA_DIR,
    DROP_PULSARS,
    SMOKE_SUBSET,
    WEN_OCARINA_18,
    WEN_CONFIGS,
    MARG_PARAMS,
    FIXED_ORIENTATION,
    LOG10_MC,
    LOG10_FGW,
    SNR_TARGET_DEFAULT,
    pulsar_unit_vector_icrs,
    _log,
    _import_healpy,
)


# ---------------------------------------------------------------------------
# Canonical fixed-source positions: Wen Table 2 galaxy cluster directions.
# Used as the K-1 held-fixed source positions when scanning source 0.
# (ICRS-ish: approximate; cos_gwtheta = sin(dec), gwphi = RA in radians.)
# ---------------------------------------------------------------------------
def _gal_cluster_dir(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    return (float(np.sin(np.deg2rad(dec_deg))), float(np.deg2rad(ra_deg)))


GALAXY_CLUSTER_DIRECTIONS: dict[str, tuple[float, float]] = {
    "Coma": _gal_cluster_dir(194.95, 27.98),
    "Fornax": _gal_cluster_dir(54.62, -35.45),
    "Hercules": _gal_cluster_dir(241.31, 17.73),
    "Norma": _gal_cluster_dir(243.55, -60.50),
    "Virgo": _gal_cluster_dir(187.70, 12.39),
}


DEFAULT_OUTPUT = Path("cgw_multi_source_localization.npz")


def compute_multi_source_localization_skymap(
    *,
    pulsar_subset=SMOKE_SUBSET,
    anchor_pulsars: tuple[str, ...] = (),
    fixed_source_skies: tuple[tuple[float, float], ...] = (),
    nside: int = 4,
    snr_target: float = SNR_TARGET_DEFAULT,
    orientation: tuple[float, float, float] = FIXED_ORIENTATION,
    pixel_chunk: int = 8,
    validate_linearity: bool = False,
):
    """Compute per-pixel per-source credible areas with K = 1 + len(fixed_source_skies).

    Source 0 is scanned over a HEALPix grid; sources 1..K-1 are held at the
    sky positions in ``fixed_source_skies``.  Same anchor mask applies to all
    K sources (the anchor set is a property of the pulsar array, not the
    sources).

    Returns
    -------
    results : dict
        ``areas_90_deg2`` and ``areas_50_deg2`` are ``(npix, K)`` arrays.
        Other keys carry metadata: K, fixed positions, pulsar mask, anchors,
        SNR, frequency, chirp mass, pulsar positions.
    """
    import jax
    import jax.numpy as jnp
    from loguru import logger

    hp = _import_healpy()
    logger.disable("pint")

    from jaxpint import load_nanograv_pta
    from jaxpint.pta.likelihood import PTAConfig
    from jaxpint.types import GlobalParams
    from jaxpint.bayes import ImproperPrior, marginalize_pta
    from jaxpint.pta.signals.cw import CWInjector
    from jaxpint.pta.cw_upper_limit import quadratic_coeffs
    from jaxpint.pta.cw_localization import (
        h0_for_snr,
        make_logL_2sky,
        gram_block_at_pair,
        assemble_joint_fisher,
        per_source_credible_areas_deg2,
    )

    K = 1 + len(fixed_source_skies)
    if K < 1:
        raise ValueError("K must be at least 1.")

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
    _log(
        f"K={K} sources. Anchor pulsars ({n_anchors}/{len(names)}): "
        f"{sorted(anchor_set) if anchor_set else '(none — all Earth-term-only)'}"
    )

    positions = jnp.asarray(np.stack([pulsar_unit_vector_icrs(pp) for pp in pp_list]))

    # ---- 3. 2K CWInjectors --------------------------------------------------
    # Prefix scheme: cw{k}t_ and cw{k}d_ for the template and data injectors of source k.
    injectors = []
    for k in range(K):
        for tag in ("t", "d"):
            injectors.append(
                CWInjector(
                    positions,
                    prefix=f"cw{k}{tag}_",
                    earth_term_only=False,
                    linear_amplitude=True,
                    pulsar_term_mask=pulsar_term_mask,
                    initial_values={"log10_fgw": LOG10_FGW},
                )
            )

    gp = GlobalParams.empty()
    for inj in injectors:
        gp = inj.register_params(gp)
    config = PTAConfig(
        toa_data_list=toa_list,
        timing_models=tm_list,
        noise_models=nm_list,
        signal_injectors=tuple(injectors),
    )

    # ---- 4. Marginalize timing params --------------------------------------
    over, priors = set(), {}
    for pn, pp in zip(names, pp_list):
        for nm in pp.free_names():
            if nm in MARG_PARAMS:
                fqn = f"{pn}_{nm}"
                over.add(fqn)
                priors[fqn] = ImproperPrior()
    _log(f"Marginalizing {len(over)} timing params across {len(names)} pulsars...")
    g, _, reduced_pp = marginalize_pta(over=over,
        priors=priors,
        config=config,
        pulsar_names=tuple(names),
        fiducial_pulsar_params=pp_list,
        fiducial_global_params=gp,
        validate_linearity=validate_linearity,
        allow_nonlinear=True,
    )

    # ---- 5. Base global params: orientation + fixed-source skies pinned -----
    cos_inc_fix, psi_fix, phase0_fix = (float(x) for x in orientation)

    def _prefix(k: int, tag: str) -> str:
        # Global-name prefix of source k's template ("t") / data ("d") injector,
        # i.e. its params are f"{_prefix(k, tag)}_{h0,cos_gwtheta,gwphi,...}".
        return f"cw{k}{tag}"

    # Orientation fixed on all 2K injectors; sources 1..K-1 pinned to their fixed
    # sky positions.  Amplitudes stay at gp's value (0); source 0's sky and the
    # active pair's amplitudes are set per pixel below.
    gp_base = gp
    for k in range(K):
        for tag in ("t", "d"):
            p = _prefix(k, tag)
            gp_base = (
                gp_base.with_value(f"{p}_cos_inc", cos_inc_fix)
                .with_value(f"{p}_psi", psi_fix)
                .with_value(f"{p}_phase0", phase0_fix)
            )
    for k in range(1, K):
        cgt, gphi = fixed_source_skies[k - 1]
        for tag in ("t", "d"):
            p = _prefix(k, tag)
            gp_base = (
                gp_base.with_value(f"{p}_cos_gwtheta", float(cgt))
                .with_value(f"{p}_gwphi", float(gphi))
            )

    # ---- 6. Closure factory: a logL pair with source 0's sky bound ---------
    # make_pair(a, tag_a, b, tag_b) -> (h_a, h_b, sky_a, sky_b) -> scalar, built
    # by make_logL_2sky: it activates exactly the chosen injector pair, leaving
    # every other injector at zero amplitude and its pinned sky.
    def make_logL_pair_factory(source_0_sky):
        gp_s0 = gp_base
        for tag in ("t", "d"):
            p = _prefix(0, tag)
            gp_s0 = (
                gp_s0.with_value(f"{p}_cos_gwtheta", source_0_sky[0])
                .with_value(f"{p}_gwphi", source_0_sky[1])
            )

        def make_pair(a_idx: int, tag_a: str, b_idx: int, tag_b: str):
            return make_logL_2sky(
                g, gp_s0, reduced_pp, _prefix(a_idx, tag_a), _prefix(b_idx, tag_b)
            )

        return make_pair

    # ---- 7. Per-pixel area function ----------------------------------------
    def areas_for_pixel(source_0_sky):
        # Build per-source truth positions (source 0 = the pixel; rest are fixed).
        source_skies = [source_0_sky]
        for cgt, gphi in fixed_source_skies:
            source_skies.append(jnp.array([cgt, gphi], dtype=jnp.float64))

        make_pair = make_logL_pair_factory(source_0_sky)

        # 7a. Per-source Y_a via quadratic_coeffs (only injector a-template active).
        Y_per_source = []
        for a in range(K):
            f_diag = make_pair(a, "t", a, "t")
            # Single-amplitude pattern: use sky_a = sky_b = source_skies[a], h_b=0.
            amp_logL = lambda h: f_diag(
                h, jnp.float64(0.0), source_skies[a], source_skies[a]
            )
            _X, Y = quadratic_coeffs(amp_logL)
            Y_per_source.append(Y)
        Y_per_source = jnp.stack(Y_per_source)
        h0_targets = jax.vmap(lambda y: h0_for_snr(jnp.float64(snr_target), y))(
            Y_per_source
        )

        # 7b. Gram blocks: K diagonals + K(K-1)/2 off-diagonals.
        gram_blocks = {}
        for a in range(K):
            f_diag = make_pair(a, "t", a, "d")
            G_aa = gram_block_at_pair(f_diag, source_skies[a], source_skies[a])
            gram_blocks[(a, a)] = G_aa
        for a in range(K):
            for b in range(a + 1, K):
                f_off = make_pair(a, "t", b, "t")
                G_ab = gram_block_at_pair(f_off, source_skies[a], source_skies[b])
                gram_blocks[(a, b)] = G_ab

        # 7c. Joint Fisher + per-source marginal areas.
        F = assemble_joint_fisher(gram_blocks, h0_targets, K)
        areas_90 = per_source_credible_areas_deg2(F, K, level=0.9)
        areas_50 = per_source_credible_areas_deg2(F, K, level=0.5)
        return areas_90, areas_50

    # ---- 8. Sky grid + jitted scan ----------------------------------------
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    sky = jnp.stack([jnp.cos(jnp.asarray(theta)), jnp.asarray(phi)], axis=1)

    @jax.jit
    def all_pixels(sky_arr):
        return jax.lax.map(areas_for_pixel, sky_arr, batch_size=pixel_chunk)

    # Warmup: same PLRedNoise cached_property pattern as Level 1.
    _ = g(gp, reduced_pp)

    _log(
        f"nside={nside} -> {npix} HEALPix pixels, K={K}, SNR={snr_target}; "
        f"K(K+1)/2={K * (K + 1) // 2} gram blocks per pixel; chunked vmap "
        f"(batch={pixel_chunk}), compiling..."
    )
    areas_90, areas_50 = all_pixels(sky)
    areas_90_np = np.asarray(areas_90)
    areas_50_np = np.asarray(areas_50)

    # Per-source diagnostics
    for k in range(K):
        a = areas_90_np[:, k]
        finite = a[np.isfinite(a)]
        n_fin = len(finite)
        if n_fin > 0:
            _log(
                f"  source {k}: 90% area finite {n_fin}/{npix} "
                f"(min/med/max = {finite.min():.3e}/{np.median(finite):.3e}/"
                f"{finite.max():.3e} deg^2)"
            )
        else:
            _log(f"  source {k}: 90% area finite 0/{npix} — all degenerate")

    return {
        "areas_90_deg2": areas_90_np,  # (npix, K)
        "areas_50_deg2": areas_50_np,  # (npix, K)
        "nside": np.int64(nside),
        "K": np.int64(K),
        "snr_target": np.float64(snr_target),
        "log10_mc": np.float64(LOG10_MC),
        "log10_fgw": np.float64(LOG10_FGW),
        "cos_inc": np.float64(cos_inc_fix),
        "psi": np.float64(psi_fix),
        "phase0": np.float64(phase0_fix),
        "fixed_source_skies": np.asarray(fixed_source_skies, dtype=np.float64),
        "pulsar_names": np.array(names),
        "anchor_pulsars": np.array(sorted(anchor_set))
        if anchor_set
        else np.array([], dtype="<U1"),
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


def run_multi_source_sweep(
    out_dir: Path,
    *,
    K: int = 2,
    fixed_source_names: tuple[str, ...] = ("Coma",),
    pulsar_subset: tuple[str, ...] | None = None,
    nside: int = 4,
    snr_target: float = SNR_TARGET_DEFAULT,
    pixel_chunk: int = 8,
) -> None:
    """Phase 2 Wen-Table-1-multi-source-analog sweep across the three Wen configs.

    For each of (standard / 25-3 / 25-6), runs
    :func:`compute_multi_source_localization_skymap` with K sources (source 0
    scanned, sources 1..K-1 held at ``fixed_source_names`` galaxy clusters).
    Outputs:

    * Per-config ``cgw_loc_multi_K{K}_{config}.npz`` with the full (npix, K)
      area arrays + metadata.
    * Per-config per-source Mollview at 90% and 50% levels.
    * ``wen_table1_multi_source_analog.csv`` and ``.txt`` with per-source
      median / min / max area across the sky per config.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(fixed_source_names) != K - 1:
        raise ValueError(
            f"Need K-1 = {K - 1} fixed-source names; got {len(fixed_source_names)}."
        )
    fixed = tuple(GALAXY_CLUSTER_DIRECTIONS[n] for n in fixed_source_names)

    subset = list(pulsar_subset) if pulsar_subset is not None else list(WEN_OCARINA_18)

    rows: list[dict] = []
    for name, anchors in WEN_CONFIGS:
        anchors_in_subset = tuple(a for a in anchors if a in subset)
        missing = tuple(a for a in anchors if a not in subset)
        if missing:
            _log(f"[warn] config {name!r}: dropping anchors not in subset: {missing}")

        out_path = out_dir / f"cgw_loc_multi_K{K}_{name}.npz"
        _log(
            f"\n=== Config: {name!r}, {len(anchors_in_subset)} anchors: "
            f"{list(anchors_in_subset)} ==="
        )
        results = compute_multi_source_localization_skymap(
            pulsar_subset=tuple(subset),
            anchor_pulsars=anchors_in_subset,
            fixed_source_skies=fixed,
            nside=nside,
            snr_target=snr_target,
            pixel_chunk=pixel_chunk,
        )
        save_results(out_path, results)

        a90 = results["areas_90_deg2"]  # (npix, K)
        a50 = results["areas_50_deg2"]
        row = {
            "config": name,
            "anchors": " ".join(anchors_in_subset),
            "K": K,
            "fixed_sources": " ".join(fixed_source_names),
            "missing_anchors": " ".join(missing),
        }
        for k in range(K):
            f90 = a90[:, k][np.isfinite(a90[:, k])]
            f50 = a50[:, k][np.isfinite(a50[:, k])]
            label = "src0" if k == 0 else fixed_source_names[k - 1]
            row[f"{label}_n_finite_90"] = int(len(f90))
            row[f"{label}_90_median"] = (
                float(np.median(f90)) if len(f90) else float("inf")
            )
            row[f"{label}_90_min"] = float(np.min(f90)) if len(f90) else float("inf")
            row[f"{label}_90_max"] = float(np.max(f90)) if len(f90) else float("inf")
            row[f"{label}_50_median"] = (
                float(np.median(f50)) if len(f50) else float("inf")
            )
        rows.append(row)

    # CSV
    import csv

    csv_path = out_dir / "wen_table1_multi_source_analog.csv"
    columns = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # Human-readable txt
    txt_path = out_dir / "wen_table1_multi_source_analog.txt"
    with open(txt_path, "w") as f:
        f.write(f"Wen et al. 2026 Table 1 multi-source analog (K={K})\n")
        f.write(f"  Source: M_c = 5e8 Msun, f_GW = 10^-8.4 Hz, SNR = {snr_target}\n")
        f.write(f"  Array: {len(subset)}-pulsar ocarina subset\n")
        f.write(f"  Source 0: scanned over HEALPix nside={nside}\n")
        f.write(f"  Sources 1..K-1: {fixed_source_names}\n")
        f.write(
            f"  Method: cross-derivative Gram joint Fisher via {2 * K} CWInjectors\n\n"
        )
        for r in rows:
            f.write(
                f"\n  Config {r['config']!r} (anchors: {r['anchors'] or '(none)'})\n"
            )
            for k in range(K):
                label = "src0(scan)" if k == 0 else fixed_source_names[k - 1]
                f.write(
                    f"    {label:>18}  90% med {r[f'{label.split("(")[0]}_90_median']:.3e}  "
                    f"min {r[f'{label.split("(")[0]}_90_min']:.3e}  "
                    f"max {r[f'{label.split("(")[0]}_90_max']:.3e}  "
                    f"(finite {r[f'{label.split("(")[0]}_n_finite_90']})\n"
                )
    print(f"Wrote {txt_path}")
    with open(txt_path) as f:
        print(f.read())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(mode="generate")
    sub = p.add_subparsers(dest="mode")

    sp = sub.add_parser("generate")
    sp.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sp.add_argument("--nside", type=int, default=4)
    sp.add_argument("--pixel-chunk", type=int, default=8)
    sp.add_argument("--snr", type=float, default=SNR_TARGET_DEFAULT)
    sp.add_argument("--anchor-pulsars", nargs="*", default=[])
    sp.add_argument(
        "--K", type=int, default=2, help="Number of CGW sources (default 2)."
    )
    sp.add_argument(
        "--fixed-source-names",
        nargs="*",
        default=["Coma"],
        help="Galaxy-cluster names from GALAXY_CLUSTER_DIRECTIONS "
        "to use as held-fixed source positions for sources 1..K-1.",
    )
    sp.add_argument(
        "--full",
        action="store_true",
        help="Use WEN_OCARINA_18 instead of SMOKE_SUBSET.",
    )
    sp.add_argument("--validate-linearity", action="store_true")

    # Sweep subparser (mirrors Level 1's run_sweep flow but for multi-source).
    sp = sub.add_parser("sweep")
    sp.add_argument("--out-dir", type=Path, default=Path("cgw_loc_multi_sweep"))
    sp.add_argument("--K", type=int, default=2)
    sp.add_argument("--fixed-source-names", nargs="*", default=["Coma"])
    sp.add_argument("--nside", type=int, default=4)
    sp.add_argument("--pixel-chunk", type=int, default=8)
    sp.add_argument("--snr", type=float, default=SNR_TARGET_DEFAULT)

    args = p.parse_args()

    if args.mode == "sweep":
        if args.K != 1 + len(args.fixed_source_names):
            raise SystemExit(
                f"--K={args.K} but --fixed-source-names has "
                f"{len(args.fixed_source_names)} entries; need K-1."
            )
        run_multi_source_sweep(
            args.out_dir,
            K=args.K,
            fixed_source_names=tuple(args.fixed_source_names),
            nside=args.nside,
            snr_target=args.snr,
            pixel_chunk=args.pixel_chunk,
        )
        return

    # generate
    if args.K != 1 + len(args.fixed_source_names):
        raise SystemExit(
            f"--K={args.K} but --fixed-source-names has {len(args.fixed_source_names)} "
            f"entries; need K-1 = {args.K - 1} fixed-source names."
        )
    fixed = tuple(GALAXY_CLUSTER_DIRECTIONS[n] for n in args.fixed_source_names)

    subset = WEN_OCARINA_18 if args.full else SMOKE_SUBSET
    results = compute_multi_source_localization_skymap(
        pulsar_subset=subset,
        nside=args.nside,
        anchor_pulsars=tuple(args.anchor_pulsars),
        fixed_source_skies=fixed,
        snr_target=args.snr,
        pixel_chunk=args.pixel_chunk,
        validate_linearity=args.validate_linearity,
    )
    save_results(args.output, results)


if __name__ == "__main__":
    main()
