"""Multi-source CW localization: per-source conditioned credible-area maps.

Injects ``S`` CW sources into the (synthetic) data and localizes **each** of them
with the conditioned scan -- scanning one source over the sky while the other
``S-1`` are baked at their truth positions
(:func:`jaxpint.pta.incoherent_ul.condition_on_statics`).  For every source it emits
two distance-marginalized credible-region maps (Tier-1 reductions over a
:func:`~jaxpint.pta.incoherent_ul.mixed_phase_A` phase grid):

* **unconditioned** -- the naive single-source matched filter ``b0`` (which sees
  *all* the signals), and
* **conditioned** -- ``b_eff = b0 - G0s @ a_static`` with the other sources baked out.

The naive map is confidently *wrong* (it locks onto a confusion/alias peak); the
conditioned map recovers each source's true sky.  The headline metric is the
**argmax offset from that source's truth**, per source.

Note: these are *conditional* credible areas -- each source given the others at fixed
positions.  The fully-marginal areas (integrating the other sources' position
uncertainty) are broader and need iterated conditioning or joint sampling.

Usage::

    python examples/cgw_conditioned_localization.py generate --data-dir DIR \\
        [--nside N] [--source-pix 64,128,...] [--source-snr 10,12,...] \\
        [--full] [--output PATH]
    python examples/cgw_conditioned_localization.py plot [--input PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ---- self-contained example config (no cross-example imports) --------------
F_GW = 27e-9  # 27 nHz
LOG10_FGW = float(np.log10(F_GW))
PI2 = float(np.pi / 2.0)
MARG_PARAMS = {
    "F0",
    "F1",
    "RAJ",
    "DECJ",
    "ELONG",
    "ELAT",
    "PMRA",
    "PMDEC",
    "PMELONG",
    "PMELAT",
}
SMOKE_SUBSET = ["J1909-3744", "J1713+0747", "J0613-0200", "J1744-1134"]
DROP_PULSARS = {
    "B1937+21ao",
    "B1937+21gbt",
    "J1600-3053gbt",
    "J1643-1224gbt",
    "J1713+0747ao",
    "J1713+0747gbt",
    "J1903+0327ao",
    "J1909-3744gbt",
}


def _log(msg):
    print(msg, flush=True)


def _import_healpy():
    try:
        import healpy as hp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "This example needs healpy (HEALPix grid + Mollweide plots). Install "
            "the optional extra:  uv pip install 'jaxpint[skymap]'."
        ) from e
    return hp


def compute_conditioned_localization(
    *,
    data_dir,
    pulsar_subset=SMOKE_SUBSET,
    nside=8,
    source_pix=None,
    source_snr=None,
    k=5.0,
    n_phase=64,
    pixel_chunk=8,
):
    """Inject ``S`` CW sources and localize each one (conditioned on the others).

    Parameters
    ----------
    data_dir : path-like
        NANOGrav-style dataset directory (par/tim), loaded by ``load_nanograv_pta``.
    pulsar_subset : list[str] or None
        Pulsar names to load; ``None`` loads all in ``data_dir``.
    nside : int
        HEALPix resolution (``npix = 12 * nside**2``).
    source_pix : list[int] or None
        HEALPix truth pixels of the ``S`` sources.  ``None`` defaults to two sources
        at ``npix // 3`` and ``2 * npix // 3``.
    source_snr : list[float] or None
        Target network matched-filter SNR per source (calibrated via
        :func:`h0_for_snr`).  ``None`` defaults to ``[10, 12, ...]``.  Must match
        ``len(source_pix)``.
    k, n_phase, pixel_chunk : phase-grid and batching knobs (see the module).

    Returns
    -------
    dict
        ``marginal_unconditioned`` / ``marginal_conditioned`` : (S, npix) maps, one
        row per source; ``offset_*_deg``, ``A68_*_deg2``, ``level_at_truth_*`` :
        (S,) per-source metrics (offset of the argmax from that source's truth is the
        headline); plus ``source_pix`` (S,), ``source_snr`` (S,), ``h0`` (S,),
        ``nside``, ``pulsar_pos``/``pulsar_names``, and ``n_phase``/``k``.
    """
    import jax
    import jax.numpy as jnp
    from loguru import logger

    from jaxpint import load_nanograv_pta
    from jaxpint.bayes import marginalize_single_pulsar, ImproperPrior
    from jaxpint.bayes.credible import credible_level_map, credible_region_area
    from jaxpint.pta.signals.cw import cw_delay_from_array, CWInjector, _KPC_TO_M, _C
    from jaxpint.pta.incoherent_ul import (
        extract_pulsar_bM,
        extract_pulsar_blocks,
        condition_on_statics,
        mixed_phase_A,
        total_logL_marg,
    )
    from jaxpint.pta.cw_localization import h0_for_snr
    from jaxpint.types import GlobalParams
    from jaxpint.utils import pulsar_unit_vector

    hp = _import_healpy()
    logger.disable("pint")

    def cw_params(ct, gp):  # face-on, unit h0
        return jnp.array([1.0, ct, gp, LOG10_FGW, 1.0, 0.0, 0.0])

    # ---- load + sky grid + sources ---------------------------------------
    psrs = load_nanograv_pta(data_dir, pulsar_names=pulsar_subset)
    keep = [i for i, n in enumerate(psrs.pulsar_names) if n not in DROP_PULSARS]
    names = [psrs.pulsar_names[i] for i in keep]
    td_list = [psrs.toa_data_list[i] for i in keep]
    tm_list = [psrs.timing_models[i] for i in keep]
    nm_list = [psrs.noise_models[i] for i in keep]
    pp_list = [psrs.pulsar_params_list[i] for i in keep]
    positions = np.stack([np.asarray(pulsar_unit_vector(pp)) for pp in pp_list])
    px = np.array([float(pp.param_value("PX")) for pp in pp_list])
    pos_j = jnp.asarray(positions)
    npsr = len(names)

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    cos_gwtheta = jnp.asarray(np.cos(theta))
    gwphi = jnp.asarray(phi)
    if source_pix is None:
        source_pix = [npix // 3, 2 * npix // 3]
    if source_snr is None:
        source_snr = [10.0 + 2.0 * i for i in range(len(source_pix))]
    if len(source_snr) != len(source_pix):
        raise ValueError("source_snr and source_pix must have the same length.")
    source_pix = [int(p) for p in source_pix]
    S = len(source_pix)
    skies = [(float(np.cos(theta[p])), float(phi[p])) for p in source_pix]

    def truth_coeff(
        ct, gp
    ):  # per-pulsar A(Δ_truth) = (1−cosΔ, sinΔ) at this sky/distance
        sin = np.sqrt(max(1.0 - ct**2, 0.0))
        om = np.array([-sin * np.cos(gp), -sin * np.sin(gp), -ct])
        cmu = positions @ om
        L = 1.0 / np.maximum(px, 1e-3)
        D = 2 * np.pi * F_GW * (L * _KPC_TO_M) * (1.0 + cmu) / _C
        return jnp.asarray(np.stack([1 - np.cos(D), np.sin(D)], axis=1))  # (npsr, 2)

    a_truth = [truth_coeff(*sk) for sk in skies]
    _log(f"Loaded {npsr} pulsars; {S} sources at pix {source_pix}.")

    # ---- per-pulsar likelihoods + per-source SNR calibration -------------
    gs = []
    for td, tm, nm, pp in zip(td_list, tm_list, nm_list, pp_list):
        over = {n for n in pp.free_names() if n in MARG_PARAMS}
        gs.append(
            marginalize_single_pulsar(
                over=over,
                priors={n: ImproperPrior() for n in over},
                toa_data=td,
                timing_model=tm,
                noise_model=nm,
                fiducial_params=pp,
                allow_nonlinear=True,
                validate_linearity=False,
            )
        )

    def templates(td, pos, ct, gp):
        cw = cw_params(ct, gp)
        e = cw_delay_from_array(
            td, pos, 1.0, cw, linear_amplitude=True, earth_term_only=True
        )
        ps = cw_delay_from_array(
            td,
            pos,
            1.0,
            cw,
            linear_amplitude=True,
            pulsar_term_only=True,
            pulsar_term_phase=PI2,
        )
        return e, ps

    def calibrate(
        ct, gp, ai_truth, snr
    ):  # M at truth (injection-free) -> h0 for target SNR
        snr2 = 0.0
        for (g, _, skel), td, pos, ai in zip(gs, td_list, pos_j, ai_truth):
            e, ps = templates(td, pos, ct, gp)
            _, M = extract_pulsar_bM(g, skel, e, ps)
            snr2 += float(ai @ M @ ai)
        return float(h0_for_snr(snr, snr2))

    h0 = [calibrate(*skies[s], a_truth[s], source_snr[s]) for s in range(S)]
    a_coeff = [
        h0[s] * a_truth[s] for s in range(S)
    ]  # per-source baked coeff, (npsr, 2)
    _log("Injecting h0 = " + ", ".join(f"{h:.3e}" for h in h0) + ".")

    # ---- inject all S sources; precompute per-pulsar g_inj + truth templates
    injs = []
    for s in range(S):
        inj = CWInjector(
            pos_j,
            prefix=f"cw{s}_",
            linear_amplitude=True,
            initial_values={
                "cos_gwtheta": skies[s][0],
                "gwphi": skies[s][1],
                "log10_fgw": LOG10_FGW,
                "cos_inc": 1.0,
                "psi": 0.0,
                "phase0": 0.0,
            },
        )
        gp_inj = inj.register_params(GlobalParams.empty()).with_value(
            f"cw{s}_h0", h0[s]
        )
        injs.append((inj, gp_inj))

    def make_ginj(g, s_inject):
        def g_inj(rp, external_delay=0.0):
            return g(rp, external_delay=external_delay + s_inject)

        return g_inj

    pp_data = []
    for a, ((g, _, skel), td, pp, pos) in enumerate(zip(gs, td_list, pp_list, pos_j)):
        s_total = sum(injs[s][0].delay(a, td, pp, injs[s][1]) for s in range(S))
        truth_tmpls = [
            templates(td, pos, *skies[s]) for s in range(S)
        ]  # (e_s, ps_s) at truth
        pp_data.append((make_ginj(g, -s_total), skel, td, pos, truth_tmpls))

    # ---- per scanned source: extract, condition, reduce both maps --------
    is_tight = jnp.zeros(npsr, bool)
    L0 = jnp.asarray(1.0 / np.maximum(px, 1e-3))
    sin_th = np.sin(theta)
    omhat = np.stack(
        [-sin_th * np.cos(phi), -sin_th * np.sin(phi), -np.cos(theta)], axis=1
    )
    cos_mu_all = jnp.asarray(omhat @ positions.T)

    def reduce_map(b_all, G_all, h0_eval):
        b_pix = jnp.transpose(jnp.asarray(b_all), (1, 0, 2))
        G_pix = jnp.transpose(jnp.asarray(G_all), (1, 0, 2, 3))

        def at(bp, Gp, cmp):
            A = mixed_phase_A(is_tight, L0, 1e-3, k, cmp, F_GW, n_phase)
            return total_logL_marg(jnp.asarray(h0_eval), bp, Gp, A)

        return np.asarray(
            jax.lax.map(
                lambda x: at(x[0], x[1], x[2]),
                (b_pix, G_pix, cos_mu_all),
                batch_size=pixel_chunk,
            )
        )

    maps_unc = np.empty((S, npix))
    maps_con = np.empty((S, npix))
    for j in range(S):
        others = [s for s in range(S) if s != j]
        b_unc = np.empty((npsr, npix, 2))
        G_arr = np.empty((npsr, npix, 2, 2))
        b_con = np.empty((npsr, npix, 2))
        for a, (g_inj, skel, td, pos, truth_tmpls) in enumerate(pp_data):
            fixed_list = [
                t for s in others for t in truth_tmpls[s]
            ]  # [e_s, ps_s, ...] fixed
            a_static = (
                jnp.concatenate([a_coeff[s][a] for s in others])
                if others
                else jnp.zeros(0)
            )

            def blocks_at(
                ct,
                gp,
                td=td,
                pos=pos,
                g_inj=g_inj,
                skel=skel,
                fixed_list=fixed_list,
                a_static=a_static,
            ):
                e_j, ps_j = templates(
                    td, pos, ct, gp
                )  # scanned source, varies per pixel
                basis = jnp.stack(
                    [e_j, ps_j] + fixed_list
                )  # scanned first, then statics
                b, G = extract_pulsar_blocks(g_inj, skel, basis)
                b_eff, _ = condition_on_statics(b, G, a_static, n_scan=2)
                return b[:2], G[:2, :2], b_eff

            out = jax.lax.map(
                lambda row: blocks_at(row[0], row[1]),
                jnp.stack([cos_gwtheta, gwphi], axis=1),
                batch_size=pixel_chunk,
            )
            b_unc[a], G_arr[a], b_con[a] = (
                np.asarray(out[0]),
                np.asarray(out[1]),
                np.asarray(out[2]),
            )
        maps_unc[j] = reduce_map(b_unc, G_arr, h0[j])
        maps_con[j] = reduce_map(b_con, G_arr, h0[j])
        _log(f"  source {j} (pix {source_pix[j]}) localized.")

    # ---- per-source metrics ----------------------------------------------
    pa_deg2 = hp.nside2pixarea(nside, degrees=True)

    def metrics(m, truth_pix):
        ip = int(np.argmax(m))
        n = hp.ang2vec(*hp.pix2ang(nside, ip))
        n0 = hp.ang2vec(*hp.pix2ang(nside, truth_pix))
        off = float(np.degrees(np.arccos(np.clip(np.dot(n, n0), -1, 1))))
        a68 = float(credible_region_area(jnp.asarray(m), pa_deg2, 0.68))
        lvl = float(np.asarray(credible_level_map(jnp.asarray(m)))[truth_pix])
        return off, a68, lvl

    mu = np.array([metrics(maps_unc[j], source_pix[j]) for j in range(S)])
    mc = np.array([metrics(maps_con[j], source_pix[j]) for j in range(S)])
    for j in range(S):
        _log(
            f"  source {j}: uncond offset {mu[j, 0]:5.1f} deg (level@truth {mu[j, 2]:.2f}) "
            f"-> cond offset {mc[j, 0]:5.1f} deg (level@truth {mc[j, 2]:.2f}), "
            f"A68 {mc[j, 1]:.0f} deg^2"
        )

    return {
        "marginal_unconditioned": maps_unc,
        "marginal_conditioned": maps_con,
        "nside": np.int64(nside),
        "source_pix": np.array(source_pix, dtype=np.int64),
        "source_snr": np.array(source_snr, dtype=np.float64),
        "h0": np.array(h0),
        "pulsar_pos": positions,
        "pulsar_names": np.array(names),
        "offset_unconditioned_deg": mu[:, 0],
        "offset_conditioned_deg": mc[:, 0],
        "A68_unconditioned_deg2": mu[:, 1],
        "A68_conditioned_deg2": mc[:, 1],
        "level_at_truth_unconditioned": mu[:, 2],
        "level_at_truth_conditioned": mc[:, 2],
        "n_phase": np.int64(n_phase),
        "k": np.float64(k),
    }


def save_results(path, results):
    """Write a :func:`compute_conditioned_localization` results dict to ``.npz``."""
    np.savez_compressed(path, **results)
    _log(f"Saved -> {path}")


def plot_results(path, outdir=".", which="conditioned"):
    """Grid of per-source HPD credible-level maps (one panel per source).

    Each panel marks its own source's truth (red star) and the other sources (cyan
    diamonds); pulsars are white dots.  ``which`` selects ``conditioned`` (default)
    or ``unconditioned`` maps.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    from jaxpint.bayes.credible import credible_level_map

    hp = _import_healpy()
    d = np.load(path, allow_pickle=False)
    nside = int(d["nside"])
    maps = np.asarray(d[f"marginal_{which}"])
    source_pix = np.asarray(d["source_pix"])
    S = len(source_pix)
    ang = [hp.pix2ang(nside, int(p)) for p in source_pix]
    pt, pp = hp.vec2ang(np.asarray(d["pulsar_pos"]))

    ncols = min(S, 3)
    nrows = int(np.ceil(S / ncols))
    fig = plt.figure(figsize=(6.5 * ncols, 4.2 * nrows))
    for j in range(S):
        lvl = np.asarray(credible_level_map(jnp.asarray(maps[j])))
        off = float(d[f"offset_{which}_deg"][j])
        a68 = float(d[f"A68_{which}_deg2"][j])
        hp.mollview(
            lvl,
            fig=fig.number,
            sub=(nrows, ncols, j + 1),
            rot=[180, 0],
            cmap="viridis_r",
            min=0.0,
            max=1.0,
            title=f"source {j} ({which})\noffset={off:.1f} deg, A68={a68:.0f} deg^2",
        )
        for s in range(S):
            mk = ("*", 220, "red") if s == j else ("D", 60, "cyan")
            hp.projscatter(
                ang[s][0], ang[s][1], marker=mk[0], s=mk[1], color=mk[2], edgecolors="k"
            )
        hp.projscatter(pt, pp, marker="o", s=16, color="white", edgecolors="k")
        hp.graticule()
    out = Path(outdir) / f"cgw_conditioned_localization_{which}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    _log(f"Plot -> {out}")


def _int_list(s):
    return [int(x) for x in s.split(",")]


def _float_list(s):
    return [float(x) for x in s.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--output", type=Path, default=Path("cgw_conditioned.npz"))
    g.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="NANOGrav-style dataset directory (par/ + tim/), e.g. the synthetic "
        "'ocarina_2' set; loaded by load_nanograv_pta.",
    )
    g.add_argument("--nside", type=int, default=8)
    g.add_argument(
        "--source-pix",
        type=_int_list,
        default=None,
        help="comma-separated HEALPix truth pixels, e.g. 64,128,200",
    )
    g.add_argument(
        "--source-snr",
        type=_float_list,
        default=None,
        help="comma-separated target SNR per source, e.g. 10,12,11",
    )
    g.add_argument("--n-phase", type=int, default=64)
    g.add_argument("--k", type=float, default=5.0)
    g.add_argument("--pixel-chunk", type=int, default=8)
    g.add_argument(
        "--full", action="store_true", help="all pulsars (else SMOKE_SUBSET)"
    )
    pl = sub.add_parser("plot")
    pl.add_argument("--input", type=Path, default=Path("cgw_conditioned.npz"))
    pl.add_argument("--outdir", type=Path, default=Path("."))
    pl.add_argument(
        "--which", choices=["conditioned", "unconditioned"], default="conditioned"
    )
    args = p.parse_args()

    if args.cmd == "generate":
        res = compute_conditioned_localization(
            data_dir=args.data_dir,
            pulsar_subset=None if args.full else SMOKE_SUBSET,
            nside=args.nside,
            source_pix=args.source_pix,
            source_snr=args.source_snr,
            k=args.k,
            n_phase=args.n_phase,
            pixel_chunk=args.pixel_chunk,
        )
        save_results(args.output, res)
    else:
        plot_results(args.input, outdir=args.outdir, which=args.which)


if __name__ == "__main__":
    main()
