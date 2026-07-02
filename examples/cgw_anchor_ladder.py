"""Anchor ladder: tight pulsar-distance priors collapse the CW pulsar-term alias.

A single CW source's distance-marginalized localization map is multimodal -- the
pulsar-term phase, marginalized over an unknown (flat) distance, forms a fringe comb,
and an alias can beat the true peak (the S=1 failure mode).  Giving a subset of
pulsars a *tight* distance prior makes their pulsar term **coherent**, adding angular
information that collapses the comb.

This driver injects one source, extracts the per-pulsar ``(b, M)`` over a HEALPix grid
**once**, then sweeps the number of *anchored* pulsars (``is_tight`` in
:func:`jaxpint.pta.incoherent_ul.mixed_phase_A`, with ``sigma_L`` sub-fringe) -- a
pure reduction sweep, anchoring **loudest-first** by per-pulsar SNR^2
``a_i^T M_i a_i`` (SkyScan's ranking) -- and reports the argmax offset from truth, the 68% HPD area,
and the credible level at truth as a function of ``n_anchors``.  The area/offset fall
as more pulsars are anchored: the value of tight pulsar distances for CW localization.

Anchoring is idealized here (each anchor's prior is a narrow ``sigma_L`` around the
pulsar's *true* distance ``1/PX``); it quantifies the reach if such distances were
available (VLBI-class), not a claim about current parallaxes.

Usage::

    python examples/cgw_anchor_ladder.py generate --data-dir DIR [--nside N] \\
        [--truth-pix P] [--snr S] [--sigma-L-pc X] [--full] [--output PATH]
    python examples/cgw_anchor_ladder.py plot [--input PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jaxpint.notebook_utils import (
    SMOKE_SUBSET,
    healpix_grid,
    import_healpy,
    load_filtered_pta,
    load_npz_results,
    log_flush as _log,
    marginalize_each_pulsar,
    overlay_pulsars,
    save_npz_results,
)

# ---- script-specific config -------------------------------------------------
F_GW = 27e-9  # 27 nHz
LOG10_FGW = float(np.log10(F_GW))
PI2 = float(np.pi / 2.0)


def compute_anchor_ladder(
    *,
    data_dir,
    pulsar_subset=SMOKE_SUBSET,
    nside=8,
    truth_pix=None,
    snr=12.0,
    sigma_L_pc=0.01,
    k=5.0,
    n_phase=64,
    pixel_chunk=8,
):
    """Inject one CW source and localize it while anchoring 0..n_psr pulsars.

    Parameters
    ----------
    data_dir : path-like
        NANOGrav-style dataset directory (par/tim), loaded by ``load_nanograv_pta``.
    pulsar_subset : list[str] or None
        Pulsar names to load; ``None`` loads all in ``data_dir``.
    nside : int
        HEALPix resolution (``npix = 12 * nside**2``).
    truth_pix : int or None
        HEALPix truth pixel of the injected source.  ``None`` -> ``2 * npix // 3``.
    snr : float
        Target network matched-filter SNR (calibrated via :func:`h0_for_snr`).
    sigma_L_pc : float
        Anchored-pulsar distance-prior width (pc), around the true distance ``1/PX``.
        Must be sub-fringe (~<=0.1 pc at 27 nHz) for the anchor to engage; otherwise
        :func:`mixed_phase_A` falls back to the flat (incoherent) grid.
    k, n_phase, pixel_chunk : phase-grid and batching knobs.

    Returns
    -------
    dict
        ``n_anchors`` : (n_psr+1,) int; ``offset_deg`` / ``A68_deg2`` /
        ``level_at_truth`` : (n_psr+1,) per-``n_anchors`` metrics; ``maps`` :
        (n_psr+1, npix) the map at each ``n_anchors``; ``anchor_order`` (n_psr,) the
        loudest-first pulsar indices, ``snr2_per_pulsar`` (n_psr,) their SNR^2; plus
        ``truth_pix``, ``nside``, ``pulsar_pos``/``pulsar_names``,
        ``snr``/``h0``/``sigma_L_pc``, ``n_phase``/``k``.
    """
    import jax
    import jax.numpy as jnp
    from loguru import logger

    from jaxpint.bayes.credible import credible_level_map, credible_region_area
    from jaxpint.pta.signals.cw import cw_delay_from_array, CWInjector, _KPC_TO_M, _C
    from jaxpint.pta.incoherent_ul import (
        extract_pulsar_bM,
        mixed_phase_A,
        total_logL_marg,
    )
    from jaxpint.pta.cw_localization import h0_for_snr
    from jaxpint.types import GlobalParams

    hp = import_healpy()
    logger.disable("pint")
    sigma_L_kpc = sigma_L_pc * 1e-3

    def cw_params(ct, gp):  # face-on, unit h0
        return jnp.array([1.0, ct, gp, LOG10_FGW, 1.0, 0.0, 0.0])

    # ---- load + sky grid + truth -----------------------------------------
    pta = load_filtered_pta(data_dir, pulsar_names=pulsar_subset)
    names = list(pta.names)
    td_list = list(pta.toa_data_list)
    pp_list = list(pta.pulsar_params_list)
    positions = pta.positions
    px = np.array([float(pp.param_value("PX")) for pp in pp_list])
    pos_j = jnp.asarray(positions)
    npsr = len(names)

    grid = healpix_grid(nside)
    npix, theta, phi = grid.npix, grid.theta, grid.phi
    if truth_pix is None:
        truth_pix = 2 * npix // 3
    ct_t, gp_t = float(np.cos(theta[truth_pix])), float(phi[truth_pix])

    sin_t = np.sqrt(max(1.0 - ct_t**2, 0.0))
    om_t = np.array([-sin_t * np.cos(gp_t), -sin_t * np.sin(gp_t), -ct_t])
    cmu_t = positions @ om_t
    L_t = 1.0 / np.maximum(px, 1e-3)
    D_t = 2 * np.pi * F_GW * (L_t * _KPC_TO_M) * (1.0 + cmu_t) / _C
    a_truth = jnp.asarray(np.stack([1 - np.cos(D_t), np.sin(D_t)], axis=1))
    _log(f"Loaded {npsr} pulsars; truth pix {truth_pix}; sigma_L={sigma_L_pc} pc.")

    # ---- per-pulsar likelihoods + SNR calibration + injection ------------
    gs = marginalize_each_pulsar(pta)

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

    snr2_i = np.empty(
        npsr
    )  # per-pulsar SNR^2 = a_i^T M_i a_i (the anchor-ranking metric)
    for a, ((g, _, skel), td, pos, ai) in enumerate(zip(gs, td_list, pos_j, a_truth)):
        e, ps = templates(td, pos, ct_t, gp_t)
        _, M = extract_pulsar_bM(g, skel, e, ps)
        snr2_i[a] = float(ai @ M @ ai)
    h0 = float(h0_for_snr(snr, float(snr2_i.sum())))
    # anchor loudest-first: the pulsars whose pulsar term carries the most coherent
    # signal power (SkyScan run_anchor_localization's ranking).
    order = np.argsort(-snr2_i)
    _log(
        f"Injecting h0={h0:.3e}; anchor order (loudest-first) "
        f"{[names[i] for i in order]}."
    )
    inj = CWInjector(
        pos_j,
        linear_amplitude=True,
        initial_values={
            "cos_gwtheta": ct_t,
            "gwphi": gp_t,
            "log10_fgw": LOG10_FGW,
            "cos_inc": 1.0,
            "psi": 0.0,
            "phase0": 0.0,
        },
    )
    gp_inj = inj.register_params(GlobalParams.empty()).with_value("cw0_h0", h0)

    # ---- extract (b, M) over the sky ONCE (anchoring is reduction-only) ---
    b_all = np.empty((npsr, npix, 2))
    M_all = np.empty((npsr, npix, 2, 2))
    for a, ((g, _, skel), td, pp, pos) in enumerate(zip(gs, td_list, pp_list, pos_j)):
        s0 = inj.delay(a, td, pp, gp_inj)

        def g_inj(rp, external_delay=0.0, _g=g, _si=-s0):
            return _g(rp, external_delay=external_delay + _si)

        def bM_at(ct, gp, td=td, pos=pos, g_inj=g_inj, skel=skel):
            e, ps = templates(td, pos, ct, gp)
            return extract_pulsar_bM(g_inj, skel, e, ps)

        out = jax.lax.map(
            lambda row: bM_at(row[0], row[1]),
            grid.sky,
            batch_size=pixel_chunk,
        )
        b_all[a] = np.asarray(out[0])
        M_all[a] = np.asarray(out[1])
        _log(f"  [{a + 1}/{npsr}] {names[a]} extracted.")

    # ---- reduction sweep over n_anchors ----------------------------------
    L0 = jnp.asarray(L_t)
    cos_mu_all = jnp.asarray(grid.omhat @ positions.T)
    b_pix = jnp.transpose(jnp.asarray(b_all), (1, 0, 2))
    M_pix = jnp.transpose(jnp.asarray(M_all), (1, 0, 2, 3))
    pa_deg2 = hp.nside2pixarea(nside, degrees=True)
    n0 = hp.ang2vec(*hp.pix2ang(nside, truth_pix))

    def reduce_map(is_tight):
        def at(bp, Mp, cmp):
            A = mixed_phase_A(is_tight, L0, sigma_L_kpc, k, cmp, F_GW, n_phase)
            return total_logL_marg(jnp.asarray(h0), bp, Mp, A)

        return np.asarray(
            jax.lax.map(
                lambda x: at(x[0], x[1], x[2]),
                (b_pix, M_pix, cos_mu_all),
                batch_size=pixel_chunk,
            )
        )

    n_anchors = np.arange(npsr + 1)
    maps = np.empty((npsr + 1, npix))
    offset = np.empty(npsr + 1)
    a68 = np.empty(npsr + 1)
    lvl = np.empty(npsr + 1)
    for na in n_anchors:
        anchored = set(order[:na].tolist())  # the na loudest pulsars
        is_tight = jnp.array([i in anchored for i in range(npsr)])
        m = reduce_map(is_tight)
        maps[na] = m
        ip = int(np.argmax(m))
        n = hp.ang2vec(*hp.pix2ang(nside, ip))
        offset[na] = float(np.degrees(np.arccos(np.clip(np.dot(n, n0), -1, 1))))
        a68[na] = float(credible_region_area(jnp.asarray(m), pa_deg2, 0.68))
        lvl[na] = float(np.asarray(credible_level_map(jnp.asarray(m)))[truth_pix])
        _log(
            f"  n_anchors={na}: offset={offset[na]:5.1f} deg, A68={a68[na]:5.0f} deg^2, "
            f"level@truth={lvl[na]:.2f}"
        )

    return {
        "n_anchors": n_anchors.astype(np.int64),
        "offset_deg": offset,
        "A68_deg2": a68,
        "level_at_truth": lvl,
        "maps": maps,
        "truth_pix": np.int64(truth_pix),
        "nside": np.int64(nside),
        "pulsar_pos": positions,
        "pulsar_names": np.array(names),
        "anchor_order": order.astype(np.int64),
        "snr2_per_pulsar": snr2_i,
        "snr": np.float64(snr),
        "h0": np.float64(h0),
        "sigma_L_pc": np.float64(sigma_L_pc),
        "n_phase": np.int64(n_phase),
        "k": np.float64(k),
    }


def plot_results(path, outdir="."):
    """Plot the anchor ladder: A68 & offset vs n_anchors, plus a Mollweide row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import jax.numpy as jnp
    from jaxpint.bayes.credible import credible_level_map

    hp = import_healpy()
    d = load_npz_results(path)
    na = np.asarray(d["n_anchors"])
    nside = int(d["nside"])
    truth_pix = int(d["truth_pix"])
    maps = np.asarray(d["maps"])

    # --- ladder curves ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(na, np.asarray(d["A68_deg2"]), "o-", color="C0")
    ax[0].set_xlabel("n_anchors")
    ax[0].set_ylabel("A68 (deg^2)")
    ax[0].set_yscale("log")
    ax[0].set_title(f"68% credible area (sigma_L={float(d['sigma_L_pc'])} pc)")
    ax[1].plot(na, np.asarray(d["offset_deg"]), "s-", color="C3")
    ax[1].set_xlabel("n_anchors")
    ax[1].set_ylabel("argmax offset from truth (deg)")
    ax[1].set_title("localization offset")
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    out_curves = Path(outdir) / "cgw_anchor_ladder_curves.png"
    fig.savefig(out_curves, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _log(f"Plot -> {out_curves}")

    # --- Mollweide row for a few representative n_anchors ---
    tt, tp = hp.pix2ang(nside, truth_pix)
    pulsar_pos = np.asarray(d["pulsar_pos"])
    sel = sorted(set([int(na[0]), int(na[len(na) // 2]), int(na[-1])]))
    figm = plt.figure(figsize=(6.2 * len(sel), 4.2))
    for i, nv in enumerate(sel):
        lvl = np.asarray(credible_level_map(jnp.asarray(maps[nv])))
        hp.mollview(
            lvl,
            fig=figm.number,
            sub=(1, len(sel), i + 1),
            rot=[180, 0],
            cmap="viridis_r",
            min=0.0,
            max=1.0,
            title=f"n_anchors={nv}\noffset={float(d['offset_deg'][nv]):.1f} deg",
        )
        hp.projscatter(tt, tp, marker="*", s=200, color="red")
        overlay_pulsars(
            pulsar_pos,
            anchor_mask=np.zeros(len(pulsar_pos), dtype=bool),
            dot_kwargs={"s": 16},
        )
        hp.graticule()
    out_maps = Path(outdir) / "cgw_anchor_ladder_maps.png"
    figm.savefig(out_maps, dpi=130, bbox_inches="tight")
    plt.close(figm)
    _log(f"Plot -> {out_maps}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--output", type=Path, default=Path("cgw_anchor_ladder.npz"))
    g.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="NANOGrav-style dataset directory (par/ + tim/), e.g. the synthetic "
        "'ocarina_2' set; loaded by load_nanograv_pta.",
    )
    g.add_argument("--nside", type=int, default=8)
    g.add_argument("--truth-pix", type=int, default=None)
    g.add_argument("--snr", type=float, default=12.0)
    g.add_argument("--sigma-L-pc", type=float, default=0.01)
    g.add_argument("--n-phase", type=int, default=64)
    g.add_argument("--k", type=float, default=5.0)
    g.add_argument("--pixel-chunk", type=int, default=8)
    g.add_argument(
        "--full", action="store_true", help="all pulsars (else SMOKE_SUBSET)"
    )
    pl = sub.add_parser("plot")
    pl.add_argument("--input", type=Path, default=Path("cgw_anchor_ladder.npz"))
    pl.add_argument("--outdir", type=Path, default=Path("."))
    args = p.parse_args()

    if args.cmd == "generate":
        res = compute_anchor_ladder(
            data_dir=args.data_dir,
            pulsar_subset=None if args.full else SMOKE_SUBSET,
            nside=args.nside,
            truth_pix=args.truth_pix,
            snr=args.snr,
            sigma_L_pc=args.sigma_L_pc,
            k=args.k,
            n_phase=args.n_phase,
            pixel_chunk=args.pixel_chunk,
        )
        save_npz_results(args.output, res)
    else:
        plot_results(args.input, outdir=args.outdir)


if __name__ == "__main__":
    main()
