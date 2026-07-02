"""Map-based CW sky LOCALIZATION: the distance-marginalized credible-region map.

Injects a coherent CW source (built with :class:`CWInjector`) at a chosen sky pixel
into the loaded data, then localizes it with the *marginal* map.  The *profile* map
(:func:`total_logL_profile`, maximize the nuisance) is emitted too, as the
frequentist alias-exposing diagnostic -- never the headline credible area.

Injection is baked into each pulsar's likelihood by a baseline
``external_delay = -h0 * s_truth`` (so the matched filter sees the signal); ``h0``
is calibrated to a target network matched-filter SNR.  Run on any NANOGrav-style
dataset via ``--data-dir`` (e.g. the synthetic ``ocarina_2`` set).

Usage::

    python examples/cgw_localization_marginal.py generate --data-dir DIR [--nside N] \\
        [--truth-pix P] [--snr S] [--n-phase N] [--k X] [--pixel-chunk N] \\
        [--full] [--output PATH]
    python examples/cgw_localization_marginal.py plot [--input PATH] \\
        [--which marginal|profile] [--outdir DIR]
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
FIXED_ORIENTATION = (1.0, 0.0, 0.0)  # cos_inc, psi, phase0 (face-on)


def compute_localization(
    *,
    data_dir,
    pulsar_subset=SMOKE_SUBSET,
    nside=8,
    truth_pix=None,
    snr=12.0,
    k=5.0,
    n_phase=64,
    pixel_chunk=8,
):
    """Inject a coherent CW at ``truth_pix`` and localize it with the marginal map.

    Builds the per-pulsar matched filter / Gram ``(b, M)`` over a HEALPix sky grid
    (real, timing-marginalized GLS via :func:`extract_pulsar_bM`), with a coherent
    CW injected at ``truth_pix`` and calibrated to a target network SNR, then reduces
    over the pulsar-term phase grid into the distance-marginalized (Bayesian) and
    profile (frequentist) log-likelihood maps and their HPD credible areas.

    Parameters
    ----------
    data_dir : path-like
        NANOGrav-style dataset directory (par/tim), loaded by ``load_nanograv_pta``.
    pulsar_subset : list[str] or None
        Pulsar names to load; ``None`` loads all in ``data_dir``.  Default
        ``SMOKE_SUBSET`` (four well-timed pulsars).
    nside : int
        HEALPix resolution; the sky grid has ``npix = 12 * nside**2`` pixels.
    truth_pix : int or None
        HEALPix pixel index of the injected source's true sky position.  ``None``
        picks a mid-sky default (``npix // 3``).
    snr : float
        Target network matched-filter SNR; the injected strain ``h0`` is calibrated
        (via :func:`h0_for_snr`) so the optimal SNR at ``truth_pix`` equals it.
    k : float
        Half-width of each pulsar's distance grid in units of ``sigma_L`` for the
        tight (coherent) branch of :func:`mixed_phase_A`.  All pulsars are flat-phase
        here, so this only sets the grid extent, not the result.
    n_phase : int
        Number of pulsar-term phase grid points per pulsar (the marginalization grid).
    pixel_chunk : int
        Batch size for the per-pixel ``jax.lax.map`` (memory/throughput knob).

    Returns
    -------
    dict
        ``marginal`` : (npix,) float array
            Distance-marginalized log-likelihood sky map -- the credible-region
            deliverable (HPD via :func:`credible_level_map`).
        ``profile`` : (npix,) float array
            Profile (max-over-phase) log-likelihood map -- the frequentist,
            alias-prone diagnostic.
        ``A68_deg2``, ``A95_deg2`` : float
            68% / 95% HPD credible-region areas of the marginal map, in deg^2.
        ``offset_deg`` : float
            Angular offset of the marginal-map argmax from the truth pixel (deg).
        ``level_at_truth`` : float
            HPD credible level at the truth pixel (0 = most probable; small = well
            localized).
        ``nside``, ``truth_pix`` : int
            HEALPix resolution and the injected true pixel.
        ``pulsar_pos`` : (n_psr, 3) float array
            Per-pulsar ICRS unit vectors (for overlay plotting).
        ``pulsar_names`` : (n_psr,) str array
        ``snr`` : float
            Target network SNR (input echo).
        ``h0`` : float
            Calibrated injected strain.
        ``n_phase``, ``k`` : phase-grid configuration (input echo).
    """
    import jax
    import jax.numpy as jnp
    from loguru import logger

    from jaxpint.bayes.credible import credible_level_map, credible_region_area
    from jaxpint.pta.signals.cw import cw_delay_from_array, CWInjector
    from jaxpint.pta.incoherent_ul import (
        extract_pulsar_bM,
        mixed_phase_A,
        total_logL_marg,
        total_logL_profile,
    )
    from jaxpint.pta.cw_localization import h0_for_snr
    from jaxpint.types import GlobalParams
    from jaxpint.pta.signals.cw import _KPC_TO_M, _C

    hp = import_healpy()
    logger.disable("pint")
    cos_inc, psi, phase0 = (float(x) for x in FIXED_ORIENTATION)

    def cw_params(ct, gp):
        return jnp.array([1.0, ct, gp, LOG10_FGW, cos_inc, psi, phase0])

    # ---- load + sky grid + truth -----------------------------------------
    pta = load_filtered_pta(data_dir, pulsar_names=pulsar_subset)
    names = list(pta.names)
    td_list = list(pta.toa_data_list)
    pp_list = list(pta.pulsar_params_list)
    positions = pta.positions
    px = np.array([float(pp.param_value("PX")) for pp in pp_list])  # parallax (mas)

    grid = healpix_grid(nside)
    npix, theta, phi = grid.npix, grid.theta, grid.phi
    if truth_pix is None:
        truth_pix = npix // 3
    ct_t, gp_t = float(np.cos(theta[truth_pix])), float(phi[truth_pix])
    # per-pulsar truth pulsar-term phase (for the coherent injection / SNR calibration)
    sin_t = np.sqrt(max(1.0 - ct_t**2, 0.0))
    omhat_t = np.array([-sin_t * np.cos(gp_t), -sin_t * np.sin(gp_t), -ct_t])
    cos_mu_t = positions @ omhat_t
    L_t = 1.0 / np.maximum(px, 1e-3)  # truth distance (kpc)
    Delta_t = 2 * np.pi * F_GW * (L_t * _KPC_TO_M) * (1.0 + cos_mu_t) / _C
    a_t = jnp.asarray(
        np.stack([1 - np.cos(Delta_t), np.sin(Delta_t)], axis=1)
    )  # (n_psr,2)
    _log(
        f"Loaded {len(names)} pulsars; truth pix {truth_pix} "
        f"(cos_gwtheta={ct_t:.3f}, gwphi={gp_t:.3f}); target SNR {snr}."
    )

    # ---- per-pulsar marginalized likelihoods (once) ----------------------
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
            pulsar_term_phase=float(np.pi / 2),
        )
        return e, ps

    # ---- calibrate h0 to the target network SNR (M at truth, injection-free) --
    snr2_unit = 0.0
    pos_j = jnp.asarray(positions)
    for (g, _, skel), td, pos, ai in zip(gs, td_list, pos_j, a_t):
        e, ps = templates(td, pos, ct_t, gp_t)
        _, M = extract_pulsar_bM(g, skel, e, ps)
        snr2_unit += float(ai @ M @ ai)  # a^T M a at h0 = 1
    h0 = float(h0_for_snr(snr, snr2_unit))  # snr / sqrt(signal power)
    _log(f"SNR^2 at h0=1 is {snr2_unit:.3e} -> injecting h0 = {h0:.3e}.")

    # ---- full-sky (b, M) with the coherent injection baked into g ---------
    # The truth source is built with CWInjector (pulsar-term distance from each
    # pulsar's PX); injecting it is a baseline external_delay = -h0 * s_truth so the
    # matched filter b sees the signal.
    inj = CWInjector(
        pos_j,
        linear_amplitude=True,
        initial_values={
            "cos_gwtheta": ct_t,
            "gwphi": gp_t,
            "log10_fgw": LOG10_FGW,
            "cos_inc": cos_inc,
            "psi": psi,
            "phase0": phase0,
        },
    )
    gp_inj = inj.register_params(GlobalParams.empty()).with_value("cw0_h0", 1.0)

    b_all = np.empty((len(names), npix, 2))
    M_all = np.empty((len(names), npix, 2, 2))
    for a, ((g, _, skel), td, pp, pos) in enumerate(zip(gs, td_list, pp_list, pos_j)):
        s_truth = inj.delay(a, td, pp, gp_inj)  # full coherent, h0=1, PX distance

        def g_inj(rp, external_delay=0.0, _g=g, _si=-h0 * s_truth):
            return _g(rp, external_delay=external_delay + _si)

        def bM_at(ct, gp, td=td, pos=pos, g_inj=g_inj, skel=skel):
            e, ps = templates(td, pos, ct, gp)
            return extract_pulsar_bM(g_inj, skel, e, ps)

        bM = jax.lax.map(
            lambda row: bM_at(row[0], row[1]),
            grid.sky,
            batch_size=pixel_chunk,
        )
        b_all[a] = np.asarray(bM[0])
        M_all[a] = np.asarray(bM[1])
        _log(f"  [{a + 1}/{len(names)}] {names[a]} extracted.")

    # ---- per-pixel marginal/profile maps ---------------------------------
    is_tight = jnp.zeros(len(names), bool)
    L0 = jnp.asarray(L_t)
    cos_mu_all = jnp.asarray(grid.omhat @ positions.T)  # (npix, n_psr)
    b_pix = jnp.transpose(jnp.asarray(b_all), (1, 0, 2))
    M_pix = jnp.transpose(jnp.asarray(M_all), (1, 0, 2, 3))

    def maps_at(b_px, M_px, cos_mu_px):
        A = mixed_phase_A(is_tight, L0, 1e-3, k, cos_mu_px, F_GW, n_phase)
        h0j = jnp.asarray(h0)
        return total_logL_marg(h0j, b_px, M_px, A), total_logL_profile(
            h0j, b_px, M_px, A
        )

    marg, prof = jax.lax.map(
        lambda x: maps_at(x[0], x[1], x[2]),
        (b_pix, M_pix, cos_mu_all),
        batch_size=pixel_chunk,
    )
    marg = np.asarray(marg)
    prof = np.asarray(prof)

    # ---- metrics ---------------------------------------------------------
    pa_deg2 = hp.nside2pixarea(nside, degrees=True)  # pixel area in deg^2
    n_truth = hp.ang2vec(*hp.pix2ang(nside, truth_pix))

    def offset_deg(m):
        n = hp.ang2vec(*hp.pix2ang(nside, int(np.argmax(m))))
        return float(np.degrees(np.arccos(np.clip(np.dot(n, n_truth), -1, 1))))

    a68 = float(credible_region_area(jnp.asarray(marg), pa_deg2, 0.68))
    a95 = float(credible_region_area(jnp.asarray(marg), pa_deg2, 0.95))
    lvl_truth = float(np.asarray(credible_level_map(jnp.asarray(marg)))[truth_pix])
    _log(
        f"Done. marginal: offset {offset_deg(marg):.1f} deg, A68={a68:.0f}, "
        f"A95={a95:.0f} deg^2, level_at_truth={lvl_truth:.3f}."
    )

    return {
        "marginal": marg,
        "profile": prof,
        "nside": np.int64(nside),
        "truth_pix": np.int64(truth_pix),
        "pulsar_pos": positions,
        "pulsar_names": np.array(names),
        "snr": np.float64(snr),
        "h0": np.float64(h0),
        "n_phase": np.int64(n_phase),
        "k": np.float64(k),
        "A68_deg2": np.float64(a68),
        "A95_deg2": np.float64(a95),
        "offset_deg": np.float64(offset_deg(marg)),
        "level_at_truth": np.float64(lvl_truth),
    }


def plot_results(path, which="marginal", outdir="."):
    """Render a Mollweide of the HPD credible-level map from a saved ``.npz``.

    Parameters
    ----------
    path : path-like
        ``.npz`` written by :func:`jaxpint.notebook_utils.save_npz_results`.
    which : {"marginal", "profile"}
        Which map to plot (``marginal`` is the credible-region deliverable;
        ``profile`` the diagnostic).
    outdir : path-like
        Directory for the output PNG ``cgw_localization_{which}.png``.

    Notes
    -----
    Colours are HPD credible levels (0 = most probable), with the truth pixel
    marked (red star) and pulsars overlaid (white dots).  Writes a PNG; returns
    ``None``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hp = import_healpy()
    from jaxpint.bayes.credible import credible_level_map
    import jax.numpy as jnp

    d = load_npz_results(path)
    nside = int(d["nside"])
    m = np.asarray(d[which])
    lvl = np.asarray(credible_level_map(jnp.asarray(m)))  # 0 = most probable
    tt, tp = hp.pix2ang(nside, int(d["truth_pix"]))
    hp.mollview(
        lvl,
        rot=[180, 0],
        cmap="viridis_r",
        min=0.0,
        max=1.0,
        title=f"{which} HPD credible level (A68={float(d['A68_deg2']):.0f} deg^2, "
        f"offset={float(d['offset_deg']):.1f} deg)",
    )
    hp.projscatter(tt, tp, marker="*", s=200, color="red", label="truth")
    pos = np.asarray(d["pulsar_pos"])
    overlay_pulsars(pos, anchor_mask=np.zeros(len(pos), dtype=bool))
    hp.graticule()
    out = Path(outdir) / f"cgw_localization_{which}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    _log(f"Plot -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--output", type=Path, default=Path("cgw_localization.npz"))
    g.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="NANOGrav-style dataset directory (par/ + tim/ subdirs, or par+tim "
        "files), e.g. the synthetic 'ocarina_2' set; loaded by load_nanograv_pta.",
    )
    g.add_argument("--nside", type=int, default=8)
    g.add_argument("--snr", type=float, default=12.0)
    g.add_argument("--truth-pix", type=int, default=None)
    g.add_argument("--n-phase", type=int, default=64)
    g.add_argument("--k", type=float, default=5.0)
    g.add_argument("--pixel-chunk", type=int, default=8)
    g.add_argument(
        "--full", action="store_true", help="all pulsars (else SMOKE_SUBSET)"
    )
    pl = sub.add_parser("plot")
    pl.add_argument("--input", type=Path, default=Path("cgw_localization.npz"))
    pl.add_argument("--which", choices=["marginal", "profile"], default="marginal")
    pl.add_argument("--outdir", type=Path, default=Path("."))
    args = p.parse_args()

    if args.cmd == "generate":
        res = compute_localization(
            pulsar_subset=None if args.full else SMOKE_SUBSET,
            data_dir=args.data_dir,
            nside=args.nside,
            truth_pix=args.truth_pix,
            snr=args.snr,
            k=args.k,
            n_phase=args.n_phase,
            pixel_chunk=args.pixel_chunk,
        )
        save_npz_results(args.output, res)
    else:
        plot_results(args.input, which=args.which, outdir=args.outdir)


if __name__ == "__main__":
    main()
