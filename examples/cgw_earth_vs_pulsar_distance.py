"""Compare Earth-term vs pulsar-term CGW distance-sensitivity sky maps.

Loads two ``.npz`` files produced by ``cgw_distance_skymap.py generate`` — one
Earth-term-only, one with ``--include-pulsar-term`` — and renders the standard
three-panel comparison:

  1. side-by-side ``log10(D_L)`` Mollweide maps,
  2. per-pixel ratio map ``D_L^PT / D_L^ET`` (expect a modest ~1.2-2x boost),
  3. overlaid per-pixel ``D_L`` histograms.

Pulsars are overlaid using the pulsar-term map's ``pulsar_term_mask`` (saved by
``compute_skymap``): **anchors** (PX > 0, carried the pulsar term) as red stars,
**non-anchors** (Earth-term only) as white dots. This makes it obvious which
pulsars actually contributed distance information on the full array.

Both inputs must share the same ``nside`` and pulsar ordering (they do when
generated from the same dataset with only ``--include-pulsar-term`` toggled).

Usage::

    python examples/cgw_earth_vs_pulsar_distance.py \\
        --earth  cgw-skymap-ocarina-real-<jobA>.npz \\
        --pulsar cgw-skymap-ocarina-real-pterm-<jobB>.npz \\
        [--outdir .] [--prefix cgw_earth_vs_pulsar]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp


def _finite_pos(m):
    """Finite, strictly-positive entries of a HEALPix map (for stats/bins)."""
    return m[np.isfinite(m) & (m > 0)]


def _pulsar_lonlat(npz):
    """(lon_deg, lat_deg) of each pulsar from the saved ICRS unit vectors."""
    pos = np.asarray(npz["pulsar_pos"])
    theta = np.arccos(np.clip(pos[:, 2], -1.0, 1.0))
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    return np.degrees(phi), 90.0 - np.degrees(theta)


def _anchor_mask(npz, n_psr):
    """Bool per-pulsar anchor flag; all-False if the .npz predates the field."""
    if "pulsar_term_mask" in npz:
        return np.asarray(npz["pulsar_term_mask"], dtype=bool)
    return np.zeros(n_psr, dtype=bool)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--earth",
        required=True,
        type=Path,
        help="Earth-term-only .npz (include_pulsar_term=False).",
    )
    ap.add_argument(
        "--pulsar",
        required=True,
        type=Path,
        help="Pulsar-term .npz (include_pulsar_term=True).",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Directory for the three output PNGs (default: cwd).",
    )
    ap.add_argument(
        "--prefix",
        default="cgw_earth_vs_pulsar",
        help="Output filename prefix (default: cgw_earth_vs_pulsar).",
    )
    args = ap.parse_args()

    bs = np.load(args.earth, allow_pickle=False)
    pt = np.load(args.pulsar, allow_pickle=False)

    d_e = np.asarray(bs["dist_ll_mpc"])
    d_p = np.asarray(pt["dist_ll_mpc"])
    nside = int(bs["nside"])
    if int(pt["nside"]) != nside:
        raise ValueError(f"nside mismatch: earth={nside}, pulsar={int(pt['nside'])}")

    psr_lon, psr_lat = _pulsar_lonlat(pt)
    n_psr = len(psr_lon)
    is_anchor = _anchor_mask(pt, n_psr)
    n_anchors = int(pt["n_anchors"]) if "n_anchors" in pt else int(is_anchor.sum())

    fe, fp = _finite_pos(d_e), _finite_pos(d_p)
    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"{n_psr} pulsars, {n_anchors} anchors (PX>0) carried the pulsar term.")
    print(
        f"Earth-term  median D_L = {np.median(fe):.1f} Mpc, "
        f"R_eff = {float(bs['r_eff_mpc']):.2f} Mpc"
    )
    print(
        f"Pulsar-term median D_L = {np.median(fp):.1f} Mpc, "
        f"R_eff = {float(pt['r_eff_mpc']):.2f} Mpc"
    )
    print(f"R_eff ratio = {float(pt['r_eff_mpc']) / float(bs['r_eff_mpc']):.3f}x")

    def _overlay_pulsars(anchor_color="red", other_color="white"):
        """Anchors as stars, non-anchors as dots, on the current mollview."""
        if is_anchor.any():
            hp.projscatter(
                psr_lon[is_anchor],
                psr_lat[is_anchor],
                lonlat=True,
                s=45,
                c=anchor_color,
                edgecolor="black",
                linewidths=0.5,
                marker="*",
            )
        if (~is_anchor).any():
            hp.projscatter(
                psr_lon[~is_anchor],
                psr_lat[~is_anchor],
                lonlat=True,
                s=18,
                c=other_color,
                edgecolor="black",
                linewidths=0.5,
                marker="o",
            )

    # ---- 1. Side-by-side log10(D_L) -------------------------------------------
    fig = plt.figure(figsize=(14, 6))
    for i, (m, title) in enumerate(
        [
            (d_e, "Earth-term only"),
            (d_p, f"Pulsar-term included ({n_anchors}/{n_psr} PX-pegged anchors)"),
        ]
    ):
        finite = _finite_pos(m)
        # Plot the LINEAR map with a log color scale (norm="log") so the colorbar
        # is in Mpc -- the same units as the median annotation. (Plotting
        # log10(D_L) instead labels the bar in dex, which makes the Mpc median in
        # the title look like it falls off the scale.)
        lin_m = np.where(np.isfinite(m) & (m > 0), m, hp.UNSEEN)
        hp.mollview(
            lin_m,
            title=f"{title}\nmedian D_L LL = {np.median(finite):.1f} Mpc",
            unit=r"$D_L$ [Mpc]",
            cmap="viridis",
            norm="log",
            min=float(finite.min()),
            max=float(finite.max()),
            sub=(1, 2, i + 1),
            fig=fig.number,
        )
        hp.graticule(dpar=30, dmer=30, color="gray", alpha=0.3)
        _overlay_pulsars()
    fig.suptitle(
        f"CGW distance sensitivity, nside={nside} "
        "(red ★ = pulsar-term anchor, white ● = Earth-term only)",
        fontsize=12,
        y=1.00,
    )
    out1 = args.outdir / f"{args.prefix}_mollviews.png"
    plt.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out1}")

    # ---- 2. Per-pixel ratio ---------------------------------------------------
    ratio = np.where(np.isfinite(d_e) & np.isfinite(d_p) & (d_e > 0), d_p / d_e, np.nan)
    fin_r = ratio[np.isfinite(ratio)]
    fig = plt.figure(figsize=(8, 5))
    hp.mollview(
        np.where(np.isfinite(ratio), ratio, hp.UNSEEN),
        title=(
            r"Pulsar-term distance boost ($D_L^{\rm PT}/D_L^{\rm ET}$)"
            "\n"
            f"median = {np.median(fin_r):.3f}x, "
            f"range {fin_r.min():.3f}x to {fin_r.max():.3f}x"
        ),
        unit=r"$D_L^{\rm PT}/D_L^{\rm ET}$",
        cmap="magma",
        fig=fig.number,
    )
    hp.graticule(dpar=30, dmer=30, color="gray", alpha=0.3)
    _overlay_pulsars(anchor_color="cyan", other_color="white")
    out2 = args.outdir / f"{args.prefix}_ratio.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out2}")

    # ---- 3. Histogram ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(
        np.log10(min(fe.min(), fp.min())), np.log10(max(fe.max(), fp.max())), 50
    )
    ax.hist(
        fe,
        bins=bins,
        alpha=0.55,
        color="C0",
        label=f"Earth-term only (median {np.median(fe):.1f} Mpc)",
    )
    ax.hist(
        fp,
        bins=bins,
        alpha=0.55,
        color="C3",
        label=f"Pulsar-term included (median {np.median(fp):.1f} Mpc)",
    )
    ax.axvline(np.median(fe), color="C0", ls="--", lw=1)
    ax.axvline(np.median(fp), color="C3", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"95% lower-limit luminosity distance $D_L$ (Mpc)")
    ax.set_ylabel(f"HEALPix pixels (nside={nside} -> {hp.nside2npix(nside)} total)")
    ax.set_title("Per-pixel CGW distance sensitivity")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    out3 = args.outdir / f"{args.prefix}_histogram.png"
    plt.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out3}")


if __name__ == "__main__":
    main()
