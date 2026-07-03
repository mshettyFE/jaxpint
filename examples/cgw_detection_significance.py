"""Frequentist CGW detection significance: F-statistics + empirical / analytic nulls.

Injects one continuous-wave source into the (synthetic) data and asks "is it
detected, and how significant?" -- the NANOGrav-style split where the F-statistic is
the *statistic* but the *significance* comes from a null distribution.  Two statistics,
each paired with the null that actually calibrates it:

* **F_e** (coherent, Earth-term, ``dof = 4``): the sky-maximized
  :func:`~jaxpint.pta.detection.fstat_skymap`.  Its significance relies on inter-pulsar
  coherence + geometry, so the null is **empirical** -- *phase shifts* (destroy the
  coherence) and *sky scrambles* (destroy the geometry).  The p-value is the fraction
  of the background exceeding the observed value.
* **F_p** (incoherent, :func:`~jaxpint.pta.detection.fstat_p`, ``dof = 2 n_psr``): the
  per-pulsar power summed.  It is sky-independent and coherence-independent, so the
  scramble nulls are *degenerate* for it; its significance is the **analytic**
  ``chi^2(2 n_psr)`` tail (:func:`~jaxpint.pta.detection.fstat_p_pvalue`).

The injection is Earth-term only (matching the F_e model); the source strain is
calibrated to a target network matched-filter SNR via ``h0_for_snr``.

Usage::

    python examples/cgw_detection_significance.py generate --data-dir DIR \\
        [--nside N] [--log10-fgw L] [--snr S] [--source-pix P] \\
        [--n-phase K] [--n-sky K] [--full] [--output PATH]
    python examples/cgw_detection_significance.py plot [--input PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ---- self-contained example config (no cross-example imports) --------------
LOG10_FGW = float(np.log10(1e-8))  # 10 nHz (resolved over a multi-year span)
FAP = 1e-3  # false-alarm probability for the reported thresholds
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


def compute_detection_significance(
    *,
    data_dir,
    pulsar_subset=SMOKE_SUBSET,
    nside=8,
    log10_fgw=LOG10_FGW,
    source_pix=None,
    snr=12.0,
    cos_inc=0.5,
    psi=0.6,
    phase0=0.9,
    n_phase=1000,
    n_sky=1000,
    fap=FAP,
    pixel_chunk=8,
    seed=0,
):
    """Inject one Earth-term CW and score F_e (empirical nulls) and F_p (chi^2 null).

    Parameters
    ----------
    data_dir : path-like
        NANOGrav-style dataset directory (par/tim), loaded by ``load_nanograv_pta``.
    pulsar_subset : list[str] or None
        Pulsar names to load; ``None`` loads all in ``data_dir``.
    nside : int
        HEALPix resolution (``npix = 12 * nside**2``) of the F_e search grid.
    log10_fgw : float
        ``log10`` GW frequency (Hz).
    source_pix : int or None
        HEALPix truth pixel of the injected source.  ``None`` -> ``npix // 2``.
    snr : float
        Target network matched-filter SNR (calibrates the injected strain).
    cos_inc, psi, phase0 : float
        Injected source orientation.
    n_phase, n_sky : int
        Number of phase-shift / sky-scramble null realizations for F_e.
    fap : float
        False-alarm probability for the reported thresholds.
    pixel_chunk : int
        Pixel batch size for the per-pulsar Gram extraction.
    seed : int
        Base PRNG seed for the background draws.

    Returns
    -------
    dict
        ``fstat_map`` (npix,), the observed ``fe_stat``/``fp_stat``, their p-values
        (``p_fe_phase``, ``p_fe_sky``, ``p_fp``), thresholds, the null samples
        (``bg_phase``/``bg_sky``), plus injection config and ``pulsar_pos``/names.
    """
    import jax
    import jax.numpy as jnp
    from loguru import logger

    from jaxpint import load_nanograv_pta
    from jaxpint.bayes import marginalize_single_pulsar, ImproperPrior
    from jaxpint.pta.signals.cw import cw_delay_from_array
    from jaxpint.pta.sensitivity import earth_term_gram, unit_noncentrality
    from jaxpint.pta.cw_localization import h0_for_snr
    from jaxpint.pta.detection import (
        quadrature_blocks,
        fstat_skymap,
        fstat_p,
        fstat_p_pvalue,
        phase_shift_background,
        sky_scramble_background,
        pvalue,
    )
    from jaxpint.sensitivity import chi2_threshold
    from jaxpint.utils import pulsar_unit_vector

    hp = _import_healpy()
    logger.disable("pint")

    # ---- load pulsars + sky grid -----------------------------------------
    psrs = load_nanograv_pta(data_dir, pulsar_names=pulsar_subset)
    keep = [i for i, n in enumerate(psrs.pulsar_names) if n not in DROP_PULSARS]
    names = [psrs.pulsar_names[i] for i in keep]
    td_list = [psrs.toa_data_list[i] for i in keep]
    tm_list = [psrs.timing_models[i] for i in keep]
    nm_list = [psrs.noise_models[i] for i in keep]
    pp_list = [psrs.pulsar_params_list[i] for i in keep]
    positions = np.stack([np.asarray(pulsar_unit_vector(pp)) for pp in pp_list])
    pos_j = jnp.asarray(positions)
    npsr = len(names)

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    cos_gwtheta = jnp.asarray(np.cos(theta))
    gwphi = jnp.asarray(phi)
    if source_pix is None:
        source_pix = npix // 2
    source_pix = int(source_pix)
    ct_truth, gp_truth = float(np.cos(theta[source_pix])), float(phi[source_pix])
    _log(
        f"Loaded {npsr} pulsars; nside={nside} (npix={npix}); source pix {source_pix}."
    )

    # ---- per-pulsar likelihoods (+ warmup) -------------------------------
    pulsars = []  # (g, skel, td, pos)
    for td, tm, nm, pp, pos in zip(td_list, tm_list, nm_list, pp_list, pos_j):
        over = {n for n in pp.free_names() if n in MARG_PARAMS}
        g, _, skel = marginalize_single_pulsar(
            over=over,
            priors={n: ImproperPrior() for n in over},
            toa_data=td,
            timing_model=tm,
            noise_model=nm,
            fiducial_params=pp,
            allow_nonlinear=True,
            validate_linearity=False,
        )
        _ = g(skel)  # warm up the noise model's cached device basis
        pulsars.append((g, skel, td, pos))

    # ---- calibrate the strain to the target SNR (Earth-term network Gram) --
    M_net = jnp.zeros((4, 4))
    for g, skel, td, pos in pulsars:
        M_net = M_net + earth_term_gram(
            g, skel, td, pos, 1.0, ct_truth, gp_truth, log10_fgw
        )
    snr2_unit = unit_noncentrality(M_net, jnp.array([[cos_inc, psi, phase0]]))[0]
    h0 = float(h0_for_snr(snr, snr2_unit))
    _log(f"Injecting SNR={snr:.1f} -> h0={h0:.3e}.")

    # ---- inject Earth-term signal; extract per-pulsar (S,C)/G -------------
    sc_l, gram_l = [], []
    cw = jnp.array([h0, ct_truth, gp_truth, log10_fgw, cos_inc, psi, phase0])
    for g, skel, td, pos in pulsars:
        s = cw_delay_from_array(
            td, pos, 1.0, cw, earth_term_only=True, linear_amplitude=True
        )
        g_inj = lambda rp, external_delay=0.0, g=g, s=s: g(
            rp, external_delay=external_delay - s
        )
        sc, gram = quadrature_blocks(g_inj, skel, td, log10_fgw)
        sc_l.append(sc)
        gram_l.append(gram)
    sc_all, gram_all = jnp.stack(sc_l), jnp.stack(gram_l)

    # ---- F_e: coherent sky-max, empirical (scramble) nulls ---------------
    fstat_map = np.asarray(fstat_skymap(sc_all, gram_all, pos_j, cos_gwtheta, gwphi))
    fe_stat = float(fstat_map.max())
    fe_pix = int(fstat_map.argmax())
    bg_phase_j = phase_shift_background(
        sc_all,
        gram_all,
        pos_j,
        cos_gwtheta,
        gwphi,
        n_phase,
        jax.random.PRNGKey(seed + 1),
    )
    bg_sky_j = sky_scramble_background(
        sc_all, gram_all, cos_gwtheta, gwphi, n_sky, jax.random.PRNGKey(seed + 2)
    )
    p_fe_phase, p_fe_sky = pvalue(fe_stat, bg_phase_j), pvalue(fe_stat, bg_sky_j)
    bg_phase, bg_sky = np.asarray(bg_phase_j), np.asarray(bg_sky_j)

    # ---- F_p: incoherent, analytic chi^2(2 n_psr) null -------------------
    fp_stat = float(fstat_p(sc_all, gram_all))
    p_fp = fstat_p_pvalue(fp_stat, npsr)
    fp_threshold = chi2_threshold(fap, 2 * npsr)

    off = float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(
                        hp.ang2vec(*hp.pix2ang(nside, fe_pix)),
                        hp.ang2vec(*hp.pix2ang(nside, source_pix)),
                    ),
                    -1,
                    1,
                )
            )
        )
    )
    _log(
        f"F_e = {fe_stat:.1f} at pix {fe_pix} ({off:.1f} deg from truth); "
        f"p(phase)={p_fe_phase:.3g}, p(sky)={p_fe_sky:.3g}."
    )
    _log(
        f"F_p = {fp_stat:.1f} (dof={2 * npsr}); p(chi2)={p_fp:.3g}, "
        f"threshold@fap={fap:.0e} is {fp_threshold:.1f}."
    )

    return {
        "fstat_map": fstat_map,
        "fe_stat": np.float64(fe_stat),
        "fe_pix": np.int64(fe_pix),
        "fe_offset_deg": np.float64(off),
        "bg_phase": bg_phase,
        "bg_sky": bg_sky,
        "p_fe_phase": np.float64(p_fe_phase),
        "p_fe_sky": np.float64(p_fe_sky),
        "fe_threshold_phase": np.float64(np.quantile(bg_phase, 1.0 - fap)),
        "fe_threshold_sky": np.float64(np.quantile(bg_sky, 1.0 - fap)),
        "fp_stat": np.float64(fp_stat),
        "p_fp": np.float64(p_fp),
        "fp_threshold": np.float64(fp_threshold),
        "fp_dof": np.int64(2 * npsr),
        "nside": np.int64(nside),
        "source_pix": np.int64(source_pix),
        "log10_fgw": np.float64(log10_fgw),
        "snr": np.float64(snr),
        "h0": np.float64(h0),
        "fap": np.float64(fap),
        "pulsar_pos": positions,
        "pulsar_names": np.array(names),
    }


def save_results(path, results):
    """Write a :func:`compute_detection_significance` results dict to ``.npz``."""
    np.savez_compressed(path, **results)
    _log(f"Saved -> {path}")


def plot_results(path, outdir="."):
    """F_e ``2F`` sky map + F_e/F_p significance panels (nulls with observed marked)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import chi2

    hp = _import_healpy()
    d = np.load(path, allow_pickle=False)
    nside = int(d["nside"])
    fstat_map = np.asarray(d["fstat_map"])
    src_ang = hp.pix2ang(nside, int(d["source_pix"]))
    max_ang = hp.pix2ang(nside, int(d["fe_pix"]))
    pt, pp = hp.vec2ang(np.asarray(d["pulsar_pos"]))

    # -- sky map --
    fig1 = plt.figure(figsize=(8, 5))
    hp.mollview(
        fstat_map,
        fig=fig1.number,
        rot=[180, 0],
        cmap="inferno",
        title=f"F_e sky map (2F), injected SNR={float(d['snr']):.0f}, "
        f"offset={float(d['fe_offset_deg']):.1f} deg",
    )
    hp.projscatter(*src_ang, marker="*", s=220, color="cyan", edgecolors="k")
    hp.projscatter(*max_ang, marker="D", s=60, color="lime", edgecolors="k")
    hp.projscatter(pt, pp, marker="o", s=16, color="white", edgecolors="k")
    hp.graticule()
    out1 = Path(outdir) / "cgw_detection_skymap.png"
    fig1.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig1)
    _log(f"Sky map -> {out1}")

    # -- significance panels --
    fig2, (axe, axp) = plt.subplots(1, 2, figsize=(13, 5))
    fap = float(d["fap"])
    for bg, lbl, col in (
        (np.asarray(d["bg_phase"]), "phase shifts", "tab:blue"),
        (np.asarray(d["bg_sky"]), "sky scrambles", "tab:orange"),
    ):
        axe.hist(
            bg,
            bins=40,
            histtype="step",
            density=True,
            color=col,
            label=f"{lbl} (p={pvalue_from(bg, float(d['fe_stat'])):.2g})",
        )
    axe.axvline(float(d["fe_stat"]), color="k", lw=2, label="observed F_e")
    axe.axvline(float(d["fe_threshold_phase"]), color="tab:blue", ls=":")
    axe.axvline(float(d["fe_threshold_sky"]), color="tab:orange", ls=":")
    axe.set(
        xlabel="sky-max 2F",
        ylabel="density",
        title=f"F_e: empirical nulls (dotted = fap={fap:.0e} threshold)",
    )
    axe.legend()

    dof = int(d["fp_dof"])
    xs = np.linspace(0, max(float(d["fp_stat"]) * 1.15, chi2.ppf(0.999, dof)), 400)
    axp.plot(xs, chi2.pdf(xs, dof), color="tab:green", label=f"chi^2({dof}) null")
    axp.axvline(
        float(d["fp_stat"]),
        color="k",
        lw=2,
        label=f"observed F_p (p={float(d['p_fp']):.2g})",
    )
    axp.axvline(
        float(d["fp_threshold"]),
        color="tab:green",
        ls=":",
        label=f"fap={fap:.0e} threshold",
    )
    axp.set(xlabel="2F_p", ylabel="density", title="F_p: analytic chi^2(2 n_psr) null")
    axp.legend()
    out2 = Path(outdir) / "cgw_detection_significance.png"
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    _log(f"Significance -> {out2}")


def pvalue_from(background, stat):
    """Fraction of ``background`` >= ``stat`` (plot-side helper)."""
    return float((np.asarray(background) >= stat).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--output", type=Path, default=Path("cgw_detection.npz"))
    g.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="NANOGrav-style dataset directory (par/ + tim/), e.g. the synthetic "
        "'ocarina_2' set; loaded by load_nanograv_pta.",
    )
    g.add_argument("--nside", type=int, default=8)
    g.add_argument("--log10-fgw", type=float, default=LOG10_FGW)
    g.add_argument("--snr", type=float, default=12.0)
    g.add_argument("--source-pix", type=int, default=None)
    g.add_argument("--n-phase", type=int, default=1000)
    g.add_argument("--n-sky", type=int, default=1000)
    g.add_argument("--pixel-chunk", type=int, default=8)
    g.add_argument(
        "--full", action="store_true", help="all pulsars (else SMOKE_SUBSET)"
    )
    pl = sub.add_parser("plot")
    pl.add_argument("--input", type=Path, default=Path("cgw_detection.npz"))
    pl.add_argument("--outdir", type=Path, default=Path("."))
    args = p.parse_args()

    if args.cmd == "generate":
        res = compute_detection_significance(
            data_dir=args.data_dir,
            pulsar_subset=None if args.full else SMOKE_SUBSET,
            nside=args.nside,
            log10_fgw=args.log10_fgw,
            snr=args.snr,
            source_pix=args.source_pix,
            n_phase=args.n_phase,
            n_sky=args.n_sky,
            pixel_chunk=args.pixel_chunk,
        )
        save_results(args.output, res)
    else:
        plot_results(args.input, outdir=args.outdir)


if __name__ == "__main__":
    main()
