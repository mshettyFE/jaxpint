"""Frequentist CGW detection sensitivity: h0_min(f) and its luminosity-distance horizon.

The *frequentist* arm (F-statistic ``2F`` ~ ``chi2(4)`` null / ``ncx2(4, lambda)``
signal), as opposed to the Bayesian upper-limit / localization maps.  For each sky
pixel and GW frequency it builds the network Earth-term orientation Gram ``M``, turns it into the
per-orientation noncentrality ``lambda_1(theta) = c(theta)^T M c(theta)``
, and solves for the strain
``h0_min`` at which the *orientation-averaged* detection probability reaches ``beta``
.  ``h0_to_distance`` maps that to a
luminosity-distance horizon for a fiducial chirp mass.

It reports the headline **sky-median ``h0_min(f)`` and ``D_L(f)`` curves** (with Q1/Q3
bands) and a **GWB-on/off contrast**: injecting a CURN covariance
(:func:`jaxpint.pta.signals.gwb.gwb_covariance`) into each pulsar's likelihood raises
the noise floor, so ``h0_min_on / h0_min_off >= 1``.

Scope (v1): **Earth term only** -- the source's Earth-term residual spans the 4-D
orientation basis (``dof = 4``), so this is the conservative ``sigma_L -> infinity``
(pulsar-distance-agnostic) bracket.  The pulsar-term / ``sigma_L`` distance sweep
(which raises the SNR and extends the horizon as distances tighten) needs the 12-D
orientation x pulsar-phase extension and is deferred.

Usage::

    python examples/cgw_distance_sensitivity.py generate --data-dir DIR \\
        [--nside N] [--n-freq K] [--n-theta T] [--no-gwb] [--full] [--output PATH]
    python examples/cgw_distance_sensitivity.py plot [--input PATH] [--skymap]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ---- self-contained example config (no cross-example imports) --------------
LOG10_MC = 9.0  # fiducial chirp mass 1e9 Msun -> horizon scale
FAP = 1e-3  # false-alarm probability -> 2F threshold
BETA = 0.95  # target orientation-averaged detection probability
DOF = 4  # Earth-term orientation-span rank
N_THETA = 64  # orientation draws for the E_theta average
N_FREQ = 8
GWB_NCOMP = 30  # CURN Fourier components
GWB_LOG10_A = -15.0  # CURN amplitude (CURN_PARAM_DEFAULTS)
GWB_GAMMA = 4.33  # CURN spectral index
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


def compute_distance_sensitivity(
    *,
    data_dir,
    pulsar_subset=SMOKE_SUBSET,
    nside=8,
    log10_fgw_grid=None,
    n_freq=N_FREQ,
    log10_mc=LOG10_MC,
    fap=FAP,
    beta=BETA,
    n_theta=N_THETA,
    pixel_chunk=8,
    gwb=True,
    gwb_log10_A=GWB_LOG10_A,
    gwb_gamma=GWB_GAMMA,
    gwb_ncomp=GWB_NCOMP,
):
    """Earth-term ``h0_min`` and distance-horizon sky maps vs frequency, GWB off/on.

    Parameters
    ----------
    data_dir : path-like
        NANOGrav-style dataset directory (par/tim), loaded by ``load_nanograv_pta``.
    pulsar_subset : list[str] or None
        Pulsar names to load; ``None`` loads all in ``data_dir``.
    nside : int
        HEALPix resolution (``npix = 12 * nside**2``).
    log10_fgw_grid : (n_freq,) array or None
        ``log10`` GW frequencies (Hz).  ``None`` -> ``linspace(log10 5e-9, log10 1e-7)``.
    n_freq : int
        Number of frequencies when ``log10_fgw_grid`` is ``None``.
    log10_mc : float
        ``log10`` fiducial chirp mass (Msun) for the distance horizon.
    fap : float
        False-alarm probability (sets the 2F threshold via ``chi2_threshold``).
    beta : float
        Target orientation-averaged detection probability.
    n_theta : int
        Orientation draws for the ``E_theta`` detection-probability average.
    pixel_chunk : int
        Pixel batch size for the per-pulsar Gram extraction (``jax.lax.map``).
    gwb : bool
        If ``True`` also compute the GWB-on (CURN) pass; else the "on" outputs equal
        the "off" ones and the penalty ratio is 1.
    gwb_log10_A, gwb_gamma : float
        CURN amplitude / spectral index.
    gwb_ncomp : int
        CURN Fourier components.

    Returns
    -------
    dict
        ``h0_min_off/on`` and ``horizon_off/on_mpc`` : (n_freq, npix) maps;
        ``*_median`` / ``*_q1`` / ``*_q3`` : (n_freq,) sky-summary curves;
        ``penalty_ratio`` : (n_freq, npix) ``h0_min_on/off``; plus ``log10_fgw_grid``,
        ``threshold``, ``rank_health_off/on`` (min eigenvalue ratio of ``M`` over
        pixels/frequencies -- ~1 is healthy rank-4, ~0 means degenerate/wrong dof),
        ``evolution_earth_ok`` (per-freq monochromatic-drift flag), config, and
        ``pulsar_pos``/``pulsar_names``.
    """
    import jax
    import jax.numpy as jnp
    from loguru import logger

    from jaxpint import load_nanograv_pta
    from jaxpint.bayes import marginalize_single_pulsar, ImproperPrior
    from jaxpint.pta.cw_upper_limit import (
        _default_extraction_orientations,
        h0_to_distance,
    )
    from jaxpint.pta.sensitivity import earth_term_gram, unit_noncentrality
    from jaxpint.pta.signals.cw import evolution_ok
    from jaxpint.pta.signals.gwb import gwb_covariance
    from jaxpint.sensitivity import chi2_threshold, h0_min_from_lambda
    from jaxpint.utils import pulsar_unit_vector

    hp = _import_healpy()
    logger.disable("pint")

    if log10_fgw_grid is None:
        log10_fgw_grid = np.linspace(np.log10(5e-9), np.log10(1e-7), n_freq)
    log10_fgw_grid = np.asarray(log10_fgw_grid, dtype=np.float64)
    n_freq = len(log10_fgw_grid)
    threshold = chi2_threshold(fap, DOF)
    orientations = _default_extraction_orientations(n_theta, seed=1)

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
    grid = jnp.asarray(
        np.stack([np.cos(theta), phi], axis=1)
    )  # (npix, 2): (cos_gwtheta, gwphi)

    # PTA-wide observing span (seconds) for the CURN Fourier basis.
    tdb = [np.asarray(td.tdb_seconds) for td in td_list]
    T_span = float(max(t.max() for t in tdb) - min(t.min() for t in tdb))
    _log(
        f"Loaded {npsr} pulsars; nside={nside} (npix={npix}); {n_freq} frequencies; "
        f"T_span={T_span / 3.15576e7:.1f} yr."
    )

    # ---- per-pulsar timing-marginalized likelihoods ----------------------
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
        # Warm up the noise model's @cached_property device basis (_fourier_basis_jax)
        # eagerly: if it first materializes inside the lax.map below (which itself wraps
        # basis_quadratics's inner lax.map) it caches a *tracer* and leaks. cf.
        # examples/cgw_localization_skymap.py:332.
        _ = g(skel)
        pulsars.append((g, skel, td, pos))

    def gwb_wrap(g, td):  # inject a CURN covariance into the per-pulsar likelihood
        U, Phi = gwb_covariance(td, gwb_ncomp, T_span, gwb_log10_A, gwb_gamma)

        def g_gwb(rp, external_delay=0.0):
            return g(rp, external_delay=external_delay, external_cov=(U, Phi))

        return g_gwb

    # ---- heavy step: network Gram M(freq, pixel), then h0_min + horizon ----
    def h0_min_maps(wrap):
        """(n_freq, npix) h0_min + horizon + per-freq worst eigenvalue ratio."""
        M_all = jnp.zeros((n_freq, npix, 4, 4))
        for g, skel, td, pos in pulsars:
            gg = wrap(g, td) if wrap else g
            for fi, lf in enumerate(log10_fgw_grid):
                lf = float(lf)
                M_a = jax.lax.map(
                    lambda row, gg=gg, td=td, pos=pos, lf=lf: earth_term_gram(
                        gg, skel, td, pos, 1.0, row[0], row[1], lf
                    ),
                    grid,
                    batch_size=pixel_chunk,
                )
                M_all = M_all.at[fi].add(M_a)

        h0 = np.empty((n_freq, npix))
        horizon = np.empty((n_freq, npix))
        rank_health = 1.0
        for fi in range(n_freq):
            lam1 = jax.vmap(lambda M: unit_noncentrality(M, orientations))(
                M_all[fi]
            )  # (npix, n_theta)
            h0[fi] = np.asarray(h0_min_from_lambda(threshold, lam1, dof=DOF, beta=beta))
            horizon[fi] = np.asarray(
                h0_to_distance(jnp.asarray(h0[fi]), log10_mc, float(log10_fgw_grid[fi]))
            )
            eig = np.linalg.eigvalsh(np.asarray(M_all[fi]))  # (npix, 4), ascending
            rank_health = min(rank_health, float(np.min(eig[:, 0] / eig[:, -1])))
        return h0, horizon, rank_health

    _log("Computing GWB-off sensitivity ...")
    h0_off, hor_off, rank_off = h0_min_maps(None)
    if rank_off < 1e-6:
        _log(
            f"  WARNING: Earth-term Gram is near-degenerate (min eig ratio {rank_off:.1e}); "
            "dof=4 may be invalid -- is the CW resolved over the span?"
        )
    if gwb:
        _log("Computing GWB-on (CURN) sensitivity ...")
        h0_on, hor_on, rank_on = h0_min_maps(gwb_wrap)
    else:
        h0_on, hor_on, rank_on = h0_off, hor_off, rank_off

    # ---- sky summaries + diagnostics -------------------------------------
    def band(x):  # (n_freq, npix) -> median, q1, q3 (each (n_freq,))
        return (np.median(x, 1), np.percentile(x, 25, 1), np.percentile(x, 75, 1))

    penalty = h0_on / h0_off
    mc_msun = 10.0**log10_mc
    evo = [evolution_ok(mc_msun, float(10.0**lf), T_span) for lf in log10_fgw_grid]
    _log(
        f"h0_min (sky median) off: {np.median(h0_off, 1)[0]:.2e} .. "
        f"{np.median(h0_off, 1)[-1]:.2e}; GWB penalty median "
        f"{np.median(penalty):.2f}x."
    )

    h0m_off, h0q1_off, h0q3_off = band(h0_off)
    h0m_on, h0q1_on, h0q3_on = band(h0_on)
    horm_off, horq1_off, horq3_off = band(hor_off)
    horm_on, horq1_on, horq3_on = band(hor_on)

    return {
        "h0_min_off": h0_off,
        "h0_min_on": h0_on,
        "horizon_off_mpc": hor_off,
        "horizon_on_mpc": hor_on,
        "penalty_ratio": penalty,
        "h0_min_median_off": h0m_off,
        "h0_min_q1_off": h0q1_off,
        "h0_min_q3_off": h0q3_off,
        "h0_min_median_on": h0m_on,
        "h0_min_q1_on": h0q1_on,
        "h0_min_q3_on": h0q3_on,
        "horizon_median_off": horm_off,
        "horizon_q1_off": horq1_off,
        "horizon_q3_off": horq3_off,
        "horizon_median_on": horm_on,
        "horizon_q1_on": horq1_on,
        "horizon_q3_on": horq3_on,
        "penalty_median": np.median(penalty, 1),
        "log10_fgw_grid": log10_fgw_grid,
        "nside": np.int64(nside),
        "threshold": np.float64(threshold),
        "fap": np.float64(fap),
        "beta": np.float64(beta),
        "dof": np.int64(DOF),
        "log10_mc": np.float64(log10_mc),
        "n_theta": np.int64(n_theta),
        "gwb": bool(gwb),
        "gwb_log10_A": np.float64(gwb_log10_A),
        "gwb_gamma": np.float64(gwb_gamma),
        "gwb_ncomp": np.int64(gwb_ncomp),
        "T_span_s": np.float64(T_span),
        "rank_health_off": np.float64(rank_off),
        "rank_health_on": np.float64(rank_on),
        "evolution_earth_ok": np.array([e["earth_ok"] for e in evo]),
        "evolution_drift_cycles": np.array([e["drift_cycles"] for e in evo]),
        "pulsar_pos": positions,
        "pulsar_names": np.array(names),
    }


def save_results(path, results):
    """Write a :func:`compute_distance_sensitivity` results dict to ``.npz``."""
    np.savez_compressed(path, **results)
    _log(f"Saved -> {path}")


def plot_results(path, outdir=".", skymap=False):
    """Sky-median ``h0_min(f)`` and ``D_L(f)`` curves (GWB off vs on, Q1/Q3 bands).

    With ``skymap=True`` also renders a Mollweide panel of the reference-frequency
    ``h0_min`` map, distance horizon, and GWB penalty ratio.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(path, allow_pickle=False)
    f_nhz = 10.0 ** np.asarray(d["log10_fgw_grid"]) / 1e-9
    gwb = bool(d["gwb"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, med, q1, q3, ylab, title in (
        (
            ax0,
            "h0_min_median",
            "h0_min_q1",
            "h0_min_q3",
            "h0_min",
            "Minimum detectable strain",
        ),
        (
            ax1,
            "horizon_median",
            "horizon_q1",
            "horizon_q3",
            "D_L horizon [Mpc]",
            "Distance horizon",
        ),
    ):
        for state, style in (("off", "-"), ("on", "--")):
            if state == "on" and not gwb:
                continue
            ax.plot(f_nhz, d[f"{med}_{state}"], style, label=f"GWB {state}")
            ax.fill_between(f_nhz, d[f"{q1}_{state}"], d[f"{q3}_{state}"], alpha=0.2)
        ax.set(
            xscale="log", yscale="log", xlabel="f_gw [nHz]", ylabel=ylab, title=title
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(
        f"Earth-term F-stat sensitivity (fap={float(d['fap']):.0e}, beta={float(d['beta']):.2f}, "
        f"log10_mc={float(d['log10_mc']):.1f})"
    )
    out = Path(outdir) / "cgw_distance_sensitivity.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    _log(f"Plot -> {out}")

    if skymap:
        hp = _import_healpy()
        rf = len(f_nhz) // 2  # reference frequency (grid middle)
        panels = [
            (np.asarray(d["h0_min_off"])[rf], "h0_min (GWB off)", "viridis"),
            (np.asarray(d["horizon_off_mpc"])[rf], "D_L horizon [Mpc]", "magma"),
            (np.asarray(d["penalty_ratio"])[rf], "GWB penalty h0_on/off", "cividis"),
        ]
        pt, pp = hp.vec2ang(np.asarray(d["pulsar_pos"]))
        fig2 = plt.figure(figsize=(6.5 * len(panels), 4.2))
        for i, (m, title, cmap) in enumerate(panels):
            hp.mollview(
                m,
                fig=fig2.number,
                sub=(1, len(panels), i + 1),
                rot=[180, 0],
                cmap=cmap,
                title=f"{title}\n(f={f_nhz[rf]:.1f} nHz)",
            )
            hp.projscatter(pt, pp, marker="o", s=16, color="white", edgecolors="k")
            hp.graticule()
        out2 = Path(outdir) / "cgw_distance_sensitivity_skymap.png"
        fig2.savefig(out2, dpi=130, bbox_inches="tight")
        plt.close(fig2)
        _log(f"Skymap -> {out2}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--output", type=Path, default=Path("cgw_distance_sensitivity.npz"))
    g.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="NANOGrav-style dataset directory (par/ + tim/), e.g. the synthetic "
        "'ocarina_2' set; loaded by load_nanograv_pta.",
    )
    g.add_argument("--nside", type=int, default=8)
    g.add_argument("--n-freq", type=int, default=N_FREQ)
    g.add_argument("--n-theta", type=int, default=N_THETA)
    g.add_argument("--log10-mc", type=float, default=LOG10_MC)
    g.add_argument("--fap", type=float, default=FAP)
    g.add_argument("--beta", type=float, default=BETA)
    g.add_argument("--pixel-chunk", type=int, default=8)
    g.add_argument("--no-gwb", action="store_true", help="skip the GWB-on (CURN) pass")
    g.add_argument(
        "--full", action="store_true", help="all pulsars (else SMOKE_SUBSET)"
    )
    pl = sub.add_parser("plot")
    pl.add_argument("--input", type=Path, default=Path("cgw_distance_sensitivity.npz"))
    pl.add_argument("--outdir", type=Path, default=Path("."))
    pl.add_argument(
        "--skymap", action="store_true", help="also render reference-freq sky maps"
    )
    args = p.parse_args()

    if args.cmd == "generate":
        res = compute_distance_sensitivity(
            data_dir=args.data_dir,
            pulsar_subset=None if args.full else SMOKE_SUBSET,
            nside=args.nside,
            n_freq=args.n_freq,
            n_theta=args.n_theta,
            log10_mc=args.log10_mc,
            fap=args.fap,
            beta=args.beta,
            pixel_chunk=args.pixel_chunk,
            gwb=not args.no_gwb,
        )
        save_results(args.output, res)
    else:
        plot_results(args.input, outdir=args.outdir, skymap=args.skymap)


if __name__ == "__main__":
    main()
