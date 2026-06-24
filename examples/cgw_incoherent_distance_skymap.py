"""Real-mode Bayesian CW distance-reach sky map with the pulsar term marginalized
over the pulsar **distance** (PX ± kσ uniform prior).

Companion to ``cgw_distance_skymap.py``.  There the pulsar term is either dropped
(Earth-term only) or pegged to the par-file PX (coherent).  Here each pulsar's
pulsar-term phase is **marginalized** over a measurement-faithful uniform distance
prior ``L ∈ [1/PX − kσ_L, 1/PX + kσ_L]`` (σ_L = σ_PX/PX²; σ_PX from the parser's
``param_uncertainty``).  Because the prior spans ≫1 phase cycle for any realistic
parallax (Δ_p ~ 1e4-1e6 rad), this collapses to the exact flat-phase limit -- so
**every** pulsar contributes its pulsar term, no PX sign/anchor restriction.  See
:mod:`jaxpint.pta.incoherent_ul` for the construction.

Fixed orientation (face-on), ``data_mode='real'``.  Usage::

    python examples/cgw_incoherent_distance_skymap.py generate [--output PATH] [--nside N] \\
        [--full] [--pixel-chunk N] [--k 5] [--n-phase 256]
    python examples/cgw_incoherent_distance_skymap.py plot [--input PATH]    # reuses the earth-vs-pulsar plotter
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Reuse the dataset/loader/constants/helpers from the sibling driver.
from examples.cgw_distance_skymap import (
    DATA_DIR, DROP_PULSARS, SMOKE_SUBSET, MARG_PARAMS, FIXED_ORIENTATION,
    LOG10_MC, LOG10_FGW, F_GW, pulsar_unit_vector_icrs, _log, _import_healpy,
)


def compute_skymap(*, pulsar_subset=SMOKE_SUBSET, nside=8, k=5.0, n_phase=256,
                   pixel_chunk=8, earth_term_only=False,
                   coherent_fraction=0.0, coherent_sigma_pc=0.1):
    """Compute the distance-reach map. Returns a results dict (see save keys).

    earth_term_only : bool
        Drop the pulsar term and route the pure Earth-term signal through the same
        numerical UL -- a *method-matched* baseline for the earth-vs-incoherent
        comparison plot (the ratio panel is then a controlled A/B).
    coherent_fraction : float
        Fraction of pulsars (those with the highest measured PX/sigma_PX) to give a
        TIGHT distance prior of width ``coherent_sigma_pc`` (pc): their pulsar-term
        phase is localized -> they contribute *coherently*, while the rest stay
        flat-phase (incoherent).  0.0 -> the all-incoherent map.
    coherent_sigma_pc : float
        Distance-prior 1-sigma (pc) for the tight subset.  ~0.01 pc localizes the
        phase (coherent); ~1 pc still spans many cycles (~ flat-phase).  Sweepable.
    """
    import jax
    import jax.numpy as jnp
    from loguru import logger

    from jaxpint import load_nanograv_pta
    from jaxpint.likelihood import single_pulsar_logL
    from jaxpint.bayes import marginalize, ImproperPrior
    from jaxpint.pta.signals.cw import cw_delay_from_array
    from jaxpint.pta.cw_upper_limit import h0_to_distance
    from jaxpint.pta.incoherent_ul import (
        extract_pulsar_bM, flat_phase_grid, h0_95_grid, _A_of_phase, earth_only_A,
        mixed_phase_A,
    )

    hp = _import_healpy()
    logger.disable("pint")

    # ---- 1. Load + filter -------------------------------------------------
    psrs = load_nanograv_pta(DATA_DIR, pulsar_names=pulsar_subset)
    keep = [i for i, n in enumerate(psrs.pulsar_names) if n not in DROP_PULSARS]
    names = [psrs.pulsar_names[i] for i in keep]
    td_list = [psrs.toa_data_list[i] for i in keep]
    tm_list = [psrs.timing_models[i] for i in keep]
    nm_list = [psrs.noise_models[i] for i in keep]
    pp_list = [psrs.pulsar_params_list[i] for i in keep]
    positions = np.stack([np.asarray(pulsar_unit_vector_icrs(pp)) for pp in pp_list])
    n_toa = int(sum(int(td.n_toas) for td in td_list))
    _log(f"Loaded {len(names)} pulsars, {n_toa} TOAs total.")

    cos_inc, psi, phase0 = (float(x) for x in FIXED_ORIENTATION)
    pi2 = float(np.pi / 2.0)

    # ---- Tight-distance subset (highest PX/sigma_PX) ----------------------
    # These pulsars get a narrow distance prior (coherent_sigma_pc), so their
    # pulsar-term phase is localized -> coherent. The rest stay flat-phase.
    def _px_sig(pp):
        try:
            px = float(pp.param_value("PX")); sig = float(pp.param_uncertainty("PX"))
        except KeyError:
            return -np.inf, np.nan
        ok = np.isfinite(px) and px > 0 and np.isfinite(sig) and sig > 0
        return (px / sig if ok else -np.inf), (px if px > 0 else np.nan)
    sig_px = np.array([_px_sig(pp)[0] for pp in pp_list])
    px_val = np.array([_px_sig(pp)[1] for pp in pp_list])
    n_tight = int(round(coherent_fraction * len(names)))
    tight_idx = np.argsort(sig_px)[::-1][:n_tight]
    tight_idx = tight_idx[sig_px[tight_idx] > -np.inf]   # never select PX<=0 / no-sigma
    is_tight = np.zeros(len(names), dtype=bool); is_tight[tight_idx] = True
    L0_kpc = np.where(px_val > 0, 1.0 / np.where(px_val > 0, px_val, 1.0), 1.0)  # kpc; dummy for non-tight
    sigma_L_kpc = float(coherent_sigma_pc) * 1e-3
    if n_tight:
        _log(f"Coherent subset: {int(is_tight.sum())} pulsars with sigma_L="
             f"{coherent_sigma_pc} pc (tight distance prior); "
             f"{[names[i] for i in tight_idx]}")

    # ---- 2. HEALPix sky grid ---------------------------------------------
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    cos_gwtheta = jnp.asarray(np.cos(theta))
    gwphi = jnp.asarray(phi)

    def cw_params(ct, gp):
        return jnp.array([1.0, ct, gp, LOG10_FGW, cos_inc, psi, phase0])

    # ---- 3. Per-pulsar (b, M) over the whole sky -------------------------
    # marginalize once per pulsar (timing params), then vmap the cheap template
    # build + autodiff extraction over sky pixels.
    _log(f"nside={nside} -> {npix} pixels; marginalizing timing params and "
         f"extracting per-pulsar (b,M) over the sky (chunk={pixel_chunk})...")
    b_all = np.empty((len(names), npix, 2))
    M_all = np.empty((len(names), npix, 2, 2))
    for a, (td, tm, nm, pp, pos) in enumerate(
        zip(td_list, tm_list, nm_list, pp_list, jnp.asarray(positions))
    ):
        over = {n for n in pp.free_names() if n in MARG_PARAMS}
        g, _, skel = marginalize(
            single_pulsar_logL, over=over,
            priors={n: ImproperPrior() for n in over},
            toa_data=td, timing_model=tm, noise_model=nm, fiducial_params=pp,
            allow_nonlinear=True, validate_linearity=False,
        )

        def bM_at(ct, gp, pos=pos, td=td, g=g, skel=skel):
            cw = cw_params(ct, gp)
            e = cw_delay_from_array(td, pos, 1.0, cw, linear_amplitude=True,
                                    earth_term_only=True)
            ps = cw_delay_from_array(td, pos, 1.0, cw, linear_amplitude=True,
                                     pulsar_term_only=True, pulsar_term_phase=pi2)
            return extract_pulsar_bM(g, skel, e, ps)

        bM = jax.lax.map(lambda row: bM_at(row[0], row[1]),
                         jnp.stack([cos_gwtheta, gwphi], axis=1),
                         batch_size=pixel_chunk)
        b_all[a] = np.asarray(bM[0])
        M_all[a] = np.asarray(bM[1])
        _log(f"  [{a+1}/{len(names)}] {names[a]} done.")

    # ---- 4. Per-pixel marginalize phase + 95% UL + distance --------------
    # Per-pulsar signal-coefficient vectors A(Delta):
    #   - earth-term-only: the singleton A=(1,0) (no marginalization);
    #   - else: tight-subset pulsars marginalize over their narrow distance prior
    #     (localized -> coherent); the rest over [0,2pi) flat-phase (incoherent).
    #     The tight grid depends on cos_mu, so A is built per pixel.
    is_tight_j = jnp.asarray(is_tight)
    L0_j = jnp.asarray(L0_kpc)
    # GW propagation direction per pixel; cos_mu = omhat . pulsar (matches cw.py).
    sin_th = np.sin(theta)
    omhat = np.stack([-sin_th * np.cos(phi), -sin_th * np.sin(phi), -np.cos(theta)], axis=1)
    cos_mu_all = jnp.asarray(omhat @ positions.T)             # (npix, n_psr)
    earth_A = jnp.broadcast_to(earth_only_A(), (len(names), 1, 2))

    # Move pixels to the leading axis so the per-pixel UL can be chunked the same
    # way the extraction is. A plain vmap over all pixels materializes
    # ~(npix * n_h0 * n_psr * n_phase) intermediates and OOMs the GPU; lax.map with
    # batch_size keeps only `pixel_chunk` pixels live at once. (The transpose is on
    # tiny ~MB arrays.)
    b_pix = jnp.transpose(jnp.asarray(b_all), (1, 0, 2))       # (npix, n_psr, 2)
    M_pix = jnp.transpose(jnp.asarray(M_all), (1, 0, 2, 3))    # (npix, n_psr, 2, 2)

    def dist_at_pixel(b_px, M_px, cos_mu_px):           # (n_psr,2),(n_psr,2,2),(n_psr,)
        if earth_term_only:
            A_px = earth_A
        else:
            A_px = mixed_phase_A(is_tight_j, L0_j, sigma_L_kpc, k, cos_mu_px,
                                 F_GW, n_phase)          # (n_psr, n_phase, 2)
        # adaptive h0_max from the per-pulsar phase-averaged Gaussian proxy (~10x cover)
        def _xy(b_a, M_a, A_a):
            return (jnp.mean(A_a @ b_a),
                    jnp.mean(jnp.einsum("ni,ij,nj->n", A_a, M_a, A_a)))
        Xbar, Ybar = jax.vmap(_xy)(b_px, M_px, A_px)
        SX, SY = jnp.sum(Xbar), jnp.clip(jnp.sum(Ybar), 1e-300, None)
        h0_max = 10.0 * (jnp.abs(SX) / SY + 5.0 / jnp.sqrt(SY))
        return h0_95_grid(b_px, M_px, A_px, h0_max)

    h0_95 = jax.lax.map(lambda bm: dist_at_pixel(bm[0], bm[1], bm[2]),
                        (b_pix, M_pix, cos_mu_all), batch_size=pixel_chunk)   # (npix,)
    dist_ll = np.asarray(h0_to_distance(h0_95, LOG10_MC, LOG10_FGW))
    r_eff = float(np.mean(dist_ll ** 3) ** (1.0 / 3.0))
    mode = "Earth-term only" if earth_term_only else "incoherent (distance-marg)"
    _log(f"Done [{mode}]. D_L lower limit min/median/max = "
         f"{dist_ll.min():.1f}/{np.median(dist_ll):.1f}/{dist_ll.max():.1f} Mpc; "
         f"R_eff = {r_eff:.2f} Mpc (nside={nside})")

    # pulsar_term_mask drives the plot's stars-vs-dots: Earth-term -> none;
    # hybrid -> the coherent (tight-distance) subset; pure incoherent -> all
    # (every pulsar contributes the pulsar term, none distinguished).
    if earth_term_only:
        pt_mask = np.zeros(len(names), dtype=bool)
    elif n_tight > 0:
        pt_mask = is_tight
    else:
        pt_mask = np.ones(len(names), dtype=bool)
    return {
        "dist_ll_mpc": dist_ll,
        "nside": np.int64(nside), "r_eff_mpc": np.float64(r_eff),
        "chirp_mass_msun": np.float64(10.0 ** LOG10_MC),
        "log10_mc": np.float64(LOG10_MC),
        "f_gw": np.float64(F_GW), "log10_fgw": np.float64(LOG10_FGW),
        "pulsar_names": np.array(names), "n_pulsars": np.int64(len(names)),
        "pulsar_pos": np.asarray(positions),
        "data_mode": np.array("real"),
        "include_pulsar_term": np.bool_(not earth_term_only),
        "pulsar_term_mask": pt_mask,
        "n_anchors": np.int64(int(pt_mask.sum())),
        "k_sigma": np.float64(k),
        "coherent_fraction": np.float64(coherent_fraction),
        "coherent_sigma_pc": np.float64(coherent_sigma_pc),
        "n_tight": np.int64(n_tight),
        "tight_pulsar_names": np.array([names[i] for i in tight_idx]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    gp = sub.add_parser("generate")
    gp.add_argument("--output", dest="path", type=Path,
                    default=Path("cgw_incoherent_distance_skymap.npz"))
    gp.add_argument("--nside", type=int, default=8)
    gp.add_argument("--pixel-chunk", type=int, default=8)
    gp.add_argument("--k", type=float, default=5.0)
    gp.add_argument("--n-phase", type=int, default=256)
    gp.add_argument("--full", action="store_true",
                    help="Use all pulsars instead of the smoke subset.")
    gp.add_argument("--earth-term-only", action="store_true",
                    help="Drop the pulsar term -> method-matched Earth-term "
                         "baseline (same numerical UL) for the comparison plot.")
    gp.add_argument("--coherent-fraction", type=float, default=0.0,
                    help="Fraction of pulsars (highest PX/sigma_PX) given a tight "
                         "distance prior (coherent). 0 -> all-incoherent.")
    gp.add_argument("--coherent-sigma-pc", type=float, default=0.1,
                    help="Tight-subset distance prior 1-sigma in pc (~0.01 localizes "
                         "the phase; ~1 ~ flat-phase). Sweepable.")
    pp = sub.add_parser("plot")
    pp.add_argument("--input", dest="path", type=Path,
                    default=Path("cgw_incoherent_distance_skymap.npz"))
    args = ap.parse_args()

    if args.cmd == "generate":
        subset = None if args.full else SMOKE_SUBSET
        res = compute_skymap(pulsar_subset=subset, nside=args.nside, k=args.k,
                             n_phase=args.n_phase, pixel_chunk=args.pixel_chunk,
                             earth_term_only=args.earth_term_only,
                             coherent_fraction=args.coherent_fraction,
                             coherent_sigma_pc=args.coherent_sigma_pc)
        np.savez_compressed(args.path, **res)
        print(f"Saved {args.path}")
    else:
        print("Plot the .npz with examples/cgw_earth_vs_pulsar_distance.py "
              "(--pulsar <this npz> --earth <earth-term npz>).")


if __name__ == "__main__":
    main()
