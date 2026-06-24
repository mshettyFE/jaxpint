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
                   pixel_chunk=8):
    """Compute the distance-reach map. Returns a results dict (see save keys)."""
    import jax
    import jax.numpy as jnp
    from loguru import logger

    from jaxpint import load_nanograv_pta
    from jaxpint.likelihood import single_pulsar_logL
    from jaxpint.bayes import marginalize, ImproperPrior
    from jaxpint.pta.signals.cw import cw_delay_from_array
    from jaxpint.pta.cw_upper_limit import h0_to_distance
    from jaxpint.pta.incoherent_ul import (
        extract_pulsar_bM, flat_phase_grid, h0_95_grid, _A_of_phase,
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
    # Realistic parallaxes span >>1 phase cycle -> flat-phase limit is exact and
    # every pulsar contributes. (distance_phase_grid is available for the rare
    # sub-cycle prior; see tests.)
    phase_grids = jnp.broadcast_to(flat_phase_grid(n_phase),
                                   (len(names), n_phase))
    A = _A_of_phase(flat_phase_grid(n_phase))            # (n_phase, 2)

    b_j = jnp.asarray(b_all); M_j = jnp.asarray(M_all)   # (n_psr, npix, 2[,2])

    def dist_at_pixel(b_px, M_px):                       # b_px (n_psr,2), M_px (n_psr,2,2)
        # adaptive h0_max from the phase-averaged Gaussian proxy (over-covers ~10x)
        Xbar = jnp.mean(jax.vmap(lambda Ba: A @ Ba)(b_px), axis=1)            # (n_psr,)
        Ybar = jnp.mean(jax.vmap(lambda Ma: jnp.einsum("ni,ij,nj->n", A, Ma, A))(M_px), axis=1)
        SX, SY = jnp.sum(Xbar), jnp.clip(jnp.sum(Ybar), 1e-300, None)
        h0_max = 10.0 * (jnp.abs(SX) / SY + 5.0 / jnp.sqrt(SY))
        h0_95 = h0_95_grid(b_px, M_px, phase_grids, h0_max)
        return h0_95

    h0_95 = jax.vmap(dist_at_pixel, in_axes=(1, 1))(b_j, M_j)   # (npix,)
    dist_ll = np.asarray(h0_to_distance(h0_95, LOG10_MC, LOG10_FGW))
    r_eff = float(np.mean(dist_ll ** 3) ** (1.0 / 3.0))
    _log(f"Done. D_L lower limit min/median/max = "
         f"{dist_ll.min():.1f}/{np.median(dist_ll):.1f}/{dist_ll.max():.1f} Mpc; "
         f"R_eff = {r_eff:.2f} Mpc (nside={nside})")

    return {
        "dist_ll_mpc": dist_ll,
        "nside": np.int64(nside), "r_eff_mpc": np.float64(r_eff),
        "chirp_mass_msun": np.float64(10.0 ** LOG10_MC),
        "log10_mc": np.float64(LOG10_MC),
        "f_gw": np.float64(F_GW), "log10_fgw": np.float64(LOG10_FGW),
        "pulsar_names": np.array(names), "n_pulsars": np.int64(len(names)),
        "pulsar_pos": np.asarray(positions),
        "data_mode": np.array("real"),
        "include_pulsar_term": np.bool_(True),
        # all pulsars contribute (distance-marginalized); mark all anchors so the
        # earth-vs-pulsar comparison plot draws them as stars.
        "pulsar_term_mask": np.ones(len(names), dtype=bool),
        "n_anchors": np.int64(len(names)),
        "k_sigma": np.float64(k),
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
    pp = sub.add_parser("plot")
    pp.add_argument("--input", dest="path", type=Path,
                    default=Path("cgw_incoherent_distance_skymap.npz"))
    args = ap.parse_args()

    if args.cmd == "generate":
        subset = None if args.full else SMOKE_SUBSET
        res = compute_skymap(pulsar_subset=subset, nside=args.nside, k=args.k,
                             n_phase=args.n_phase, pixel_chunk=args.pixel_chunk)
        np.savez_compressed(args.path, **res)
        print(f"Saved {args.path}")
    else:
        print("Plot the .npz with examples/cgw_earth_vs_pulsar_distance.py "
              "(--pulsar <this npz> --earth <earth-term npz>).")


if __name__ == "__main__":
    main()
