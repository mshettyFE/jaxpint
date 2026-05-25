"""CGW distance lower-limit sky map — no-MCMC Bayesian approximation of Fig. 8.

Reproduces, approximately, the sky map of the 95% *lower limit on the distance*
to a continuous-GW source of fixed chirp mass and frequency from the NANOGrav
15-yr individual-SMBHB paper (arXiv:2306.16222, Fig. 8), without running MCMC.

Method (see ``jaxpint/pta/cw_upper_limit.py`` for the math):

1. The Earth-term CW residual is exactly linear in the strain ``h0``, so the
   timing-marginalized Gaussian log-likelihood is exactly quadratic in ``h0``:
   ``logL(h0) = logL(0) + h0*X - 0.5*h0**2*Y``.
2. A single CWInjector(linear_amplitude=True, earth_term_only=True) carries
   the sky position / orientation
   / linear amplitude as global params; ``marginalize(pta_logL, ...)`` analytically
   marginalizes the (linear) timing-model parameters once.  Autodiff of the
   marginalized likelihood w.r.t. the amplitude gives ``X=(d|s_hat)`` and
   ``Y=(s_hat|s_hat)`` exactly.
3. The CGW orientation ``(cos_inc, psi, phase0)`` is held *fixed* (first-pass;
   default face-on/optimal), so per sky pixel the posterior is one truncated
   Gaussian and the 95% strain UL is closed form → invert to a distance lower
   limit.  Report ``R_eff = [<D_L^3>]^(1/3)``.

Assumptions: Earth-term only; white noise only (diagonal EFAC/EQUAD; ECORR and
red noise dropped); ``M = 1e9 Msun``, ``f_GW = 27 nHz``; CGW orientation fixed
(not marginalized); timing-model parameters marginalized.

Usage
-----
    python examples/cgw_distance_skymap.py generate [--output PATH] [--nside N] [--full]
    python examples/cgw_distance_skymap.py plot     [--input  PATH]
    python examples/cgw_distance_skymap.py both

Requires the optional 'skymap' extra (healpy + matplotlib):
    uv pip install 'jaxpint[skymap]'
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _log(msg: str) -> None:
    """Flushed progress line (keeps the SLURM .out live under block buffering)."""
    print(msg, flush=True)


def _import_healpy():
    """Import healpy with a clear hint if the optional extra isn't installed."""
    try:
        import healpy as hp
    except ImportError as e:  # pragma: no cover - import-time guard
        raise ImportError(
            "This example needs healpy (HEALPix sky grid + Mollweide plots). "
            "Install the optional extra:  uv pip install 'jaxpint[skymap]'  "
            "(or: pip install healpy matplotlib)."
        ) from e
    return hp

# ---- Configuration ---------------------------------------------------------
# Ocarina par/tim directory. Override via $JAXPINT_OCARINA_DIR (the SLURM job
# points this at the staged /scratch copy); falls back to the local path.
DATA_DIR = Path(
    os.environ.get("JAXPINT_OCARINA_DIR", "/home/hector/NYU/PTA/jax_pint/ocarina")
).expanduser()

# Fixed CGW source (matches Fig. 8).
LOG10_MC = 9.0                      # chirp mass 1e9 Msun
F_GW = 27e-9                        # 27 nHz
LOG10_FGW = float(np.log10(F_GW))

# B1937+21 appears three times in ocarina (combined + per-telescope); keep one.
DROP_PULSARS = {"B1937+21ao", "B1937+21gbt"}

# Small, well-timed default subset for the smoke test (Earth-term needs no PX).
SMOKE_SUBSET = ["J1909-3744", "J1713+0747", "J0613-0200", "J1744-1134"]

# Linear timing-model params to marginalize analytically (improper priors).
# The dominant low-frequency degeneracies; all linear in the residuals.
# Pulsar distance (parallax PX) is deliberately EXCLUDED — it is held fixed at
# its fiducial/par-file value rather than marginalized (first-pass choice).
MARG_PARAMS = {
    "F0", "F1", "RAJ", "DECJ", "ELONG", "ELAT",
    "PMRA", "PMDEC", "PMELONG", "PMELAT",
}

DEFAULT_DATA_PATH = Path("cgw_distance_skymap.npz")

# IAU 2006 obliquity at J2000.0 (for ELONG/ELAT -> ICRS).
_OBLIQ = np.deg2rad(84381.406 / 3600.0)
_COS_EPS, _SIN_EPS = np.cos(_OBLIQ), np.sin(_OBLIQ)


def pulsar_unit_vector_icrs(pp):
    """ICRS Cartesian unit vector from RAJ/DECJ or ELONG/ELAT (PINT convention)."""
    if "RAJ" in pp.names and "DECJ" in pp.names:
        ra, dec = float(pp.param_value("RAJ")), float(pp.param_value("DECJ"))
        return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    if "ELONG" in pp.names and "ELAT" in pp.names:
        elong, elat = float(pp.param_value("ELONG")), float(pp.param_value("ELAT"))
        x = np.cos(elat) * np.cos(elong)
        y_ec, z_ec = np.cos(elat) * np.sin(elong), np.sin(elat)
        return np.array([x, _COS_EPS * y_ec - _SIN_EPS * z_ec, _SIN_EPS * y_ec + _COS_EPS * z_ec])
    raise KeyError(f"Pulsar lacks (RAJ,DECJ) and (ELONG,ELAT): {pp.names}")


# CGW orientation (cos_inc, psi, phase0) held fixed for this first-pass map,
# rather than marginalized. Face-on (cos_inc=1) maximizes the antenna response,
# so this is the *optimal*-orientation reach (an optimistic bound); the
# orientation-averaged limit would be a factor of a few shallower. Override via
# compute_skymap(orientation=...).
FIXED_ORIENTATION = (1.0, 0.0, 0.0)  # cos_inc, psi, phase0


def compute_skymap(
    *,
    pulsar_subset=SMOKE_SUBSET,
    nside=8,
    orientation=FIXED_ORIENTATION,
    validate_linearity=False,
    data_mode="expected",
    pixel_chunk=64,
):
    """Compute the 95% distance lower-limit sky map. Returns a results dict.

    The linear timing-model parameters are marginalized analytically (improper
    priors), but the CGW orientation is *fixed* to ``orientation`` =
    (cos_inc, psi, phase0) rather than marginalized — a first-pass sensitivity.
    With a single orientation the per-pixel posterior is one truncated Gaussian,
    so the limit is closed form (:func:`h0_95_closed_form`).

    data_mode:
      - "expected": set the matched filter X=(d|s_hat)=0, so the limit depends
        only on Y=(s_hat|s_hat). Gives the *expected* (noise-realization-
        independent) sensitivity sky map. REQUIRED here because the synthetic
        ocarina TOAs contain red noise that the white-noise-only model does not
        account for; the real-residual matched filter would otherwise be
        dominated by that unmodeled red noise (huge spurious "detections").
      - "real": use X from the actual residuals. Only meaningful once the noise
        model matches the data (e.g. red noise included).

    pixel_chunk: number of sky pixels vmapped together per chunk (the rest are
    scanned). Trades memory for speed — raise it to go faster if memory allows,
    lower it if the per-chunk grad tape OOMs on the full PTA.
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
    from jaxpint.noise.noise_model import NoiseModel
    from jaxpint.bayes import ImproperPrior, marginalize
    from jaxpint.pta.signals.cw import CWInjector
    from jaxpint.pta.cw_upper_limit import (
        quadratic_coeffs, h0_95_closed_form, h0_to_distance,
    )

    # ---- 1. Load + filter --------------------------------------------------
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"DATA_DIR {DATA_DIR} not found.")
    psrs = load_nanograv_pta(DATA_DIR, pulsar_names=pulsar_subset)
    keep = [i for i, n in enumerate(psrs.pulsar_names) if n not in DROP_PULSARS]
    names = [psrs.pulsar_names[i] for i in keep]
    toa_list = tuple(psrs.toa_data_list[i] for i in keep)
    pp_list = tuple(psrs.pulsar_params_list[i] for i in keep)
    tm_list = tuple(psrs.timing_models[i] for i in keep)
    # White-noise-only: keep each pulsar's EFAC/EQUAD, drop correlated (ECORR/red).
    nm_list = tuple(
        NoiseModel(white_noise=psrs.noise_models[i].white_noise, correlated=())
        for i in keep
    )
    n_toa_total = int(sum(int(td.n_toas) for td in toa_list))
    _log(f"Loaded {len(names)} pulsars, {n_toa_total} TOAs total.")

    positions = jnp.asarray(np.stack([pulsar_unit_vector_icrs(pp) for pp in pp_list]))

    # ---- 2. Injector + config + global params ------------------------------
    # Earth-term-only, linear-amplitude CW template: residual linear in h0 so
    # logL is exactly quadratic in it (the analytic-UL requirement).
    injector = CWInjector(
        positions, prefix="cw_", earth_term_only=True, linear_amplitude=True,
        initial_values={"log10_fgw": LOG10_FGW},
    )
    gp = injector.register_params(GlobalParams.empty())
    config = PTAConfig(
        toa_data_list=toa_list, timing_models=tm_list,
        noise_models=nm_list, signal_injectors=(injector,),
    )

    # ---- 3. Timing-model marginalization (improper priors) -----------------
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

    # ---- 4. logL(amp) closure with sky/orientation as globals --------------
    idx = {k: gp._name_to_index[f"cw_{k}"] for k in
           ("h0", "cos_gwtheta", "gwphi", "cos_inc", "psi", "phase0")}
    base_vals = gp.values

    def logL_at(amp, cos_gwtheta, gwphi, cos_inc, psi, phase0):
        v = (base_vals
             .at[idx["h0"]].set(amp)
             .at[idx["cos_gwtheta"]].set(cos_gwtheta)
             .at[idx["gwphi"]].set(gwphi)
             .at[idx["cos_inc"]].set(cos_inc)
             .at[idx["psi"]].set(psi)
             .at[idx["phase0"]].set(phase0))
        gp_new = eqx.tree_at(lambda gg: gg.values, gp, v)
        return g(gp_new, reduced_pp)

    cos_inc_fix, psi_fix, phase0_fix = (float(x) for x in orientation)

    def xy_for_pixel(cos_gwtheta, gwphi):
        f = lambda amp: logL_at(amp, cos_gwtheta, gwphi, cos_inc_fix, psi_fix, phase0_fix)
        return quadratic_coeffs(f)  # scalar X, Y at the fixed orientation

    # ---- 5. HEALPix sky grid (RING ordering, exactly equal-area) -----------
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))   # colatitude, longitude
    sky = jnp.stack([jnp.cos(jnp.asarray(theta)), jnp.asarray(phi)], axis=1)  # (npix, 2)

    # Chunked vmap over pixels: lax.map(batch_size=...) vectorizes `pixel_chunk`
    # pixels at a time and scans over the chunks, so the grad-of-likelihood tape
    # is replicated only chunk-wide (bounds memory) while removing per-pixel
    # dispatch. A plain vmap over all npix would OOM on the full PTA.
    @jax.jit
    def all_xy(sky_arr):
        return jax.lax.map(
            lambda row: xy_for_pixel(row[0], row[1]),
            sky_arr, batch_size=pixel_chunk,
        )

    _log(f"nside={nside} -> {npix} HEALPix pixels, fixed orientation "
         f"(cos_inc={cos_inc_fix}, psi={psi_fix}, phase0={phase0_fix}), "
         f"data_mode={data_mode}; chunked vmap (batch={pixel_chunk}), compiling...")

    Xs, Ys = all_xy(sky)                              # (npix,), (npix,) on device
    if data_mode == "expected":
        Xs = jnp.zeros_like(Xs)                       # X=0: noise-only sensitivity

    # ---- 6. Limits + distances, vectorized over the whole map (one transfer) -
    h0_95 = h0_95_closed_form(Xs, Ys)                 # elementwise truncated-Gaussian UL
    dist_ll = np.asarray(h0_to_distance(h0_95, LOG10_MC, LOG10_FGW))

    # HEALPix pixels are exactly equal-area, so R_eff = <D_L^3>^(1/3) is exact.
    r_eff = float(np.mean(dist_ll ** 3) ** (1.0 / 3.0))
    _log(f"Done. D_L lower limit min/median/max = "
         f"{dist_ll.min():.1f}/{np.median(dist_ll):.1f}/{dist_ll.max():.1f} Mpc; "
         f"R_eff = {r_eff:.2f} Mpc (nside={nside}, data_mode={data_mode})")

    # dist_ll_mpc is a HEALPix map (RING ordering); nside lets plot use hp.mollview.
    # pulsar_pos (ICRS unit vectors) lets plot_results_with_pulsars overlay them.
    return {
        "dist_ll_mpc": dist_ll,
        "nside": np.int64(nside), "r_eff_mpc": np.float64(r_eff),
        # Fixed CGW source parameters (for plot annotation), in handy units:
        "chirp_mass_msun": np.float64(10.0 ** LOG10_MC),
        "log10_mc": np.float64(LOG10_MC),
        "f_gw": np.float64(F_GW), "log10_fgw": np.float64(LOG10_FGW),
        "cos_inc": np.float64(cos_inc_fix), "psi": np.float64(psi_fix),
        "phase0": np.float64(phase0_fix), "orientation": np.array(orientation),
        "pulsar_names": np.array(names), "n_pulsars": np.int64(len(names)),
        "pulsar_pos": np.asarray(positions),
        "data_mode": np.array(data_mode),
    }


def save_results(path: Path, results: dict) -> None:
    np.savez_compressed(path, **results)
    print(f"Saved {path} ({path.stat().st_size / 1e3:.1f} kB).")


def load_results(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {k: (v.item() if v.ndim == 0 else v) for k, v in data.items()}


def plot_results(results: dict) -> None:
    hp = _import_healpy()
    import matplotlib.pyplot as plt

    dist = results["dist_ll_mpc"]   # HEALPix map, RING ordering
    hp.mollview(
        dist,
        title=(
            f"95% lower limit on $D_L$  ($\\mathcal{{M}}=10^9 M_\\odot$, $f=27$ nHz)\n"
            f"$R_{{eff}}$={float(results['r_eff_mpc']):.1f} Mpc, "
            f"{int(results['n_pulsars'])} pulsars"
        ),
        unit="$D_L$ lower limit [Mpc]",
        cmap="viridis",
        # Center on RA=12h (180 deg); with healpy's default astro flip RA then
        # increases to the left, matching NANOGrav Fig 8's convention.
        rot=[180, 0],
    )
    hp.graticule()
    plt.savefig("cgw_distance_skymap.png", dpi=130, bbox_inches="tight")
    print("Wrote cgw_distance_skymap.png")


def _positions_from_par(names, data_dir=DATA_DIR):
    """ICRS unit vectors for ``names``, read from ocarina .par files (no TOAs).

    Fallback for plotting an .npz that predates the ``pulsar_pos`` field. Reads
    only the par files (sky position is all we need), so it's fast and doesn't
    touch the .tim data.
    """
    import pint.models as pm
    from jaxpint.bridge import pint_model_to_params

    par_dir = Path(data_dir) / "par"
    out = []
    for name in names:
        # Match the par whose stem (before the first '_') equals the pulsar name,
        # mirroring the loader's keying so B1937+21 doesn't grab its variants.
        cands = [p for p in sorted(par_dir.glob(f"{name}*.par"))
                 if p.stem.split("_", 1)[0] == name]
        if not cands:
            raise FileNotFoundError(f"no par file for {name!r} under {par_dir}")
        pp = pint_model_to_params(pm.get_model(str(cands[0]))).params
        out.append(pulsar_unit_vector_icrs(pp))
    return np.stack(out)


def plot_results_with_pulsars(results: dict, data_dir=None,
                              output: str = "cgw_distance_skymap_pulsars.png") -> None:
    """Like :func:`plot_results`, but overlays each pulsar as a red star.

    Pulsar positions come from ``results['pulsar_pos']`` when present (saved by
    :func:`compute_skymap`); for older .npz files lacking that field they are
    read from the ocarina par files via :func:`_positions_from_par` (pass
    ``data_dir`` or rely on ``$JAXPINT_OCARINA_DIR`` / the default).
    """
    hp = _import_healpy()
    import matplotlib.pyplot as plt

    dist = results["dist_ll_mpc"]   # HEALPix map, RING ordering
    if "pulsar_pos" in results:
        pos = np.atleast_2d(np.asarray(results["pulsar_pos"]))
    else:
        names = [str(n) for n in np.atleast_1d(results["pulsar_names"])]
        pos = _positions_from_par(names, data_dir or DATA_DIR)

    # ICRS unit vector (x, y, z) -> healpy (theta=colatitude, phi=longitude).
    theta = np.arccos(np.clip(pos[:, 2], -1.0, 1.0))
    phi = np.arctan2(pos[:, 1], pos[:, 0])

    hp.mollview(
        dist,
        title=(
            f"95% lower limit on $D_L$  ($\\mathcal{{M}}=10^9 M_\\odot$, $f=27$ nHz)\n"
            f"$R_{{eff}}$={float(results['r_eff_mpc']):.1f} Mpc, "
            f"{int(results['n_pulsars'])} pulsars (red stars)"
        ),
        unit="$D_L$ lower limit [Mpc]",
        cmap="viridis",
        # Center on RA=12h (180 deg); with healpy's default astro flip RA then
        # increases to the left, matching NANOGrav Fig 8's convention.
        rot=[180, 0],
    )
    hp.graticule()
    # projscatter draws onto the current mollview projection (theta/phi in rad).
    hp.projscatter(theta, phi, marker="*", s=120, color="red",
                   edgecolors="black", linewidths=0.5, zorder=5)
    plt.savefig(output, dpi=130, bbox_inches="tight")
    print(f"Wrote {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.set_defaults(mode="both")
    sub = p.add_subparsers(dest="mode")
    for name in ("generate", "both"):
        sp = sub.add_parser(name)
        sp.add_argument("--output", dest="path", type=Path, default=DEFAULT_DATA_PATH)
        sp.add_argument("--nside", type=int, default=8,
                        help="HEALPix nside (npix = 12*nside^2; default 8 -> 768 pixels).")
        sp.add_argument("--pixel-chunk", type=int, default=64,
                        help="Pixels vmapped per chunk (memory<->speed; default 64).")
        sp.add_argument("--full", action="store_true",
                        help="Use all pulsars (drop B1937+21 variants) instead of the smoke subset.")
        sp.add_argument("--validate-linearity", action="store_true")
        sp.add_argument("--data-mode", choices=("expected", "real"), default="expected",
                        help="'expected': X=(d|s_hat)=0, noise-realization-independent "
                             "sensitivity (use this — ocarina has unmodeled red noise). "
                             "'real': matched filter from residuals (needs matching noise model).")
    sp = sub.add_parser("plot")
    sp.add_argument("--input", dest="path", type=Path, default=DEFAULT_DATA_PATH)
    sp.add_argument("--pulsars", action="store_true",
                    help="Overlay pulsar locations as red stars.")

    args = p.parse_args()
    if args.mode == "plot":
        results = load_results(args.path)
        if getattr(args, "pulsars", False):
            plot_results_with_pulsars(results)
        else:
            plot_results(results)
        return

    subset = None if getattr(args, "full", False) else SMOKE_SUBSET
    results = compute_skymap(
        pulsar_subset=subset, nside=args.nside, pixel_chunk=args.pixel_chunk,
        validate_linearity=args.validate_linearity, data_mode=args.data_mode,
    )
    save_results(args.path, results)
    if args.mode == "both":
        plot_results(results)


if __name__ == "__main__":
    main()
