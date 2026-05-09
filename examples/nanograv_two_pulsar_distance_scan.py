"""NANOGrav PTA likelihood: 2D distance scan of the two most-constraining pulsars.

Real-data analogue of `likelihood_contour_pulsar0_vs_pulsar1_distance.ipynb`.
We

1. load the NANOGrav 15-yr narrowband dataset via `jaxpint.load_nanograv_pta`,
2. inject a single continuous-wave (CW) source so distance becomes
   identifiable through the pulsar-term phase,
3. rank pulsars by how sharply the log-likelihood depends on their distance
   (curvature of `single_pulsar_pta_logL` w.r.t. each pulsar's `PX`), and
4. sweep `scan_logL` over a 2D grid of the top-2 pulsars' distances, holding
   everything else at the injected truth.

Data: download the Zenodo tarball (`10.5281/zenodo.8423265`,
`NANOGrav15yr_PulsarTiming_v2.0.0.tar.gz`) and point `DATA_DIR` below at the
extracted `narrowband/` directory.

Both the ranking and the sweep exploit the per-pulsar decomposition of the
PTA likelihood: `pta_logL` is a sum of independent per-pulsar contributions,
so any scan whose axes touch only a few pulsars' parameters can pre-compute
the rest as constants. `jaxpint.pta.scan.scan_logL` does this dependency
analysis automatically; for a 5-pulsar 400x400 distance scan with two
varying pulsars, that's 803 single-pulsar evaluations instead of 800,000.

The pulsar-term phase
    phi_p = phi_e - 2*pi * f_gw * (d/c) * (1 + cos(mu))
makes per-pulsar `logL` periodic in distance with period
    delta_d = c / [f_gw * (1 + cos(mu))].
The 2D map will show a grid of peaks set by the two pulsars' opening angles
to the injected source.

Usage
-----
    python examples/nanograv_two_pulsar_distance_scan.py generate [--output PATH]
    python examples/nanograv_two_pulsar_distance_scan.py plot     [--input  PATH]
    python examples/nanograv_two_pulsar_distance_scan.py both     [--data   PATH]   # default

`generate` runs the GPU scan and writes a compressed `.npz`. `plot` reads the
`.npz` and draws the figures with no JAX/GPU import on the path. `both` is the
end-to-end behaviour the script had before this CLI was added.
"""
from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- Configuration ---------------------------------------------------------
DATA_DIR = Path(
    os.environ.get(
        "JAXPINT_DATA_DIR",
        "/home/hector/NYU/PTA/jax_pint/minish/jpg00017/NANOGrav15yr_PulsarTiming_v2.0.0/narrowband",
    )
).expanduser()
#DATA_DIR = Path("/home/hector/NYU/PTA/jax_pint/ocarina").expanduser()

# CW injection (Earth-frame). Picked to fall inside the PTA band.
TRUE_LOG10_H = -14.0
TRUE_LOG10_FGW = -8.0          # 10 nHz
TRUE_COS_GWTHETA = 0.3
TRUE_GWPHI = 1.7
SEED = 0

# 2D sweep grid (kpc, half-widths around each truth).
HALF_WINDOW_KPC = 1e-3
N_GRID = 400                   # 400 x 400 = 160k logL evals

DEFAULT_DATA_PATH = Path("nanograv_two_pulsar_distance_scan.npz")

# IAU 2006 obliquity of the ecliptic at J2000.0
OBLIQUITY_RAD = np.deg2rad(84381.406 / 3600.0)
COS_EPS = np.cos(OBLIQUITY_RAD)
SIN_EPS = np.sin(OBLIQUITY_RAD)


def pulsar_unit_vector_icrs(pp):
    """ICRS Cartesian unit vector from either RAJ/DECJ or ELONG/ELAT.

    NANOGrav par files use either equatorial (RAJ/DECJ) or ecliptic
    (ELONG/ELAT) coordinates depending on which gives a better-conditioned
    timing fit. CWInjector wants ICRS unit vectors, so we rotate the
    ecliptic ones by the J2000 obliquity (PINT's ELONG/ELAT convention).
    """
    if "RAJ" in pp.names and "DECJ" in pp.names:
        ra = float(pp.param_value("RAJ"))
        dec = float(pp.param_value("DECJ"))
        return np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ])
    if "ELONG" in pp.names and "ELAT" in pp.names:
        elong = float(pp.param_value("ELONG"))
        elat = float(pp.param_value("ELAT"))
        # Ecliptic Cartesian, then rotate +eps about x to reach ICRS.
        x = np.cos(elat) * np.cos(elong)
        y_ec = np.cos(elat) * np.sin(elong)
        z_ec = np.sin(elat)
        return np.array([x, COS_EPS * y_ec - SIN_EPS * z_ec, SIN_EPS * y_ec + COS_EPS * z_ec])
    raise KeyError(
        f"Pulsar params lack both (RAJ, DECJ) and (ELONG, ELAT); names={pp.names}"
    )


class _TeeStream:
    """File-like that fans writes out to several streams.

    Used to redirect stdout/stderr through a tee so terminal output is
    preserved while a copy lands on disk. Relies on the underlying
    streams being line-buffered (we set that in `main`) so a `\\n` is
    flushed to the kernel before the next line is produced — that's
    what survives most crashes.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def fileno(self):
        return self._streams[0].fileno()

    def isatty(self):
        try:
            return self._streams[0].isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


def _attach_log_file(log_path: Path) -> None:
    """Tee stdout/stderr to `log_path` and add a loguru sink alongside.

    Survives Python exceptions and most segfaults (line-buffered file
    handle flushes per newline). It does NOT survive SIGKILL from the
    kernel OOM-killer; nothing in-process can. For that, run with
    `python ... 2>&1 | tee scan.log` at the shell instead.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(log_path, "a", buffering=1, encoding="utf-8")
    sys.stdout = _TeeStream(sys.stdout, fp)
    sys.stderr = _TeeStream(sys.stderr, fp)
    # `enqueue=True` hands records off to a separate writer thread, so
    # log lines flush even if the main thread dies. Goes to a sibling
    # file to avoid two writers fighting over the same fd.
    loguru_path = log_path.with_name(log_path.name + ".loguru")
    from loguru import logger
    logger.add(str(loguru_path), enqueue=True, level="DEBUG")
    print(f"[crash-logging] stdout/stderr -> {log_path}, loguru -> {loguru_path}")


def _phase_cm(records: list, gpu_device, enabled: bool):
    """Build a context manager that records wall/cpu/RSS/GPU stats per phase.

    Each delta is "additional peak observed during this phase". Both
    `ru_maxrss` and JAX's `peak_bytes_in_use` are monotonic, so a phase
    that doesn't push either watermark higher reports a zero — the cost
    showed up in an earlier phase.
    """
    @contextmanager
    def _phase(name: str):
        if not enabled:
            yield
            return
        wall0 = time.perf_counter()
        cpu0 = time.process_time()
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        gpu0 = gpu_device.memory_stats()["peak_bytes_in_use"] if gpu_device is not None else 0
        try:
            yield
        finally:
            gpu1 = gpu_device.memory_stats()["peak_bytes_in_use"] if gpu_device is not None else 0
            records.append({
                "name": name,
                "wall_s": time.perf_counter() - wall0,
                "cpu_s": time.process_time() - cpu0,
                "rss_delta_mb": (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1024.0,
                "gpu_peak_delta_mb": (gpu1 - gpu0) / 1e6,
            })
    return _phase


def _print_phase_summary(records: list) -> None:
    if not records:
        return
    print("\n=== Profile ===")
    header = f"{'phase':<16} {'wall':>9} {'cpu':>9} {'dRSS':>11} {'dGPU peak':>12}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    tot_wall = tot_cpu = tot_rss = tot_gpu = 0.0
    for r in records:
        print(f"{r['name']:<16} "
              f"{r['wall_s']:>8.2f}s {r['cpu_s']:>8.2f}s "
              f"{r['rss_delta_mb']:>8.1f} MB {r['gpu_peak_delta_mb']:>9.1f} MB")
        tot_wall += r["wall_s"]
        tot_cpu += r["cpu_s"]
        tot_rss += r["rss_delta_mb"]
        tot_gpu += r["gpu_peak_delta_mb"]
    print(sep)
    print(f"{'total':<16} "
          f"{tot_wall:>8.2f}s {tot_cpu:>8.2f}s "
          f"{tot_rss:>8.1f} MB {tot_gpu:>9.1f} MB")


def compute_logL_2d(debug: bool = False, profile: bool = False) -> dict:
    """Run the full GPU pipeline and return everything the plot step needs.

    `debug=True` raises the moment a non-finite value is produced
    (`JAX_DEBUG_NANS` + `JAX_DEBUG_INFS`) and turns off JAX's traceback
    filtering so internal frames show up in errors.

    `profile=True` times three phases (load_pta, build_config, scan) and
    prints a wall/cpu/RSS/GPU summary at the end. Forces a host sync on
    the GPU result before stopping the scan clock so the timing reflects
    actual compute, not async dispatch.
    """
    # Disable JAX's default 75% GPU preallocation so the heavy-pulsar
    # transpose's transient autotuner workspace (~700 MB) has room to
    # breathe. Must be set before `import jax`.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if debug:
        # Same rule as the XLA flag above: must be set before `import jax`.
        os.environ["JAX_DEBUG_NANS"] = "1"
#        os.environ["JAX_DEBUG_INFS"] = "1"
        os.environ["JAX_TRACEBACK_FILTERING"] = "off"
        print("[debug] JAX_DEBUG_NANS , traceback filtering OFF")

    import jax
    import jax.numpy as jnp

    from loguru import logger
    logger.disable("pint")

    from jaxpint import load_nanograv_pta
    from jaxpint.pta import PerPulsarScanAxis, scan_logL
    from jaxpint.pta.signals.cw import CWInjector
    from jaxpint.notebook_utils import inject_and_build_config

    gpu_device = jax.devices("gpu")[0]
    jax.config.update("jax_default_device", gpu_device)

    profile_records: list = []
    phase = _phase_cm(profile_records, gpu_device, profile)

    # ---- 1. Load the NANOGrav PTA -----------------------------------------
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"DATA_DIR = {DATA_DIR} does not exist. "
            "Download https://zenodo.org/records/8423265/files/"
            "NANOGrav15yr_PulsarTiming_v2.0.0.tar.gz, extract, and point DATA_DIR "
            "at the extracted narrowband/ directory."
        )

    # Diagnostic subset (~20 pulsars). The full 76-pulsar load needs >15 GB
    # of system RAM combined with GPU preallocation + the first pta_logL
    # compile, which OOMs on smaller workstations. Set PULSAR_SUBSET = None
    # to load all 76.
    # PULSAR_SUBSET = [
    #     "B1855+09",       # BT
    #     "J0023+0923",     # ELL1 + FB orbital -> exercises PB-from-FB synthesis
    #     # ... see git history for the full annotated list ...
    # ]
    PULSAR_SUBSET = None

    with phase("load_pta"):
        psrs = load_nanograv_pta(DATA_DIR, pulsar_names=PULSAR_SUBSET)

    print(f"Loaded {len(psrs.pulsar_names)} pulsars from {DATA_DIR}.")
    for name, td in zip(psrs.pulsar_names[:5], psrs.toa_data_list[:5]):
        print(f"  {name:>14s}: {int(td.mjd_int.shape[0]):>5d} TOAs")
    print(f"  ... ({len(psrs.pulsar_names) - 5} more)")

    # ---- 2. Restrict to pulsars whose `.par` provides a parallax ----------
    candidate_indices = [
        i for i, pp in enumerate(psrs.pulsar_params_list) if "PX" in pp.names
    ]
    print(
        f"{len(candidate_indices)} / {len(psrs.pulsar_names)} pulsars carry PX in their .par file."
    )

    # ---- 3. Inject one CW source and build the `PTAConfig` ----------------
    with phase("build_config"):
        positions_np = np.stack([pulsar_unit_vector_icrs(pp) for pp in psrs.pulsar_params_list])
        positions = jnp.asarray(positions_np)

        n_eq = sum("RAJ" in pp.names for pp in psrs.pulsar_params_list)
        n_ec = sum("ELONG" in pp.names for pp in psrs.pulsar_params_list)
        print(f"Coordinate system: {n_eq} equatorial, {n_ec} ecliptic")

        cw_injector = CWInjector(
            positions,
            prefix="cw_",
            initial_values={
                "log10_h": TRUE_LOG10_H,
                "log10_fgw": TRUE_LOG10_FGW,
                "cos_gwtheta": TRUE_COS_GWTHETA,
                "gwphi": TRUE_GWPHI,
            },
        )

        gp, config = inject_and_build_config(psrs, (cw_injector,))
        pp_tuple = psrs.pulsar_params_list

    print(f"PTAConfig built: {config.n_pulsars} pulsars, {gp.n_params} global params.")

    # ---- 4. Pick the two pulsars to sweep ---------------------------------
    PULSAR_A, PULSAR_B = candidate_indices[0], candidate_indices[1]
    print(
        f"Sweeping over: A = {psrs.pulsar_names[PULSAR_A]} (idx {PULSAR_A}), "
        f"B = {psrs.pulsar_names[PULSAR_B]} (idx {PULSAR_B})."
    )

    # ---- 5. 2D log-likelihood sweep over the two pulsars' distances ------
    true_px_a = float(pp_tuple[PULSAR_A].param_value("PX"))
    true_px_b = float(pp_tuple[PULSAR_B].param_value("PX"))
    true_dist_a = 1.0 / true_px_a
    true_dist_b = 1.0 / true_px_b

    dist_a_grid = np.linspace(true_dist_a - HALF_WINDOW_KPC, true_dist_a + HALF_WINDOW_KPC, N_GRID)
    dist_b_grid = np.linspace(true_dist_b - HALF_WINDOW_KPC, true_dist_b + HALF_WINDOW_KPC, N_GRID)
    px_a_mas_grid = jnp.asarray(1.0 / dist_a_grid)
    px_b_mas_grid = jnp.asarray(1.0 / dist_b_grid)

    print(f"Computing {N_GRID} x {N_GRID} = {N_GRID*N_GRID} log-likelihood "
          f"values via dependency-aware scan_logL...")
    print(f"  pulsar A = {psrs.pulsar_names[PULSAR_A]}, "
          f"pulsar B = {psrs.pulsar_names[PULSAR_B]}")
    print(f"  Other {len(pp_tuple) - 2} pulsars contribute constants computed once.")
    with phase("scan"):
        logL_2d = scan_logL(
            gp, pp_tuple, config,
            axes=[
                PerPulsarScanAxis(pulsar_idx=PULSAR_A, param_name="PX",
                                  values=px_a_mas_grid),
                PerPulsarScanAxis(pulsar_idx=PULSAR_B, param_name="PX",
                                  values=px_b_mas_grid),
            ],
            indexing="xy",  # -> shape (N_GRID, N_GRID) = (n_b, n_a)
            chunk_size=25,
        )
        # Force host sync so the phase timer reflects actual compute, not
        # async dispatch latency.
        logL_2d = np.asarray(logL_2d)
    print(f"Done. Output shape: {logL_2d.shape}")

    _print_phase_summary(profile_records)

    return {
        "logL_2d": logL_2d,
        "dist_a_grid": dist_a_grid,
        "dist_b_grid": dist_b_grid,
        "px_a_mas_grid": np.asarray(px_a_mas_grid),
        "px_b_mas_grid": np.asarray(px_b_mas_grid),
        "true_dist_a": np.float64(true_dist_a),
        "true_dist_b": np.float64(true_dist_b),
        "true_px_a": np.float64(true_px_a),
        "true_px_b": np.float64(true_px_b),
        "pulsar_name_a": np.array(psrs.pulsar_names[PULSAR_A]),
        "pulsar_name_b": np.array(psrs.pulsar_names[PULSAR_B]),
        "pulsar_idx_a": np.int64(PULSAR_A),
        "pulsar_idx_b": np.int64(PULSAR_B),
        "true_log10_h": np.float64(TRUE_LOG10_H),
        "true_log10_fgw": np.float64(TRUE_LOG10_FGW),
        "true_cos_gwtheta": np.float64(TRUE_COS_GWTHETA),
        "true_gwphi": np.float64(TRUE_GWPHI),
        "n_pulsars": np.int64(config.n_pulsars),
        "n_grid": np.int64(N_GRID),
        "half_window_kpc": np.float64(HALF_WINDOW_KPC),
    }


def save_results(path: Path, results: dict) -> None:
    np.savez_compressed(path, **results)
    print(f"Saved scan to {path} ({path.stat().st_size / 1e6:.2f} MB).")


def load_results(path: Path) -> dict:
    """Load a `.npz` written by `save_results`, unwrapping 0-d arrays."""
    data = np.load(path, allow_pickle=False)
    out = {}
    for k in data.files:
        v = data[k]
        out[k] = v.item() if v.ndim == 0 else v
    return out


def plot_results(results: dict) -> None:
    # Lazy import: keeps `plot` mode off the JAX/CUDA init path until
    # `jaxpint.notebook_utils` is actually needed.
    from jaxpint.notebook_utils import plot_2d_delta_logL

    logL_2d = results["logL_2d"]
    dist_a_grid = results["dist_a_grid"]
    dist_b_grid = results["dist_b_grid"]
    px_a_mas_grid = results["px_a_mas_grid"]
    px_b_mas_grid = results["px_b_mas_grid"]
    true_dist_a = results["true_dist_a"]
    true_dist_b = results["true_dist_b"]
    true_px_a = results["true_px_a"]
    true_px_b = results["true_px_b"]
    name_a = results["pulsar_name_a"]
    name_b = results["pulsar_name_b"]

    # ---- 6. 2D contour ----------------------------------------------------
    # distance = 1 / PX is monotonically decreasing in PX, so reverse both
    # grid axes and `logL_2d` to present in ascending kpc.
    fig, ax = plt.subplots(figsize=(9, 8))
    dist_a_plot = dist_a_grid[::-1]
    dist_b_plot = dist_b_grid[::-1]
    logL_2d_plot = logL_2d[::-1, ::-1]

    mesh = plot_2d_delta_logL(ax, dist_a_plot, dist_b_plot, logL_2d_plot)
    ax.plot(true_dist_a, true_dist_b, "r*", markersize=15, label="Injected truth")
    ax.set_xlabel(f"{name_a} distance (kpc)", fontsize=13)
    ax.set_ylabel(f"{name_b} distance (kpc)", fontsize=13)
    ax.set_title(
        f"NANOGrav PTA logL vs distance ({name_a}, {name_b})\n"
        f"CW injected at log10_h={results['true_log10_h']}, "
        f"log10_fgw={results['true_log10_fgw']}",
        fontsize=12,
    )
    fig.colorbar(mesh, ax=ax, label=r"$\Delta$ log-likelihood")
    ax.legend(loc="upper right")
    fig.tight_layout()

    # ---- 7. 1D projections at the other pulsar's true distance ------------
    idx_b_true = int(np.argmin(np.abs(np.asarray(px_b_mas_grid) - true_px_b)))
    idx_a_true = int(np.argmin(np.abs(np.asarray(px_a_mas_grid) - true_px_a)))

    logL_1d_a = logL_2d[idx_b_true, :]
    logL_1d_b = logL_2d[:, idx_a_true]
    delta_a = logL_1d_a - logL_1d_a.max()
    delta_b = logL_1d_b - logL_1d_b.max()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax0.plot(np.asarray(dist_a_grid)[::-1], delta_a[::-1], lw=1.2)
    ax0.axvline(true_dist_a, color="r", ls="--", lw=1, label=f"truth = {true_dist_a:.4f} kpc")
    ax0.set_xlabel(f"{name_a} distance (kpc)")
    ax0.set_ylabel(r"$\Delta$ log-likelihood")
    ax0.set_title(f"{name_a} (slice at {name_b} truth)")
    ax0.legend()

    ax1.plot(np.asarray(dist_b_grid)[::-1], delta_b[::-1], lw=1.2)
    ax1.axvline(true_dist_b, color="r", ls="--", lw=1, label=f"truth = {true_dist_b:.4f} kpc")
    ax1.set_xlabel(f"{name_b} distance (kpc)")
    ax1.set_ylabel(r"$\Delta$ log-likelihood")
    ax1.set_title(f"{name_b} (slice at {name_a} truth)")
    ax1.legend()

    fig.tight_layout()
    plt.show()


def cmd_generate(args: argparse.Namespace) -> None:
    save_results(args.path, compute_logL_2d(debug=args.debug, profile=args.profile))


def cmd_plot(args: argparse.Namespace) -> None:
    plot_results(load_results(args.path))


def cmd_both(args: argparse.Namespace) -> None:
    results = compute_logL_2d(debug=args.debug, profile=args.profile)
    save_results(args.path, results)
    plot_results(results)


def main() -> None:
    # Always-on, cheap crash safety: line-buffered stdout/stderr (so each
    # print() flushes to the kernel on `\n`) and faulthandler for C-level
    # segfaults / aborts (CUDA driver crashes most commonly).
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    faulthandler.enable()
    # Surface the loader's per-pulsar `log.info("Loading %s ...")` so that
    # if construction fails we can see which pulsar was in flight.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="NANOGrav 2-pulsar distance scan: compute, save, and/or plot."
    )
    # Default when no subcommand is given: run the full pipeline (`both`).
    parser.set_defaults(
        func=cmd_both, path=DEFAULT_DATA_PATH, debug=False, profile=False, log=None,
    )
    subparsers = parser.add_subparsers(dest="mode")

    debug_help = (
        "Set JAX_DEBUG_NANS, JAX_DEBUG_INFS, and JAX_TRACEBACK_FILTERING=off "
        "before importing jax. Raises immediately on non-finite values and "
        "leaves JAX-internal frames visible in tracebacks."
    )
    profile_help = (
        "Time three phases (load_pta, build_config, scan) and print a "
        "wall/CPU/RSS/GPU-peak summary. Forces a host sync on the scan "
        "result so timing reflects actual compute, not async dispatch."
    )
    log_help = (
        "Tee stdout/stderr to PATH and add a loguru sink at PATH.loguru "
        "(enqueue=True). Survives Python exceptions and most segfaults; "
        "does NOT survive SIGKILL — for that, pipe to `tee` at the shell."
    )

    p_gen = subparsers.add_parser(
        "generate",
        help="Run the GPU scan and save the result to disk; no plotting.",
    )
    p_gen.add_argument("--output", dest="path", type=Path, default=DEFAULT_DATA_PATH)
    p_gen.add_argument("--debug", action="store_true", help=debug_help)
    p_gen.add_argument("--profile", action="store_true", help=profile_help)
    p_gen.add_argument("--log", type=Path, default=None, help=log_help)
    p_gen.set_defaults(func=cmd_generate)

    p_plot = subparsers.add_parser(
        "plot",
        help="Load a saved scan and plot it; no JAX/GPU required.",
    )
    p_plot.add_argument("--input", dest="path", type=Path, default=DEFAULT_DATA_PATH)
    p_plot.set_defaults(func=cmd_plot)

    p_both = subparsers.add_parser(
        "both",
        help="Run the scan, save the result, and plot in one process.",
    )
    p_both.add_argument("--data", dest="path", type=Path, default=DEFAULT_DATA_PATH)
    p_both.add_argument("--debug", action="store_true", help=debug_help)
    p_both.add_argument("--profile", action="store_true", help=profile_help)
    p_both.add_argument("--log", type=Path, default=None, help=log_help)
    p_both.set_defaults(func=cmd_both)

    args = parser.parse_args()

    if getattr(args, "log", None) is not None:
        _attach_log_file(args.log)

    args.func(args)


if __name__ == "__main__":
    main()
