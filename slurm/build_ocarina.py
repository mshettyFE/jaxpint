#!/usr/bin/env python3
"""Build the ocarina synthetic dataset.

For each pulsar in the NANOGrav 15yr narrowband release:

1. Strip its .par file to bare-minimum: sky location + spindown + noise.
   Drop binary, DM/DMX, FD, JUMP, chromatic/wavex, glitches.
2. Generate N uniform-cadence synthetic TOAs (1 µs error, 1400 MHz, GBT)
   using JaxPINT's make_fake_toas with the original noise model.

Outputs:
    ocarina/par/<orig>.par   76 stripped .par files
    ocarina/tim/<orig>.tim   76 synthetic .tim files
"""
from __future__ import annotations

import argparse
import io
from collections import Counter
from glob import glob
from pathlib import Path

import astropy.units as u
import jax
import numpy as np
import pint.models as pm
import pint.simulation as psim

from jaxpint.bridge import build_timing_model, pint_model_to_params, pint_toas_to_jax
from jaxpint.simulation import make_fake_toas


REPO = Path("/home/hector/NYU/PTA/jax_pint")
SRC = REPO / "minish/jpg00017/NANOGrav15yr_PulsarTiming_v2.0.0/narrowband"
DST = REPO / "ocarina_2"


KEEP_TOKENS = {
    # metadata / SSB transform configuration
    "PSR", "EPHEM", "CLOCK", "CLK", "UNITS",
    "TIMEEPH", "T2CMETHOD", "DILATEFREQ", "DMDATA",
    "ECL", "START", "FINISH", "POSEPOCH", "PEPOCH",
    "TZRMJD", "TZRSITE", "TZRFRQ", "INFO", "MODE", "NTOA", "CHI2",
    # astrometry (Roemer)
    "RAJ", "DECJ", "ELONG", "ELAT",
    "PMRA", "PMDEC", "PMELONG", "PMELAT", "PX",
    # noise model
    "EFAC", "EQUAD",
    # white-only: ECORR + red noise dropped so ocarina_2 matches ocarina's model
    # "ECORR",
    # "RNAMP", "RNIDX", "TNRedAmp", "TNRedGam", "TNRedC",
}


def _is_spin_token(tok: str) -> bool:
    return len(tok) >= 2 and tok[0] == "F" and tok[1:].isdigit()


def strip_par(par_path: Path) -> str:
    out: list[str] = []
    saw_planet_shapiro = False
    saw_correct_tropo = False
    for raw in par_path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        tok = stripped.split()[0]
        if tok == "PLANET_SHAPIRO":
            out.append(f"{tok:<28}N")
            saw_planet_shapiro = True
            continue
        if tok == "CORRECT_TROPOSPHERE":
            out.append(f"{tok:<28}N")
            saw_correct_tropo = True
            continue
        if tok in KEEP_TOKENS or _is_spin_token(tok):
            out.append(line)
    if not saw_planet_shapiro:
        out.append(f"{'PLANET_SHAPIRO':<28}N")
    if not saw_correct_tropo:
        out.append(f"{'CORRECT_TROPOSPHERE':<28}N")
    return "\n".join(out) + "\n"


def inspect_tim(tim_path: Path) -> tuple[int, str]:
    n = 0
    flag_counter: Counter[str] = Counter()
    with tim_path.open() as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            head = stripped.split(maxsplit=1)[0]
            if head in ("FORMAT", "MODE", "SKIP", "NOSKIP", "INCLUDE", "JUMP", "C"):
                continue
            if head.startswith("#"):
                continue
            tokens = stripped.split()
            if "-cut" in tokens:
                continue
            n += 1
            for i, t in enumerate(tokens):
                if t == "-f" and i + 1 < len(tokens):
                    flag_counter[tokens[i + 1]] += 1
                    break
    if not flag_counter:
        return n, ""
    return n, flag_counter.most_common(1)[0][0]


def synthesize(par_text: str, n_toas: int, dominant_flag: str, key):
    pint_model = pm.get_model(io.StringIO(par_text))
    start = float(pint_model.START.value)
    finish = float(pint_model.FINISH.value)
    flags = {"f": dominant_flag} if dominant_flag else None
    pint_toas = psim.make_fake_toas_uniform(
        startMJD=start, endMJD=finish, ntoas=n_toas,
        model=pint_model, error=1.0 * u.us,
        freq=1400.0 * u.MHz, obs="gbt",
        add_noise=False, add_correlated_noise=False,
        flags=flags,
    )
    toa_data = pint_toas_to_jax(pint_toas, model=pint_model)
    params = pint_model_to_params(pint_model).params
    jax_model, noise_model = build_timing_model(pint_model, pint_toas)
    synthetic = make_fake_toas(
        jax_model, toa_data, params, key,
        noise_components=tuple(noise_model.components),
    )
    return synthetic, pint_model


def _format_mjd(whole: int, frac: float) -> str:
    """Format an int/frac MJD pair as <int>.<17 fractional digits>."""
    while frac < 0.0:
        whole -= 1
        frac += 1.0
    while frac >= 1.0:
        whole += 1
        frac -= 1.0
    frac_str = f"{frac:.17f}"
    if frac_str.startswith("1."):
        whole += 1
        digits = "0" * 17
    else:
        digits = frac_str.split(".", 1)[1][:17].ljust(17, "0")
    return f"{whole}.{digits}"


def write_tim(path: Path, synthetic, pint_model, dominant_flag: str) -> None:
    psr = pint_model.PSR.value
    mjd_int = np.asarray(synthetic.mjd_int)
    mjd_frac = np.asarray(synthetic.mjd_frac)
    flag_part = f"-f {dominant_flag}" if dominant_flag else ""
    lines = ["FORMAT 1"]
    for i in range(int(synthetic.n_toas)):
        mjd_str = _format_mjd(int(mjd_int[i]), float(mjd_frac[i]))
        line = f" {psr}  1400.000000  {mjd_str}  1.000  gbt {flag_part}".rstrip()
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")


def main(seed: int, out_dir: Path):
    (out_dir / "par").mkdir(parents=True, exist_ok=True)
    (out_dir / "tim").mkdir(parents=True, exist_ok=True)

    # Record the seed so the noise draw is reproducible from the dataset alone.
    (out_dir / "seed.txt").write_text(f"{seed}\n")

    par_files = sorted(glob(str(SRC / "par" / "*.par")))
    print(f"Processing {len(par_files)} pulsars into {out_dir} (seed={seed})")

    base_key = jax.random.PRNGKey(seed)
    failures: list[tuple[str, str]] = []

    for i, par_str in enumerate(par_files):
        par_path = Path(par_str)
        tim_path = SRC / "tim" / par_path.name.replace(".par", ".tim")
        psr_name = par_path.stem.split("_")[0]
        try:
            simplified = strip_par(par_path)
            (out_dir / "par" / par_path.name).write_text(simplified)
            n_toas, dominant_flag = inspect_tim(tim_path)
            key = jax.random.fold_in(base_key, i)
            synthetic, pint_model = synthesize(simplified, n_toas, dominant_flag, key)
            tim_out = out_dir / "tim" / tim_path.name
            write_tim(tim_out, synthetic, pint_model, dominant_flag)
            print(f"[{i+1:2d}/{len(par_files)}] {psr_name}: N={n_toas} -f={dominant_flag}")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[{i+1:2d}/{len(par_files)}] {psr_name}: FAILED — {msg}")
            failures.append((psr_name, msg))

    if failures:
        print(f"\n{len(failures)} failures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")
    else:
        print("\nAll pulsars processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, default=2501,
        help="PRNG seed for the noise draw (default: 2501)",
    )
    parser.add_argument(
        "--out", type=Path, default=DST,
        help=f"output dataset directory (default: {DST})",
    )
    args = parser.parse_args()
    main(args.seed, args.out)
