#!/usr/bin/env python3
"""Build the ocarina synthetic dataset.

For each pulsar in the NANOGrav 15yr narrowband release:

1. Strip its .par file to bare-minimum: sky location + spindown + noise.
   Drop binary, DM/DMX, FD, JUMP, chromatic/wavex, glitches.
2. Generate synthetic TOAs at the *real* per-pulsar sampling — the actual epochs,
   frequencies, receivers and per-TOA errors from the source .tim — placed on
   pulse against the stripped model (PINT's leap-aware clock chain, so they
   round-trip on reload). JaxPINT's white-noise model (per-receiver EFAC/EQUAD on
   the real errors) then adds heteroscedastic white noise.

Outputs:
    ocarina/par/<orig>.par   76 stripped .par files
    ocarina/tim/<orig>.tim   76 synthetic .tim files
"""

from __future__ import annotations

import argparse
import io
from glob import glob
from pathlib import Path

import astropy.units as u
import jax
import numpy as np
import pint.models as pm
import pint.simulation as psim
from astropy.time import TimeDelta

from jaxpint.bridge import build_timing_model, pint_model_to_params, pint_toas_to_jax
from jaxpint.simulation import simulate_noise


REPO = Path("/home/hector/NYU/PTA/jax_pint")
SRC = REPO / "minish/jpg00017/NANOGrav15yr_PulsarTiming_v2.0.0/narrowband"
DST = REPO / "ocarina_2"


KEEP_TOKENS = {
    # metadata / SSB transform configuration
    "PSR",
    "EPHEM",
    "CLOCK",
    "CLK",
    "UNITS",
    "TIMEEPH",
    "T2CMETHOD",
    "DILATEFREQ",
    "DMDATA",
    "ECL",
    "START",
    "FINISH",
    "POSEPOCH",
    "PEPOCH",
    "TZRMJD",
    "TZRSITE",
    "TZRFRQ",
    "INFO",
    "MODE",
    "NTOA",
    "CHI2",
    # astrometry (Roemer)
    "RAJ",
    "DECJ",
    "ELONG",
    "ELAT",
    "PMRA",
    "PMDEC",
    "PMELONG",
    "PMELAT",
    "PX",
    # noise model: white (EFAC/EQUAD) + red spin noise (RNAMP/RNIDX or TNRed*),
    # so red-noise-dominated pulsars get their realistic low-frequency limit.
    "EFAC",
    "EQUAD",
    "RNAMP",
    "RNIDX",
    "TNRedAmp",
    "TNRedGam",
    "TNRedC",
    # ECORR (per-epoch white jitter) still dropped; uncomment to include it too.
    # "ECORR",
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


def synthesize(par_text: str, tim_path: Path, key):
    """Return a PINT TOAs object: real-sampling placement + noise realization.

    PINT generates fake TOAs at the *real* source-.tim sampling — the actual
    observation epochs, frequencies, receivers (``-f`` flags) and per-TOA errors —
    placed on integer pulse phase against the stripped model with its full,
    leap-second-aware clock chain (so the written MJDs round-trip on reload).
    Using the real sampling gives the array its realistic, heterogeneous sky
    sensitivity instead of the flat response of a uniform 1 us cadence.

    The noise is drawn from JaxPINT's noise model over every component the par
    declares — white (per-receiver EFAC/EQUAD on the real errors) plus red spin
    noise (PLRedNoise from RNAMP/RNIDX) — and applied via PINT's leap-correct
    writer. NOTE: real-mode analysis must model the SAME components (white + red);
    a white-only covariance would read the injected red noise as a spurious signal.
    """
    pint_model = pm.get_model(io.StringIO(par_text))
    pint_toas = psim.make_fake_toas_fromtim(
        str(tim_path),
        model=pint_model,
        add_noise=False,
        add_correlated_noise=False,
    )
    toa_data = pint_toas_to_jax(pint_toas, model=pint_model)
    params = pint_model_to_params(pint_model).params
    _, noise_model = build_timing_model(pint_model, pint_toas)
    noise_s = np.asarray(
        simulate_noise(toa_data, params, key, tuple(noise_model.components))
    )
    pint_toas.adjust_TOAs(TimeDelta(noise_s * u.s))
    return pint_toas


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
            key = jax.random.fold_in(base_key, i)
            fake_toas = synthesize(simplified, tim_path, key)
            tim_out = out_dir / "tim" / tim_path.name
            fake_toas.write_TOA_file(str(tim_out), format="tempo2")
            print(f"[{i + 1:2d}/{len(par_files)}] {psr_name}: N={fake_toas.ntoas}")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[{i + 1:2d}/{len(par_files)}] {psr_name}: FAILED — {msg}")
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
        "--seed",
        type=int,
        default=2501,
        help="PRNG seed for the noise draw (default: 2501)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DST,
        help=f"output dataset directory (default: {DST})",
    )
    args = parser.parse_args()
    main(args.seed, args.out)
