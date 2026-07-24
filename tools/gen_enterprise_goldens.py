"""Regenerate the committed enterprise-generated golden references.

The dependency-free test suite pins JaxPINT outputs against **enterprise's
own numbers**, frozen into ``tests/data/enterprise_goldens.json``.  This
script is the single place those numbers come from: each ``SPECS`` entry
names the enterprise function, the parameters, and the evaluation grid, and
the committed JSON records all three alongside the values — so a test and
its golden can never silently disagree about what was run (the loader,
:func:`tests.helpers.enterprise_golden`, cross-checks them).

Unlike ``gen_tempo2_goldens.py`` there is no subprocess machinery: enterprise
is a pure-Python import (``pip install jaxpint[enterprise]``) needed only to
RUN THIS SCRIPT, never to run the tests.  Goldens are regenerated
deliberately — never at test time, so an enterprise version bump cannot
silently move the reference under the suite (drift against *current*
enterprise is the job of the live tests/enterprise_checks/ suite).

Usage::

    python tools/gen_enterprise_goldens.py

then commit the rewritten JSON and say why in the commit message.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "data"
OUT_FILE = OUT_PATH / "enterprise_goldens.json"

# The PSD-weight fixture grid shared with tests/test_spectral_models.py:
# n_comp linearly spaced Fourier bins k/T. The grid parameters are recorded
# in every entry and re-asserted by the loader, so the test fixture and this
# script cannot drift apart unnoticed.
PSD_GRID = {"t_span_s": 1.0e8, "n_components": 5}

# name -> (enterprise gp_priors function, parameters). Parameters are passed
# verbatim as keyword arguments; the interleaved (sin, cos) frequency array
# is the only positional argument. Enterprise folds df and the sin/cos
# repeat into the returned weights, so entries compare directly against
# SpectralModel.psd_weights output.
SPECS: dict[str, dict] = {
    "powerlaw_weights": {
        "function": "powerlaw",
        "grid": PSD_GRID,
        "params": {
            "log10_A": -14.0,
            "gamma": 4.33,
        },
    },
    "turnover_weights": {
        "function": "turnover",
        "grid": PSD_GRID,
        "params": {
            "log10_A": -14.0,
            "gamma": 4.33,
            "lf0": -8.5,
            "kappa": 10.0 / 3.0,
            "beta": 0.5,
        },
    },
    "turnover_knee_weights": {
        "function": "turnover_knee",
        "grid": PSD_GRID,
        "params": {
            "log10_A": -14.0,
            "gamma": 4.33,
            "lfb": -8.65,
            "lfk": -7.5,
            "kappa": 10.0 / 3.0,
            "delta": -1.0,
        },
    },
}


def _psd_freqs(grid: dict) -> np.ndarray:
    """Interleaved (sin, cos) frequency array for the enterprise call."""
    freqs = np.arange(1, grid["n_components"] + 1) / grid["t_span_s"]
    return np.repeat(freqs, 2)


def generate() -> dict:
    import enterprise
    from enterprise.signals import gp_priors

    out: dict[str, object] = {
        "_meta": {
            "generator": "tools/gen_enterprise_goldens.py",
            "enterprise_version": getattr(enterprise, "__version__", "unknown"),
        }
    }
    for name, spec in SPECS.items():
        fn = getattr(gp_priors, spec["function"])
        # json floats round-trip float64 exactly (shortest-repr).
        vals = [
            float(v) for v in np.asarray(fn(_psd_freqs(spec["grid"]), **spec["params"]))
        ]
        out[name] = {
            "function": f"enterprise.signals.gp_priors.{spec['function']}",
            "grid": spec["grid"],
            "params": spec["params"],
            "values": vals,
        }
        print(f"{name}: {len(vals)} values from {spec['function']}")
    return out


def main() -> None:
    goldens = generate()
    OUT_FILE.write_text(json.dumps(goldens, indent=1) + "\n")
    print(f"wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
