# JaxPINT

JAX-accelerated pulsar timing — a port of [PINT](https://github.com/nanograv/PINT) built on [JAX](https://github.com/jax-ml/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

## Overview

JaxPINT reimplements PINT's pulsar timing model as pure JAX code, enabling:

- **Automatic differentiation** — design matrices (Jacobians) are computed via `jax.jacobian` instead of hand-coded partial derivatives
- **JIT compilation** — all timing model evaluations compile to optimized XLA kernels
- **GPU acceleration** — switch from CPU to GPU by changing a single JAX config flag
- **Composable components** — delays, phases, and noise sources are Equinox modules that compose as JAX pytrees

JaxPINT targets numerical compatibility with PINT while providing a functional, differentiable interface for parameter fitting and inference.

## Comparison with PINT

JaxPINT began as a JAX layer on top of PINT, but the `.par`/`.tim` loading path has since been ported, so **PINT is now an optional dependency** (`pip install jaxpint[pint]`). The two libraries still serve complementary roles:

| | PINT | JaxPINT |
|---|---|---|
| **I/O** | Reads `.par`/`.tim` files, handles units via Astropy | Native `.par`/`.tim` parser (TEMPO2 format); optional PINT bridge for other formats |
| **Ephemeris & clock** | Bundles observatory + clock data, computes barycentric corrections | Native clock corrections, TT→TDB, and barycentric positions (Astropy/ERFA + JPL kernels), with auto-updating IPTA clock data |
| **Timing model** | Object-oriented, mutable state, Astropy units throughout | Pure functional, immutable Equinox modules, plain float64 arrays |
| **Derivatives** | Hand-coded analytical partial derivatives | Automatic via `jax.jacobian` |
| **Fitting** | WLS/GLS with numpy | WLS/GLS with JAX (JIT-compiled, autodiff design matrices) |
| **Hardware** | CPU only | CPU or GPU via JAX |

**Typical workflow:** Parse `.par`/`.tim` natively into JaxPINT types, then fit or simulate entirely in JAX. PINT is only pulled in for the optional bridge adapters (converting an in-memory PINT model, or reading legacy TOA formats the native parser doesn't handle).

## Installation

Requires Python >= 3.12 and the package manager uv.

Pick exactly one JAX flavor (`cuda` or `cpu`), and add `dev` if you plan to hack on the code:

| Use case | Command |
|---|---|
| GPU workstation, read-only | `uv sync --extra cuda` |
| GPU workstation, developing | `uv sync --extra cuda --extra dev` |
| CPU laptop, read-only | `uv sync --extra cpu` |
| CPU laptop, developing | `uv sync --extra cpu --extra dev` |

The native `.par`/`.tim` path needs none of PINT. Add `--extra pint` (or `pip install jaxpint[pint]`) only if you want the optional PINT-bridge adapters; `--extra dev` already includes it.

## Quick Start

No PINT required — JaxPINT parses the files natively (the `.tim` must be TEMPO2 format):

```python
import jaxpint.par as par
from jaxpint import native, build_model, WLSFitter

# Parse and build entirely in JaxPINT
parsed = par.get_model("pulsar.par")               # ParResult: parameters + components
toa_data = native.get_TOAs("pulsar.tim", parsed)   # TOAData (clock-corrected, barycentered)
timing_model, noise_model = build_model(parsed, toa_data)   # TimingModel + NoiseModel

# `parsed.params` is the ParameterVector — the only differentiable leaf of the model
fitter = WLSFitter(timing_model, toa_data, parsed.params, noise_model=noise_model)

result = fitter.fit_toas(maxiter=1)    # first call JIT-compiles (warmup)
result = fitter.fit_toas(maxiter=99)   # subsequent calls reuse the cached kernel

print(f"Chi-squared: {result.chi2:.2f}")
print(f"Degrees of freedom: {result.dof}")
print(f"Reduced chi-squared: {result.reduced_chi2:.4f}")
```

`native.get_model_and_toas("pulsar.par", "pulsar.tim")` collapses the three parsing lines into one call (mirroring PINT's `get_model_and_toas`), returning `(model, noise, toa_data)`.

> **Don't have data handy?** PINT's bundled example files work as test data. Install the extra (`pip install jaxpint[pint]`) and locate a TEMPO2-format pair via `from pint.config import examplefile` — e.g. `examplefile("B1855+09_NANOGrav_dfg+12.tim")` and `examplefile("B1855+09_NANOGrav_dfg+12_TAI.par")`. The files are read by JaxPINT's *native* parser; PINT is used only to find them on disk. (Note: `NGC6440E.tim` is the older Princeton format, which the native parser does not read.)

If you want more involved usages, see the `examples/` directory. To load via PINT instead (legacy formats, or an in-memory PINT model), see the [loading-data guide](docs/guides/loading_data.rst).

## Architecture

JaxPINT is built around three component types, all Equinox modules:

- **DelayComponent** — Computes a time delay contribution (seconds). Applied sequentially; each component sees the accumulated delay from prior components. Examples: `AstrometryEquatorial`, `DispersionDM`, `BinaryDD`.

- **PhaseComponent** — Computes a phase contribution (cycles, as integer + fractional parts). All phase components are summed; order does not matter. Examples: `Spindown`, `Glitch`, `PhaseJump`.

- **NoiseComponent** — Describes stochastic noise via a Woodbury decomposition `(N_diag, U, Phi_diag)` for efficient covariance inversion. Examples: `ScaleToaError`, `EcorrNoise`, `PLRedNoise`.

These are orchestrated by `TimingModel`, which chains delays and sums phases. The only dynamic (differentiable) leaf in the entire pytree is `ParameterVector.values` — a flat `float64` array. All component fields are static metadata, making the model fully JIT-traceable.

Inputs are produced by the **native parser** (`jaxpint.par`, `jaxpint.tim`, `jaxpint.clock`, surfaced through `jaxpint.native`), which reads `.par`/`.tim` files straight into the JaxPINT types above. The optional **bridge layer** (`jaxpint.bridge`) does the same job starting from an in-memory PINT model. Astropy units only appear during parsing (the bridge, and the native ephemeris/clock stage); once parsing is done, the rest of the pipeline is plain `float64`.

## Testing

Run the fast test suite (unit tests only):

```bash
pytest
```

Slow tests (PINT integration, fitting, noise whitening) are skipped by default.
Run the full suite including slow tests with:

```bash
pytest --runslow
```

Run only the slow tests:

```bash
pytest -m slow --runslow
```

Note: the full suite takes 10+ minutes due to numerical validation against PINT.

JaxPINT uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. Three profiles are available:

```bash
# Default (no deadline, for interactive use)
pytest

# CI (deterministic seeds, prints reproduction blobs)
HYPOTHESIS_PROFILE=ci pytest

# Fuzzing (1000 examples per test)
HYPOTHESIS_PROFILE=fuzzing pytest
```

## Documentation

The vast majority of the documentation is autogenerated from Sphinx-style comments. The `dev` extra already pulls in Sphinx and `sphinx-autobuild` via `jaxpint[docs]`, so no extra sync is needed if you installed with `--extra dev`.

**One-off build:**

```bash
JAX_PLATFORMS=cpu uv run sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.

**Live-reload while editing (recommended):**

```bash
JAX_PLATFORMS=cpu .venv/bin/sphinx-autobuild docs docs/_build/html --watch jaxpint --open-browser
```

Serves on `http://127.0.0.1:8000`. Saves to `.rst` files or docstrings in `jaxpint/` trigger a rebuild and the browser reloads automatically.

Docs are also built and deployed to GitHub Pages on every push to `main` via `.github/workflows/docs.yml`.
Before pushing to main, run the following command at the root of JaxPINT: 

```
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

This converts sphinx warnings to errors, which forces you to fix them.

## Disclosure

Claude code was used to generate the documentation. It also helped designing and implementing JaxPINT. 

## License

MIT


