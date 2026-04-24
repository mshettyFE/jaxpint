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

JaxPINT is not a replacement for PINT. The two libraries serve complementary roles:

| | PINT | JaxPINT |
|---|---|---|
| **I/O** | Reads `.par`/`.tim` files, handles units via Astropy | Delegates all I/O to PINT via a bridge layer |
| **Ephemeris** | Computes solar system ephemerides, barycentric corrections | Consumes pre-computed positions from PINT |
| **Timing model** | Object-oriented, mutable state, Astropy units throughout | Pure functional, immutable Equinox modules, plain float64 arrays |
| **Derivatives** | Hand-coded analytical partial derivatives | Automatic via `jax.jacobian` |
| **Fitting** | WLS/GLS with numpy | WLS/GLS with JAX (JIT-compiled, autodiff design matrices) |
| **Hardware** | CPU only | CPU or GPU via JAX |

**Typical workflow:** Load data with PINT, convert to JaxPINT types via the bridge layer, then fit or simulate entirely in JAX.

## Installation

Requires Python >= 3.12 and the package manager uv.

Pick exactly one JAX flavor (`cuda` or `cpu`), and add `dev` if you plan to hack on the code:

| Use case | Command |
|---|---|
| GPU workstation, read-only | `uv sync --extra cuda` |
| GPU workstation, developing | `uv sync --extra cuda --extra dev` |
| CPU laptop, read-only | `uv sync --extra cpu` |
| CPU laptop, developing | `uv sync --extra cpu --extra dev` |

## Quick Start

```python
# Used to load example data
import pint.models as pm
import pint.toa as pt
from pint.config import examplefile

from jaxpint import (
    build_timing_model,
    pint_model_to_params,
    pint_toas_to_jax,
    ParameterVector,
    compute_time_residuals,
    WLSFitter
)

## Temporarily have PINT shut up with DEBUG and INFO messages 
from loguru import logger
logger.disable("pint")      


# Load example data from pint
par_file = examplefile("NGC6440E.par")
tim_file = examplefile("NGC6440E.tim")
pint_model = pm.get_model(par_file)
pint_toas = pt.get_TOAs(tim_file, ephem="DE421")

#  convert PINT model and TOA data to JAX primitives
# Actual TOA data
toa_data = pint_toas_to_jax(pint_toas, model=pint_model)
# Value store for all the possible parameters in the fitter
params = pint_model_to_params(pint_model).params
# Actual differentiable models. Split into deterministic and stochastic (re: corrrelation matrix) contributions
timing_model, noise_model = build_timing_model(pint_model, pint_toas)

fitter = WLSFitter(timing_model, toa_data, params, noise_model=noise_model)
# JIT Warmup
print("Warming up the fitter...")
result = fitter.fit_toas(maxiter=1)
print("Running...")
result = fitter.fit_toas(maxiter=99)

print(f"Chi-squared: {result.chi2:.2f}")
print(f"Degrees of freedom: {result.dof}")
print(f"Reduced chi-squared: {result.reduced_chi2:.4f}")
```

If you want more involved usages, see examples directory

## Architecture

JaxPINT is built around three component types, all Equinox modules:

- **DelayComponent** — Computes a time delay contribution (seconds). Applied sequentially; each component sees the accumulated delay from prior components. Examples: `AstrometryEquatorial`, `DispersionDM`, `BinaryDD`.

- **PhaseComponent** — Computes a phase contribution (cycles, as integer + fractional parts). All phase components are summed; order does not matter. Examples: `Spindown`, `Glitch`, `PhaseJump`.

- **NoiseComponent** — Describes stochastic noise via a Woodbury decomposition `(N_diag, U, Phi_diag)` for efficient covariance inversion. Examples: `ScaleToaError`, `EcorrNoise`, `PLRedNoise`.

These are orchestrated by `TimingModel`, which chains delays and sums phases. The only dynamic (differentiable) leaf in the entire pytree is `ParameterVector.values` — a flat `float64` array. All component fields are static metadata, making the model fully JIT-traceable.

The **bridge layer** (`jaxpint.bridge`) converts between PINT objects and JaxPINT types. It is the only part of the codebase that touches Astropy units.

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

## Disclosure

Claude code was used to generate the documentation. It also helped designing and implementing JaxPINT. 

## License

MIT


