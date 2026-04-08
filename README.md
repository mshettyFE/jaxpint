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

Requires Python >= 3.12.

```bash
git clone <repo-url>
cd JaxPINT
pip install -e .
```

### Dependencies

- `jax[cpu] >= 0.4.20` — array computation and autodiff
- `equinox >= 0.11.0` — pytree-based neural network / module library
- `pint-pulsar >= 1.0` — I/O, ephemeris computation, data conversion
- `jaxtyping >= 0.3.9` — shape/dtype annotations for JAX arrays
- `hypothesis >= 6.0` — property-based testing

For GPU support, install JAX with CUDA following the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Quick Start

```python
import pint.models as pm
import pint.toa as pt
from jaxpint import (
    build_timing_model,
    pint_model_to_params,
    pint_toas_to_jax,
    compute_time_residuals,
    WLSFitter,
)

# Load data using PINT
pint_model = pm.get_model("pulsar.par")
pint_toas = pt.get_TOAs("pulsar.tim")

# Convert to JaxPINT types
toa_data = pint_toas_to_jax(pint_toas, pint_model)
params = pint_model_to_params(pint_model, pint_toas)
timing_model = build_timing_model(pint_model, pint_toas)

# Compute residuals (pure JAX — JIT-compatible)
residuals = compute_time_residuals(timing_model, toa_data, params)

# Fit parameters
fitter = WLSFitter(timing_model, toa_data, params)
result = fitter.fit_toas(maxiter=5)

print(f"Chi-squared: {result.chi2:.2f}")
print(f"Reduced chi-squared: {result.reduced_chi2:.4f}")
```

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

Note: the full suite takes 5+ minutes due to numerical validation against PINT.

JaxPINT uses [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. Three profiles are available:

```bash
# Default (no deadline, for interactive use)
pytest

# CI (deterministic seeds, prints reproduction blobs)
HYPOTHESIS_PROFILE=ci pytest

# Fuzzing (1000 examples per test)
HYPOTHESIS_PROFILE=fuzzing pytest
```

## License

MIT
