JaxPINT Documentation
=====================

JaxPINT is a JAX-accelerated port of `PINT <https://github.com/nanograv/PINT>`_,
the standard NANOGrav pulsar timing package. It reimplements PINT's timing
model as pure JAX code built on `Equinox <https://github.com/patrick-kidger/equinox>`_,
trading PINT's Astropy-units interface for plain ``float64`` arrays in
exchange for JIT compilation, automatic differentiation, and GPU execution.

JaxPINT parses ``.par`` / ``.tim`` files **natively** — clock corrections,
TT→TDB timescale conversion, barycentric ephemeris positions, flag-based
parameter masks, the TZR absolute-phase anchor, and troposphere geometry are
all reimplemented in JaxPINT (see :mod:`jaxpint.native`). PINT is therefore an
**optional** dependency: install it with ``pip install jaxpint[pint]`` only if
you want the :mod:`jaxpint.bridge` adapters (e.g. to convert an in-memory PINT
model, or to read the legacy TOA formats the native reader does not parse).

After parsing, every downstream computation (delays, phases, noise,
likelihoods, fits) is plain JAX, so design matrices fall out of
:func:`jax.jacobian` instead of being hand-coded, and the whole forward model
JIT-compiles into a single XLA kernel. JaxPINT is validated for numerical
parity against PINT but is otherwise independent of it at runtime.

Installation
------------

Requires Python ≥ 3.12 and the `uv <https://docs.astral.sh/uv/>`_ package
manager. Pick one JAX flavor:

.. code-block:: bash

   uv sync --extra cpu    # CPU-only
   uv sync --extra cuda   # NVIDIA GPU

Append ``--extra dev`` if you intend to hack on the source. Add ``--extra pint``
(``pip install jaxpint[pint]``) only if you need the optional PINT-bridge
adapters — the native ``.par`` / ``.tim`` path does not require it.

Quickstart
----------

Fit a real ``.par`` / ``.tim`` pair — no PINT required:

.. code-block:: python

   import jaxpint.par as par
   from jaxpint import native, build_model, WLSFitter

   # Parse and build entirely in JaxPINT (Tempo2, Princeton or Parkes .tim)
   parsed = par.get_model("pulsar.par")               # ParResult: parameters + components
   toa_data = native.get_TOAs("pulsar.tim", parsed)   # TOAData (clock-corrected, barycentered)
   timing_model, noise_model = build_model(parsed, toa_data)  # TimingModel + NoiseModel

   # `parsed.params` is the ParameterVector — the only differentiable leaf
   fitter = WLSFitter(timing_model, toa_data, parsed.params, noise_model=noise_model)
   result = fitter.fit_toas(maxiter=99)   # first call JIT-compiles; later calls are cached

   print(f"Reduced χ² = {result.reduced_chi2:.4f}  ({result.dof} dof)")

:func:`jaxpint.native.get_model_and_toas` wraps the three parsing lines into a
single call, mirroring PINT's ``get_model_and_toas``.

.. tip::

   Don't have a ``.par`` / ``.tim`` pair handy? PINT ships example files that
   make good test data. Install the extra (``pip install jaxpint[pint]``) and
   locate a TEMPO2-format pair with ``from pint.config import examplefile``,
   e.g. ``examplefile("B1855+09_NANOGrav_dfg+12.tim")`` — the files are read by
   JaxPINT's *native* parser; PINT is used only to find them on disk.

For loading details (the native vs. PINT-bridge paths) see
:doc:`guides/loading_data`; for an end-to-end walkthrough of the PTA
signal-processing pipeline see :doc:`guides/pta_likelihood_flow`.

Where to next
-------------

- :doc:`guides/index` — narrative guides explaining how the pieces fit together.
- :doc:`api/index` — full API reference, auto-generated from docstrings.
- `Source on GitHub <https://github.com/mshettyFE/jaxpint>`_ — issues, contributing, examples.

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   guides/index
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
