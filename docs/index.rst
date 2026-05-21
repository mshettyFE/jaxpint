JaxPINT Documentation
=====================

JaxPINT is a JAX-accelerated port of `PINT <https://github.com/nanograv/PINT>`_,
the standard NANOGrav pulsar timing package. It reimplements PINT's timing
model as pure JAX code built on `Equinox <https://github.com/patrick-kidger/equinox>`_,
trading PINT's Astropy-units interface for plain ``float64`` arrays in
exchange for JIT compilation, automatic differentiation, and GPU execution.

JaxPINT is **not** a replacement for PINT. PINT continues to own all I/O —
``.par`` / ``.tim`` parsing, observatory database, clock corrections,
ephemeris lookups — and JaxPINT consumes those via a thin bridge layer
(:mod:`jaxpint.bridge`). After conversion every downstream computation
(delays, phases, noise, likelihoods, fits) is plain JAX, so design matrices
fall out of :func:`jax.jacobian` instead of being hand-coded, and the whole
forward model JIT-compiles into a single XLA kernel.

Installation
------------

Requires Python ≥ 3.12 and the `uv <https://docs.astral.sh/uv/>`_ package
manager. Pick one JAX flavor:

.. code-block:: bash

   uv sync --extra cpu    # CPU-only
   uv sync --extra cuda   # NVIDIA GPU

Append ``--extra dev`` if you intend to hack on the source.

Quickstart
----------

Fit a real ``.par`` / ``.tim`` pair using a PINT example file:

.. code-block:: python

   import pint.models as pm
   import pint.toa as pt
   from pint.config import examplefile

   from jaxpint import (
       build_timing_model,
       pint_model_to_params,
       pint_toas_to_jax,
       WLSFitter,
   )

   # Load with PINT, then convert to JaxPINT types
   pint_model = pm.get_model(examplefile("NGC6440E.par"))
   pint_toas = pt.get_TOAs(examplefile("NGC6440E.tim"), ephem="DE421")

   toa_data = pint_toas_to_jax(pint_toas, model=pint_model)
   params = pint_model_to_params(pint_model).params
   timing_model, noise_model = build_timing_model(pint_model, pint_toas)

   # Fit (first call JIT-compiles; subsequent calls are cached)
   fitter = WLSFitter(timing_model, toa_data, params, noise_model=noise_model)
   result = fitter.fit_toas(maxiter=99)

   print(f"Reduced χ² = {result.reduced_chi2:.4f}  ({result.dof} dof)")

For an end-to-end walkthrough of the PTA signal-processing pipeline, see
:doc:`guides/pta_likelihood_flow`.

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
