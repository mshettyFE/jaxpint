API Reference
=============

Core
----

.. autosummary::
   :toctree: generated

   jaxpint.types
   jaxpint.components
   jaxpint.model
   jaxpint.likelihood
   jaxpint.simulation
   jaxpint.utils
   jaxpint.constants

Phase Components
----------------

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.phase

Delay Components
----------------

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.delay

Binary Models
-------------

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.binary

Noise Models
------------

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.noise

Fitters
-------

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.fitters

Native Pipeline (PINT-free)
---------------------------

The native ``.par``/``.tim`` loading path and the parameter core it builds on.
These require no PINT (see :mod:`jaxpint.bridge` for the PINT-backed adapters).

.. autosummary::
   :toctree: generated

   jaxpint.native
   jaxpint.clock
   jaxpint.model_builder
   jaxpint.par.parser
   jaxpint.par.core
   jaxpint.par.registry
   jaxpint.par.result
   jaxpint.par.spec
   jaxpint.par.raw_params

PINT Bridge
-----------

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.bridge

PTA
---

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.pta

Bayesian Inference
------------------

Analytic marginalization of nuisance (timing-model) parameters. The opt-in
NumPyro sampler layer (``jaxpint.bayes.samplers``: prior specification via
NumPyro distributions + a NUTS runner) is documented in-source.

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.bayes

Frequentist Inference
---------------------

Detection statistics, empirical null calibrations, and detection-sensitivity
forecasts — the frequentist sibling of :mod:`jaxpint.bayes`, built on top of
the PTA likelihood machinery.

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.frequentist

Statistical Primitives
----------------------

Arm-neutral numerics (grid reductions, upper limits, credible/confidence
regions) shared by the Bayesian machinery, the frequentist detection
statistics, and the PTA CW products.

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.stats

Data Loaders
------------

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.loaders
