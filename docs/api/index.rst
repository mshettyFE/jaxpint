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

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.bayes

Data Loaders
------------

.. autosummary::
   :toctree: generated
   :recursive:

   jaxpint.loaders
