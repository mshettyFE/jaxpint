High Level Overview of the Signal Processing Pipeline of JAXPint (WIP)
######################################################################

The PTA signal processing pipeline from raw TOA data to log likelihood evaluation is a bit confusing to an outsider. This is my best attempt at explaining this from end to end; this is also my mental model that I had in mind when building JAXPint. 

Input Data 
**********

Each pulsar in the array is represented by two different files: a .par file and .tim file 

.tim File
=========

The ``.tim`` file is a list of one time-of-arrival (TOA) per line, plus a handful of optional file-level directives. JaxPINT reads these via PINT, so anything PINT understands works here. Since PTAs have been around for a while, there are a smorgasbord of different formats. 

For simplicity, let's just look at one of these formats: TEMPO2 format. The first line must be ``FORMAT 1``, and every subsequent TOA line has five positional fields followed by zero or more ``-flag value`` metadata pairs:

.. code-block:: text

   name  freq_MHz  MJD  uncertainty_us  site  [-flag value]*

Example:

.. code-block:: text
   :caption: Example ``.tim`` file (TEMPO2 format)

   FORMAT 1
   J1834-0701_L.guppi.1  1500.000  58423.5001352123456  0.312  gbt  -be GUPPI -fe Rcvr1_2 -pta NANOGrav
   J1834-0701_L.guppi.2  1500.000  58423.5001352123456  0.298  gbt  -be GUPPI -fe Rcvr1_2 -pta NANOGrav
   J1834-0701_S.guppi.1  2100.000  58424.7614293182746  0.405  gbt  -be GUPPI -fe Rcvr2_3 -pta NANOGrav

The trailing ``-flag value`` pairs are free-form metadata attached to each TOA. PTAs use them to tag the backend, frontend, observing group, receiver, etc. JaxPINT's noise model selects per-TOA ``EFAC``, ``EQUAD``, and ``ECORR`` values by matching against these flags, so the same flag keys have to appear consistently in the ``.tim`` file and the ``.par`` file.

TEMPO2 also defines a small set of in-file commands (``TIME`` offsets, ``EFAC``, ``EQUAD``, ``MODE``, ``SKIP``/``NOSKIP``, ``INCLUDE``, ``JUMP``/``NOJUMP``) that PINT applies on read. Unlike the ``.par`` parameter set, there is no single enumerated catalog of flag names -- each PTA defines its own conventions. See the `TEMPO2 manual <https://bitbucket.org/psrsoft/tempo2/src/master/documentation/>`_ for the authoritative format specification, and the `PINT explanation page <https://nanograv-pint.readthedocs.io/en/latest/explanation.html>`_ for PINT-specific notes on what happens when the file is loaded.

Some important observations
---------------------------

- The fundamental unit of data for a TOA is the triple

.. math::

   (\nu,\; t,\; \sigma_t)

consisting of the observing frequency :math:`\nu` (MHz), the pulse arrival time :math:`t` (MJD), and the measurement uncertainty :math:`\sigma_t` (µs). Everything else is metadata which is either documentation, or flags which alter one of these values

- Each frequency gets assigned its own TOA, even if they were observed at the same MJD.
- There is no gaurentee on uniform cadence between TOAs (which makes sense, since the telescope could be down for maintainance or someething)

Parsing .tim files in JAXPint 
-----------------------------

I made the pragmatic decision to offload the TOA parsing to PINT. I didn't want to deal with all the nuances of different time standards, flags and backends and probably other things that I'm forgetting.

The .tim parsing workflow is the read in .tim files via PINT, which results in a `pint.toa.TOAs <https://nanograv-pint.readthedocs.io/en/latest/_autosummary/pint.toa.TOAs.html>`_ object (an ``astropy.table.Table`` with one row per TOA) -- and then converted that into a JAX-compatible container, :class:`~jaxpint.types.TOAData`, via :func:`~jaxpint.bridge.toa_conversion.pint_toas_to_jax`.

In principle, it would be straightforward to skip the PINT parsing dependency and just write directly to  :class:`~jaxpint.types.TOAData`.

.par File
=========

This is just a plain table of values which specifies how to build a particular pulsar model.

.. code-block:: text
   :caption: Example ``.par`` file

   PSR           J1834-0701_00
   RAJ           18:34:29.80260
   DECJ          -07:01:17.6778
   F0            443.4391679646
   F1            -2.481624e-15
   PEPOCH        57000.0
   DM            49.0249
   PX            1.359724
   EPHEM         DE440
   CLK           TT(BIPM2019)
   UNITS         TDB
   EFAC tel gbt 1.0
   EQUAD tel gbt 0.1

Most of the parameters in PINT are supported (with the exception of some obscure stuff). See the `PINT supported parameters table
<https://nanograv-pint.readthedocs.io/en/latest/timingmodels.html#supported-parameters>`_
for the full list of accepted keywords, units, and aliases.

Parsing .par files in JAXPint
-----------------------------

JAXPint currently outsources the initial parsing and construction of the pulsar timing model to PINT. This was done for reasons exactly analagous to PINT. I don't want to be writing parsers that have already been written.

The flow mirrors the ``.tim`` case:

1. The raw ``.par`` file is read by `pint.models.get_model <https://nanograv-pint.readthedocs.io/en/latest/_autosummary/pint.models.get_model.html>`_, which returns a `pint.models.TimingModel <https://nanograv-pint.readthedocs.io/en/latest/_autosummary/pint.models.TimingModel.html>`_ object. That object holds every parameter as an ``astropy.units.Quantity`` and tracks PINT's component hierarchy (delays, phases, noise).

2. The PINT ``TimingModel`` is then handed to the JaxPINT bridge layer, which produces two JAX-native objects:

   - :func:`~jaxpint.bridge.model_conversion.pint_model_to_params` extracts every numerical parameter into a flat :class:`~jaxpint.types.ParameterVector` -- the only differentiable leaf of the pytree.
   - :func:`~jaxpint.bridge.component_builder.build_timing_model` constructs the JAX-native :class:`~jaxpint.model.TimingModel`. Astropy units are stripped and every scalar becomes plain ``float64``.

Once this conversion is done, the PINT object is no longer needed at runtime -- the rest of the pipeline operates entirely on the JaxPINT types.

Synthetic Versus Real Data 
========================== 

As an aside, notice that the pipeline doesn't care about the origin of the .par and .tim files. Hence, you could run JAXPint on real pulsar data, or you can generate synthetic .par and .tim files.

For synthetic data generation,  you could use `PINT's model construction facilities <https://nanograv-pint.readthedocs.io/en/latest/_autosummary/pint.models.model_builder.get_model.html#pint.models.model_builder.get_model>`_, generate a uniform time series of observations with `pint.simulation.make_fake_toas_uniform <https://nanograv-pint.readthedocs.io/en/latest/_autosummary/pint.simulation.make_fake_toas_uniform.html>`_ (or one of the other generators in `pint.simulation <https://nanograv-pint.readthedocs.io/en/latest/_autosummary/pint.simulation.html>`_), and then pass the two through the pipeline to get the JAX-compatible model and time data.
