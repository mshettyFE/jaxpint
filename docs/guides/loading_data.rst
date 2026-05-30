Loading Data
============

Everything downstream of parsing -- residuals, phases, noise, fits,
likelihoods -- operates on three JaxPINT objects:

- a :class:`~jaxpint.types.ParameterVector` (the flat, differentiable parameter
  array),
- a :class:`~jaxpint.types.TOAData` (the per-TOA times, frequencies, masks, and
  precomputed geometry), and
- a :class:`~jaxpint.model.TimingModel` (plus a :class:`~jaxpint.noise.NoiseModel`).

This guide covers how to produce those from raw ``.par`` / ``.tim`` files. There
are two routes: the **native** path (the default, PINT-free) and the optional
**PINT bridge**.

The native path (default)
-------------------------

The native path reads ``.par`` / ``.tim`` files with no PINT dependency. The
public entry points live in :mod:`jaxpint.native` and deliberately mirror PINT's
``get_model`` / ``get_TOAs`` / ``get_model_and_toas`` names.

The explicit three-step form makes the data flow clear:

.. code-block:: python

   import jaxpint.par as par
   from jaxpint import native, build_model

   parsed = par.get_model("pulsar.par")               # -> ParResult
   toa_data = native.get_TOAs("pulsar.tim", parsed)   # -> TOAData
   model, noise = build_model(parsed, toa_data)       # -> (TimingModel, NoiseModel)

   params = parsed.params                             # -> ParameterVector

1. :func:`jaxpint.par.parser.get_model` (re-exported as ``jaxpint.par.get_model``)
   tokenizes the ``.par`` file, detects the active components and binary model,
   and assembles a :class:`~jaxpint.par.result.ParResult`. Its ``params``
   attribute is the :class:`~jaxpint.types.ParameterVector` you pass to fitters
   and likelihoods.
2. :func:`jaxpint.native.get_TOAs` reads the ``.tim`` file into a
   :class:`~jaxpint.types.TOAData`. Passing the parsed ``.par`` is optional but
   recommended -- it supplies the astrometry direction (for barycentric
   frequencies), the TZR absolute-phase anchor, the flag-mask selectors, the
   troposphere configuration, and the default ephemeris / clock settings.
3. :func:`jaxpint.model_builder.build_model` (re-exported as
   ``jaxpint.build_model``) turns the ``ParResult`` -- together with the
   ``TOAData``, which the TOA-dependent noise bases need -- into the JAX-native
   :class:`~jaxpint.model.TimingModel` and :class:`~jaxpint.noise.NoiseModel`.

For convenience, :func:`jaxpint.native.get_model_and_toas` runs all three steps
in a single call and returns ``(model, noise, toa_data)``:

.. code-block:: python

   from jaxpint import native

   model, noise, toa_data = native.get_model_and_toas("pulsar.par", "pulsar.tim")

If you only need the model (no TOAs), :func:`jaxpint.native.get_model` parses a
``.par`` and returns ``(model, noise)`` -- but note that without TOAs the
TOA-dependent noise components (ECORR and the power-law red/DM/chromatic/solar-wind
noise) are omitted, since they need a ``TOAData`` to build their Fourier and
quantization bases.

What native parsing computes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The native ``.tim`` path reproduces, in plain ``float64``, the pieces PINT used
to handle on read:

- per-site **clock corrections** (observatory → TT(BIPM)) from the
  auto-updating IPTA clock data (see :doc:`clock_data`),
- the **TT→TDB** timescale conversion,
- **barycentric** position/velocity vectors from a JPL ephemeris,
- per-TOA observing-**frequency barycentering**,
- the flag-based **parameter masks** (``TOAData.flag_masks``) used by JUMP / EFAC
  / EQUAD / ECORR selection,
- the **TZR** absolute-phase anchor, and
- the **troposphere** delay geometry.

Optional keyword arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both :func:`jaxpint.native.get_TOAs` and
:func:`jaxpint.native.get_model_and_toas` accept the same keyword overrides;
when left at their defaults they fall back to the settings declared in the
``.par`` file (``EPHEM``, ``CLK``, etc.).

``ephem``
   JPL ephemeris name (e.g. ``"DE440"``). The native default is **DE440**
   (Astropy's ``DE421`` download URL is dead, so do not request it).

``include_bipm`` / ``bipm_version``
   Whether to apply the TT(BIPM) realization and which one (e.g.
   ``"BIPM2023"``).

``planets``
   Whether to compute individual planet position/velocity vectors (needed only
   for the planetary Shapiro delay).

``limits``
   ``"warn"`` (default) or ``"error"`` -- governs what happens when a TOA falls
   outside a clock file's covered MJD range (see :doc:`clock_data`).

Supported ``.tim`` formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The native reader understands the **TEMPO2** line format
(``name freq MJD err site -flag value ...``) plus the common in-file commands
(``TIME`` / ``PHASE`` offsets, ``EFAC``, ``EQUAD``, ``EMIN`` / ``EMAX``,
``FMIN`` / ``FMAX``, ``SKIP`` / ``NOSKIP``, ``INCLUDE``, ``JUMP``, ``INFO``;
``MODE`` is recognized but ignored). The legacy fixed-column formats
(Princeton / Parkes / ITOA) are **not** parsed natively and raise
``NotImplementedError`` -- use the PINT bridge below for those.

The optional PINT bridge
------------------------

The bridge (:mod:`jaxpint.bridge`) does the same job but starting from PINT
objects instead of raw files. It requires the optional extra
(``pip install jaxpint[pint]``); importing any bridge symbol without PINT
installed raises a clear ``ImportError`` naming the extra.

.. code-block:: python

   import pint.models as pm
   import pint.toa as pt

   from jaxpint import build_timing_model, pint_model_to_params, pint_toas_to_jax

   pint_model = pm.get_model("pulsar.par")
   pint_toas = pt.get_TOAs("pulsar.tim", model=pint_model, ephem="DE440")

   toa_data = pint_toas_to_jax(pint_toas, model=pint_model)
   params = pint_model_to_params(pint_model).params
   model, noise = build_timing_model(pint_model, pint_toas)

- :func:`jaxpint.bridge.pint_toas_to_jax` converts a ``pint.toa.TOAs`` into a
  :class:`~jaxpint.types.TOAData`.
- :func:`jaxpint.bridge.pint_model_to_params` extracts the parameters into a
  :class:`~jaxpint.types.ParameterVector` (via the same
  :class:`~jaxpint.par.result.ParResult` contract the native path uses).
- :func:`jaxpint.bridge.build_timing_model` builds the
  :class:`~jaxpint.model.TimingModel` (and noise model) from the PINT model.

Reach for the bridge when you already have a PINT ``TimingModel`` in memory
(for example after building one with PINT's simulation helpers), or when you
need to read a TOA format the native parser does not support. Either way, once
parsing is done the PINT objects are no longer used -- the rest of the pipeline
runs entirely on the JaxPINT types.

Once you have the data
----------------------

With ``model``, ``noise``, ``toa_data``, and ``params`` in hand you can fit:

.. code-block:: python

   from jaxpint import WLSFitter

   fitter = WLSFitter(model, toa_data, params, noise_model=noise)
   result = fitter.fit_toas(maxiter=10)
   print(result.reduced_chi2, result.dof)

See :doc:`pta_likelihood_flow` for how these objects flow through the full
per-pulsar forward model and into the PTA log-likelihood.
