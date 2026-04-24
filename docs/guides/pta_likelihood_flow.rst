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

If you have some external signal (re: CW gravitational wave of a stochastic GW background), you could also generate a mock timeseries to reflect these injected signals.

TimingModel
***********

:class:`~jaxpint.model.TimingModel` is the deterministic side of the pipeline, and it is best thought of as a **forward model**: given a :class:`~jaxpint.types.ParameterVector` and a :class:`~jaxpint.types.TOAData`, it predicts a pulsar rotational phase for each TOA. The output has the same shape as the input timestamps -- ``(n_toas,)`` -- but the values are *phases* (cycles), not modified times.

Every TOA marks a moment where the pulse beam swept past Earth, so by construction the *observed* phase is always a pulse peak -- i.e. integer phase, modulo one cycle. :class:`~jaxpint.model.TimingModel`  predicts what phase the pulsar *should* have been at at that same moment. If the prediction lands at e.g. :math:`12{,}345{,}678{,}901.2346` cycles, the integer portion tells us *which* pulse we caught and the fractional :math:`0.2346` tells us how many cycles off the model was from reality. Dividing that sub-cycle mismatch by :math:`F_0` converts it into the time residual the fitter actually minimises.

Internally, TimingModel holds three tuples of components -- delays, phases, and dispersion -- and combines them in different ways.

Delay components
================

A "delay" here is the correction between the **topocentric TOA** (what the telescope recorded, after PINT's clock corrections) and the time at which the pulsar's intrinsic spin model -- the :math:`F_0, F_1, \ldots` polynomial -- actually applies. That reference time is effectively the pulse emission time in the pulsar's rest frame. Concretely, if :math:`t` is the topocentric TOA and :math:`\Delta t_\mathrm{total}(t)` is the full delay, the phase model is evaluated at

.. math::

   t_\mathrm{eval} = t - \Delta t_\mathrm{total}(t).

Each individual :math:`\Delta t_k` accounts for one physical reason the pulse took longer to reach the telescope than the pulsar's own clock would suggest:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Component
     - Correcting for
   * - Astrometry (Rømer) -- :class:`~jaxpint.delay.astrometry.AstrometryEquatorial` / :class:`~jaxpint.delay.astrometry.AstrometryEcliptic`
     - Light-travel time between the solar-system barycenter (SSB) and the observatory -- i.e. Earth's position in its orbit. Biggest term, up to ~500 s.
   * - Shapiro (solar system) -- :class:`~jaxpint.delay.shapiro.SolarSystemShapiroDelay`
     - Gravitational time dilation as the pulse passes near the Sun.
   * - Dispersion (ISM) -- :class:`~jaxpint.delay.dispersion_dm.DispersionDM`
     - Frequency-dependent slowdown from free electrons along the line of sight.
   * - Troposphere -- :class:`~jaxpint.delay.troposphere.TroposphereDelay`
     - Signal slowdown through Earth's atmosphere.
   * - Binary (Rømer / Einstein / Shapiro) -- :class:`~jaxpint.binary.ell1.BinaryELL1`, :class:`~jaxpint.binary.dd.BinaryDD`, and siblings in :mod:`jaxpint.binary`
     - For binary pulsars: light-travel, gravitational redshift, and Shapiro delay inside the pulsar's own orbit.

Walking the ``delay_components`` tuple in order effectively steps the reference frame outward along the signal path: **observatory → SSB → pulsar system barycenter → pulsar surface**.

The accumulating chain
----------------------

:meth:`~jaxpint.model.TimingModel.compute_delay` walks the ``delay_components`` tuple sequentially. Each component contributes an additive term, but *sees the accumulated delay from prior components* as an input. The final value is still a simple sum,

.. math::

   \Delta t_\mathrm{total}(t) = \sum_k \Delta t_k\bigl(t,\; \Delta t_{<k}\bigr),

where

- :math:`t` is the topocentric TOA (the observation time read from the ``.tim`` file, after clock corrections).
- :math:`k` indexes the entries of the ``delay_components`` tuple, walked in order.
- :math:`\Delta t_k(\cdot)` is the delay contribution produced by the :math:`k`-th component, in seconds. Each component is a callable that takes the TOA data and the accumulated delay so far.
- :math:`\Delta t_{<k} \equiv \sum_{j<k} \Delta t_j` is the running sum of all component contributions *before* :math:`k`. Passing this in as the second argument is what lets later components (e.g. binary orbital delays) operate in the frame the earlier components have already corrected for.
- :math:`\Delta t_\mathrm{total}(t)` is the final summed delay at :math:`t`, returned by :meth:`~jaxpint.model.TimingModel.compute_delay`.

The order matters because :math:`\Delta t_k` is in general a *function of* :math:`\Delta t_{<k}`, not just an additive constant. A binary-orbit Rømer delay, for instance, has to be computed in the SSB frame -- which only exists once the astrometric correction has already been peeled off. If two components genuinely commute (their contributions don't depend on prior delay), their ordering is irrelevant; in practice most components are weakly order-dependent.

Phase components
================

:meth:`~jaxpint.model.TimingModel.compute_phase` sums the contributions from every component in ``phase_components``. Addition is commutative, so order is irrelevant here. Each component receives the total delay from above as input, then returns its own phase contribution as a :class:`~jaxpint.dual_float.DualFloat` (an integer + fractional cycle split, kept separate to preserve double-precision over long baselines).

Some examples of phase components:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - What it contributes
   * - :class:`~jaxpint.phase.spin.Spindown`
     - The bulk of the phase: a Taylor expansion :math:`\varphi(t) = F_0 \Delta t + \tfrac{1}{2} F_1 \Delta t^2 + \ldots` around the reference epoch ``PEPOCH``. Present in essentially every pulsar model.
   * - :class:`~jaxpint.phase.glitch.Glitch`
     - Discrete spin-up events: a step in :math:`F_0` (and optionally :math:`F_1`) at a specified epoch, occasionally with an exponentially decaying recovery term.
   * - :class:`~jaxpint.phase.jump.PhaseJump`
     - Arbitrary additive phase offsets applied to subsets of TOAs selected by ``-flag value`` matching. Used to absorb unknown instrumental or observational offsets.

After summing, the model subtracts the phase at the TZR reference TOA to get absolute phase, then applies the ``PHOFF`` offset if present.

When this absolute phase is fed into :func:`~jaxpint.fitters._base.compute_phase_residuals`, the integer portion is reconciled against ``delta_pulse_number`` (which pins each TOA to a specific integer pulse). **Only the fractional part survives as the residual** -- the integer pulse count cancels out, leaving the sub-cycle timing mismatch that the fitter tries to minimise.

Dispersion components
=====================

Dispersion components are a specialisation (:class:`~jaxpint.components.DispersionDelayComponent`) that participate twice:

1. **In the delay chain.** Every dispersion component is also a delay component, so its timing contribution is folded into the sequential ``compute_delay`` chain described above.
2. **In a separate DM sum.** :meth:`~jaxpint.model.TimingModel.compute_dm` sums each component's :math:`\mathrm{DM}` contribution across the ``dispersion_components`` tuple to produce a per-TOA DM (in pc/cm³). This is used by the wideband fitter (:class:`~jaxpint.fitters.wideband.WidebandGLSFitter`) to form DM residuals alongside the time residuals.

For narrowband fits, only the first role matters; ``compute_dm`` is simply not called.

.. _noisemodel:

NoiseModel
**********

:class:`~jaxpint.noise.noise_model.NoiseModel` is the stochastic side. Every correlated noise source contributes a block to a single, unified pulsar-level covariance matrix expressed in Woodbury form:

.. math::

   C = \mathrm{diag}(N_\mathrm{diag}) + U \, \mathrm{diag}(\Phi_\mathrm{diag}) \, U^{\mathsf{T}}.

- :math:`N_\mathrm{diag}` comes from the white-noise component (:class:`~jaxpint.noise.white.ScaleToaError` applying ``EFAC``/``EQUAD``). When absent, the raw TOA uncertainties are used.
- :math:`U` is the horizontal concatenation of the basis matrices from every correlated component (``ECORR``, red noise, DM noise, chromatic noise, ...).
- :math:`\Phi_\mathrm{diag}` is the concatenation of the corresponding per-basis weights.

Each individual noise component contributes linearly. Adding a new noise source means appending a few columns to :math:`U` and a few entries to :math:`\Phi_\mathrm{diag}`, nothing else.

:meth:`~jaxpint.noise.noise_model.NoiseModel.covariance` returns the triple :math:`(N_\mathrm{diag},\; U,\; \Phi_\mathrm{diag})`, which the GLS fitters then feed directly into the Woodbury identity to avoid ever materialising the dense :math:`n_\mathrm{toas} \times n_\mathrm{toas}` covariance.

Fitting
*******

Once the raw ``.par``/``.tim`` files have been converted to JAX-native objects, everything downstream is pure JAX code -- no more Astropy units, no more PINT dependency at runtime. A fit takes four inputs:

- :class:`~jaxpint.model.TimingModel` -- the deterministic timing model (chained delays + summed phases).
- :class:`~jaxpint.types.TOAData` -- observation times, frequencies, uncertainties, SSB positions, etc.
- :class:`~jaxpint.types.ParameterVector` -- current parameter values, plus a frozen/free mask that determines which parameters actually move.
- :class:`~jaxpint.noise.noise_model.NoiseModel` -- optional; carries ``EFAC``/``EQUAD``/``ECORR``/red-noise contributions.

Residuals
---------

Residuals come in two flavors, both pure functions of ``(model, toa_data, params)``:

- :func:`~jaxpint.fitters._base.compute_phase_residuals` returns the fractional part of the model phase (in cycles), with ``delta_pulse_number`` offsets accounted for.
- :func:`~jaxpint.fitters._base.compute_time_residuals` wraps the phase residuals and divides by the spin frequency :math:`F_0` to return residuals in seconds.

Both return a ``jax.Array`` of shape ``(n_toas,)``.

PTA Likelihood Construction
***************************

All of the above holds on a per pulsar level. For PTA, you run this procedure for each pulsar timeseries in your PTA to generate a set of time-residuals (ie. What offset do you need to apply to the given MJDs which best matches the given pulsar model?). 

Denote the i-th pulsar timing residual as :math:`\Delta t_i` and its pulsar-level covariance (built by the :class:`~jaxpint.noise.noise_model.NoiseModel`) as :math:`C_i`. The number of toas per pulsar is denoted as :math:`N_{i}`.

Per-pulsar Gaussian log-likelihood
==================================

Under the usual assumption that each pulsar's residuals are a zero-mean Gaussian with covariance :math:`C_i`, the per pulsar log-likelihood is

.. math::

   \ln \mathcal{L}_i = -\tfrac{1}{2}\,\Delta t_i^{\mathsf{T}}\, C_i^{-1}\, \Delta t_i
       \;-\; \tfrac{1}{2}\,\ln \det C_i
       \;-\; \tfrac{N_{i}}{2}\,\ln(2\pi).

Because :math:`C_i` has the Woodbury structure :math:`C_i = \mathrm{diag}(N_i) + U_i\,\mathrm{diag}(\Phi_i)\,U_i^{\mathsf{T}}` (see the :ref:`NoiseModel <noisemodel>` section), both the quadratic form and the log-determinant can be evaluated in :math:`\mathcal{O}(n_\mathrm{toas}\,n_\mathrm{basis}^2)` time without ever materialising the dense :math:`n_\mathrm{toas}\times n_\mathrm{toas}` matrix. 

Uncorrelated PTA log-likelihood
===============================

When the pulsars are treated as independent (no cross-pulsar correlations), the full PTA log-likelihood is just the sum:

.. math::

   \ln \mathcal{L}_\mathrm{PTA} \;=\; \sum_i \ln \mathcal{L}_i.

This is what :func:`~jaxpint.pta.likelihood.pta_logL` computes. It takes three things:

- A :class:`~jaxpint.pta.params.GlobalParams` -- the parameters that are shared across pulsars (e.g. CW source sky location, common red-noise spectral index, …).
- A ``tuple`` of per-pulsar :class:`~jaxpint.types.ParameterVector` objects -- timing and noise parameters for each pulsar.
- A :class:`~jaxpint.pta.likelihood.PTAConfig` -- a static bundle of the per-pulsar :class:`~jaxpint.types.TOAData`, :class:`~jaxpint.model.TimingModel`, :class:`~jaxpint.noise.noise_model.NoiseModel`, plus a tuple of :class:`~jaxpint.pta.likelihood.SignalInjector` objects (see below). 

Internally, ``pta_logL`` loops over pulsars; for each one it asks every ``SignalInjector`` for (i) a deterministic delay contribution to subtract from the residuals and (ii) a ``(U, Phi)`` covariance augmentation to append to the noise model, then hands everything to :func:`~jaxpint.likelihood.single_pulsar_logL`.

Signal injectors
----------------

:class:`~jaxpint.pta.likelihood.SignalInjector` is an abstract base class that lets you plug in PTA-wide signals without touching the core likelihood. Each injector implements one or both of:

- ``delay(p, toa_data, pulsar_params, global_params)`` -- returns a per-TOA delay :math:`\delta t_{i,\alpha}` (shape ``(n_toas,)``) contributed by injector :math:`\alpha` to pulsar :math:`i`. Used for deterministic signals such as a single continuous-wave (CW) source.
- ``covariance(p, toa_data, pulsar_params, global_params)`` -- returns a ``(U, Phi)`` pair that augments pulsar :math:`i`'s noise model. Used for stochastic signals such as a common-spectrum red process.

How the injector contributions enter the per-pulsar likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each pulsar :math:`i`, ``pta_logL`` collects the ``delay`` outputs from every injector :math:`\alpha` and **subtracts their sum** from the timing residual that goes into :math:`\ln \mathcal{L}_i`:

.. math::

   \Delta t_i \;\longrightarrow\; \Delta t_i \;-\; \sum_\alpha \delta t_{i,\alpha}.

(Positive delay means a later arrival, hence the subtraction.) This is how a deterministic signal like CW gets removed from the residuals before the Gaussian likelihood is evaluated -- the injector is saying "given these global parameters, this is the waveform that should already be in the data."

For stochastic signals, ``pta_logL`` **horizontally concatenates** the ``(U_α, Φ_α)`` contributions onto the pulsar's own noise basis, producing a single augmented Woodbury form:

.. math::

   C_i
   \;\longrightarrow\;
   \mathrm{diag}(N_i)
   \;+\;
   \bigl[\, U_i \;\big|\; U_{i,1} \;\big|\; U_{i,2} \;\big|\; \ldots \,\bigr]\,
   \mathrm{diag}\!\bigl[\, \Phi_i ;\; \Phi_{i,1} ;\; \Phi_{i,2} ;\; \ldots \,\bigr]\,
   \bigl[\, \cdot \,\bigr]^{\mathsf{T}}.

So each stochastic injector just **appends a few basis columns** and a few weight entries -- the structure of the per-pulsar likelihood evaluation is unchanged, only the widths of :math:`U_i` and :math:`\Phi_i` grow. This is the same additive trick the :ref:`NoiseModel <noisemodel>` uses internally; the injectors simply extend it out to PTA-wide signals.

Under the hood, ``pta_logL`` passes the summed delay as the ``external_delay`` argument to :func:`~jaxpint.likelihood.single_pulsar_logL` and the concatenated ``(U, Φ)`` as ``external_cov``. The core per-pulsar code know of care about the source of the external delay and convariance. 

Concrete injectors shipped today:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Injector
     - Signal
   * - :class:`~jaxpint.pta.signals.cw.CWInjector` / :class:`~jaxpint.pta.signals.cw.CWInjectorStack`
     - Single or multiple continuous-wave sources (deterministic, via ``delay``).
   * - :class:`~jaxpint.pta.signals.gwb.CURNInjector`
     - Common uncorrelated red noise -- the same power-law spectrum in every pulsar, no cross-correlations (stochastic, via ``covariance``).

Correlated PTA log-likelihood
=============================

A real gravitational-wave background does not leave the pulsars independent: it induces cross-pulsar correlations with an angular dependence described by an **overlap reduction function** (ORF), most famously the Hellings-Downs curve for an isotropic stochastic background. The correlated log-likelihood is

.. math::

   \ln \mathcal{L}_\mathrm{PTA}^{\,\mathrm{corr}} = -\tfrac{1}{2}\,\Delta t^{\mathsf{T}}\, C^{-1}\, \Delta t
       \;-\; \tfrac{1}{2}\,\ln \det C
       \;-\; \tfrac{n_\mathrm{tot}}{2}\,\ln(2\pi),

where :math:`\Delta t` is the concatenation of all per-pulsar residuals and the global covariance has the block form

.. math::

   C \;=\; D \;+\; V\, \Phi_\mathrm{GWB}\, V^{\mathsf{T}},
   \qquad
   \Phi_\mathrm{GWB} \;=\; \Gamma \otimes \mathrm{diag}(S),

with :math:`D = \mathrm{blockdiag}(C_1, \ldots, C_N)` (the per-pulsar noise), :math:`V = \mathrm{blockdiag}(F_1, \ldots, F_N)` (the per-pulsar Fourier bases), :math:`\Gamma` the ORF matrix of shape ``(n_pulsars, n_pulsars)``, and :math:`S` the GWB power-law PSD.

This is what :func:`~jaxpint.pta.correlated_likelihood.pta_logL_correlated` computes, using a two-tier Woodbury scheme:

1. An **inner** per-pulsar Woodbury solve handles :math:`D` (white + per-pulsar correlated noise).
2. An **outer** dense Cholesky on the compressed Fourier-basis system couples pulsars through :math:`\Gamma`.

This avoids ever forming :math:`C` itself (which would be :math:`n_\mathrm{tot}\times n_\mathrm{tot}`) while still capturing the cross-pulsar physics. The static bundle is :class:`~jaxpint.pta.correlated_likelihood.CorrelatedPTAConfig`, and the corresponding injector ABC is :class:`~jaxpint.pta.correlated_likelihood.CorrelatedSignalInjector`.

Correlated injector shipped today:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Injector
     - Signal
   * - :class:`~jaxpint.pta.signals.correlated_gwb.HDCorrelatedGWBInjector`
     - Isotropic stochastic GWB with Hellings-Downs cross-correlations (:func:`~jaxpint.pta.signals.orf.hd_orf`). Related ORFs in :mod:`jaxpint.pta.signals.orf`: :func:`~jaxpint.pta.signals.orf.monopole_orf`, :func:`~jaxpint.pta.signals.orf.dipole_orf`.

Putting it together
===================

A minimal end-to-end PTA log-likelihood evaluation:

.. code-block:: python

   from jaxpint.pta.likelihood import PTAConfig, pta_logL
   from jaxpint.pta.params import GlobalParams
   from jaxpint.pta.signals.cw import CWInjector

   # one entry per pulsar (from the bridge layer)
   toa_data_list  = (toa1, toa2, toa3)
   timing_models  = (tm1,  tm2,  tm3)
   noise_models   = (nm1,  nm2,  nm3)
   pulsar_params  = (p1,   p2,   p3)

   injectors = (CWInjector(...),)                         # deterministic CW
   config    = PTAConfig(toa_data_list, timing_models,
                         noise_models, injectors)
   global_params = GlobalParams.empty()
   for inj in injectors:
       global_params = inj.register_params(global_params)

   logL = pta_logL(global_params, pulsar_params, config)  # scalar jax.Array

Swapping ``pta_logL`` for :func:`~jaxpint.pta.correlated_likelihood.pta_logL_correlated` (with a :class:`~jaxpint.pta.correlated_likelihood.CorrelatedPTAConfig` and one or more correlated injectors) turns on the Hellings-Downs coupling. Because both functions are pure JAX, the whole thing plays nicely with ``jax.jit``, ``jax.grad``, and downstream samplers like BlackJAX.
