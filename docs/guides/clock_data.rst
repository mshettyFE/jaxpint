Clock Data & Configuration
==========================

Converting a raw site arrival time into a barycentric TDB time needs
**observatory clock-correction files** -- the per-telescope tables that map each
site's local clock onto TT(BIPM). JaxPINT carries these natively (it no longer
borrows PINT's copies), sourced from the
`IPTA pulsar-clock-corrections <https://github.com/ipta/pulsar-clock-corrections>`_
repository. This guide explains where that data lives, how it stays current,
and how to make a run reproducible.

The time-standard chain
-----------------------

A pulse arrival time is recorded against a telescope's **local clock** and must
end up as **TDB** -- the coordinate time at the solar-system barycentre, the
inertial reference in which the pulsar's rotation is regular. Getting there walks
a chain of well-defined time standards, each step handled either by a
clock-correction *file* or by an analytic conversion:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - From → To
     - Transformation
     - Where it happens
   * - site clock → GPS (or local ref)
     - per-telescope clock table
     - site ``*.clk`` / ``*.dat`` files
   * - GPS → UTC
     - GPS−UTC offset (~tens of ns)
     - ``gps2utc.clk`` (per-obs ``apply_gps2utc``)
   * - UTC → TAI
     - add leap seconds
     - erfa leap-second table
   * - TAI → TT
     - ``+ 32.184 s`` (definitional)
     - constant
   * - TT(TAI) → TT(BIPM)
     - µs-level realization refinement
     - ``tai2tt_bipm<year>.clk`` (``include_bipm``)
   * - TT → TDB
     - periodic relativistic terms (~1.6 ms)
     - erfa, from the JPL ephemeris

The standards, briefly:

- **Local / site clock** -- the raw timestamp the backend recorded. Observatories
  discipline it to an external reference, commonly **GPS** (sometimes a hydrogen
  maser or NIST).
- **GPS** -- Global Positioning System time: a continuous atomic scale used for
  time dissemination, offset from UTC by the accumulated leap seconds (and from
  TAI by a fixed 19 s).
- **UTC** -- Coordinated Universal Time: atomic, but with **leap seconds** inserted
  irregularly so it tracks Earth's rotation. The civil standard.
- **TAI** -- International Atomic Time: continuous atomic time with **no** leap
  seconds.
- **TT** -- Terrestrial Time: the idealized time on Earth's geoid that the timing
  model uses, defined as ``TT = TAI + 32.184 s``. It has two realizations:
  **TT(TAI)**, available in real time, and **TT(BIPM<year>)**, a more stable
  version the BIPM recomputes each year by re-analysing the global atomic-clock
  ensemble (``BIPM2019``, ``BIPM2021``, ``BIPM2023``, ...).
- **TDB** -- Barycentric Dynamical Time: a coordinate time at the barycentre,
  differing from TT only by small **periodic relativistic terms** (~1.6 ms
  amplitude) from Earth's motion through the Sun's gravity well.

The **clock chain** (``jaxpint.clock.correction``) assembles the *file-based*
legs -- the per-site tables, ``gps2utc``, and ``tai2tt_bipm`` -- and records their
sum as the per-TOA ``clkcorr``, landing the time on **TT(BIPM)**. The leap-second
(UTC→TAI) and ``+32.184 s`` (TAI→TT) steps are analytic, and the final **TT → TDB**
conversion is a separate relativistic step (``jaxpint.clock.timescale``). Note
that this chain only fixes the *timescale*; the geometric **barycentering** -- the
Roemer light-travel delay from the observatory to the barycentre -- is a
timing-model delay, not a clock correction, even though both are needed to reach a
true barycentric time.

Where the data lives
--------------------

A clock *snapshot* (the ``.clk`` / ``.dat`` files plus a metadata manifest) is
cached on disk. By default it lives inside the installed package
(``jaxpint/data/clock``); set ``JAXPINT_CLOCK_DIR`` to relocate it to a writable
directory you control. The cache is consulted lazily -- only when a TOA actually
needs a correction, never at import time -- and network failures are non-fatal
(JaxPINT keeps using the cached snapshot and, if it is old, warns).

Staying current (auto-update)
-----------------------------

By default JaxPINT refreshes the snapshot from IPTA when it is older than
``JAXPINT_CLOCK_TTL_DAYS`` (default **7** days; ``0`` checks on every run). This
keeps corrections current as new IPTA data lands without any action on your
part. If you are offline and the cached snapshot has grown genuinely old,
you'll get a ``StaleClockWarning`` rather than a hard failure.

To force a refresh immediately (the rare manual override):

.. code-block:: python

   from jaxpint.clock import update_clocks

   diff = update_clocks()          # pull IPTA main HEAD
   diff = update_clocks(ref="...") # or a specific commit SHA
   # diff -> {ref_old, ref_new, added, removed}

Reproducibility (pinning)
-------------------------

For runs you need to reproduce exactly, **pin a commit** by setting
``JAXPINT_CLOCK_REF`` to an IPTA ``pulsar-clock-corrections`` commit SHA. Pinning
freezes the snapshot to that exact commit *and disables auto-update*, so the
clock data can never shift underneath you:

.. code-block:: bash

   export JAXPINT_CLOCK_REF=c6731ec...   # exact IPTA commit

The commit JaxPINT ships against is available as
``jaxpint.clock.SEED_CLOCK_REF`` (with ``SEED_CLOCK_DATE``), which is a
convenient value to pin to for a fully offline, deterministic run.

Environment variables
---------------------

All clock configuration is via ``JAXPINT_CLOCK_*`` environment variables (so it
works identically on Linux, macOS, and Windows, and is easy to set in CI):

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Variable
     - Default
     - Meaning
   * - ``JAXPINT_CLOCK_DIR``
     - packaged dir
     - Override the clock cache directory.
   * - ``JAXPINT_CLOCK_REF``
     - *(unset)*
     - Pin an exact IPTA commit SHA. Makes runs reproducible and disables
       auto-update.
   * - ``JAXPINT_CLOCK_TTL_DAYS``
     - ``7``
     - Auto-update cadence in days; the cache is refreshed when older than this
       (``0`` = check every run).

``jaxpint.clock.config.describe()`` prints this table with the currently
effective values, which is handy when debugging a surprising correction.

Out-of-range TOAs
-----------------

A clock file covers a finite MJD range. If a TOA falls *past* the last entry of
a clock file -- typically very recent data, ahead of the latest published
corrections -- the interpolation clamps to the endpoint value, and JaxPINT
flags it. This is controlled by the ``limits`` keyword on
:func:`jaxpint.native.get_TOAs` / :func:`jaxpint.native.get_model_and_toas`:

- ``limits="warn"`` (default) emits a ``ClockCorrectionOutOfRange`` warning and
  proceeds with the clamped value;
- ``limits="error"`` raises instead.

If you hit this, the fix is usually to refresh the clock data
(``update_clocks()``), since the published corrections simply haven't caught up
to your TOAs yet. Note this is distinct from ``StaleClockWarning``, which is
about the *snapshot's age*, not about a specific TOA being past the data.
