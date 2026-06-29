Adding a Timing-Model Component
===============================

A timing-model component is an `Equinox <https://github.com/patrick-kidger/equinox>`_
module that contributes either a **delay** (seconds), a **phase** (cycles), or a
**noise** term (a Woodbury covariance triple). This guide walks through adding a
new one, using the existing :class:`~jaxpint.phase.Spindown` phase component as
the worked example.

Adding a component touches four files: the class itself, the
:class:`~jaxpint.par.registry.Component` enum, the component table in
``jaxpint.par.registry_table``, and the builder in
:mod:`jaxpint.model_builder` (plus the package ``__init__`` exports). The table
is the **single source of truth**: the parser vocabulary, auto-detection,
execution order, and the optional PINT-bridge name all *derive* from one
``ComponentSpec`` entry, so there are no longer separate lists to keep in sync.

Every component carries two related-but-distinct descriptions of its parameters,
explained in detail in :mod:`jaxpint.components`:

- ``PARAMS`` -- the static *schema* (what parameters exist, their units / prefixes
  / aliases), used at parse time;
- ``*_name`` / ``*_names`` *fields* -- the concrete parameter names this
  configured instance reads from the :class:`~jaxpint.types.ParameterVector` at
  runtime.

1. Write the component class
----------------------------

Subclass one of :class:`~jaxpint.components.PhaseComponent`,
:class:`~jaxpint.components.DelayComponent`, or
:class:`~jaxpint.components.NoiseComponent`. Declare a class-level ``PARAMS``
schema, add static ``eqx.field`` instances for the configured parameter names,
and implement the physics:

.. code-block:: python

   import equinox as eqx
   from jaxpint.components import ParamDecl, PhaseComponent

   class Spindown(PhaseComponent):
       # Static schema: what this component models (used by the .par parser)
       PARAMS = (
           ParamDecl("F0", unit="Hz", prefix="F"),   # prefix => F0, F1, F2, ...
           ParamDecl("PEPOCH", kind="mjd"),
       )

       # Runtime config: the concrete names this instance reads at evaluation
       spin_param_names: tuple[str, ...] = eqx.field(static=True)
       pepoch_name: str = eqx.field(static=True, default="PEPOCH")

       def __check_init__(self):                     # optional validation
           if not self.spin_param_names:
               raise ValueError("Spindown requires at least F0")

       def __call__(self, toa_data, params, delay):
           # Phase components return a DualFloat (int + frac cycles).
           pepoch = params.epoch_dual(self.pepoch_name)
           dt = toa_data.tdb - pepoch
           ...

The ``__call__`` signature depends on the base class -- all three take
``(self, toa_data, params, delay)``, but a :class:`~jaxpint.components.DelayComponent`
returns a seconds array, a :class:`~jaxpint.components.PhaseComponent` returns a
:class:`~jaxpint.types.DualFloat`, and a
:class:`~jaxpint.components.NoiseComponent` instead implements ``covariance(...)``
returning the Woodbury triple ``(N_diag, U, Phi_diag)``. Read the base classes
in :mod:`jaxpint.components` for the exact contracts.

The only rule for JIT-traceability: **all fields must be static metadata**
(``eqx.field(static=True)``). The sole dynamic leaf in the whole model is the
:class:`~jaxpint.types.ParameterVector`; components hold *names*, not values, and
look values up at call time.

2. Give it a Component identity
-------------------------------

Add an entry to the :class:`~jaxpint.par.registry.Component` enum in
``jaxpint/par/registry.py``. This enum is the shared vocabulary that the parser,
the component table, and the builder all key off:

.. code-block:: python

   class Component(Enum):
       ...
       SPINDOWN = "Spindown"

3. Register it in the component table
-------------------------------------

This is the consolidated step. In ``jaxpint.par.registry_table`` -- the
single source of truth -- do two small edits:

**(a)** Add a ``jaxpint.par.registry_table.ComponentSpec`` to the
``COMPONENTS`` tuple. This one line carries the component's identity, its PINT
class name(s) (for the optional bridge), and its execution order:

.. code-block:: python

   COMPONENTS = (
       ...
       ComponentSpec(C.SPINDOWN, ("Spindown",), order=19),
       ...
   )

**(b)** Add the class to the dict returned by ``_param_classes()`` (a lazy,
function-local import so the table stays import-light and PINT-free):

.. code-block:: python

   def _param_classes():
       from jaxpint.phase.spin import Spindown
       ...
       return {
           ...
           C.SPINDOWN: (Spindown,),   # tuple: binary maps several classes to one Component
           ...
       }

Everything else is **derived** from these two edits at import time (validated by
``_validate()``, which checks the table still covers the whole ``Component``
enum):

- **Parsing** -- the class's ``PARAMS`` are aggregated into the parser's
  vocabulary (``PARAM_SPEC`` / ``ALIAS_MAP`` / ``TRIGGER_MAP`` / ``BINARY_PARAMS``
  in :mod:`jaxpint.par.spec`).
- **Detection** -- automatic: a parameter *uniquely owned* by exactly one
  non-binary component activates ("triggers") that component when it appears in a
  ``.par`` file. You never declare triggers by hand.
- **Execution order** -- ``DEFAULT_ORDER`` / ``PRIORITY`` in
  ``jaxpint/_component_order.py`` derive from the ``order=`` field (see step 4).
- **PINT bridge** -- ``PINT_COMPONENT_MAP`` in ``jaxpint.par.components``
  derives from the ``pint_names`` tuple; the optional bridge uses it to detect
  the component from an in-memory PINT model. Omit ``pint_names`` (defaults to
  ``()``) if there is no corresponding PINT class.

4. Set its execution order
--------------------------

The ``order=`` integer on the ``ComponentSpec`` is the component's position in
``DEFAULT_ORDER`` (it mirrors PINT's ordering). Delays are chained in this order
-- each sees the accumulated delay from earlier components -- so placement matters
for delays; phases are summed, so their relative order is irrelevant. The values
are currently contiguous, so to insert a delay *between* two existing ones, bump
the ``order`` of the entries that follow.

Components that are detected/activated but never executed in the delay/phase
chain (``PHASE_OFFSET``, the admin-only ``NONE``) omit ``order=`` entirely.
Binary models set ``is_binary=True`` instead, which routes their params to
``BINARY_PARAMS`` and excludes them from the trigger map.

5. Build it from a parsed model
-------------------------------

In :mod:`jaxpint.model_builder`, write a ``_build_<comp>(ctx)`` function and
register it in the ``_BUILDERS`` dispatch table. The function receives a
:class:`~jaxpint.model_builder.BuildContext` (the parsed ``par``, optional
``toa_data``, and the pre-resolved astrometry names), reads the relevant
parameters off ``ctx.par``, and returns the constructed instance -- or ``None``
when the parameters are absent:

.. code-block:: python

   def _build_spindown(ctx: BuildContext):
       from jaxpint.phase.spin import Spindown

       par = ctx.par
       spin_names = ["F0"]
       for pname in par.params.names:
           if pname.startswith("F") and pname != "F0" and pname[1:].isdigit():
               spin_names.append(pname)
       spin_names.sort(key=lambda n: int(n[1:]))
       return Spindown(spin_param_names=tuple(spin_names))


   _BUILDERS = {
       ...
       Component.SPINDOWN: _build_spindown,
       ...
   }

You do **not** choose which list the result goes in: ``build_model`` runs the
builders in execution order and routes each return value to the delay, phase, or
noise bucket by its base class (``DelayComponent`` / ``PhaseComponent`` /
``NoiseComponent``). Noise objects are further partitioned by type into the
white / DM-white / correlated slots automatically. (The one explicit case is
``ScaleDmError``, which scales DM-domain uncertainties and so subclasses
``eqx.Module`` rather than ``NoiseComponent``; the router names it directly.)

6. Export it
------------

Add the class to its subpackage's ``__init__`` (e.g. ``jaxpint/phase/__init__.py``)
and to the top-level ``jaxpint/__init__.py`` so users can import it directly.

Testing
-------

The repo validates components for numerical parity against PINT. A new component
should come with a test that compares its contribution (or end-to-end residuals)
against PINT for a representative ``.par`` / ``.tim`` pair, alongside any
unit-level checks of its math. See the existing ``tests/`` for the patterns
(differential parity, Hypothesis property tests). ``tests/test_registry_consistency.py``
additionally checks that every ``ComponentSpec`` resolves and that the table
covers the ``Component`` enum, so a half-registered component fails fast there.
