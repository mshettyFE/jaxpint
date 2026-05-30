Adding a Timing-Model Component
===============================

A timing-model component is an `Equinox <https://github.com/patrick-kidger/equinox>`_
module that contributes either a **delay** (seconds), a **phase** (cycles), or a
**noise** term (a Woodbury covariance triple). This guide walks through adding a
new one, using the existing :class:`~jaxpint.phase.Spindown` phase component as
the worked example.

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
:class:`~jaxpint.dual_float.DualFloat`, and a
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
the execution-order table, and the builder all key off:

.. code-block:: python

   class Component(Enum):
       ...
       SPINDOWN = "Spindown"

3. Register it for parsing & detection
--------------------------------------

Add the class to ``_component_classes()`` in :mod:`jaxpint.par.spec`. That
function is the single list of components whose ``PARAMS`` get aggregated into
the parser's vocabulary, so registering it there is what makes the ``.par``
parser recognize the component's parameters.

Detection is then **automatic**: a parameter that is *uniquely owned* by exactly
one non-binary component activates ("triggers") that component when it appears in
a ``.par`` file. You do not declare triggers by hand -- they're derived in
``jaxpint.par.spec`` from the aggregated schemas. (If you also want the optional
PINT bridge to detect the component from an in-memory PINT model, add a
PINT-class-name → ``Component`` entry to ``PINT_COMPONENT_MAP`` in
``jaxpint.par.components``.)

4. Slot it into the execution order
-----------------------------------

Add the new ``Component`` to ``DEFAULT_ORDER`` in ``jaxpint/_component_order.py``
at the right physical position. Delays are chained in this order (each sees the
accumulated delay from earlier components), so placement matters for delays;
phases are summed, so their order is irrelevant.

5. Build it from a parsed model
-------------------------------

Add a branch to :func:`jaxpint.model_builder.build_model` (it ``match``-es on the
detected ``Component`` set) that reads the relevant parameters out of the
:class:`~jaxpint.par.result.ParResult` and constructs your instance with the
right ``*_name`` fields:

.. code-block:: python

   case Component.SPINDOWN:
       spin_names = _collect_prefix_indices(par, "F")  # F0, F1, ...
       phase_components.append(Spindown(spin_param_names=tuple(spin_names)))

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
(differential parity, Hypothesis property tests).
