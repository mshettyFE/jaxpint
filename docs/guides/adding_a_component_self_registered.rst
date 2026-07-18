Adding a Timing-Model Component
===============================

A timing-model component is an `Equinox <https://github.com/patrick-kidger/equinox>`_
module that contributes a **delay** (seconds), a **phase** (cycles), or a
**noise** term (a Woodbury covariance triple). Components **self-register**: a
class declares its identity and how to build it *on the class itself*, and the
parser vocabulary, auto-detection, model builder, and optional PINT bridge all
*derive* from that one declaration. There are no parallel tables to keep in sync.

Adding a 1:1 component touches three places:

- the component class -- its ``PARAMS`` schema, a ``@register_component``
  decorator, and a ``build`` classmethod;
- one member on the :class:`~jaxpint.par.registry.Component` enum;
- the package ``__init__`` exports -- which is what makes the module import, and
  therefore register.

Order-sensitive delays additionally name their ``Component`` in one ordered list
(step 5). Everything else is derived.

Every component carries two related-but-distinct descriptions of its parameters,
explained in detail in :mod:`jaxpint.components`:

- ``PARAMS`` -- the static *schema* (what parameters exist; their units /
  prefixes / aliases), used at parse time;
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
       # Static schema: what this component models (used by the .par parser).
       PARAMS = (
           ParamDecl("F0", unit="Hz", prefix="F"),   # prefix => F0, F1, F2, ...
           ParamDecl("PEPOCH", kind="mjd"),
       )

       # Runtime config: the concrete names this instance reads at evaluation.
       spin_param_names: tuple[str, ...] = eqx.field(static=True)
       pepoch_name: str = eqx.field(static=True, default="PEPOCH")

       def __call__(self, toa_data, params, delay):
           # Phase components return a DualFloat (integer + fractional cycles).
           ...

The ``__call__`` contract depends on the base class: a
:class:`~jaxpint.components.DelayComponent` returns a seconds array, a
:class:`~jaxpint.components.PhaseComponent` returns a
:class:`~jaxpint.types.DualFloat`, and a
:class:`~jaxpint.components.NoiseComponent` instead implements ``covariance(...)``
returning the Woodbury triple ``(N_diag, U, Phi_diag)``. Read the base classes
in :mod:`jaxpint.components` for the exact contracts.

The one rule for JIT-traceability: **all fields must be static metadata**
(``eqx.field(static=True)``). The sole dynamic leaf in the model is the
:class:`~jaxpint.types.ParameterVector`; components hold parameter *names*, not
values, and look values up at call time.

2. Give it a Component identity
-------------------------------

Add a member to the :class:`~jaxpint.par.registry.Component` enum in
``jaxpint/par/registry.py``. This enum is the shared key the parser, the
registry, and the builder all use:

.. code-block:: python

   class Component(Enum):
       ...
       SPINDOWN = "Spindown"

3. Self-register the component
------------------------------

Decorate the class with ``@register_component`` (from
``jaxpint.par._component_registry``) and give it a ``build`` classmethod. The
decorator records the component's identity and optional PINT class name(s); the
``build`` classmethod constructs a configured instance from a parsed model:

.. code-block:: python

   from typing import TYPE_CHECKING

   from jaxpint.par._component_registry import register_component
   from jaxpint.par.registry import Component

   if TYPE_CHECKING:
       from jaxpint._build_context import BuildContext


   @register_component(component=Component.SPINDOWN, pint_names=("Spindown",))
   class Spindown(PhaseComponent):
       ...

       @classmethod
       def build(cls, ctx: "BuildContext") -> "Spindown":
           idx = ctx.par.params.prefix_indices("F")
           spin_names = ["F0"] + [f"F{i}" for i in idx if i != 0]
           return cls(spin_param_names=tuple(spin_names))

``build`` receives a ``BuildContext`` (from ``jaxpint._build_context``) carrying
the parsed ``par``, the optional ``toa_data``, and the astrometry parameter
names resolved once up front (``raj`` / ``decj`` / ``pmra`` / ``pmdec`` /
``posepoch`` / ``obliquity_arcsec``). It reads what it needs off ``ctx.par`` and
returns the constructed instance -- or ``None`` when the component's parameters
are absent, so a component that doesn't apply to a given ``.par`` simply isn't
built.

``jaxpint._build_context`` also provides small helpers for common build chores,
imported locally inside ``build`` to keep the module import-light:
``opt_name(par, name)`` (the name if it is set, else ``None``),
``value(par, name)``, ``epoch_or_pepoch(par, name)`` (an epoch parameter,
falling back to ``PEPOCH``), and ``basis_seconds`` / ``span_seconds`` for
Fourier-basis and ECORR builders.

``pint_names`` is optional (defaults to ``()``): give the PINT component class
name(s) when there is a corresponding PINT component, so the optional bridge can
detect it from an in-memory PINT model. Omit it when there is none.

You do **not** choose which bucket the result lands in: ``build_model`` runs the
builders in execution order and routes each returned instance to the delay,
phase, or noise slot by its base class. Noise objects are further partitioned by
type into the white / DM-white / correlated slots automatically.

4. Export it
------------

Add the class to its subpackage's ``__init__`` (e.g. ``jaxpint/phase/__init__.py``)
and to the top-level ``jaxpint/__init__.py``. This is not merely for users'
import convenience: **registration is a side effect of importing the component
module**, and the package ``__init__`` is what imports it. A component whose
module is never imported never registers, and the registry's coverage check
(step 6) fails with the name of the ``Component`` that never showed up.

5. Set its execution order
--------------------------

Execution order is a *global arrangement* -- how delays chain -- not a
per-component fact, so it lives in one place: the ``EXECUTION_ORDER`` tuple in
``jaxpint.par.registry_table``. If your component is a **delay whose position in
the chain matters**, insert its ``Component`` member at the right spot:

.. code-block:: python

   EXECUTION_ORDER = (
       ...
       C.DISPERSION_DM,
       C.DISPERSION_DMX,
       ...
   )

Position *is* the order (it mirrors PINT's ordering) -- no integers to renumber,
no collisions. Delays are chained in this order (each sees the accumulated delay
from earlier components); **phases are summed**, so their relative order is
irrelevant and phase / noise components can be left out of ``EXECUTION_ORDER``
entirely (anything absent sorts to the end).

6. Testing
----------

The repo validates components for numerical parity against PINT. A new component
should come with a test comparing its contribution (or end-to-end residuals)
against PINT for a representative ``.par`` / ``.tim`` pair, plus any unit-level
checks of its math -- see the existing ``tests/`` for the patterns (differential
parity, Hypothesis property tests). ``tests/test_registry_consistency.py``
independently checks that the registry covers the whole ``Component`` enum, that
every registered class carries ``PARAMS``, and that ``pint_names`` are real PINT
component names, so a half-added component fails fast there.

Component families
------------------

A few components do not fit the one-class-to-one-``Component`` mould: the binary
models (``BinaryBT`` / ``BinaryDD`` / ``BinaryDDK`` / ``BinaryDDGR`` /
``BinaryELL1``) all map to a single ``Component.BINARY`` and dispatch on the
``BINARY``-line model name, so the 1:1 decorator does not apply. These register
as a **family** via ``register_family`` in a small registration-only module
(``jaxpint/binary/_build.py``):

.. code-block:: python

   from jaxpint.par._component_registry import register_family
   from jaxpint.par.registry import Component

   def build_binary(ctx):
       ...   # dispatch on ctx.par.binary_model; return the right Binary* instance

   register_family(
       component=Component.BINARY,
       classes=(BinaryBT, BinaryDD, BinaryDDK, BinaryDDGR, BinaryELL1),
       build=build_binary,
       is_binary=True,
   )

``register_family`` takes the full ``classes`` tuple (every class contributes
its ``PARAMS`` to the one component), a module-level ``build`` dispatcher, and
``is_binary`` (which routes those params to ``BINARY_PARAMS`` and excludes them
from the trigger map). Because such a module defines no class the package
``__init__`` would otherwise export, import it explicitly from the package
``__init__`` (``from jaxpint.binary import _build``) so the registration fires.

.. note::

   The registry is assembled **lazily** and validated on first use (typically
   the first parse), not at bare ``import jaxpint``. Self-registration means a
   component's decorator and ``build`` are recorded when its module imports;
   reading the registry eagerly during import could re-enter a component
   mid-import (a cycle), so the parser and bridge derivations defer to first use
   -- after every component package has finished importing. In practice this is
   invisible: any real use reads the registry, which triggers validation.

What you get for free
---------------------

From the single ``@register_component`` declaration (plus the enum member and
the export), everything else is *derived*:

- **Parsing** -- the class's ``PARAMS`` are aggregated into the parser
  vocabulary (``PARAM_SPEC`` / ``ALIAS_MAP`` / ``TRIGGER_MAP`` / ``BINARY_PARAMS``
  in :mod:`jaxpint.par.spec`).
- **Detection** -- automatic: a parameter *uniquely owned* by exactly one
  non-binary component activates ("triggers") that component when it appears in a
  ``.par`` file. You never declare triggers by hand.
- **Building** -- the model builder's dispatch table *is* the registry's
  ``build`` callables; there is nothing to wire up separately.
- **PINT bridge** -- the PINT-name map derives from ``pint_names``.
- **Execution order** -- ``PRIORITY`` derives from ``EXECUTION_ORDER`` (step 5).
