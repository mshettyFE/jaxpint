"""Component ordering mirroring PINT's DEFAULT_ORDER.

Defines the canonical execution order for timing model components.
Used by :func:`jaxpint.model_builder.build_model` to process
components via a priority queue so that delays are chained in the
correct physical order.
"""

from __future__ import annotations

from jaxpint.par import registry_table
from jaxpint.par.registry import Component

# Mirrors PINT's DEFAULT_ORDER (timing_model.py:119-135), derived from the
# single-source-of-truth registry's ``order`` field (which encodes the same
# sequence).  Deriving the order does not import any component class.
DEFAULT_ORDER: tuple[Component, ...] = registry_table.derive_default_order()

# Priority lookup: Component → position in DEFAULT_ORDER.
# Components not in DEFAULT_ORDER get len(DEFAULT_ORDER) (sort to end).
PRIORITY: dict[Component, int] = {comp: i for i, comp in enumerate(DEFAULT_ORDER)}
