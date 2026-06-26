"""Backward-compatible re-export of :class:`~jaxpint.types.GlobalParams`.

``GlobalParams`` is a foundational named-vector pytree (a sibling of
:class:`~jaxpint.types.ParameterVector`) and now lives in
:mod:`jaxpint.types.global_params`. It is re-exported here so existing
``from jaxpint.pta.params import GlobalParams`` / ``from jaxpint.pta import
GlobalParams`` imports keep working.
"""

from jaxpint.types.global_params import GlobalParams

__all__ = ["GlobalParams"]
