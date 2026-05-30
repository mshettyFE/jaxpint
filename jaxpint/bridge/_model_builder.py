"""Back-compat shim.

``build_model`` (and its helpers) moved to :mod:`jaxpint.model_builder` — it is
PINT-free and no longer belongs under the PINT bridge.  This module re-exports
the public names so existing imports keep working.
"""

from jaxpint.model_builder import *  # noqa: F401,F403
from jaxpint.model_builder import build_model, _build_binary  # noqa: F401
