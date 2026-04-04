"""DD model with SHAPMAX parameterization (DDS).

Uses ``SHAPMAX = -ln(1 - sin(i))`` instead of ``SINI`` for the
Shapiro delay.

Reference
---------
Kramer et al. (2006), Science, 314, 97.
"""

from __future__ import annotations

import equinox as eqx

from jaxpint.binary.dd import BinaryDD


class BinaryDDS(BinaryDD):
    """DD model with SHAPMAX parameterization (DDS).

    Uses ``SHAPMAX = -ln(1 - sin(i))`` instead of ``SINI``.
    """

    shapiro_mode: str = eqx.field(static=True, default="shapmax")
