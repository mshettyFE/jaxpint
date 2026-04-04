"""DD model with H3/STIGMA Shapiro parameterization (DDH).

Uses orthometric Shapiro parameters ``H3`` and ``STIGMA`` instead
of ``M2`` and ``SINI``.

Reference
---------
Freire & Wex (2010), MNRAS, 409, 199.
"""

from __future__ import annotations

import equinox as eqx

from jaxpint.binary.dd import BinaryDD


class BinaryDDH(BinaryDD):
    """DD model with H3/STIGMA Shapiro parameterization (DDH).

    Uses ``H3`` and ``STIGMA`` instead of ``M2`` and ``SINI``.
    """

    shapiro_mode: str = eqx.field(static=True, default="h3stigma")
