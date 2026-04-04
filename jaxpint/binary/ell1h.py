"""ELL1 model with H3/STIGMA or H3/H4 Shapiro delay (ELL1H).

Uses harmonic decomposition of the Shapiro delay for
low-eccentricity orbits.

Reference
---------
Freire & Wex (2010), MNRAS, 409, 199.
"""

from __future__ import annotations

import equinox as eqx

from jaxpint.binary.ell1 import BinaryELL1


class BinaryELL1H(BinaryELL1):
    """ELL1 model with H3/STIGMA or H3/H4 Shapiro delay (ELL1H).

    Uses harmonic decomposition from Freire & Wex (2010).
    """

    shapiro_mode: str = eqx.field(static=True, default="h3stigma")
