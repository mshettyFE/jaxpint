"""ELL1 model with OMDOT/LNEDOT for short-period binaries (ELL1k).

Uses periastron advance (OMDOT) and eccentricity growth rate (LNEDOT)
instead of EPS1DOT/EPS2DOT.

Reference
---------
Susobhanan et al. (2018), A&A, 618, A185.
"""

from __future__ import annotations

from jaxpint.binary.ell1 import BinaryELL1


class BinaryELL1k(BinaryELL1):
    """ELL1 model with OMDOT/LNEDOT for short-period binaries (ELL1k).

    Susobhanan et al. (2018) — uses OMDOT and LNEDOT instead of
    EPS1DOT/EPS2DOT.
    """
    pass
