"""Shared parameter declarations for the binary models.

The orbital parameters common to every supported binary model live here so each
``Binary*`` class can splice them into its ``PARAMS`` (``PARAMS = (*BINARY_CORE,
*extras)``) without duplication.  Binary params do not ``trigger`` a component —
the binary model is selected from the ``BINARY`` line, not from param presence.
"""

from __future__ import annotations

from jaxpint.components import ParamDecl

# Common to BT / BT_piecewise / DD(+DDS/DDH) / DDK / DDGR / ELL1(+ELL1H/ELL1k).
# Units are kept only where the runtime needs them (deg -> rad); the rest are
# documentation the parser ignores.  PBDOT/A1DOT/EDOT carry PINT unit_scale.
BINARY_CORE = (
    ParamDecl("PB"),
    ParamDecl("A1"),
    ParamDecl("ECC", aliases=("E",)),
    ParamDecl("OM", unit="deg"),
    ParamDecl("OMDOT", unit="deg / yr"),
    ParamDecl("PBDOT", scale=1e-12, scale_threshold=1e-7),
    ParamDecl("A1DOT", aliases=("XDOT",), scale=1e-12, scale_threshold=1e-7),
    ParamDecl("EDOT", scale=1e-12, scale_threshold=1e-7),
    ParamDecl("FB0", prefix="FB"),
)
