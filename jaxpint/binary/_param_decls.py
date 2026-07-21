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

BINARY_CORE_NAMES: frozenset[str] = frozenset(d.name for d in BINARY_CORE)


# ---------------------------------------------------------------------------
# Per-model parameter sets
#
# Which parameters each *model* accepts, beyond BINARY_CORE.  This cannot be
# read off the classes: ten models are implemented by six classes (DD/DDS/DDH
# all build ``BinaryDD``; ELL1/ELL1H/ELL1k all build ``BinaryELL1``), so a
# class's ``PARAMS`` is the *union* over its variants -- ``BinaryDD.PARAMS``
# holds SHAPMAX and H3/STIGMA together and cannot say which belongs to which.
#
# It lives here, next to the models, so the split is declared once.  The parser
# derives ``spec.BINARY_MODEL_PARAMS`` from it to resolve tempo2's ``BINARY T2``
# (pick the simplest model whose parameter set covers the file) and the
# no-BINARY-line fallback.  Keyed by ``BinaryModel`` *value* strings so this
# module stays dependency-free; ``test_model_params_cover_every_binary_model``
# pins the keys against the enum.
# ---------------------------------------------------------------------------

_BT_EXTRA = frozenset({"T0", "GAMMA"})
_DD_EXTRA = _BT_EXTRA | {"A0", "B0", "DR", "DTH", "M2", "SINI"}
_ELL1_EXTRA = frozenset({"TASC", "EPS1", "EPS2", "EPS1DOT", "EPS2DOT", "M2", "SINI"})

MODEL_EXTRA_PARAMS: dict[str, frozenset[str]] = {
    "BT": _BT_EXTRA,
    "BT_piecewise": _BT_EXTRA | {"A1X_0000", "T0X_0000", "XR1_0000", "XR2_0000"},
    "ELL1": _ELL1_EXTRA,
    "ELL1H": _ELL1_EXTRA | {"H3", "H4", "STIGMA", "NHARMS"},
    "ELL1k": _ELL1_EXTRA | {"LNEDOT"},
    "DD": _DD_EXTRA,
    "DDK": _DD_EXTRA | {"KIN", "KOM", "K96"},
    "DDGR": _DD_EXTRA | {"MTOT", "XOMDOT", "XPBDOT"},
    "DDS": _DD_EXTRA | {"SHAPMAX"},
    "DDH": _DD_EXTRA | {"H3", "STIGMA"},
}
