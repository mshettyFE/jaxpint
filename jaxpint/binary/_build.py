"""Binary family construction + self-registration.

five model
classes (``BinaryBT/DD/DDK/DDGR/ELL1``) map to one ``Component.BINARY`` and the
construction dispatches on the ``BINARY``-line model name (``BT/DD/DDS/DDH/…``),
not on a single class. So it registers via :func:`register_family` with a
module-level ``build_binary`` dispatcher rather than a per-class ``build``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jaxpint._build_context import opt_name as _opt_name, param_is_set as _param_is_set
from jaxpint.binary.bt import BinaryBT
from jaxpint.binary.bt_piecewise import BinaryBTPiecewise
from jaxpint.binary.dd import BinaryDD
from jaxpint.binary.ddgr import BinaryDDGR
from jaxpint.binary.ddk import BinaryDDK
from jaxpint.binary.ell1 import BinaryELL1
from jaxpint.par._component_registry import register_family
from jaxpint.par.registry import BinaryModel, Component
from jaxpint.par.result import ParResult

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


def _dd_common_kwargs(par: ParResult) -> dict:
    return dict(
        pb_name="PB",
        t0_name="T0",
        a1_name="A1",
        ecc_name="ECC",
        om_name="OM",
        pbdot_name=_opt_name(par, "PBDOT"),
        omdot_name=_opt_name(par, "OMDOT"),
        edot_name=_opt_name(par, "EDOT"),
        a1dot_name=_opt_name(par, "A1DOT"),
        xpbdot_name=_opt_name(par, "XPBDOT"),
        gamma_name=_opt_name(par, "GAMMA"),
        dr_name=_opt_name(par, "DR"),
        dth_name=_opt_name(par, "DTH"),
        a0_name=_opt_name(par, "A0"),
        b0_name=_opt_name(par, "B0"),
    )


def _ell1_common_kwargs(par: ParResult) -> dict:
    return dict(
        pb_name="PB",
        tasc_name="TASC",
        a1_name="A1",
        eps1_name="EPS1",
        eps2_name="EPS2",
        pbdot_name=_opt_name(par, "PBDOT"),
        a1dot_name=_opt_name(par, "A1DOT"),
        xpbdot_name=_opt_name(par, "XPBDOT"),
    )


def build_binary(ctx: "BuildContext") -> object:
    """Construct the appropriate binary delay component (family dispatcher)."""
    par = ctx.par
    bname = par.binary_model
    if bname is None:
        raise ValueError("No BINARY model specified in .par file")

    match bname:
        case BinaryModel.BT:
            return BinaryBT(
                pb_name="PB",
                t0_name="T0",
                a1_name="A1",
                ecc_name="ECC",
                om_name="OM",
                pbdot_name=_opt_name(par, "PBDOT"),
                omdot_name=_opt_name(par, "OMDOT"),
                edot_name=_opt_name(par, "EDOT"),
                a1dot_name=_opt_name(par, "A1DOT"),
                gamma_name=_opt_name(par, "GAMMA"),
                xpbdot_name=_opt_name(par, "XPBDOT"),
            )

        case BinaryModel.DD:
            return BinaryDD(
                **_dd_common_kwargs(par),
                m2_name=_opt_name(par, "M2"),
                sini_name=_opt_name(par, "SINI"),
                shapiro_mode="standard",
            )

        case BinaryModel.DDS:
            return BinaryDD(
                **_dd_common_kwargs(par),
                m2_name=_opt_name(par, "M2"),
                shapmax_name="SHAPMAX",
                shapiro_mode="shapmax",
            )

        case BinaryModel.DDH:
            return BinaryDD(
                **_dd_common_kwargs(par),
                h3_name="H3",
                stigma_name="STIGMA",
                shapiro_mode="h3stigma",
            )

        case BinaryModel.ELL1:
            return BinaryELL1(
                **_ell1_common_kwargs(par),
                eps1dot_name=_opt_name(par, "EPS1DOT"),
                eps2dot_name=_opt_name(par, "EPS2DOT"),
                m2_name=_opt_name(par, "M2"),
                sini_name=_opt_name(par, "SINI"),
                shapiro_mode="standard" if _param_is_set(par, "M2") else "none",
            )

        case BinaryModel.ELL1H:
            if _param_is_set(par, "STIGMA"):
                shapiro_mode = "h3stigma"
            elif _param_is_set(par, "H4"):
                shapiro_mode = "h3h4"
            elif _param_is_set(par, "H3"):
                shapiro_mode = "h3nharms"
            else:
                shapiro_mode = "none"
            nharms = par.int_params.get("NHARMS", 7)
            return BinaryELL1(
                **_ell1_common_kwargs(par),
                eps1dot_name=_opt_name(par, "EPS1DOT"),
                eps2dot_name=_opt_name(par, "EPS2DOT"),
                h3_name=_opt_name(par, "H3"),
                stigma_name=_opt_name(par, "STIGMA"),
                h4_name=_opt_name(par, "H4"),
                shapiro_mode=shapiro_mode,
                nharms=nharms,
            )

        case BinaryModel.ELL1k:
            return BinaryELL1(
                **_ell1_common_kwargs(par),
                omdot_name=_opt_name(par, "OMDOT"),
                lnedot_name=_opt_name(par, "LNEDOT"),
                m2_name=_opt_name(par, "M2"),
                sini_name=_opt_name(par, "SINI"),
                shapiro_mode="standard" if _param_is_set(par, "M2") else "none",
            )

        case BinaryModel.DDK:
            k96 = par.bool_params.get("K96", False)
            return BinaryDDK(
                **_dd_common_kwargs(par),
                m2_name=_opt_name(par, "M2"),
                kin_name="KIN",
                kom_name="KOM",
                px_name="PX",
                raj_name=ctx.raj,
                decj_name=ctx.decj,
                pmra_name=ctx.pmra,
                pmdec_name=ctx.pmdec,
                posepoch_name=ctx.posepoch,
                k96=k96,
            )

        case BinaryModel.DDGR:
            return BinaryDDGR(
                pb_name="PB",
                t0_name="T0",
                a1_name="A1",
                ecc_name="ECC",
                om_name="OM",
                mtot_name="MTOT",
                m2_name="M2",
                edot_name=_opt_name(par, "EDOT"),
                a1dot_name=_opt_name(par, "A1DOT"),
                xomdot_name=_opt_name(par, "XOMDOT"),
                xpbdot_name=_opt_name(par, "XPBDOT"),
                a0_name=_opt_name(par, "A0"),
                b0_name=_opt_name(par, "B0"),
            )

        case BinaryModel.BT_PIECEWISE:
            t0x_names = par.params.names_with_prefix("T0X_")
            a1x_names = par.params.names_with_prefix("A1X_")
            xr1_names = par.params.names_with_prefix("XR1_")
            xr2_names = par.params.names_with_prefix("XR2_")
            n_pieces = len(xr1_names)

            return BinaryBTPiecewise(
                pb_name="PB",
                t0_name="T0",
                a1_name="A1",
                ecc_name="ECC",
                om_name="OM",
                pbdot_name=_opt_name(par, "PBDOT"),
                omdot_name=_opt_name(par, "OMDOT"),
                edot_name=_opt_name(par, "EDOT"),
                a1dot_name=_opt_name(par, "A1DOT"),
                gamma_name=_opt_name(par, "GAMMA"),
                xpbdot_name=_opt_name(par, "XPBDOT"),
                n_pieces=n_pieces,
                t0x_names=tuple(t0x_names),
                a1x_names=tuple(a1x_names),
                xr1_names=tuple(xr1_names),
                xr2_names=tuple(xr2_names),
            )

        case _:
            raise NotImplementedError(
                f"Binary model {bname!r} is not yet ported to JaxPINT"
            )


# --- Family registration (the carve-out): metadata + classes + family build ---
register_family(
    component=Component.BINARY,
    classes=(BinaryBT, BinaryDD, BinaryDDK, BinaryDDGR, BinaryELL1),
    build=build_binary,
    is_binary=True,
)
register_family(
    component=Component.BINARY_BT_PIECEWISE,
    classes=(BinaryBTPiecewise,),
    build=build_binary,
    is_binary=True,
)
