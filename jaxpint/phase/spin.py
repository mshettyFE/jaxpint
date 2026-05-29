"""Spindown phase component: polynomial spin model.

Ports PINT's ``Spindown`` class as a pure Equinox module.  The pulse phase
is modelled as a Taylor expansion of frequency derivatives about PEPOCH:

    phase(t) = F0*dt + F1*dt^2/2! + F2*dt^3/3! + ...

where dt = (t_TDB - PEPOCH) in seconds, minus accumulated delay.

All hand-coded derivatives are omitted; ``jax.jacobian`` through
``__call__`` replaces PINT's ``d_phase_d_F`` and ``d_spindown_phase_d_delay``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl, PhaseComponent
from jaxpint.constants import SECS_PER_DAY
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import taylor_horner, taylor_horner_deriv, taylor_horner_phase


class Spindown(PhaseComponent):
    """Polynomial spindown phase component.

    Parameters
    ----------
    spin_param_names : tuple[str, ...]
        Names of the spin-frequency parameters in the ``ParameterVector``,
        ordered by derivative index.  E.g. ``("F0",)`` or ``("F0", "F1", "F2")``.
    pepoch_name : str
        Name of the reference-epoch parameter (default ``"PEPOCH"``).

    Raises
    ------
    ValueError
        If no spin terms are provided (``spin_param_names`` is empty).
    ValueError
        If the first spin term is not ``'F0'``.
    """

    PARAMS = (
        ParamDecl("F0", unit="Hz", prefix="F"),
        ParamDecl("PEPOCH", kind="mjd"),
    )

    spin_param_names: tuple[str, ...] = eqx.field(static=True)
    pepoch_name: str = eqx.field(static=True, default="PEPOCH")

    def __check_init__(self):
        if len(self.spin_param_names) == 0:
            raise ValueError("Spindown requires at least one spin term (F0)")
        if self.spin_param_names[0] != "F0":
            raise ValueError(
                f"First spin term must be 'F0', got '{self.spin_param_names[0]}'"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_spin_coeffs(
        self, params: ParameterVector
    ) -> Float[Array, " n_terms_plus_1"]:
        """Assemble ``[0.0, F0, F1, ..., FN]`` for :func:`taylor_horner`.

        The leading zero is the constant phase term (no phase offset at PEPOCH).
        """
        f_values = jnp.array(
            [params.param_value(name) for name in self.spin_param_names]
        )
        return jnp.concatenate([jnp.zeros(1), f_values])

    def _get_spin_coeffs_scaled(
        self, params: ParameterVector
    ) -> Float[Array, " n_terms_plus_1"]:
        """Same as ``_get_spin_coeffs`` but with each F_k pre-divided by k!.

        Hands :func:`taylor_horner_phase` coefficients in the form
        ``[0, F0, F1/2!, F2/3!, ...]`` so its Horner loop doesn't need
        per-step division — keeping ``c_int * x_int_s`` an exact
        integer × integer product.
        """
        f_values = jnp.array(
            [params.param_value(name) for name in self.spin_param_names]
        )
        n = f_values.shape[0]
        # cumprod([1, 2, 3, ..., n]) = [1!, 2!, 3!, ..., n!]
        factorials = jnp.cumprod(jnp.arange(1, n + 1, dtype=jnp.float64))
        f_scaled = f_values / factorials
        return jnp.concatenate([jnp.zeros(1), f_scaled])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> DualFloat:
        """Compute spindown phase contribution.

        Implements Eq. 120 of Edwards, Hobbs & Manchester (2006),
        MNRAS 372, 1549 (Tempo2 Paper II).

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times, etc.).
        params : ParameterVector
            Timing-model parameters containing F0, F1, ..., and PEPOCH.
        delay : array, shape (n_toas,)
            Accumulated signal delay in **seconds**.

        Returns
        -------
        DualFloat
            Pulse phase in cycles (dimensionless), split as int + frac.
        """
        pepoch = params.epoch_dual(self.pepoch_name)
        dt = toa_data.tdb - pepoch
        coeffs = self._get_spin_coeffs_scaled(params)
        return taylor_horner_phase(dt.int, dt.frac, delay, coeffs)

    def change_pepoch(
        self,
        params: ParameterVector,
        new_epoch_int: float,
        new_epoch_frac: float,
    ) -> ParameterVector:
        """Re-derive spin parameters at a new reference epoch.

        The physical prediction is invariant: only the Taylor coefficients
        and PEPOCH change so that the expansion is centred at the new epoch.

        .. note::
            Not JIT-compatible (modifies static ``epoch_int_values``).
            Intended for model setup, not inner-loop computation.

        Parameters
        ----------
        params : ParameterVector
            Current parameter values.
        new_epoch_int : float
            Integer MJD day of the new epoch.
        new_epoch_frac : float
            Fractional MJD day of the new epoch.

        Returns
        -------
        ParameterVector
            Updated with new PEPOCH and recomputed F-terms.
        """
        old_int, old_frac = params.epoch_value(self.pepoch_name)
        dt_days = (new_epoch_int - old_int) + (new_epoch_frac - old_frac)
        dt_seconds = jnp.asarray(dt_days * SECS_PER_DAY)

        coeffs = self._get_spin_coeffs(params)

        new_params = params
        for i, name in enumerate(self.spin_param_names):
            new_val = taylor_horner_deriv(dt_seconds, coeffs, deriv_order=i + 1)
            new_params = new_params.with_value(name, new_val)

        # Update PEPOCH: fractional part in values, integer part in static dict.
        # epoch_int_values is static (not a pytree leaf), so we reconstruct
        # the ParameterVector rather than using eqx.tree_at.
        new_params = new_params.with_value(self.pepoch_name, new_epoch_frac)
        new_epoch_ints = dict(new_params.epoch_int_values)
        new_epoch_ints[self.pepoch_name] = new_epoch_int
        new_params = ParameterVector(
            values=new_params.values,
            frozen_mask=new_params.frozen_mask,
            names=new_params.names,
            units=new_params.units,
            epoch_int_values=new_epoch_ints,
            marginalized_mask=new_params.marginalized_mask,
        )

        return new_params
