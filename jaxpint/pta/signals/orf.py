"""Overlap reduction functions (ORFs) for inter-pulsar correlations.

Ported from Discovery's ``signals.py``.  These are pure geometry functions
mapping pulsar-pair angular separation to a correlation coefficient.

References
----------
.. [orf_mod_hd83] Hellings & Downs (1983), "Upper limits on the isotropic
   gravitational radiation background from pulsar timing analysis",
   ApJL 265, L39.  Eq. 2 (Hellings-Downs curve).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def hd_orf(pos1: Float[Array, "3"], pos2: Float[Array, "3"]) -> Float[Array, ""]:
    """Hellings-Downs overlap reduction function.

    Implements Eq. 2 of Hellings & Downs (1983) [orf_hd83]_:

    .. math::
        C(\\xi) = \\frac{3}{2} x \\ln x - \\frac{x}{4} + \\frac{1}{2},
        \\quad x = \\frac{1 - \\cos\\xi}{2}

    Parameters
    ----------
    pos1, pos2 : (3,) arrays
        Unit vectors pointing to the two pulsars.

    Returns
    -------
    float
        HD correlation coefficient: cross-pulsar values in [−1/8, 1/2],
        with Γ_aa = 1.0 on the diagonal (``pos1 == pos2``).

    Notes
    -----
    The diagonal is the **auto**-correlation \Gamma_aa = 1.0 — twice the \zeta→0
    limit of the cross-correlation curve (0.5), because a single pulsar's
    GWB response is fully correlated with itself (both the plus and cross
    response terms survive, whereas for two distinct co-located pulsars
    only the average does).  Matches discovery/enterprise's ``hd_orf``
    special case; without it every pulsar receives half the GWB
    auto-power and amplitude posteriors are biased.

    References
    ----------
    .. [orf_hd83] Hellings & Downs (1983), ApJL 265, L39.
    """
    omc2 = (1.0 - jnp.dot(pos1, pos2)) / 2.0
    # Clip guards log(0) in the (masked) cross-correlation branch when
    # pos1 == pos2; the diagonal takes the explicit 1.0 branch below.
    omc2 = jnp.clip(omc2, 1e-30)
    cross = 1.5 * omc2 * jnp.log(omc2) - 0.25 * omc2 + 0.5
    return jnp.where(jnp.all(pos1 == pos2), 1.0, cross)


def monopole_orf(pos1: Float[Array, "3"], pos2: Float[Array, "3"]) -> Float[Array, ""]:
    """Monopole ORF (isotropic, unit correlation for all pairs).

    Returns 1.0 for every distinct pulsar pair and ``1.0 + 1e-6`` on the
    diagonal — the conditioning trick from enterprise/discovery.  Without
    it Γ is the all-ones matrix (rank 1), and the ``inv``/``cholesky`` in
    the correlated outer tier returns silent NaNs under JAX.

    Parameters
    ----------
    pos1, pos2 : (3,) arrays
        Unit vectors pointing to the two pulsars.

    Returns
    -------
    float
        ``1.0 + 1e-6`` when ``pos1 == pos2``, else 1.0.
    """
    return jnp.where(jnp.all(pos1 == pos2), 1.0 + 1.0e-6, 1.0)


def dipole_orf(pos1: Float[Array, "3"], pos2: Float[Array, "3"]) -> Float[Array, ""]:
    """Dipole ORF (correlation proportional to cos(angle)).

    Returns the cosine of the angular separation between the two pulsars,
    ``dot(pos1, pos2)``, with ``1.0 + 1e-6`` on the diagonal — the
    conditioning trick from enterprise/discovery.  Without it
    Γ = P Pᵀ has rank ≤ 3 for more than 3 pulsars, and the
    ``inv``/``cholesky`` in the correlated outer tier returns silent NaNs
    under JAX.

    Parameters
    ----------
    pos1, pos2 : (3,) arrays
        Unit vectors pointing to the two pulsars.

    Returns
    -------
    float
        ``1.0 + 1e-6`` when ``pos1 == pos2``, else the cosine of the
        angular separation, in [-1, 1].
    """
    return jnp.where(jnp.all(pos1 == pos2), 1.0 + 1.0e-6, jnp.dot(pos1, pos2))
