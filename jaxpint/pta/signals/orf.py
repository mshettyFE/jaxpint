"""Overlap reduction functions (ORFs) for inter-pulsar correlations.

Ported from Discovery's ``signals.py``.  These are pure geometry functions
mapping pulsar-pair angular separation to a correlation coefficient.

References
----------
.. [1] Hellings & Downs (1983), "Upper limits on the isotropic
   gravitational radiation background from pulsar timing analysis",
   ApJL 265, L39.  Eq. 2 (Hellings-Downs curve).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def hd_orf(
    pos1: Float[Array, "3"], pos2: Float[Array, "3"]
) -> Float[Array, ""]:
    """Hellings-Downs overlap reduction function.

    Implements Eq. 2 of Hellings & Downs (1983) [1]_:

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
        HD correlation coefficient in [−1/8, 1/2].

    References
    ----------
    .. [1] Hellings & Downs (1983), ApJL 265, L39.
    """
    omc2 = (1.0 - jnp.dot(pos1, pos2)) / 2.0
    # Guard against log(0) when pos1 == pos2 (self-correlation → 0.5)
    omc2 = jnp.clip(omc2, 1e-30)
    return 1.5 * omc2 * jnp.log(omc2) - 0.25 * omc2 + 0.5


def monopole_orf(
    pos1: Float[Array, "3"], pos2: Float[Array, "3"]
) -> Float[Array, ""]:
    """Monopole ORF (isotropic, unit correlation for all pairs)."""
    return jnp.where(jnp.allclose(pos1, pos2), 1.0, 1.0)


def dipole_orf(
    pos1: Float[Array, "3"], pos2: Float[Array, "3"]
) -> Float[Array, ""]:
    """Dipole ORF (correlation proportional to cos(angle))."""
    return jnp.dot(pos1, pos2)
