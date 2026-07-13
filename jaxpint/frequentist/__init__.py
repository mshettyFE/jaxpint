"""Frequentist-inference arm for JaxPINT.

The sibling of :mod:`jaxpint.bayes`: detection statistics, their null
calibrations, and detection-sensitivity forecasts, built on top of the PTA
likelihood machinery in :mod:`jaxpint.pta`.  Importing from
``jaxpint.frequentist`` flags that a script frames its results as test
statistics, p-values, and detection probabilities rather than posteriors.

Layout
------
- ``jaxpint.frequentist.stats`` — the PTA-agnostic distribution layer
  (``chi2``/``ncx2`` thresholds, detection probabilities, ``h0_min`` solves).
- ``jaxpint.frequentist.sensitivity`` — the noncentrality producer that
  feeds it from the real timing-marginalized likelihood.
- ``jaxpint.frequentist.detection`` — F-statistics (coherent sky-maximized
  ``F_e``, incoherent ``F_p``) and their empirical backgrounds.
- ``jaxpint.frequentist.nulls`` — statistic-agnostic null-calibration
  primitives (phase rotations, isotropic scrambles, empirical p-values).
- ``jaxpint.frequentist.optimal`` — the optimal statistic, the GWB
  cross-correlation amplitude estimator over pulsar pairs.

Both inference arms sit *above* :mod:`jaxpint.pta` in the import graph and
share the arm-neutral numerics in :mod:`jaxpint.stats`; nothing in
``jaxpint.pta`` may import from this package.
"""

from __future__ import annotations

from jaxpint.frequentist.detection import (
    fstat,
    fstat_p,
    fstat_p_pvalue,
    fstat_skymap,
    phase_shift_background,
    quadrature_blocks,
    sky_scramble_background,
)
from jaxpint.frequentist.nulls import (
    isotropic_positions,
    pvalue,
    rotate_quadratures,
)
from jaxpint.frequentist.optimal import (
    OptimalStatistic,
    optimal_statistic,
)
from jaxpint.frequentist.sensitivity import (
    earth_term_gram,
    unit_noncentrality,
)
from jaxpint.frequentist.stats import (
    chi2_threshold,
    detection_probability,
    h0_min_from_lambda,
)

__all__ = [
    # Distribution layer (PTA-agnostic)
    "chi2_threshold",
    "detection_probability",
    "h0_min_from_lambda",
    # Noncentrality producer
    "earth_term_gram",
    "unit_noncentrality",
    # Detection statistics + empirical backgrounds
    "quadrature_blocks",
    "fstat",
    "fstat_skymap",
    "fstat_p",
    "fstat_p_pvalue",
    "phase_shift_background",
    "sky_scramble_background",
    # Null-calibration primitives
    "isotropic_positions",
    "rotate_quadratures",
    "pvalue",
    # Optimal statistic (GWB cross-correlation detector)
    "optimal_statistic",
    "OptimalStatistic",
]
