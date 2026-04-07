"""PhaseResult: backward-compatible alias for DualFloat.

PhaseResult was the original name for the int/frac split type, used
exclusively for pulse phase (cycles). It is now a type alias for the
more general ``DualFloat`` class, which supports both ``[-0.5, 0.5)``
(cycles) and ``[0, 1)`` (days) normalization.

Existing code that imports ``PhaseResult`` and calls ``PhaseResult.create()``
continues to work unchanged.
"""

from jaxpint.dual_float import DualFloat

PhaseResult = DualFloat
