# Reference residuals copied from PINT's test data

Verbatim copies of `PINT/tests/datafile/*.{tempo2_test,tempo_test,libstempo}`,
vendored so `tests/test_cross_implementation.py` does not depend on a PINT
*source checkout* sitting next to this repo (the installed `pint-pulsar` wheel
ships none of these files).

| suffix | producer | used |
|---|---|---|
| `.tempo2_test` | tempo2 | yes — 6 pairs |
| `.tempo_test` | TEMPO (tempo1) | yes — 4 pairs, the only non-tempo2 reference |
| `.libstempo` | libstempo 2022-05-24 | not yet; near-duplicate of our generated J1713 golden |

**These carry no provenance.** Upstream records bare numbers with no ephemeris,
clock, or version header — which is why a disagreement on
`B1855+09_NANOGrav_9yv1.gls.par.tempo2_test` could not be attributed until we
generated our own (see `../tempo2_goldens/`, and the stale-golden test).

**The par/tim inputs are NOT vendored.** They are ~10.5 MB against 1.5 MB for
these goldens, so the comparisons still read pars and tims from the sibling
PINT checkout and skip without it. Vendoring those too would make the suite
fully self-contained at that cost.
