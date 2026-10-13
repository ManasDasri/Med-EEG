# Amendment 1 to the ds008836 pre-registration: repairing a degenerate model

**Committed before the repaired inference was computed.** The original point
estimate (−0.067) was already public (phase 6), so this amendment is not
blind to it. It is written so that the choice of repair depends only on the
identification problem below, never on the resulting p-value.

## The problem

The pre-registered model (§5) z-scores depth, the ratio and time **within
each recording**, then fits

    depth ~ minimal_ratio + time,   random intercept + random slope per participant

Within-recording z-scoring sets every recording's mean depth to exactly 0
(largest absolute mean: 4 × 10⁻¹⁶). A per-participant random intercept is
therefore not identifiable. Every optimizer fails to converge (audit finding
#1; lbfgs, bfgs and powell all fail), so the original verdict ("replicated,
one-sided p = 0.036") rests on an invalid fit and is **withdrawn**.

Dropping the intercept (random slope only) converges, but the slope variance
is estimated at exactly 0: the data show no detectable between-person
variation in the slope, so that random effect is degenerate too.

Only variance components were inspected to establish this, not
coefficients or p-values of any alternative model.

## The repair (fixed now)

**Repaired primary test.** Same 830 SILENT-block reports, same z-scoring,
same predictors, same one-sided decision rule (negative slope, one-sided
p < 0.05 → replicated):

    depth ~ minimal_ratio + time      (ordinary least squares)

with **standard errors clustered by participant**. This is the standard
estimator when the random effects carry no identifiable variance. Clustering
accounts for repeated reports from the same person, and with zero
random-slope variance it targets the same within-person slope as §5.

**Cross-check (reported regardless).** One slope per participant (depth on
ratio and time, within that participant's reports), then a one-sample t-test
of the 39 slopes against 0, one-sided. This is assumption-light and treats
participants as the unit of replication.

**Verdict:** from the repaired primary test alone. If the cross-check
disagrees in direction or significance, both are reported and the result is
described as **inconclusive**, not replicated.

**Secondary analyses S1–S5** get the same repair (cluster-robust OLS) and stay
exploratory. S6 (per-person slope signs) is unchanged.

## What is not changed

Data, preprocessing, windows, predictor, outcome, z-scoring, the report
subset and the one-sided alternative are all exactly as pre-registered (§2–§4).
