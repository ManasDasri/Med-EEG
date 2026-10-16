# Search protocol: improving the depth decoder (phase 8)

**Committed before any candidate below was run.** Phase 7 tested five
pre-registered models on all 39 people. This phase *searches* for a better
one. Searching and scoring on the same people inflates the winner's score,
so the people are split first and the held-out group is used exactly once.

## Split (fixed now)
- **Held-out test set: 13 of the 39 people**, drawn by
  `np.random.default_rng(42).permutation(sorted(subjects))[:13]`:
  sub-005, sub-006, sub-008, sub-011, sub-019, sub-024, sub-025, sub-026,
  sub-029, sub-030, sub-031, sub-035, sub-039.
- **Search set: the other 26.** Every candidate is scored there by
  leave-one-subject-out within-person AUC (the phase 7 metric).
- Disclosure: the phase 7 run already used all 39 people, including the
  held-out 13, for its five fixed models. The held-out set guards the
  *choices made in this search*, not phase 7.

## Candidates (the whole search space, each run once)
Stage 1, on the search set:

| # | Name | What changes from `riemann_ridge` |
|---|---|---|
| 0 | `riemann` | nothing (reference) |
| 1 | `riemann_theta` / `_alpha` / `_beta` | covariance of one band only (4–8, 8–13, 13–30 Hz), from the cross-spectrum |
| 2 | `riemann_filterbank` | broadband + theta + alpha + beta tangent vectors side by side |
| 3 | `riemann_plus_spectral` | tangent vector + the 67 spectral + 15 spectrum-shape features |
| 4 | `ensemble` | mean of `riemann` and `boosting_spectral_plus` predictions (each z-scored within recording) |
| 5 | `riemann_context` | each report's tangent vector averaged with the previous report's in the same recording (EEG only, no labels) |
| 6 | `riemann_ztarget` | training ratings z-scored within each training recording |
| 7 | `riemann_logistic` | logistic regression (C chosen inside the fold) on "above own average" |

Stage 2: if at least one feature option (1–5) and one target option (6–7)
each beat the reference on the search set, the best of each is combined and
scored as one more candidate. Nothing else is added.

## Selection
The highest search-set AUC wins; within 0.005 of the best, the simpler one.

## Final test (once)
- The winner and the reference `riemann` are each trained on all 26
  search-set people and applied once to the 13 held-out people.
- Reported: held-out within-person AUC; p from 1,000 circular shifts of the
  held-out ratings within each recording (model fixed); paired per-person
  difference vs the reference and vs `baseline_context`.
- **The winner counts as an improvement only if its held-out paired
  difference vs `riemann` has a 95% interval above 0.** With 13 people that
  interval will be wide; an inconclusive result is reported as such.
- For the dashboard, the winner is also run with the phase 7 setup (all 39,
  LOSO, 200 permutations), labelled *selected by search, exploratory*.
