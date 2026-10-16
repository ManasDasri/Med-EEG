# Amendment to the depth search protocol (round 2)

**Written after round 1, before round 2 was run.**

## What round 1 showed
Round 1 ran as committed (outputs kept under `round1/`):
- Search set: `riemann` 0.639 won; no variant beat it. `riemann_ztarget`
  scored 0.513, far below a near-equivalent fit.
- The final held-out test therefore ran on `riemann` itself: **held-out AUC
  0.521 (p = 0.24)** vs `baseline_context` 0.488. This has been seen; it is
  reported as is.

## Bug found
`RidgeCV` chooses alpha by its built-in leave-one-out shortcut. Here the
features (and, for `riemann_ztarget`, the target) are centred within each
recording, so a report equals minus the sum of the rest of its recording.
With more features than reports (2,080 vs ~900), a nearly unregularised fit
then "predicts" the left-out report from its own recording. Reproduced on
pure noise (test in `tests/test_explore_depth.py`): leave-one-out picks the
smallest alpha (0.1), i.e. it fits noise. That is what sank
`riemann_ztarget`. With the raw target it doesn't leak (it picks the
largest alpha), so phase 7's `riemann_ridge` is not distorted by it; its
alpha did sit at the top edge of the grid (1e5).

## Change
- Every ridge candidate chooses alpha by **5-fold CV grouped by person**,
  inside each training fold, over a wider grid (0.1 to 1e7).
- Round 2 re-runs the **same candidate list** with the same selection rule
  and the same stage-2 rule. Nothing is added.
- Final test as before: winner vs `riemann` (now with the fixed alpha
  selection) on the same 13 held-out people. Disclosure: the held-out score
  of the round-1 `riemann` is already known (above). Selection still uses
  only the search set.
- The all-39 dashboard run of the winner uses **100** permutations (each
  fit now takes ~7 s; 200 would not finish overnight). Smallest possible p
  becomes 0.0099.
