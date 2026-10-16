# Results: searching for a better depth decoder (phase 8)

Protocols, each committed before the round it governs:
[`search_protocol_depth.md`](search_protocol_depth.md) (22:17),
[amendment, round 2](search_protocol_depth_amendment.md) (22:30),
[round 3, calibration](search_protocol_depth_calibration.md) (22:42; one fix at 22:44).
Every candidate was scored on 26 *search* people; only winners met the 13
*held-out* people.

## Verdict

**No improvement over phase 7's `riemann_ridge` was confirmed.** Nine
variants (single frequency bands, a filter bank, added spectral features, an
ensemble, temporal context, two target treatments) all tied with or lost to
the plain covariance model on the search set. One rated calibration session
helps a little (+0.01 to +0.02 AUC, in both the search and held-out people),
but not by enough to be distinguished from noise.

The search also found and fixed a real bug (round 2), and it showed that the
phase 7 headline varies a lot with *which* people are scored.

## Round 1 (as committed)
Search-set winner: `riemann` (0.639). Held-out: **0.521** (p = 0.24).
`riemann_ztarget` scored 0.513, which exposed the bug below.

## Bug: RidgeCV's leave-one-out leaks here
`RidgeCV` picks alpha by a leave-one-out shortcut. The features are centred
within each recording, so a report equals minus the sum of the rest of its
recording. With more features than reports and a target also centred per
recording, a barely regularised model "predicts" the left-out report from its
own recording. On pure noise it picks the smallest alpha
(`tests/test_explore_depth.py` reproduces this). Fix: choose alpha by 5-fold
CV grouped by person, over a wider grid (0.1–1e7).

It doesn't affect phase 7's `riemann_ridge`, which uses the raw ratings.
Re-running that model on all 39 people with the fixed alpha selection gives
**0.607** (p = 0.010, 100 permutations) against phase 7's 0.609.

## Round 2 (fixed alpha selection, same candidates)

| Candidate | Search-set AUC (26 people) |
|---|---|
| **`riemann`** (reference, winner) | **0.632** |
| `riemann_filterbank` (broadband + θ + α + β) | 0.632 |
| `riemann_ztarget` | 0.629 |
| `riemann_plus_spectral` | 0.629 |
| `riemann_context` (averaged with previous report) | 0.618 |
| `riemann_logistic` | 0.612 |
| `ensemble` (riemann + boosting) | 0.596 |
| `riemann_alpha` | 0.556 |
| `riemann_theta` | 0.543 |
| `riemann_beta` | 0.529 |

Held-out (trained on the 26, tested once on the 13): `riemann` **0.551**
(95% 0.473–0.623; p = 0.057), `baseline_context` 0.488; difference +0.062
(−0.015 to 0.142).

What this says:
- **Broadband beats every single band.** No one band (θ, α, β) carries the
  signal; combining them adds nothing on top of the broadband covariance.
- **Spectral features add nothing** once the covariance is in.
- **No stage 2.** No feature option beat the reference.

## Round 3: one calibration session

Train on everyone else plus the person's *other* session (weight *w*),
predict this session. sub-029 is left out (only one usable session).

| *w* | Search-set AUC | Held-out AUC (12 people) |
|---|---|---|
| 0 (no calibration) | 0.626 | 0.552 (p = 0.062) |
| 1 | 0.637 | |
| **5** (chosen) | **0.646** | **0.565 (p = 0.021)** |
| 20 | 0.645 | |

Held-out per-person gain from calibrating: **+0.013 (−0.006 to 0.033)**.
Not confirmed. It points the same way as the search set, and it's small.

## Why the held-out people score lower
In phase 7's own run (every model trained without the person being scored),
the 13 held-out people average **0.544** and the other 26 average **0.642**.
The held-out set is simply harder, not a sign of overfitting. It shows that
a 13-person sample can swing the headline by ±0.05 or so, and the phase 7
figure (0.609 over 39 people) should be read with that in mind.

## Where this leaves accuracy
Within-person AUC ≈ 0.60–0.61 looks like roughly what scalp EEG and these
ratings support with this data. The obvious modelling routes are now tested.
What would most likely move it is more and better labels: more reports per
person, or ratings with less noise. Changing the model further is unlikely
to help much.

## Files
`audit-work/evaluation/ds008836-phase8/` (`round1/`, `search_table.csv`,
`heldout_*.csv`, `winner_all39.csv`, `calibration/`).
