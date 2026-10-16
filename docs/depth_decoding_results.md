# Results: decoding meditation depth in ds008836 (phase 7)

Plan: [`analysis_plan_depth_decoding.md`](analysis_plan_depth_decoding.md),
committed 2026-10-16 20:03, before any model was run. Code committed 20:07,
before the run. The run (`depth_decoding.py`, 200 permutations per model)
finished 2026-10-17 00:17. Nothing in the plan was changed.

## Verdict

**Decoded, by the plan's criterion.** The primary model, `riemann_ridge`,
tells a person's deeper-than-usual reports from their shallower ones at
within-person AUC **0.609** (95% interval 0.565 to 0.649; chance 0.502;
Holm-corrected p = 0.035). It beats the better non-EEG baseline
(time + block + report type) by **+0.091** per person (0.046 to 0.132).

That is a modest effect: given two of a person's reports, one deeper than
their average and one shallower, the model ranks them correctly about 61% of
the time. It is the project's first EEG result that clears both chance and
the non-EEG baselines.

## All models

1,707 reports from 39 people; every prediction from a model that never saw
that person; Holm correction across all seven rows.

| Model | Within-person AUC | 95% interval | Chance | p | Holm p | vs `baseline_context` |
|---|---|---|---|---|---|---|
| **`riemann_ridge`** (primary) | **0.609** | 0.565 – 0.649 | 0.502 | 0.005 | **0.035** | **+0.091 (0.046 to 0.132)** |
| `boosting_spectral_plus` | 0.572 | 0.533 – 0.609 | 0.507 | 0.005 | 0.035 | +0.053 (−0.012 to 0.112) |
| `linear_spectral_plus` | 0.563 | 0.525 – 0.599 | 0.501 | 0.005 | 0.035 | +0.044 (−0.020 to 0.101) |
| `linear_spectral` | 0.558 | 0.522 – 0.593 | 0.501 | 0.010 | 0.040 | +0.039 (−0.022 to 0.092) |
| `epoch_ridge` | 0.549 | 0.508 – 0.586 | 0.500 | 0.015 | 0.045 | +0.031 (−0.033 to 0.090) |
| `baseline_context` (no EEG) | 0.519 | 0.475 – 0.564 | 0.500 | 0.179 | 0.358 | |
| `baseline_time` (no EEG) | 0.462 | 0.417 – 0.506 | 0.500 | 0.920 | 0.920 | |

0.005 is the smallest p 200 permutations can give. Every EEG model beats
chance after correction. Only the covariance model also clearly beats the
non-EEG baseline per person; the others are secondary results.

## What it means

- **The spatial pattern carries more than the spectrum.** The four
  spectrum-based models land at 0.55–0.57; the covariance model, which
  sees how all 64 channels co-vary, reaches 0.61. The extra spectrum-shape
  features (1/f exponent, alpha peak) add little (0.558 → 0.563).
- **Segments don't help.** Training on 2-s segments (`epoch_ridge`) is the
  weakest EEG model, consistent with depth being a slow state rather than
  something visible in every 2 s.
- **Not a time or context effect.** Report time alone is below chance
  (people don't simply rate deeper later), and block/report type add little.
- **Not the γ/θ ratio.** Phase 6 found no replicable γ/θ link to depth; this
  signal comes from the whole-scalp covariance instead.

## Caveats

- **People differ a lot.** Per-person AUCs range widely, and a later check
  (phase 8) found that 13 people chosen at random average only 0.544 in this
  same run, against 0.642 for the other 26. A model trained on the 26 and
  tested on those 13 scored 0.55 (p = 0.057). Expect a new group's result to
  land anywhere from about chance to the mid-0.6s.
- **Alpha at the grid edge.** RidgeCV chose its regularisation by the
  built-in leave-one-out shortcut, and picked the top of the grid (1e5) for
  `riemann_ridge`. Phase 8 found that this shortcut can leak when features
  are centred per recording (it did for a centred target, not for the raw
  ratings used here); with alpha chosen by person-grouped CV instead, the
  same model scores similarly on the phase 8 search set (0.632 vs 0.639).
- Retrospective, one dataset, 39 experienced Vipassana meditators; EEG is
  normalised using each whole recording, which a live system would have to
  approximate with a running estimate.

## Files

Run: `audit-work/evaluation/ds008836-phase7/decoding/`
(`model_comparison_depth.csv`, `subject_scores_depth.csv`,
`baseline_comparisons_depth.csv`, `predictions_depth.csv`, manifest).
Published to the dashboard via `publish_results.py --depth_run`.
