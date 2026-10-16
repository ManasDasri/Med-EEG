# Analysis plan: decoding meditation depth in ds008836 (phase 7)

**Committed before any model below was run on ds008836.** ds008836 was
already used for one pre-registered slope test (the gamma/theta ratio, not
replicated). This plan is a separate question: can richer EEG features
*decode* a person's self-rated depth? It is exploratory with respect to
that earlier test, but fixed here so the result can't be steered.

## Data
- OpenNeuro ds008836: 39 experienced Vipassana meditators, two sessions,
  depth reports 1–5 in all four 35-min blocks (silent/tones × prompted/
  spontaneous), ~1,700 reports. Same preprocessing as before (corrected
  pipeline, after the ICA-sampling fix); the pre-set quality exclusion
  applies (currently 1 recording).
- Each report uses the window already defined in `ds008836.py` (up to 120 s
  before the report). The window length is **not** tuned.

## Target and metric
- **Within-person AUC**, as for concentration in ds001787: for each person,
  how well the model separates their above-own-average depth reports from
  the rest, averaged over people. 0.5 = chance.
- Leave-one-subject-out: every prediction comes from a model that never saw
  that person.
- Chance and p-value from 200 recording-aware permutations (ratings
  circularly shifted within each recording). **Holm correction across all
  models and baselines below.**

## Models (all fixed now)
EEG features are normalised within each recording, using that recording's
EEG only, never labels.

| Name | Features | Model |
|---|---|---|
| `linear_spectral` | the existing 67 regional band-power/slope features | Ridge (α = 10), as in ds001787: reference point |
| `linear_spectral_plus` | the 67 + per region: individual alpha peak frequency, alpha peak height above the 1/f background, broadband 1/f exponent (2–40 Hz, 7–14 Hz excluded) | RidgeCV, α chosen inside each training fold |
| `boosting_spectral_plus` | same as above | gradient boosting (fixed conservative settings) |
| **`riemann_ridge`** (**primary**) | 64×64 covariance of each report's EEG window (OAS shrinkage), re-centred per recording, mapped to the tangent space at the identity (matrix log, upper triangle) | RidgeCV, α chosen inside each training fold |
| `epoch_ridge` | the 67 features per 2-s segment instead of per report | RidgeCV on segments; predictions averaged per report |

Non-EEG baselines, scored identically:
- `baseline_time`: report time within the session
- `baseline_context`: time + block type (tones/silent) + report type (prompted/spontaneous)

## Success criterion
The **primary model (`riemann_ridge`)** decodes depth if **both** hold:
1. Holm-corrected permutation p < 0.05, **and**
2. its paired per-person comparison with the **better** non-EEG baseline
   has a 95% interval above 0.

The other EEG models are reported with the same tests as secondary results.
A secondary model passing where the primary fails is reported as exploratory,
not as success.

## What would change the plan
Only outcome-independent bug fixes (a crash, a wrong file), listed in the
results. No feature, model, window or setting changes after results are seen.
