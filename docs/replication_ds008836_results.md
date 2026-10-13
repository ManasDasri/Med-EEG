# Results: pre-registered replication on OpenNeuro ds008836

Pre-registration: [`preregistration_ds008836.md`](preregistration_ds008836.md)
(commit `280830b`, before any ds008836 EEG was loaded).
Amendment: [`preregistration_ds008836_amendment.md`](preregistration_ds008836_amendment.md)
(commit `a099e1d`, before the repaired result was computed).

## Verdict: NOT REPLICATED

The pre-registered mixed model is **unidentifiable** after within-recording
z-scoring: every recording's mean depth is exactly 0, so the per-participant
random intercept can't be estimated and no optimizer converges (audit
finding #1). Amendment 1 replaced it with OLS with participant-clustered
standard errors, plus a per-participant cross-check, without changing the
data, predictor, z-scoring or decision rule. Run once on the corrected
features (after the ICA-sampling fix), commit `5c48b24`:

| | Estimate | 95% CI | one-sided p | Reports | People |
|---|---|---|---|---|---|
| **Primary: OLS, participant-clustered SE** | **−0.072** | −0.166 to +0.022 | **0.064** | 830 | 39 |
| Cross-check: one slope per participant, t-test | −0.074 | −0.184 to +0.036 | 0.089 | | 39 (26 negative) |
| Original mixed model (for the record) | −0.059 | | invalid fit | 830 | 39 |

Both tests agree: the slope is negative, as predicted, but neither reaches the
pre-registered one-sided p < 0.05. **By the pre-registered rule, lead 1
did not replicate.** Units are within-person SD of depth per within-recording
SD of the ratio; the ds001787 experts showed ≈ −0.21.

## Exploratory analyses (same repaired method)

| | Estimate | 95% CI | p two-sided | p one-sided (neg.) | n |
|---|---|---|---|---|---|
| S1 all four blocks (+ block type, variant) | −0.086 | −0.152 to −0.020 | 0.012 | 0.006 | 1707 / 39 |
| S2 SILENT, prompted reports only | −0.026 | −0.167 to +0.115 | 0.71 | 0.35 | 334 / 38 |
| S3 gamma power alone | −0.032 | −0.138 to +0.074 | 0.54 | 0.27 | 830 / 39 |
| S3 theta power alone | +0.083 | −0.006 to +0.172 | 0.068 | — | 830 / 39 |
| S4 ratio + temple-gamma covariate | −0.069 | −0.162 to +0.025 | 0.15 | 0.074 | 830 / 39 |
| S5 `in_analysis == yes` only | −0.075 | −0.176 to +0.026 | 0.14 | 0.071 | 776 / 37 |
| S6 per-person slopes | median −0.090 | | | | 25 of 39 negative |

## Interpretation

- The **direction** holds (negative in every ratio analysis, about two thirds
  of people negative), but the **evidence is too weak** for the pre-registered
  test. The likely effect, if real, is about a third the size seen in ds001787.
- Pooling all four blocks (S1, 1,707 reports) gives p = 0.006. It's
  exploratory, but it suggests a small effect that this design was
  under-powered to confirm with the SILENT blocks alone. A new pre-registered
  test with more reports per person would be the honest next step.
- The earlier claim that **theta carries the effect** no longer holds up
  (p = 0.068 with valid inference).

---

## Superseded: the original phase-6 run (invalid inference)

Kept for the record. These numbers came from a mixed model that **never
converged** (warnings were suppressed), on features computed before the
ICA-sampling fix. Their p-values and the "REPLICATED" verdict are not valid
inference.


Pre-registration: [`preregistration_ds008836.md`](preregistration_ds008836.md)
(commit `280830b`, 10:04, before any ds008836 EEG was loaded).
Analysis run once, after commit `b154f94` (15:43).

### Original primary result

| | Estimate | 95% CI | one-sided p | Reports | Participants |
|---|---|---|---|---|---|
| **depth ~ gamma/theta ratio (Fz+Pz), SILENT blocks** | **−0.067** | −0.141 to +0.006 | **0.036** | 830 | 39 |

Units: within-person SD of depth per within-recording SD of the ratio.
ds001787 experts: ≈ −0.21 in the same units.

**Verdict by the pre-registered rule (negative slope, one-sided p < 0.05):
REPLICATED.** The effect is about one third of the size seen in
ds001787, and the 95% CI just includes zero; a two-sided test (p = 0.072)
would not pass.

### Original exploratory analyses

| | Estimate | 95% CI | p two-sided | p one-sided (neg.) | n |
|---|---|---|---|---|---|
| S1 all four blocks (+ block type, variant) | −0.082 | −0.162 to −0.003 | 0.043 | 0.021 | 1707 / 39 |
| S2 SILENT, prompted reports only | −0.028 | −0.138 to +0.082 | 0.62 | 0.31 | 334 / 38 |
| S3 gamma power alone | −0.032 | −0.104 to +0.039 | 0.38 | 0.19 | 830 / 39 |
| S3 theta power alone | **+0.086** | +0.008 to +0.164 | **0.030** | — | 830 / 39 |
| S4 ratio + temple-gamma covariate | −0.063 | −0.141 to +0.014 | 0.11 | 0.055 | 830 / 39 |
| S5 `in_analysis == yes` only | −0.069 | −0.144 to +0.006 | 0.070 | 0.035 | 776 / 37 |
| S6 per-person slopes | median −0.083 | | | | 25 of 39 negative |

### Original interpretation

- The **direction** of the pre-registered prediction holds in an independent
  dataset: different lab, country, meditation tradition (Vipassana), EEG
  system (BrainAmp), rating scale (1–5) and reference.
- The **size** is much smaller, and the result is fragile. It doesn't survive
  a two-sided test, the prompted-only subset, or (narrowly) the temple-gamma
  control.
- The **mechanism differs.** In ds001787 the effect was carried by gamma
  (more gamma → less concentration). Here gamma alone shows nothing, and
  **theta** carries it: more theta at Fz/Pz during deeper reported
  meditation, consistent with the frontal-midline-theta literature on
  meditation. The ratio replicates because both a gamma rise and a theta
  drop push it the same way.
- Overall: *the gamma/theta ratio tracks reported meditative depth in
  experienced meditators, weakly, across two datasets; which band drives it
  is not consistent between them.*

### Data quality

78 recordings, all preprocessed without errors; median epoch rejection 0.4%;
median 7 bad channels per recording (more than ds001787's 2, all repaired by
interpolation). One recording excluded by the pre-set rule (sub-029 ses-2,
28 of 64 channels bad). 834 SILENT reports; 4 dropped by within-recording
z-scoring (recordings where every depth rating was identical).

### Changes after pre-registration (original run)

All were outcome-independent bug fixes, made before the analysis was run,
as allowed by §7:

| Commit | Time | Fix |
|---|---|---|
| `d458572` | 10:07 | Analysis code written. A depth rating without its own prompt marker is dropped (1 case, SAPO block; not in the primary data) |
| (same) | | A test caught a dropped 0-rating letting a later orphan rating borrow its window. Fixed before any EEG was loaded |
| `981dd85` | 10:27 | Mains frequency read via BIDS inheritance: the notch was falling back to 50 Hz, while the dataset is 60 Hz as pre-registered. Found by the technical pilot |
| `b154f94` | 15:43 | Feature extraction assumed ds001787's rating columns and crashed. Fixed before the analysis ran |

No pre-registered parameter (§2–§5) was changed.
