# Results: pre-registered replication on OpenNeuro ds008836

Pre-registration: [`preregistration_ds008836.md`](preregistration_ds008836.md)
(commit `280830b`, 10:04, before any ds008836 EEG was loaded).
Analysis run once, after commit `b154f94` (15:43).

## Primary result

| | Estimate | 95% CI | one-sided p | Reports | Participants |
|---|---|---|---|---|---|
| **depth ~ gamma/theta ratio (Fz+Pz), SILENT blocks** | **−0.067** | −0.141 to +0.006 | **0.036** | 830 | 39 |

Units: within-person SD of depth per within-recording SD of the ratio.
ds001787 experts: ≈ −0.21 in the same units.

**Verdict by the pre-registered rule (negative slope, one-sided p < 0.05):
REPLICATED.** The effect is about one third of the size seen in
ds001787, and the 95% CI just includes zero; a two-sided test (p = 0.072)
would not pass.

## Exploratory analyses (pre-registration §6)

| | Estimate | 95% CI | p two-sided | p one-sided (neg.) | n |
|---|---|---|---|---|---|
| S1 all four blocks (+ block type, variant) | −0.082 | −0.162 to −0.003 | 0.043 | 0.021 | 1707 / 39 |
| S2 SILENT, prompted reports only | −0.028 | −0.138 to +0.082 | 0.62 | 0.31 | 334 / 38 |
| S3 gamma power alone | −0.032 | −0.104 to +0.039 | 0.38 | 0.19 | 830 / 39 |
| S3 theta power alone | **+0.086** | +0.008 to +0.164 | **0.030** | — | 830 / 39 |
| S4 ratio + temple-gamma covariate | −0.063 | −0.141 to +0.014 | 0.11 | 0.055 | 830 / 39 |
| S5 `in_analysis == yes` only | −0.069 | −0.144 to +0.006 | 0.070 | 0.035 | 776 / 37 |
| S6 per-person slopes | median −0.083 | | | | 25 of 39 negative |

## Interpretation

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

## Data quality

78 recordings, all preprocessed without errors; median epoch rejection 0.4%;
median 7 bad channels per recording (more than ds001787's 2, all repaired by
interpolation). One recording excluded by the pre-set rule (sub-029 ses-2,
28 of 64 channels bad). 834 SILENT reports; 4 dropped by within-recording
z-scoring (recordings where every depth rating was identical).

## Changes after pre-registration

All were outcome-independent bug fixes, made before the analysis was run,
as allowed by §7:

| Commit | Time | Fix |
|---|---|---|
| `d458572` | 10:07 | Analysis code written. A depth rating without its own prompt marker is dropped (1 case, SAPO block; not in the primary data) |
| (same) | | A test caught a dropped 0-rating letting a later orphan rating borrow its window. Fixed before any EEG was loaded |
| `981dd85` | 10:27 | Mains frequency read via BIDS inheritance: the notch was falling back to 50 Hz, while the dataset is 60 Hz as pre-registered. Found by the technical pilot |
| `b154f94` | 15:43 | Feature extraction assumed ds001787's rating columns and crashed. Fixed before the analysis ran |

No pre-registered parameter (§2–§5) was changed.
