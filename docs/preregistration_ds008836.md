# Pre-registration: replicating lead 1 on OpenNeuro ds008836

**Written and committed before any EEG from ds008836 was loaded or analysed.**
Only the dataset's metadata (README, `participants.tsv`, `*_events.tsv`,
`*_channels.tsv`, JSON sidecars) had been read. The git commit timestamp of
this file is the record. Anything not specified here, or decided after the
EEG was opened, will be reported as exploratory.

## 1. The finding being tested

In ds001787 (Delorme & Brandmeyer; 23 people after quality exclusion), among
**experienced meditators**, moments with a higher gamma/theta power ratio at
Fz + Pz were rated as *less* concentrated/deep, within each person:

- experts only, same model as §5: **−0.12 per SD** (95% CI −0.21 to −0.03,
  p = 0.008, 11 people, 486 probes); per-person slopes negative in 10 of 11
- replicated across first vs later sessions; robust to leave-one-person-out
  and to controlling temple-electrode gamma (phase 5)

ds008836 has only experienced meditators, so it tests this **expert-side
slope**, not the expert-vs-novice interaction.

## 2. Data

- **Dataset:** OpenNeuro ds008836 v1.0.1+ (Reggente et al.), 39 experienced
  Vipassana meditators (≥ 5 years of practice), two sessions each, 64-channel
  BrainAmp at 500 Hz, online reference FCz, 60 Hz mains.
- **Outcome:** the depth rating ("Please rate the depth of your meditation since
  your last response", 1 = shallow, 5 = deepest), `trial_type == depth_rating`.
  Ratings of 0 are removed; ratings above 5 are set to 5 (the dataset's
  documented convention; 5 such rows).
- **Primary reports:** depth ratings in the two **SILENT** blocks
  (`silent_probe` and `silent_emerge`): meditation without tones, the
  closest match to ds001787.
- **Participants:** all 39, minus recordings failing the quality rule set in
  ds001787 *before* its results (> 25% bad channels or > 50% epochs rejected).

## 3. Preprocessing (identical to ds001787 unless listed)

Band-pass 1–45 Hz, notch at the sidecar's mains frequency (60 Hz), resample to
256 Hz, LOF bad channels → interpolation, average reference, ICA (20
components; eye rule correlation ≥ 0.5 with Fp1/Fp2 **and** ≥ 40% frontal-pole
map weight; MNE muscle detection), 2 s epochs with 1 s overlap, peak-to-peak
rejection at 150 µV. Same code (`src/preprocess.py`), same `config.py` values.

Dataset-specific differences, fixed now:
- EEGLAB `.set` files; channel types from `channels.tsv`; standard 10-05 montage.
- **Window per report:** ends at the start of that report's prompt/report
  window (`prompt_window_begin` for probe, `report_window_begin` for emerge,
  the last one before the rating in the same block). It starts at the latest
  of: 120 s earlier; 10 s after the previous depth/confidence rating in the
  same block; the block's start. (ds001787's "10 s after *any* event" rule
  can't be used: the tone blocks have a tone every second.)

## 4. Predictor

`minimal_ratio`: mean Welch PSD over Fz and Pz in 30–45 Hz divided by that in
4–8 Hz, averaged over the window's epochs: `src/extract_features.py`,
unchanged.

## 5. Primary test (confirmatory)

Linear mixed model, one row per depth report:

    depth ~ minimal_ratio + time,   random intercept + random minimal_ratio slope per participant

with `minimal_ratio` and `time` (report onset) **z-scored within each
recording**, and depth **z-scored within each recording** (so the estimate is
in within-person SD units, comparable to ds001787's ≈ −0.21 within-person SD
of rating per SD of ratio). REML, `src/mixed_effects_stats.fit_lme`.

- **Hypothesis H1:** the `minimal_ratio` coefficient is **negative**.
- **Decision rule:** one-sided p < 0.05 (= two-sided p < 0.10 with a negative
  estimate) → **replicated**. A positive estimate, or one-sided p ≥ 0.05 →
  **not replicated**. The estimate and 95% CI are reported either way.

## 6. Secondary analyses (exploratory, reported regardless of outcome)

1. All four 35-min blocks (SILENT + SAPO), with block type and report variant
   (probe / emerge) as covariates.
2. Probe-prompted SILENT reports only.
3. Gamma power alone and theta power alone in place of the ratio.
4. Ratio with temple gamma (mean of left/right temporal region gamma, log)
   as an additional covariate.
5. Participants with `in_analysis == yes` only.
6. Per-person slopes: count negative vs positive.

## 7. What would and wouldn't count

- A significant negative slope supports the expert-side finding in a
  different lab, system, country, meditation tradition and rating scale.
- A null result means the effect doesn't generalise at this sample size, or
  is specific to ds001787; it will be reported as a failed replication, not
  rescued with post-hoc choices.
- No parameter in §2–§5 will be changed after the EEG is opened. Bug fixes
  that don't depend on the outcome (e.g. a file failing to load) are allowed
  and will be listed in the results.
