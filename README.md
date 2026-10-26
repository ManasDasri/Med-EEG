# Med-EEG — NeuroDial

**Interpretable gamma/theta EEG biomarkers for meditation depth**

**Live results dashboard: [www.algorithmicbit.tech/neurodial](https://www.algorithmicbit.tech/neurodial/)**.
It loads [`results/latest/results.json`](results/latest/results.json) from this repository
on every visit, so it always shows the latest published run (see *Publishing results* below).

[![Dashboard](https://img.shields.io/badge/dashboard-live-0ca30c)](https://www.algorithmicbit.tech/neurodial/)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MNE](https://img.shields.io/badge/MNE--Python-1.7%2B-1f77b4)](https://mne.tools/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-LME-4B8BBE)](https://www.statsmodels.org/)
[![SHAP](https://img.shields.io/badge/explainability-SHAP-ff0051)](https://github.com/shap/shap)
[![Dataset](https://img.shields.io/badge/OpenNeuro-ds001787-00a0b0)](https://openneuro.org/datasets/ds001787)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/ManasDasri/Med-EEG)](https://github.com/ManasDasri/Med-EEG/commits/main)

Can a two-number biomarker — the **gamma/theta power ratio** and **theta–gamma
phase-amplitude coupling** at a few frontal-midline electrodes — predict
moment-to-moment self-reported concentration during meditation nearly as well as
a black-box model trained on the full 64-channel spectrum?

This is a retrospective research analysis. Evidence from a few electrodes would
motivate a prospective headband study; it would not establish live feedback performance.

---

## Dataset

Delorme & Brandmeyer, *EEG meditation study* —
[OpenNeuro ds001787](https://openneuro.org/datasets/ds001787) /
[Zenodo 2536267](https://zenodo.org/records/2536267).

- 24 subjects: 12 experienced meditators, 12 novices
- 64-channel BioSemi EEG, `.bdf`, BIDS format (~5 GB)
- ~1 hour of seated meditation, interrupted every ~2 min by self-report probes
  (concentration, mind wandering, tiredness)

The probes give repeated, **within-subject** ground truth rather than just a
group label — that's what this project exploits.

## What's in the analysis

| # | Idea | Why it matters |
|---|------|----------------|
| 1 | **Phase-amplitude coupling** (Tort Modulation Index) alongside power ratio | Tests whether gamma is *locked* to theta phase, not just co-occurring |
| 2 | **Electrode minimalism** | Best 1, 2, 4 … 64 electrodes (chosen on training people only) vs full scalp, plus predefined clusters |
| 3 | **Interpretable vs. black-box shootout** | 2-feature linear model vs. linear and gradient-boosting models on 67 regional spectral features, with permutation tests and non-EEG baselines |
| 4 | **Linear mixed-effects models** | Predictors normalised within each recording, time-on-task as covariate, subject as random effect |
| 5 | **SHAP audit** | Checks what the black box actually relies on (theta/gamma, or artifacts like delta) |
| 6 | **NeuroDial dashboard** | Interactive, serverless replay of results as a neurofeedback-style dial |

## Pipeline

```
raw .bdf (BIDS)                 mne.io.read_raw_bdf, 10-20 channel names
    │
Bandpass 1–45 Hz + mains notch  line frequency read from the BIDS sidecar
Bad channels (LOF) → interpolate
Average reference               BioSemi records reference-free (CMS/DRL)
ICA                             remove eye + muscle components
    │
Epoch before each probe         2 s windows, 50% overlap, up to 2 min back,
                                never overlapping the previous probe's Q&A
Reject noisy epochs             peak-to-peak > 150 µV
    │
Welch spectrum per epoch        band power (absolute + relative) per region,
                                gamma/theta ratio, 1/f slope, alpha asymmetry
Theta–gamma PAC per cluster     Tort MI, bias-corrected + surrogate z-score
    │
Feature tables                  one row per probe + one row per epoch
    │
Stats: LME  concentration ~ ratio + PAC + (1 | subject)
ML:    subject-wise LOSO-CV, linear and gradient-boosting models, SHAP audit
    │
dashboard_data.json → dashboard.html
```

## Repository layout

```
src/
  config.py               shared constants: bands, channel maps, event codes, paths
  inspect_dataset.py      sanity-check the BIDS layout, subjects, probe events
  preprocess.py           filter, re-reference, ICA, epoch before each probe
  extract_features.py     spectral features per region/cluster + PAC, per probe and per epoch
  mixed_effects_stats.py  core within-person mixed-effects models (ratio, PAC, time)
  electrode_minimalism.py how few electrodes keep the signal (nested selection)
  robustness_checks.py    session-split, leave-one-out and muscle checks of the leads
  ds008836.py             adapter for the replication dataset (loader + report windows)
  replication_ds008836.py the pre-registered replication test
  depth_decoding.py       decode depth in ds008836: spectral, Riemannian, segment models
  band_covariances.py     per-report theta/alpha/beta covariances (phase 8 search)
  explore_depth.py        phase 8 search: candidates on 26 people, one held-out test on 13
  calibrate_depth.py      phase 8: does one rated calibration session help?
  fidelity.py             phase 9: per-person brain-report fidelity, stability, practice
  horizon.py              phase 9: decoding depth from bins before the report
  horizon_equal_bins.py   phase 9 exploratory: equal-length bins
  publish_results.py      bundle finished runs into results/latest/results.json
  model_comparison.py     LOSO models + permutation tests + non-EEG baselines + SHAP
  export_dashboard.py     bundle results into dashboard_data.json
  followup_analysis.py    follow-ups: delta power, tiredness, random slopes, expertise
tests/
  test_preprocess.py      probe/answer decoding checks (python tests/test_preprocess.py)
  test_features.py        feature checks on synthetic signals (python tests/test_features.py)
  test_models.py          signal vs no-signal model checks (python tests/test_models.py)
  test_electrode_minimalism.py  nested electrode selection checks
  test_robustness_checks.py     cross-session electrode transfer checks
  test_ds008836.py              replication report-window parser
  test_replication.py           pre-registered decision rule
  test_depth_decoding.py        Riemannian re-centring and segment-model checks
  test_explore_depth.py         search learners + the alpha-selection leak
  test_calibrate_depth.py       calibration never sees the test session
  test_phase9.py                kernel ridge = sklearn ridge; bins; fidelity helpers
dashboard.html            NeuroDial results dashboard (open directly in a browser)
dashboard_data.sample.json  sample data for the dashboard
knn_prototype.py          retirement notice for the invalid historical prototype
requirements.txt
```

## Getting started

```bash
git clone https://github.com/ManasDasri/Med-EEG.git
cd Med-EEG
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt -c constraints.txt

# download the dataset (~6 GB)
openneuro-py download --dataset ds001787 --target_dir data/ds001787
```

Run the steps in this order. Each writes intermediates to
`data/ds001787/derivatives/neurodial/`, so you can resume mid-pipeline.

```bash
python src/inspect_dataset.py      --bids_root data/ds001787
python src/preprocess.py           --bids_root data/ds001787   # --subjects / --limit for a quick run
python src/extract_features.py     --bids_root data/ds001787   # --window_s 30 to use only the last 30 s
python src/mixed_effects_stats.py  --bids_root data/ds001787
python src/electrode_minimalism.py --bids_root data/ds001787 --task mind_wandering   # or concentration
python src/robustness_checks.py    --bids_root data/ds001787

# replication (phase 6): OpenNeuro ds008836
python src/preprocess.py           --bids_root data/ds008836 --dataset ds008836
python src/extract_features.py     --bids_root data/ds008836
python src/replication_ds008836.py --bids_root data/ds008836
python src/model_comparison.py     --bids_root data/ds001787 --task expertise  # or concentration, mind_wandering
python src/export_dashboard.py     --bids_root data/ds001787
python src/followup_analysis.py    --bids_root data/ds001787   # optional
```

After preprocessing, check `preprocessing_qc.csv` in the derivatives folder:
it lists bad channels, removed ICA components and the epoch rejection rate
for every recording.

### Dashboard

Open `dashboard.html` in a browser and load `dashboard_data.json` (or
`dashboard_data.sample.json` to try it without the dataset). Dependencies and fonts are local; a direct file open works offline.

## Historical results (original analysis)

The following values describe the original analysis run and are retained as historical context. They do not use the corrected ICA sampling, recording-preserving nulls, fit-validity checks, or 100-permutation evaluation. See the final corrected-run report linked below before interpreting them.

## Results (all 24 subjects)

Leave-one-subject-out throughout; every score is compared with shuffled-label
runs (chance level, p-value) **and with non-EEG baselines**. One subject
(sub-010, poor electrode contact) is excluded by a rule set before any
results, leaving 23 people and 929 probes.

Published run: corrected features (after the ICA-sampling fix), 200
permutations per model, p-values Holm-corrected across all models and
baselines within each task. Same numbers as the dashboard.

| Question | Best EEG result | Non-EEG baseline | Verdict |
|---|---|---|---|
| Expert vs novice | 78.3% of people (18/23); p = 0.025, corrected 0.10 | **sex alone: 82.6%** | Not supported: experts are 9/11 male, novices 10/12 female, and the model mostly detects sex |
| Concentration, moment to moment | within-person AUC 0.529; corrected p = 0.64 | **time-on-task: 0.561** | No signal |
| Mind wandering, moment to moment | within-person AUC 0.587 (γ/θ ratio + PAC); p = 0.015, corrected 0.060 | time-on-task: 0.533 | Weak, just short of significance after correction; the single-electrode result does not replicate across sessions |

With 24 people, a model with **no signal** reaches ≥ 70% accuracy about 8% of
the time, so a bare accuracy figure without a chance level means little. Full
analysis: the phase 3 breakdown, and `model_comparison_<task>.csv` in the
derivatives folder.

*Phase 4–5 numbers below: re-run on the corrected features (after the
ICA-sampling fix), 17 Oct.*

Further within-person analyses (mixed models, phase 4): concentration rises
over a session (p = 0.0004), and higher delta (slow-wave) power goes with
lower concentration (p = 0.031); its link to tiredness, seen before the fix,
is no longer significant (p = 0.10). Exploratory but robust to per-person random
slopes and sex: **in experts a higher gamma/theta ratio goes with lower
concentration, in novices it doesn't** (interaction p = 0.0003). Electrode
minimalism: for mind wandering one electrode (AF8, chosen on training people
only) matches the full scalp.

Robustness (phase 5, `robustness_checks.py`): the expertise × ratio effect
**replicates separately in first sessions (p = 0.012) and later sessions
(p = 0.002)**, survives dropping any single person (weakest p = 0.010), is
carried by gamma power, and survives controlling for temple-electrode gamma, a
muscle marker (p = 0.002). The one-electrode mind-wandering detector (AF8,
AUC 0.629 vs 0.589 for all 64) does **not** replicate across sessions
(p = 0.06 and 0.09, and a different electrode is chosen each way).

### Independent replication (phase 6)

Pre-registered before any EEG was loaded
([`docs/preregistration_ds008836.md`](docs/preregistration_ds008836.md)) and
tested on OpenNeuro ds008836 (39 experienced Vipassana meditators, different
lab, EEG system and rating scale). The prediction: in experienced
meditators, a higher gamma/theta ratio at Fz + Pz goes with lower reported
meditation depth.

**Not replicated.** The pre-registered mixed model turned out to be
unidentifiable, so it never converged; an earlier "replicated" verdict rested
on that invalid fit and is withdrawn. Under the documented repair
([amendment](docs/preregistration_ds008836_amendment.md)), the slope is
negative as predicted (−0.072 within-person SD) but not significant
(one-sided p = 0.064; per-participant cross-check p = 0.089). Pooling all
blocks gives p = 0.006 (exploratory). Details:
[`docs/replication_ds008836_results.md`](docs/replication_ds008836_results.md).

### Decoding meditation depth (phase 7)

Planned before any run ([`docs/analysis_plan_depth_decoding.md`](docs/analysis_plan_depth_decoding.md)),
on ds008836 (39 people, 1,707 depth reports). **Decoded:** a Riemannian
model on each report's 64-channel covariance tells a person's
deeper-than-usual reports from their shallower ones at within-person AUC
**0.609** (95% 0.565–0.649; chance 0.502; Holm p = 0.035), beating the best
non-EEG baseline by +0.091 per person (0.046–0.132). Spectrum-based models
reach 0.55–0.57. Modest, and people vary a lot: 13 randomly chosen people
average 0.54. Details: [`docs/depth_decoding_results.md`](docs/depth_decoding_results.md).

**Searching for more (phase 8).** Nine variants (frequency bands, filter
bank, extra spectral features, ensemble, temporal context, target
treatments) were scored on 26 people and none beat the plain covariance
model. One rated calibration session helped a little (+0.013 on 12 held-out
people, interval −0.006 to 0.033), which isn't confirmed. It also fixed a
RidgeCV alpha-selection leak; the phase 7 model re-run with the fix gives
0.607. Details: [`docs/depth_search_results.md`](docs/depth_search_results.md).

### Introspective fidelity and the lead-up horizon (phase 9)

Pre-registered ([`docs/preregistration_phase9.md`](docs/preregistration_phase9.md)).
Two new questions on ds008836, beyond decoding:
- **The lead-up horizon (supported):** the depth a meditator is about to
  report is already in their EEG **60–120 s before the report** (AUC 0.584,
  Holm p = 0.020), and it doesn't fade with distance: equal 10-s bins
  score ~0.54 both 0–10 s and 110–120 s before. Depth behaves like a slow state.
- **Introspective fidelity (not supported, and why):** how well a person's
  reports track their own EEG doesn't correlate detectably across their two
  sessions (ρ = 0.15, p = 0.19), or with practice. About 73% of the
  per-session variation is sampling noise at ~21 reports per session.
  Measuring it reliably would need ~130 reports per session.

Details: [`docs/phase9_results.md`](docs/phase9_results.md).

## Publishing results

After a new analysis run, bundle it and push. The dashboard picks it up within
a few minutes, with no change to the site:

```bash
python src/publish_results.py \
    --models_run <model_comparison run dir> --replication_run <replication run dir> \
    --qc ds001787=<preprocessing_qc.csv> --qc ds008836=<preprocessing_qc.csv> \\
    --depth_run <depth_decoding run dir>
git add results/latest/results.json && git commit -m "Publish results" && git push
```

The bundle holds only summary tables (scores, chance levels, p-values,
per-person scores, the replication table, QC summaries) and each source run's
provenance. Dashboard source: [ManasDasri/neurodial](https://github.com/ManasDasri/neurodial).

## Limitations

- **Expertise is confounded with sex** (and partly age) in this dataset; only
  a sex-balanced sample can answer the expertise question.
- Self-report probes are subjective and arrive every ~90 s, so the labels are noisy.
- Scalp gamma (30–45 Hz) can't be fully separated from muscle activity.
- 23 usable people is small; treat every result as hypothesis-generating.

## Citation

If you use the dataset, cite the original authors:

> Brandmeyer, T., & Delorme, A. (2018). Reduced mind wandering in experienced
> meditators and associated EEG correlates. *Experimental Brain Research*.
> OpenNeuro ds001787.

## License

[MIT](LICENSE)

Analysis commands write a new `derivatives/neurodial/runs/` directory by default.
Use `--output_dir RUN_DIR` to collect distinct analysis stages in one run; repeating
a stage in that directory is rejected. Export with `--results_dir RUN_DIR`. Each
stage records its input hash, code revision, arguments and package versions.
Historical headline p-values require regeneration from a preserved run.

State-task nulls now circularly shift ratings within each recording; trait labels
remain shuffled between people. This preserves session membership and circular
serial order. It assumes approximate stationarity in probe order, not equal
clock-time spacing; p-values remain exploratory for irregular/nonstationary
ratings. This outcome-independent audit change must be distinguished from the
historical unrestricted-shuffle results.

Model outputs retain predictions for every model, per-person scores, descriptive
subject-bootstrap intervals and paired EEG-minus-baseline differences. These
intervals condition on the fitted cross-validation predictions; overlapping
training folds mean they are not independent-refit confidence bounds. Holm
adjustment covers every model/baseline tested within each task. Comparing across
tasks requires a separately declared combined family; the highest observed
score is descriptive, not an independently validated winning model.

### Retrospective evaluation contract

The held-out person's labels are excluded from model training, but their full
recording supplies EEG normalization statistics. Filtering and ICA also use the
continuous recording. These scores apply to retrospective analysis, not causal
real-time prediction. Prospective validation requires a separate past-only
preprocessing/calibration protocol, fixed before testing, and forward-time
evaluation on new recordings. Such a live predictor is outside this repository's
current claims; the exported evaluation scope makes this distinction explicit.

SHAP tables describe the gradient-boosting model fitted on the full dataset and
are explicitly marked `in_sample_full_dataset`. They do not explain held-out
predictions or independently establish generalization.

### Portable run workflow

Use Python 3.14 with `requirements.txt` and the tested `constraints.txt`. From
a fresh clone, the maintained downloader and pipeline are in `scripts/`:

```bash
python scripts/download_dataset.py --dataset ds001787 --bids_root ../ds001787
python scripts/run_pipeline.py --dataset ds001787 --bids_root ../ds001787 --run_dir ../runs/discovery-001 --n_permutations 100
python scripts/run_pipeline.py --dataset ds008836 --bids_root ../ds008836 --run_dir ../runs/replication-001 --stages preprocess,features,stats,export
python scripts/run_tests.py
```

Each run owns its intermediate directory, logs, result tables and completion
markers. For analysis of existing corrected intermediates, pass `--derivatives
PATH --stages models,stats,export`. A named run refuses completed stages; choose
a new directory for a new experiment. Legacy shell entry points live in
`scripts/` and accept `EEG_PYTHON` / `EEG_BIDS_ROOT`; they no longer watch log
text or mark failed subjects as complete. Keep raw EEG and large generated
derivatives outside Git, and preserve selected small result summaries/manifests
with the research report. Downloads use complete pinned object manifests;
ETags pin object identity, and multipart ETags are not represented as MD5 hashes.
