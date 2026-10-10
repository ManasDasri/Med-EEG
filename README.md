# Med-EEG — NeuroDial

**Interpretable gamma/theta EEG biomarkers for meditation depth**

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

If yes, a meditation-feedback device doesn't need a deep net or a research cap:
a 2–4 electrode headband would do.

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
ML:    LOSO-CV, interpretable vs RF vs MLP, SHAP audit
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
  model_comparison.py     LOSO models + permutation tests + non-EEG baselines + SHAP
  export_dashboard.py     bundle results into dashboard_data.json
  followup_analysis.py    follow-ups: delta power, tiredness, random slopes, expertise
tests/
  test_preprocess.py      probe/answer decoding checks (python tests/test_preprocess.py)
  test_features.py        feature checks on synthetic signals (python tests/test_features.py)
  test_models.py          signal vs no-signal model checks (python tests/test_models.py)
  test_electrode_minimalism.py  nested electrode selection checks
dashboard.html            NeuroDial results dashboard (open directly in a browser)
dashboard_data.sample.json  sample data for the dashboard
knn_prototype.py          original single-file KNN prototype
requirements.txt
```

## Getting started

```bash
git clone https://github.com/ManasDasri/Med-EEG.git
cd Med-EEG
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

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
python src/model_comparison.py     --bids_root data/ds001787 --task expertise  # or concentration, mind_wandering
python src/export_dashboard.py     --bids_root data/ds001787
python src/followup_analysis.py    --bids_root data/ds001787   # optional
```

After preprocessing, check `preprocessing_qc.csv` in the derivatives folder:
it lists bad channels, removed ICA components and the epoch rejection rate
for every recording.

### Dashboard

Open `dashboard.html` in a browser and load `dashboard_data.json` (or
`dashboard_data.sample.json` to try it without the dataset). No server needed.

## Results (all 24 subjects)

Leave-one-subject-out throughout; every score is compared with shuffled-label
runs (chance level, p-value) **and with non-EEG baselines**. One subject
(sub-010, poor electrode contact) is excluded by a rule set before any
results, leaving 23 people and 929 probes.

| Question | Best EEG result | Non-EEG baseline | Verdict |
|---|---|---|---|
| Expert vs novice | 78.3% of people (18/23), p = 0.012 | **sex alone: 82.6%** | Not supported: experts are 9/11 male, novices 10/12 female, and the model mostly detects sex |
| Concentration, moment to moment | within-person AUC 0.54 (n.s.) | time-on-task: 0.56 | No signal |
| Mind wandering, moment to moment | within-person AUC 0.59 (gamma/theta ratio), p = 0.004; **one electrode (AF8) alone: 0.63** | time-on-task: 0.53 | Weak; borderline after multiple-comparison correction; the signal sits at the right forehead above the eye, so likely muscle or eye-movement artifact rather than brain |

With 24 people, a model with **no signal** reaches ≥ 70% accuracy about 8% of
the time, so a bare accuracy figure without a chance level means little. Full
analysis: the phase 3 breakdown, and `model_comparison_<task>.csv` in the
derivatives folder.

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
