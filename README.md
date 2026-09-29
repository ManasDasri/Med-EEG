# Med-EEG — NeuroDial

**Interpretable gamma/theta EEG biomarkers for meditation depth**

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MNE](https://img.shields.io/badge/MNE--Python-1.6%2B-1f77b4)](https://mne.tools/)
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
| 2 | **Electrode minimalism** | Finds the smallest electrode set that keeps ~90% of full-scalp decoding accuracy |
| 3 | **Interpretable vs. black-box shootout** | 2-feature linear model vs. Random Forest / MLP on the full spectrum, same LOSO-CV |
| 4 | **Linear mixed-effects models** | Subject as random effect — tests the within-person relationship, not the confounded expert/novice split |
| 5 | **SHAP audit** | Checks what the black box actually relies on (theta/gamma, or artifacts like delta) |
| 6 | **NeuroDial dashboard** | Interactive, serverless replay of results as a neurofeedback-style dial |

## Pipeline

```
raw .bdf (BIDS)                 mne_bids.read_raw_bids
    │
Bandpass 1–45 Hz + notch        mne.filter
Bad-channel interp + ICA        eye / muscle artifact removal
    │
Epoch before each probe         2 s windows, 50% overlap, ~2 min lookback
    │
Welch PSD → delta…gamma power, gamma/theta ratio, Tort MI (PAC)
    │
Feature table (subject × probe × electrode set)
    │
Stats: LME  concentration ~ ratio + PAC + (1 | subject)
ML:    LOSO-CV, interpretable vs RF vs MLP, SHAP audit
    │
dashboard_data.json → dashboard.html
```

## Repository layout

```
src/
  utils.py                     shared constants: bands, channel maps, paths
  01_load_bids.py              sanity-check the BIDS layout, subjects, probe events
  02_preprocess.py             filter, ICA, epoch around probes
  03_features.py               band power, gamma/theta ratio, PAC per electrode/cluster
  04_stats_analysis.py         mixed-effects models + electrode-minimalism curve
  05_ml_pipeline.py            LOSO-CV model shootout + SHAP audit
  06_export_dashboard_data.py  bundle results into dashboard_data.json
  07_followup_analysis.py      follow-ups: delta power, tiredness, random slopes, expertise
dashboard.html                 NeuroDial results dashboard (open directly in a browser)
dashboard_data.sample.json     sample data for the dashboard
meditation_pipeline.py         original single-file KNN prototype (hard-coded paths)
requirements.txt
```

## Getting started

```bash
git clone https://github.com/ManasDasri/Med-EEG.git
cd Med-EEG
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# download the dataset (~5 GB)
openneuro-py download --dataset ds001787 --target_dir data/ds001787
```

Run the steps in order. Each writes intermediates to
`data/ds001787/derivatives/neurodial/`, so you can resume mid-pipeline.

```bash
python src/01_load_bids.py              --bids_root data/ds001787
python src/02_preprocess.py             --bids_root data/ds001787   # --subjects / --limit for a quick run
python src/03_features.py               --bids_root data/ds001787
python src/04_stats_analysis.py         --bids_root data/ds001787
python src/05_ml_pipeline.py            --bids_root data/ds001787 --task classify   # or --task regress
python src/06_export_dashboard_data.py  --bids_root data/ds001787
python src/07_followup_analysis.py      --bids_root data/ds001787   # optional
```

### Dashboard

Open `dashboard.html` in a browser and load `dashboard_data.json` (or
`dashboard_data.sample.json` to try it without the dataset). No server needed.

## What success looks like

- **LME:** ratio and/or PAC significantly predict trial-level concentration,
  controlling for subject — reported with effect sizes, not just p-values.
- **ML:** the 2-feature interpretable model lands within ~5–10% of the
  black-box models under leave-one-subject-out CV.
- **Electrode minimalism:** a 2–4 electrode subset recovers most of full-scalp
  performance.

All evaluation uses **leave-one-subject-out** CV — random k-fold leaks subject
identity and inflates accuracy.

## Limitations

- 24 subjects is small for a 64-channel feature space; hence the emphasis on
  simple models and few electrodes.
- Self-report probes are subjective and retrospective — noisy labels, not gold
  standard.
- Expertise is confounded with age and other traits; the within-subject LME is
  the stronger claim, not the group contrast.

## Citation

If you use the dataset, cite the original authors:

> Brandmeyer, T., & Delorme, A. (2018). Reduced mind wandering in experienced
> meditators and associated EEG correlates. *Experimental Brain Research*.
> OpenNeuro ds001787.

## License

[MIT](LICENSE)
