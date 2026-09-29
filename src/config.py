"""Shared constants and small helpers for the meditation EEG pipeline.

Dataset: Delorme & Brandmeyer, "EEG meditation study" (OpenNeuro ds001787 /
Zenodo 2536267). 24 subjects (12 experienced meditators, 12 novices), 64 EEG
channels + misc, originally 2048 Hz downsampled to 256 Hz, .bdf format,
probed roughly every 2 minutes with self-report questions during ~1 hour of
seated meditation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Frequency bands ---------------------------------------------------
THETA_BAND = (4.0, 8.0)
GAMMA_BAND = (30.0, 45.0)
# Extra bands kept around for the "black box" feature set / sanity checks
DELTA_BAND = (1.0, 4.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
ALL_BANDS = {
    "delta": DELTA_BAND,
    "theta": THETA_BAND,
    "alpha": ALPHA_BAND,
    "beta": BETA_BAND,
    "gamma": GAMMA_BAND,
}

# --- BioSemi 64 numbered-channel -> standard 10-20/10-10 name mapping ---
# This dataset's channels.tsv uses BioSemi's own A1-A32/B1-B32 numbering,
# not standard 10-20 names. This is the standard, well-documented BioSemi
# 64-channel equivalence (same order as MNE's built-in "biosemi64" montage).
# preprocess.py renames channels using this dict right after loading the
# raw file, so everything downstream can just use normal names like "Fz".
BIOSEMI64_TO_1020 = {
    "A1": "Fp1", "A2": "AF7", "A3": "AF3", "A4": "F1", "A5": "F3",
    "A6": "F5", "A7": "F7", "A8": "FT7", "A9": "FC5", "A10": "FC3",
    "A11": "FC1", "A12": "C1", "A13": "C3", "A14": "C5", "A15": "T7",
    "A16": "TP7", "A17": "CP5", "A18": "CP3", "A19": "CP1", "A20": "P1",
    "A21": "P3", "A22": "P5", "A23": "P7", "A24": "P9", "A25": "PO7",
    "A26": "PO3", "A27": "O1", "A28": "Iz", "A29": "Oz", "A30": "POz",
    "A31": "Pz", "A32": "CPz",
    "B1": "Fpz", "B2": "Fp2", "B3": "AF8", "B4": "AF4", "B5": "AFz",
    "B6": "Fz", "B7": "F2", "B8": "F4", "B9": "F6", "B10": "F8",
    "B11": "FT8", "B12": "FC6", "B13": "FC4", "B14": "FC2", "B15": "FCz",
    "B16": "Cz", "B17": "C2", "B18": "C4", "B19": "C6", "B20": "T8",
    "B21": "TP8", "B22": "CP6", "B23": "CP4", "B24": "CP2", "B25": "P2",
    "B26": "P4", "B27": "P6", "B28": "P8", "B29": "P10", "B30": "PO8",
    "B31": "PO4", "B32": "O2",
}
# Non-EEG channels present in this dataset's channels.tsv -- these get
# marked with their real type (eog/misc/etc) in preprocess.py so
# filtering/ICA/band-power code doesn't treat them as EEG signal.
# EXG1/EXG2 are typically mastoid references, EXG3-6 are EOG electrodes.
NON_EEG_CHANNELS = {
    "EXG1": "misc", "EXG2": "misc", "EXG3": "eog", "EXG4": "eog",
    "EXG5": "eog", "EXG6": "eog", "EXG7": "misc", "EXG8": "misc",
    "GSR1": "gsr", "GSR2": "gsr", "Erg1": "misc", "Erg2": "misc",
    "Resp": "resp", "Plet": "misc", "Temp": "misc",
}

# --- Electrode clusters for the "electrode minimalism" experiment ------
# Standard 10-20 / 10-10 labels -- these are used AFTER renaming with
# BIOSEMI64_TO_1020 above, so plain "Fz" etc. is correct here.
FRONTAL_MIDLINE_THETA_CHS = ["Fz", "FCz", "Cz"]
POSTERIOR_GAMMA_CHS = ["Pz", "POz", "Oz"]
MINIMAL_CH_SET = FRONTAL_MIDLINE_THETA_CHS[:1] + POSTERIOR_GAMMA_CHS[:1]  # e.g. Fz + Pz

# --- Preprocessing ------------------------------------------------------
# Fallback only: preprocess.py reads the real value from each recording's
# *_eeg.json sidecar ("PowerLineFrequency" -- 50 Hz for this dataset,
# recorded in Toulouse, France).
NOTCH_FREQ_HZ = 50.0
BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 45.0
RESAMPLE_HZ = 256.0  # dataset is already delivered at 256 Hz; higher gets downsampled

# ICA: 20 components on 64 channels captures the big artifact sources (blinks,
# eye movements, muscle) without splitting brain activity into noise.
ICA_N_COMPONENTS = 20
ICA_RANDOM_STATE = 42
# The dataset declares no EOG channels (EOGChannelCount=0 in the sidecar), so
# blink/eye components are found by correlation with the most frontal scalp
# electrodes -- they sit right above the eyes and pick up blinks strongly.
EOG_PROXY_CHS = ["Fp1", "Fp2"]
# A component is "eye" only if BOTH hold (checked against the scalp maps of 6
# recordings: every eye component had r >= 0.6 and share >= 0.54; every brain
# component with r >= 0.45 had share <= 0.14):
#   - |correlation| with EOG_PROXY_CHS (1-10 Hz) >= EOG_MIN_CORR
#   - share of the component's squared map weight on the frontal-pole
#     electrodes (EYE_MAP_CHS) >= EOG_MIN_FRONTAL_SHARE
# MNE's default (z-score > 3 among components) missed eye components when the
# eye signal was split across two or three of them, and removed posterior
# ~10 Hz alpha components -- brain signal this project needs.
EOG_MIN_CORR = 0.5
EYE_MAP_CHS = ["Fp1", "Fpz", "Fp2", "AF7", "AF8"]
EOG_MIN_FRONTAL_SHARE = 0.4

# Epochs whose peak-to-peak amplitude on any EEG channel exceeds this (after
# ICA) are dropped as residual artifact. 150 uV is a common post-ICA limit;
# lower it if features look noisy, raise it if too many epochs get dropped
# (the QC report shows the drop rate per recording).
REJECT_PEAK_TO_PEAK_V = 150e-6
FLAT_PEAK_TO_PEAK_V = 1e-6  # anything flatter than this is a dead channel

# --- Epoching ------------------------------------------------------------
EPOCH_LENGTH_S = 2.0
EPOCH_OVERLAP_S = 1.0
PROBE_LOOKBACK_S = 120.0  # analyze up to ~2 min preceding each self-report probe
# Probes are as little as 16 s apart (median ~95 s), so a fixed 120 s lookback
# would reach back into the PREVIOUS probe's question/answer period (voice
# prompts, button presses, eye movements). Each window therefore starts no
# earlier than the previous probe's last event plus this settling buffer.
POST_PROBE_BUFFER_S = 10.0

# --- Expected BIDS event / probe naming ----------------------------------
# CONFIRM these against the actual *_events.tsv trial_type column for this
# dataset before trusting downstream code -- BIDS event naming for probe
# questions varies by upload version. inspect_dataset.py prints unique
# trial_type values for exactly this reason; don't skip that step.
PROBE_EVENT_KEYWORDS = ("probe", "concentration", "mind_wandering", "rating")

# --- Probe / self-report event decoding -----------------------------------
# Verified against the dataset's own experiment script
# (code/run_mw_experiment6.m) and all 40 events.tsv files. A probe is a
# trial_type="stimulus" row with value=128, followed by trial_type="response"
# rows answering up to 3 voice-prompted questions in this FIXED order:
#   Q1: "rate your meditation"     -> concentration_rating
#   Q2: "rate your mind wandering" -> mind_wandering_rating
#   Q3: "rate how tired you are"   -> tiredness_rating
# Pressing digit key k sends code 2**k, so rating = log2(code); in practice
# only 2/4/8 (ratings 1-3) occur. Only ~37% of probes have all 3 answers:
# a non-digit key cancels the remaining questions WITHOUT sending a code, so
# missing answers are always the trailing ones and position still tells you
# which question an answer belongs to.
PROBE_STIMULUS_VALUE = 128
RESPONSE_VALUE_TO_RATING = {1: 0, 2: 1, 4: 2, 8: 3}
N_PROBE_QUESTIONS = 3
# Answers normally arrive 3-10 s apart (99th percentile 9.3 s). A "response"
# arriving longer than this after the previous event is a stray button press
# during meditation, not an answer, and ends the probe's answer sequence.
MAX_RESPONSE_GAP_S = 30.0

GROUP_MAP_HINT = (
    "Group (experienced vs novice) typically lives in participants.tsv, "
    "not in the events. Check participants.tsv columns after downloading."
)


@dataclass
class Paths:
    bids_root: Path
    derivatives: Path

    @classmethod
    def from_root(cls, bids_root: str | Path) -> "Paths":
        root = Path(bids_root)
        deriv = root / "derivatives" / "neurodial"
        deriv.mkdir(parents=True, exist_ok=True)
        return cls(bids_root=root, derivatives=deriv)


def band_power_columns() -> list[str]:
    return [f"{band}_power" for band in ALL_BANDS]
