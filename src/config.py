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
NOTCH_FREQ_HZ = 50.0  # EU mains; use 60.0 if recorded in the US
BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 45.0
RESAMPLE_HZ = 256.0  # dataset is already delivered at 256 Hz

# --- Epoching ------------------------------------------------------------
EPOCH_LENGTH_S = 2.0
EPOCH_OVERLAP_S = 1.0
PROBE_LOOKBACK_S = 120.0  # analyze the ~2 min preceding each self-report probe

# --- Expected BIDS event / probe naming ----------------------------------
# CONFIRM these against the actual *_events.tsv trial_type column for this
# dataset before trusting downstream code -- BIDS event naming for probe
# questions varies by upload version. inspect_dataset.py prints unique
# trial_type values for exactly this reason; don't skip that step.
PROBE_EVENT_KEYWORDS = ("probe", "concentration", "mind_wandering", "rating")

# --- Probe / self-report event decoding -----------------------------------
# From Brandmeyer & Delorme (2018), the source paper for this dataset: a
# probe interrupts meditation with 3 questions, answered on a 0-3 scale via
# button press. In the BIDS events.tsv this shows up as a trial_type=
# "stimulus" row with value=128 (the probe onset), immediately followed by
# 3 trial_type="response" rows -- one per question, in this fixed order:
#   Q1: "rate the depth of your meditation"     -> concentration_rating
#   Q2: "rate the depth of your mind wandering" -> mind_wandering_rating
#   Q3: "rate how tired you are"                -> tiredness_rating
# The response `value` column holds a button-box bit code (1/2/4/8), not
# the 0-3 rating directly -- standard encoding is button k -> 2^(k-1), so
# rating = log2(code). Verify this against your own data: if ratings look
# skewed or never hit 0, double check by finding a row with value==1.
PROBE_STIMULUS_VALUE = 128
RESPONSE_VALUE_TO_RATING = {1: 0, 2: 1, 4: 2, 8: 3}
N_PROBE_QUESTIONS = 3

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
