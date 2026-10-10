"""Dataset adapter for OpenNeuro ds008836 (Reggente et al.): experienced
Vipassana meditators, 64-channel BrainAmp EEG (EEGLAB .set), depth reports
(1-5) during 35-min meditation blocks. Used by preprocess.py --dataset ds008836
for the pre-registered replication (docs/preregistration_ds008836.md).

Everything after loading and windowing -- filtering, ICA, epoching, features
-- is the same code as for ds001787.
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

# The four 35-min meditation blocks with depth reports.
REPORT_BLOCKS = ["silent_probe", "silent_emerge", "sapo_probe", "sapo_emerge"]
WINDOW_STARTS = ("prompt_window_begin", "report_window_begin")
RATING_EVENTS = ("depth_rating", "confidence_rating")
LOOKBACK_S = 120.0       # as ds001787 (config.PROBE_LOOKBACK_S)
AFTER_REPORT_S = 10.0    # as ds001787 (config.POST_PROBE_BUFFER_S)
CHANNEL_TYPES = {"EEG": "eeg", "EMG": "emg", "ECG": "ecg", "RESP": "resp",
                 "GSR": "gsr", "PPG": "misc"}


def load_recording(raw_path: Path) -> mne.io.BaseRaw:
    """EEGLAB file -> Raw with channel types and bad channels from the BIDS
    channels.tsv, and the standard 10-05 electrode positions."""
    raw = mne.io.read_raw_eeglab(raw_path, preload=True)
    channels = pd.read_csv(sidecar(raw_path, "_channels.tsv"), sep="\t")
    raw.set_channel_types({r.name: CHANNEL_TYPES.get(r.type, "misc")
                           for r in channels.itertuples() if r.name in raw.ch_names},
                          on_unit_change="ignore")
    raw.info["bads"] = [r.name for r in channels.itertuples()
                        if r.status == "bad" and r.name in raw.ch_names]
    raw.set_montage("standard_1005", on_missing="warn")
    return raw


def sidecar(raw_path: Path, suffix: str) -> Path:
    """sub-X_ses-Y_task-Z_eeg.set -> sub-X_ses-Y_task-Z<suffix>"""
    return raw_path.with_name(raw_path.name.rsplit("_eeg", 1)[0] + suffix)


def parse_reports(events_df: pd.DataFrame) -> pd.DataFrame:
    """One row per depth report in the four meditation blocks, in the same
    shape as preprocess.parse_probes (probe_index, probe_onset_s,
    window_start_s) plus depth_rating, block, block_type and variant.

    The window ends where the report's prompt/report window began (so the
    response itself is excluded) and starts at the latest of: LOOKBACK_S
    earlier, AFTER_REPORT_S after the previous rating in the same block, or
    the block's start. Ratings of 0 are dropped and values above 5 set to 5,
    the dataset's documented convention. A rating without its own
    prompt/report marker (1 case in the dataset) is dropped: its window
    can't be known."""
    ev = events_df.sort_values("onset", kind="stable").reset_index(drop=True)
    kind, block, onset = ev["trial_type"], ev["block"], ev["onset"].astype(float)
    rows = []
    for b in REPORT_BLOCKS:
        starts = onset[kind == f"{b}_begin"]
        if starts.empty:
            continue
        in_block = block == b
        windows = onset[in_block & kind.isin(WINDOW_STARTS)].to_numpy()
        ratings = onset[in_block & kind.isin(RATING_EVENTS)].to_numpy()
        used: set[float] = set()
        for i in ev.index[in_block & (kind == "depth_rating")]:
            before = windows[windows <= onset[i]]
            if not before.size or before.max() in used:
                continue  # no prompt/report marker of its own: window unknown
            end = before.max()
            used.add(end)  # claimed even if the rating is dropped below
            depth = pd.to_numeric(ev.at[i, "depth"], errors="coerce")
            if np.isnan(depth) or depth == 0:
                continue
            earlier = ratings[ratings < end]
            start = max(end - LOOKBACK_S, starts.iloc[0],
                        earlier.max() + AFTER_REPORT_S if earlier.size else -np.inf)
            rows.append({"probe_onset_s": end, "window_start_s": start,
                         "depth_rating": min(float(depth), 5.0), "block": b,
                         "block_type": b.split("_")[0], "variant": b.split("_")[1]})
    if not rows:
        raise ValueError("no depth reports in the meditation blocks")
    out = pd.DataFrame(rows).sort_values("probe_onset_s").reset_index(drop=True)
    out.insert(0, "probe_index", range(len(out)))
    return out
