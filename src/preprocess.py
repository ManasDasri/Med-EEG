"""Step 2: filter, remove eye/muscle artifacts with ICA, and cut epochs from
the ~2 minutes preceding each self-report probe.

Writes one -epo.fif file per subject/session into derivatives/, plus a
probe_labels.csv mapping each epoch to its subject, session, probe time, and
(once you've filled them in from the *_events.tsv rating codes) concentration
/ mind-wandering scores.

Usage:
    python src/preprocess.py --bids_root data/ds001787
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BANDPASS_HIGH_HZ, BANDPASS_LOW_HZ, BIOSEMI64_TO_1020,
                    EPOCH_LENGTH_S, EPOCH_OVERLAP_S, N_PROBE_QUESTIONS,
                    NOTCH_FREQ_HZ, NON_EEG_CHANNELS, PROBE_LOOKBACK_S,
                    PROBE_STIMULUS_VALUE, RESPONSE_VALUE_TO_RATING,
                    Paths)  # noqa: E402

mne.set_log_level("WARNING")


def rename_and_type_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """This dataset uses BioSemi's A1-A32/B1-B32 numbering instead of
    standard 10-20 names, and includes non-EEG sensors (EOG, GSR,
    respiration, etc). Rename to standard names, mark non-EEG channel
    types so filtering/ICA/band-power code only ever touches real EEG,
    and attach the standard BioSemi-64 montage so downstream steps that
    need real electrode positions (like interpolate_bads) actually work
    -- renaming alone doesn't give MNE any 3D position information."""
    raw = raw.copy()
    rename_map = {old: new for old, new in BIOSEMI64_TO_1020.items()
                  if old in raw.ch_names}
    if rename_map:
        raw.rename_channels(rename_map)
    type_map = {ch: t for ch, t in NON_EEG_CHANNELS.items()
                if ch in raw.ch_names}
    if type_map:
        raw.set_channel_types(type_map)
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing="warn", match_case=False)
    return raw


def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Bandpass + notch filter, then ICA to remove eye/muscle components."""
    raw = rename_and_type_channels(raw)
    raw = raw.load_data()
    raw.filter(BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, fir_design="firwin")
    raw.notch_filter(NOTCH_FREQ_HZ, fir_design="firwin")

    # Mark and interpolate flat/noisy channels before ICA so they don't
    # dominate components.
    raw.info["bads"] = mne.preprocessing.find_bad_channels_lof(
        raw, picks="eeg", n_neighbors=8) if hasattr(
        mne.preprocessing, "find_bad_channels_lof") else raw.info["bads"]
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True)

    ica = mne.preprocessing.ICA(n_components=20, method="fastica",
                                 random_state=42, max_iter="auto")
    # picks='eeg' skips GSR/respiration/temperature/etc entirely (they're
    # not brain signal, no reason to spend time on them). decim=5 fits
    # ICA on every 5th time sample instead of every sample -- ICA finds
    # the same underlying components either way since it doesn't need
    # every single sample, and this is the single biggest speed win here.
    ica.fit(raw, picks="eeg", decim=5)
    try:
        eog_idx, _ = ica.find_bads_eog(raw)
        ica.exclude.extend(eog_idx)
    except (RuntimeError, ValueError):
        warnings.warn("No EOG channel found for automatic EOG-component "
                       "detection -- inspect ICA components by hand for "
                       "this subject if results look noisy.")
    ica.apply(raw)
    return raw


def find_probe_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows of events_df that are probe onsets: trial_type=='stimulus'
    with value==128 (per Brandmeyer & Delorme 2018 -- see config.py for the
    full explanation). Returns rows with their original integer position
    preserved as the index, since decode_probe_responses() needs it to look
    ahead at the following response rows."""
    required = {"trial_type", "value"}
    if not required.issubset(events_df.columns):
        raise ValueError(
            f"events.tsv is missing expected columns {required} -- open it "
            f"and check the actual column names for this download.")
    mask = ((events_df["trial_type"].astype(str).str.lower() == "stimulus")
            & (pd.to_numeric(events_df["value"], errors="coerce")
               == PROBE_STIMULUS_VALUE))
    probes = events_df.loc[mask]
    if probes.empty:
        raise ValueError(
            f"No probe events found (trial_type=='stimulus', "
            f"value=={PROBE_STIMULUS_VALUE}). Open this subject's "
            f"events.tsv and confirm the stimulus/value convention matches "
            f"config.py -- some re-exports may use a different value.")
    return probes


def decode_probe_responses(events_df: pd.DataFrame, probe_row_idx: int
                            ) -> dict:
    """Given the integer position of a probe (stimulus/128) row in
    events_df, read the up-to-N_PROBE_QUESTIONS immediately following
    'response' rows and decode their button codes into 0-3 ratings for
    the 3 fixed questions (meditation depth, mind-wandering depth,
    tiredness). Missing/malformed responses become NaN rather than
    silently defaulting to 0, since 0 is itself a valid rating."""
    ratings = [np.nan] * N_PROBE_QUESTIONS
    j = probe_row_idx + 1
    n = len(events_df)
    count = 0
    while j < n and count < N_PROBE_QUESTIONS:
        row = events_df.iloc[j]
        if str(row.get("trial_type", "")).lower() != "response":
            break
        try:
            code = int(float(row["value"]))
        except (TypeError, ValueError):
            code = None
        ratings[count] = RESPONSE_VALUE_TO_RATING.get(code, np.nan)
        count += 1
        j += 1
    return {
        "concentration_rating": ratings[0],       # Q1: depth of meditation
        "mind_wandering_rating": ratings[1],       # Q2: depth of mind wandering
        "tiredness_rating": ratings[2],             # Q3: how tired
        "n_responses_found": count,
    }


def epoch_pre_probe_window(raw: mne.io.BaseRaw, probe_onset_s: float
                            ) -> mne.Epochs:
    """Cut fixed-length overlapping epochs from the PROBE_LOOKBACK_S window
    immediately before a probe."""
    start = max(0.0, probe_onset_s - PROBE_LOOKBACK_S)
    stop = probe_onset_s
    seg = raw.copy().crop(tmin=start, tmax=stop)
    events = mne.make_fixed_length_events(
        seg, duration=EPOCH_LENGTH_S,
        overlap=EPOCH_OVERLAP_S)
    epochs = mne.Epochs(seg, events, tmin=0, tmax=EPOCH_LENGTH_S,
                         baseline=None, preload=True, reject_by_annotation=True)
    return epochs


def process_subject(bids_root: Path, sub: str, deriv_dir: Path) -> list[dict]:
    """Process one subject's recording(s). Returns metadata rows for the
    probe_labels.csv manifest. Adapt the glob pattern if your download uses
    a different raw extension (.bdf is standard for this dataset)."""
    rows: list[dict] = []
    raw_files = sorted((bids_root / sub).rglob("*_eeg.bdf"))
    if not raw_files:
        raw_files = sorted((bids_root / sub).rglob("*_eeg.set"))
    if not raw_files:
        print(f"[skip] no raw EEG file found for {sub}")
        return rows

    for raw_path in raw_files:
        session_tag = raw_path.stem
        events_path = raw_path.parent / raw_path.name.replace(
            "_eeg.bdf", "_events.tsv").replace("_eeg.set", "_events.tsv")
        if not events_path.exists():
            print(f"[skip] no events.tsv next to {raw_path.name}")
            continue

        print(f"[..] {sub} / {session_tag}")
        raw = (mne.io.read_raw_bdf(raw_path, preload=False)
               if raw_path.suffix == ".bdf"
               else mne.io.read_raw_eeglab(raw_path, preload=False))
        raw_clean = preprocess_raw(raw)

        events_df = pd.read_csv(events_path, sep="\t").reset_index(drop=True)
        probes = find_probe_events(events_df)

        for i, probe_row_idx in enumerate(probes.index):
            onset_s = float(events_df.loc[probe_row_idx, "onset"])
            epochs = epoch_pre_probe_window(raw_clean, onset_s)
            if len(epochs) == 0:
                continue
            ratings = decode_probe_responses(events_df, probe_row_idx)
            if ratings["n_responses_found"] < N_PROBE_QUESTIONS:
                print(f"[warn] {sub} probe {i} at {onset_s:.1f}s only found "
                      f"{ratings['n_responses_found']}/{N_PROBE_QUESTIONS} "
                      f"response rows -- some ratings will be NaN")
            out_path = (deriv_dir /
                        f"{sub}_{session_tag}_probe-{i:02d}-epo.fif")
            epochs.save(out_path, overwrite=True)
            rows.append({
                "subject": sub,
                "session": session_tag,
                "probe_index": i,
                "probe_onset_s": onset_s,
                "epochs_file": str(out_path),
                "n_epochs": len(epochs),
                "concentration_rating": ratings["concentration_rating"],
                "mind_wandering_rating": ratings["mind_wandering_rating"],
                "tiredness_rating": ratings["tiredness_rating"],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subject IDs to process, e.g. "
                              "'sub-001,sub-002' -- use this to test the "
                              "pipeline quickly before running all 24 "
                              "subjects, since preprocessing (ICA) is slow.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N subjects found -- "
                              "another quick way to test on a subset.")
    args = parser.parse_args()

    paths = Paths.from_root(args.bids_root)
    subjects = sorted(p.name for p in paths.bids_root.glob("sub-*")
                       if p.is_dir())

    if args.subjects:
        wanted = {s.strip() for s in args.subjects.split(",")}
        missing = wanted - set(subjects)
        if missing:
            raise SystemExit(f"Requested subject(s) not found: {missing}")
        subjects = [s for s in subjects if s in wanted]
    elif args.limit:
        subjects = subjects[:args.limit]

    print(f"[ok] processing {len(subjects)} subject(s): {subjects}")

    out_csv = paths.derivatives / "probe_labels.csv"
    all_rows: list[dict] = []
    failed_subjects: list[str] = []

    for sub in subjects:
        try:
            all_rows.extend(process_subject(paths.bids_root, sub,
                                             paths.derivatives))
        except Exception as e:
            failed_subjects.append(sub)
            print(f"[ERROR] {sub} failed and was skipped: {type(e).__name__}: {e}")
            print(f"[ERROR] {sub}'s data may have a quirk not seen in other "
                  f"subjects -- note this down and tell me the error so we "
                  f"can fix it, then re-run just this subject later with "
                  f"--subjects {sub}")
        finally:
            # Save progress after every subject, success or failure, so a
            # crash on subject N never loses the work already done on
            # subjects 1..N-1. Safe to re-run this script at any time.
            if all_rows:
                pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    manifest = pd.DataFrame(all_rows)
    manifest.to_csv(out_csv, index=False)
    print(f"\n[ok] wrote {len(manifest)} probe-epoch rows to {out_csv}")
    if failed_subjects:
        print(f"[warn] {len(failed_subjects)} subject(s) failed and were "
              f"skipped entirely: {failed_subjects}")
    n_nan = manifest["concentration_rating"].isna().sum()
    if n_nan:
        print(f"[warn] {n_nan}/{len(manifest)} rows have NaN "
              f"concentration_rating (missing/malformed response events) "
              f"-- these will be dropped automatically in 04/05.")
    if args.subjects or args.limit:
        print("[note] this was a partial run (--subjects/--limit used). "
              "Once you've confirmed the output looks sane, re-run without "
              "those flags to process all subjects -- this will overwrite "
              "probe_labels.csv with the full set.")


if __name__ == "__main__":
    main()
