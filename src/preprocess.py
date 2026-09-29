"""Step 2: turn raw EEG recordings into clean, labelled epochs from the
window before each self-report probe.

For every recording, in this order (the order matters -- each step assumes
the previous one already happened):
  1. load the .bdf, rename BioSemi channels to 10-20 names, type non-EEG sensors
  2. resample (only if above RESAMPLE_HZ), band-pass, notch out mains noise
  3. find bad channels (Local Outlier Factor) and interpolate them
  4. re-reference to the average -- BioSemi records reference-free (CMS/DRL),
     so without this every channel carries a large shared noise signal
  5. ICA: remove eye-movement/blink and muscle components
  6. cut 2 s epochs from each probe's pre-probe window, drop noisy epochs

Writes into <bids_root>/derivatives/neurodial/:
  - <sub>_<ses>_probe-NN-epo.fif  one file per probe; each epoch carries
                                  metadata (probe_index, seconds_before_probe)
  - probe_labels.csv              one row per probe: ratings + epochs file
  - preprocessing_qc.csv          one row per recording: what got removed

Usage:
    python src/preprocess.py --bids_root data/ds001787
    python src/preprocess.py --bids_root data/ds001787 --subjects sub-001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne_bids import get_entities_from_fname

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BANDPASS_HIGH_HZ, BANDPASS_LOW_HZ, BIOSEMI64_TO_1020,
                    EOG_PROXY_CHS, EPOCH_LENGTH_S, EPOCH_OVERLAP_S,
                    FLAT_PEAK_TO_PEAK_V, ICA_N_COMPONENTS, ICA_RANDOM_STATE,
                    MAX_RESPONSE_GAP_S, N_PROBE_QUESTIONS, NON_EEG_CHANNELS,
                    NOTCH_FREQ_HZ, POST_PROBE_BUFFER_S, PROBE_LOOKBACK_S,
                    PROBE_STIMULUS_VALUE, REJECT_PEAK_TO_PEAK_V,
                    RESAMPLE_HZ, RESPONSE_VALUE_TO_RATING,
                    Paths)  # noqa: E402

mne.set_log_level("WARNING")

RATING_COLUMNS = ["concentration_rating", "mind_wandering_rating",
                  "tiredness_rating"]  # Q1, Q2, Q3 -- see config.py


# --- 1. Loading ------------------------------------------------------------

def rename_and_type_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """This dataset uses BioSemi's A1-A32/B1-B32 numbering instead of
    standard 10-20 names, and includes non-EEG sensors (EOG, GSR,
    respiration, etc). Rename to standard names, mark non-EEG channel
    types so filtering/ICA/band-power code only ever touches real EEG,
    and attach the standard BioSemi-64 montage so steps that need real
    electrode positions (interpolation, muscle detection) actually work
    -- renaming alone doesn't give MNE any 3D position information."""
    raw = raw.copy()
    rename_map = {old: new for old, new in BIOSEMI64_TO_1020.items()
                  if old in raw.ch_names}
    if rename_map:
        raw.rename_channels(rename_map)
    type_map = {ch: t for ch, t in NON_EEG_CHANNELS.items()
                if ch in raw.ch_names}
    if type_map:
        raw.set_channel_types(type_map, on_unit_change="ignore")
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing="warn", match_case=False)
    return raw


def read_line_freq(raw_path: Path) -> float:
    """Mains frequency from the recording's BIDS sidecar, so the notch filter
    is right even for data recorded on a 60 Hz grid."""
    sidecar = raw_path.with_name(raw_path.name.replace("_eeg.bdf", "_eeg.json"))
    if sidecar.exists():
        freq = json.loads(sidecar.read_text()).get("PowerLineFrequency")
        if isinstance(freq, (int, float)):
            return float(freq)
    print(f"[warn] no PowerLineFrequency in {sidecar.name}, "
          f"using {NOTCH_FREQ_HZ} Hz from config.py")
    return NOTCH_FREQ_HZ


# --- 2-5. Cleaning the continuous signal -------------------------------------

def clean_continuous(raw: mne.io.BaseRaw, line_freq: float
                     ) -> tuple[mne.io.BaseRaw, dict]:
    """Filter, repair bad channels, average-reference, and ICA-clean the whole
    recording in place. Returns the raw plus a QC dict of what was removed."""
    if raw.info["sfreq"] > RESAMPLE_HZ:
        raw.resample(RESAMPLE_HZ)
    raw.filter(BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, picks="eeg",
               fir_design="firwin")
    raw.notch_filter(line_freq, picks="eeg", fir_design="firwin")

    # Bad channels must be repaired BEFORE average referencing, or one dead
    # or noisy electrode leaks into every other channel through the average.
    bads = list(mne.preprocessing.find_bad_channels_lof(raw, picks="eeg"))
    raw.info["bads"] = bads
    if bads:
        raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference("average", projection=False)

    ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS,
                                method="fastica",
                                random_state=ICA_RANDOM_STATE,
                                max_iter="auto")
    # decim=5 fits on every 5th sample: same components, ~5x faster.
    ica.fit(raw, picks="eeg", decim=5)
    eog_idx, _ = ica.find_bads_eog(raw, ch_name=EOG_PROXY_CHS)
    # Muscle matters most for this project: EMG is broadband and lands right
    # in the 30-45 Hz gamma band the biomarker is built on.
    muscle_idx, _ = ica.find_bads_muscle(raw)
    ica.exclude = sorted(set(eog_idx) | set(muscle_idx))
    ica.apply(raw)

    return raw, {
        "line_freq_hz": line_freq,
        "n_bad_channels": len(bads),
        "bad_channels": ";".join(bads),
        "n_ica_eog_removed": len(eog_idx),
        "n_ica_muscle_removed": len(muscle_idx),
    }


# --- Event decoding ------------------------------------------------------------

def parse_probes(events_df: pd.DataFrame) -> pd.DataFrame:
    """One row per probe: its onset, decoded ratings, and where its clean
    pre-probe window starts.

    Answers are the 'response' rows directly after the probe, in question
    order (config.py explains why position is reliable). A response arriving
    more than MAX_RESPONSE_GAP_S after the previous event is a stray button
    press, not an answer. The window starts PROBE_LOOKBACK_S before the probe
    but never earlier than POST_PROBE_BUFFER_S after the previous event of
    any kind, so it never contains an earlier probe's questions, answers, or
    a stray button press. Missing answers stay NaN, since 0 is a valid rating.
    """
    required = {"onset", "trial_type", "value"}
    if not required.issubset(events_df.columns):
        raise ValueError(f"events.tsv is missing expected columns "
                         f"{required - set(events_df.columns)}")
    ev = events_df.sort_values("onset", kind="stable").reset_index(drop=True)
    kind = ev["trial_type"].astype(str).str.lower().to_numpy()
    value = pd.to_numeric(ev["value"], errors="coerce").to_numpy()
    onset = ev["onset"].astype(float).to_numpy()

    probe_rows = np.flatnonzero((kind == "stimulus")
                                & (value == PROBE_STIMULUS_VALUE))
    if probe_rows.size == 0:
        raise ValueError(f"No probe events (trial_type=='stimulus', "
                         f"value=={PROBE_STIMULUS_VALUE}) in events.tsv")

    rows = []
    for probe_index, r in enumerate(probe_rows):
        ratings = [np.nan] * N_PROBE_QUESTIONS
        last_t, j, n = onset[r], r + 1, 0
        while (j < len(ev) and n < N_PROBE_QUESTIONS
               and kind[j] == "response"
               and onset[j] - last_t <= MAX_RESPONSE_GAP_S):
            ratings[n] = RESPONSE_VALUE_TO_RATING.get(value[j], np.nan)
            last_t, j, n = onset[j], j + 1, n + 1

        prev_event_end = onset[r - 1] + POST_PROBE_BUFFER_S if r > 0 else 0.0
        rows.append({
            "probe_index": probe_index,
            "probe_onset_s": onset[r],
            "window_start_s": max(onset[r] - PROBE_LOOKBACK_S,
                                  prev_event_end, 0.0),
            **dict(zip(RATING_COLUMNS, ratings)),
            "n_responses_found": n,
        })
    return pd.DataFrame(rows)


# --- 6. Epoching ----------------------------------------------------------------

def epoch_probes(raw: mne.io.BaseRaw, probes: pd.DataFrame
                 ) -> tuple[mne.Epochs | None, pd.Series]:
    """Cut overlapping EPOCH_LENGTH_S epochs from every probe's window in one
    pass over the recording, drop epochs that are still too noisy (or flat),
    and tag each epoch with its probe and its distance to the probe onset so
    later steps can pick a narrower window without re-running ICA.
    Returns (epochs or None, epochs planned per probe)."""
    step = EPOCH_LENGTH_S - EPOCH_OVERLAP_S
    starts, meta = [], []
    for p in probes.itertuples(index=False):
        # +1e-9 keeps an epoch that ends exactly at the probe onset.
        for t in np.arange(p.window_start_s,
                           p.probe_onset_s - EPOCH_LENGTH_S + 1e-9, step):
            starts.append(t)
            meta.append({"probe_index": p.probe_index,
                         "seconds_before_probe": p.probe_onset_s - t})
    metadata = pd.DataFrame(meta, columns=["probe_index",
                                           "seconds_before_probe"])
    planned = metadata["probe_index"].value_counts()
    if not starts:
        return None, planned

    sfreq = raw.info["sfreq"]
    samples = raw.first_samp + np.round(np.array(starts) * sfreq).astype(int)
    events = np.column_stack([samples, np.zeros_like(samples),
                              np.ones_like(samples)])
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=EPOCH_LENGTH_S - 1.0 / sfreq,
        baseline=None, picks="eeg", metadata=metadata, preload=True,
        reject={"eeg": REJECT_PEAK_TO_PEAK_V},
        flat={"eeg": FLAT_PEAK_TO_PEAK_V})
    return epochs, planned


# --- Orchestration ----------------------------------------------------------------

def process_recording(raw_path: Path, deriv_dir: Path
                      ) -> tuple[list[dict], dict]:
    """Preprocess one recording end to end. Returns (probe_labels rows,
    QC row)."""
    entities = get_entities_from_fname(raw_path)
    sub, ses = f"sub-{entities['subject']}", f"ses-{entities['session']}"
    events_path = raw_path.with_name(
        raw_path.name.replace("_eeg.bdf", "_events.tsv"))
    if not events_path.exists():
        raise FileNotFoundError(f"no events.tsv next to {raw_path.name}")

    print(f"[..] {sub} / {ses}")
    probes = parse_probes(pd.read_csv(events_path, sep="\t"))
    raw = rename_and_type_channels(mne.io.read_raw_bdf(raw_path, preload=True))
    raw, qc = clean_continuous(raw, read_line_freq(raw_path))
    epochs, planned = epoch_probes(raw, probes)

    rows = []
    for p in probes.itertuples(index=False):
        n_planned = int(planned.get(p.probe_index, 0))
        probe_epochs = (epochs[f"probe_index == {p.probe_index}"]
                        if epochs is not None else [])
        if len(probe_epochs) == 0:
            continue
        out_path = deriv_dir / f"{sub}_{ses}_probe-{p.probe_index:02d}-epo.fif"
        probe_epochs.save(out_path, overwrite=True)
        rows.append({"subject": sub, "session": ses,
                     **p._asdict(),
                     "epochs_file": str(out_path),
                     "n_epochs": len(probe_epochs),
                     "n_epochs_rejected": n_planned - len(probe_epochs)})

    n_planned_total = int(planned.sum())
    n_kept = sum(r["n_epochs"] for r in rows)
    qc.update({
        "subject": sub, "session": ses,
        "n_probes": len(probes),
        "n_probes_kept": len(rows),
        "n_epochs_kept": n_kept,
        "epoch_reject_rate": (1 - n_kept / n_planned_total
                              if n_planned_total else np.nan),
    })
    print(f"[ok] {sub} / {ses}: {len(rows)}/{len(probes)} probes, "
          f"{n_kept} epochs kept ({qc['epoch_reject_rate']:.0%} rejected), "
          f"{qc['n_bad_channels']} bad channels, ICA removed "
          f"{qc['n_ica_eog_removed']} eye + {qc['n_ica_muscle_removed']} "
          f"muscle components")
    return rows, qc


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
    qc_csv = paths.derivatives / "preprocessing_qc.csv"
    all_rows: list[dict] = []
    qc_rows: list[dict] = []
    failed: list[str] = []

    for sub in subjects:
        recordings = sorted((paths.bids_root / sub).rglob("*_eeg.bdf"))
        if not recordings:
            print(f"[skip] no .bdf recording found for {sub}")
        for raw_path in recordings:
            try:
                rows, qc = process_recording(raw_path, paths.derivatives)
                all_rows.extend(rows)
                qc_rows.append(qc)
            except Exception as e:
                failed.append(raw_path.name)
                print(f"[ERROR] {raw_path.name} failed and was skipped: "
                      f"{type(e).__name__}: {e}")
            finally:
                # Save after every recording, success or failure, so a crash
                # on recording N never loses the work done on 1..N-1.
                if all_rows:
                    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
                if qc_rows:
                    pd.DataFrame(qc_rows).to_csv(qc_csv, index=False)

    if not all_rows:
        raise SystemExit("[ERROR] no probes were preprocessed -- see errors "
                         "above.")
    manifest = pd.DataFrame(all_rows)
    print(f"\n[ok] wrote {len(manifest)} probe rows to {out_csv}")
    print(f"[ok] wrote QC for {len(qc_rows)} recording(s) to {qc_csv}")
    if failed:
        print(f"[warn] {len(failed)} recording(s) failed: {failed} -- "
              f"re-run just those with --subjects once fixed")
    n_nan = manifest["concentration_rating"].isna().sum()
    if n_nan:
        print(f"[warn] {n_nan}/{len(manifest)} probes have no concentration "
              f"answer (probe skipped by the participant) -- dropped "
              f"automatically in later steps.")
    if args.subjects or args.limit:
        print("[note] this was a partial run (--subjects/--limit used); "
              "re-running without them overwrites probe_labels.csv with "
              "the full set.")


if __name__ == "__main__":
    main()
