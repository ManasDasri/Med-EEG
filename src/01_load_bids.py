"""Step 1: point at the downloaded BIDS dataset and sanity-check its layout.

This does NOT touch raw signal data yet -- it just confirms subjects,
sessions, channel names, and event/probe structure match what the rest of
the pipeline assumes, and prints anything that needs a manual check.

Usage:
    python src/01_load_bids.py --bids_root data/ds001787
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROBE_EVENT_KEYWORDS, Paths  # noqa: E402


def find_subjects(bids_root: Path) -> list[str]:
    return sorted(p.name for p in bids_root.glob("sub-*") if p.is_dir())


def inspect_participants_tsv(bids_root: Path) -> pd.DataFrame | None:
    f = bids_root / "participants.tsv"
    if not f.exists():
        print(f"[warn] no participants.tsv at {f} -- group labels (expert/"
              f"novice) will need to come from elsewhere.")
        return None
    df = pd.read_csv(f, sep="\t")
    print(f"[ok] participants.tsv columns: {list(df.columns)}")
    print(df.head())
    return df


def inspect_events(bids_root: Path, subjects: list[str]) -> None:
    print("\n--- scanning *_events.tsv for probe/rating trial_type values ---")
    seen_types: set[str] = set()
    n_events_files = 0
    for sub in subjects:
        for events_file in (bids_root / sub).rglob("*_events.tsv"):
            n_events_files += 1
            df = pd.read_csv(events_file, sep="\t")
            if "trial_type" in df.columns:
                seen_types.update(df["trial_type"].dropna().unique().tolist())
    print(f"[ok] found {n_events_files} events.tsv files across "
          f"{len(subjects)} subjects")
    print(f"[ok] unique trial_type values seen: {sorted(seen_types)}")

    probe_like = [t for t in seen_types
                  if any(k in str(t).lower() for k in PROBE_EVENT_KEYWORDS)]
    if probe_like:
        print(f"[ok] probe-like event labels: {probe_like}")
    else:
        print("[action needed] none of the trial_type values obviously "
              "match probe/concentration/rating keywords. Open one "
              "*_events.tsv by hand and update PROBE_EVENT_KEYWORDS in "
              "utils.py, or hardcode the exact label(s) used for the "
              "3 self-report probes in 02_preprocess.py.")


def inspect_channels(bids_root: Path, subjects: list[str]) -> None:
    if not subjects:
        return
    sample = subjects[0]
    ch_files = list((bids_root / sample).rglob("*_channels.tsv"))
    if not ch_files:
        print(f"[warn] no *_channels.tsv found for {sample}")
        return
    df = pd.read_csv(ch_files[0], sep="\t")
    print(f"\n--- channel names for {sample} ({ch_files[0].name}) ---")
    print(df["name"].tolist() if "name" in df.columns else df.iloc[:, 0].tolist())
    print("[action needed] confirm FRONTAL_MIDLINE_THETA_CHS / "
          "POSTERIOR_GAMMA_CHS in utils.py exist under these exact names.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path,
                         help="Path to the downloaded ds001787 / BIDS_EEG_"
                              "meditation_experiment folder")
    args = parser.parse_args()

    if not args.bids_root.exists():
        raise SystemExit(
            f"{args.bids_root} does not exist. Download the dataset first "
            f"(see README.md) -- it's several GB, this script won't fetch "
            f"it for you.")

    paths = Paths.from_root(args.bids_root)
    subjects = find_subjects(paths.bids_root)
    print(f"[ok] found {len(subjects)} subjects: {subjects}")
    if len(subjects) != 24:
        print(f"[warn] expected 24 subjects for this dataset, found "
              f"{len(subjects)} -- double check bids_root points at the "
              f"dataset top level.")

    inspect_participants_tsv(paths.bids_root)
    inspect_events(paths.bids_root, subjects)
    inspect_channels(paths.bids_root, subjects)

    print("\nDone. Resolve any [action needed] items above before running "
          "02_preprocess.py -- this dataset has gone through a couple of "
          "BIDS re-exports over the years and exact event/channel naming "
          "can differ by download source (Zenodo vs OpenNeuro vs version).")


if __name__ == "__main__":
    main()
