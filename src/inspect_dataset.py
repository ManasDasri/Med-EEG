"""Step 1: point at the downloaded BIDS dataset and sanity-check its layout.

This does NOT touch raw signal data yet -- it just confirms subjects,
sessions, channel names, and event/probe structure match what the rest of
the pipeline assumes, and prints anything that needs a manual check.

Usage:
    python src/inspect_dataset.py --bids_root data/ds001787
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ASYMMETRY_PAIR, BIOSEMI64_TO_1020, PAC_CLUSTERS, REGIONS)  # noqa: E402


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


def event_inventory(events: pd.DataFrame, dataset: str) -> dict:
    if dataset == "ds001787":
        required = {"trial_type", "value"}
        if not required.issubset(events.columns):
            raise ValueError(f"events missing columns: {required - set(events.columns)}")
        values = pd.to_numeric(events["value"], errors="coerce")
        probe_count = int((events["trial_type"].astype(str).str.lower().eq("stimulus") & values.eq(128)).sum())
        return {"probes": probe_count, "event_types": sorted(events.trial_type.dropna().astype(str).unique())}
    required = {"trial_type", "depth"}
    if not required.issubset(events.columns):
        raise ValueError(f"events missing columns: {required - set(events.columns)}")
    types = events["trial_type"].astype(str)
    return {"probes": int(types.eq("depth_rating").sum()),
            "report_markers": int(types.isin(["prompt_window_begin", "report_window_begin"]).sum()),
            "event_types": sorted(events.trial_type.dropna().astype(str).unique())}


def inspect_events(bids_root: Path, subjects: list[str], dataset: str) -> None:
    print("\n--- checking the dataset-specific probe event codes ---")
    total, files, seen = 0, 0, set()
    for sub in subjects:
        for path in (bids_root / sub).rglob("*_events.tsv"):
            inventory = event_inventory(pd.read_csv(path, sep="\t"), dataset)
            files += 1
            total += inventory["probes"]
            seen.update(inventory["event_types"])
    print(f"[ok] {files} event files; {total} probe/rating events")
    print(f"[ok] event types: {sorted(seen)}")
    if total == 0:
        raise ValueError("No expected dataset-specific probe/rating events found")


def inspect_channels(bids_root: Path, subjects: list[str], dataset: str) -> None:
    if not subjects: return
    files = list((bids_root / subjects[0]).rglob("*_channels.tsv"))
    if not files:
        print(f"[warn] no *_channels.tsv found for {subjects[0]}")
        return
    channels = pd.read_csv(files[0], sep="\t")
    names = set(channels.name.astype(str))
    if dataset == "ds001787":
        names = {BIOSEMI64_TO_1020.get(name, name) for name in names}
        expected = set().union(*REGIONS.values(), *PAC_CLUSTERS.values(), set(ASYMMETRY_PAIR))
    else:
        names = set(channels.loc[channels.type == "EEG", "name"].astype(str))
        expected = set().union(*PAC_CLUSTERS.values())
    missing = sorted(expected - names)
    print(f"[ok] checked channel names for {files[0].name}")
    if missing:
        print(f"[warn] expected analysis electrodes not present: {missing}")
    else:
        print("[ok] analysis electrodes are present")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path,
                         help="Path to the downloaded ds001787 / BIDS_EEG_"
                              "meditation_experiment folder")
    parser.add_argument("--derivatives", type=Path, default=None,
                        help="Explicit intermediate directory (preserve prior runs)")
    parser.add_argument("--dataset", choices=["ds001787", "ds008836"], default="ds001787")
    args = parser.parse_args()

    if not args.bids_root.exists():
        raise SystemExit(
            f"{args.bids_root} does not exist. Download the dataset first "
            f"(see README.md) -- it's several GB, this script won't fetch "
            f"it for you.")

    bids_root = args.bids_root
    subjects = find_subjects(bids_root)
    print(f"[ok] found {len(subjects)} subjects: {subjects}")
    if len(subjects) != 24:
        print(f"[warn] expected 24 subjects for this dataset, found "
              f"{len(subjects)} -- double check bids_root points at the "
              f"dataset top level.")

    inspect_participants_tsv(bids_root)
    inspect_events(bids_root, subjects, args.dataset)
    inspect_channels(bids_root, subjects, args.dataset)

    print("\nDone. Resolve any warnings before processing this dataset.")


if __name__ == "__main__":
    main()
