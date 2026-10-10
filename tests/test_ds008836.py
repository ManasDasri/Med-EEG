"""Checks for the ds008836 report parser. Run: python tests/test_ds008836.py"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ds008836 import parse_reports  # noqa: E402

B = "silent_probe"
events = pd.DataFrame([
    # onset, trial_type, block, depth
    (100.0, "silent_probe_begin", B, None),
    (130.0, "prompt_window_begin", B, None),   # report 0: block started 30 s ago
    (133.0, "depth_rating", B, 4),
    (138.0, "confidence_rating", B, None),
    (300.0, "prompt_window_begin", B, None),   # report 1: 120 s lookback
    (302.0, "depth_rating", B, 7),             # > 5 -> 5
    (306.0, "confidence_rating", B, None),
    (320.0, "prompt_window_begin", B, None),   # report 2: previous rating 14 s ago
    (322.0, "depth_rating", B, 0),             # 0 -> dropped
    (330.0, "depth_rating", B, 2),             # no marker of its own -> dropped
    (500.0, "prompt_window_begin", "break", None),  # outside report blocks
    (501.0, "depth_rating", "break", 3),
], columns=["onset", "trial_type", "block", "depth"])

r = parse_reports(events)
assert len(r) == 2, r
assert r["depth_rating"].tolist() == [4.0, 5.0]
assert r["probe_onset_s"].tolist() == [130.0, 300.0]       # ends where the prompt began
assert r.loc[0, "window_start_s"] == 100.0                  # clipped at block start
assert r.loc[1, "window_start_s"] == 300.0 - 120.0          # full lookback
assert set(r["block_type"]) == {"silent"} and set(r["variant"]) == {"probe"}

# Previous rating clips the window: a report 30 s after the last rating.
e2 = pd.concat([events.iloc[:7], pd.DataFrame([
    (336.0, "prompt_window_begin", B, None), (338.0, "depth_rating", B, 3)],
    columns=events.columns)])
r2 = parse_reports(e2)
assert r2.iloc[-1]["window_start_s"] == 306.0 + 10.0        # 10 s after confidence

print("ok: ds008836 parser")
