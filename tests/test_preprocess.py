"""Checks for the probe/answer decoding in preprocess.py, built from real
patterns in the dataset's events.tsv files. Run: python tests/test_preprocess.py"""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import POST_PROBE_BUFFER_S, PROBE_LOOKBACK_S  # noqa: E402
from preprocess import parse_probes  # noqa: E402

events = pd.DataFrame([
    # onset, trial_type, value
    (12.0, "STATUS", 254),      # recording-start marker, not a probe
    (200.0, "stimulus", 128),   # probe 0: all 3 answers
    (203.0, "response", 4),
    (206.0, "response", 2),
    (209.0, "response", 8),
    (240.0, "stimulus", 128),   # probe 1: only 40 s after probe 0, 1 answer
    (243.0, "response", 2),
    (400.0, "stimulus", 128),   # probe 2: answer, then a stray press 79 s later
    (404.0, "response", 2),
    (483.0, "response", 8),
    (600.0, "stimulus", 128),   # probe 3: skipped entirely
], columns=["onset", "trial_type", "value"])

p = parse_probes(events).set_index("probe_index")

# Ratings are log2(code), in question order.
assert p.loc[0, ["concentration_rating", "mind_wandering_rating",
                 "tiredness_rating"]].tolist() == [2, 1, 3]
# Missing trailing answers stay NaN, never 0 (0 is a real rating).
assert p.loc[1, "concentration_rating"] == 1
assert math.isnan(p.loc[1, "mind_wandering_rating"])
# A press 79 s after the last answer is not an answer.
assert p.loc[2, "n_responses_found"] == 1
assert math.isnan(p.loc[2, "mind_wandering_rating"])
assert p.loc[3, "n_responses_found"] == 0

# Windows never reach back into the previous probe's Q&A.
assert p.loc[0, "window_start_s"] == 200.0 - PROBE_LOOKBACK_S
assert p.loc[1, "window_start_s"] == 209.0 + POST_PROBE_BUFFER_S
# ...nor into a stray button press.
assert p.loc[3, "window_start_s"] == 483.0 + POST_PROBE_BUFFER_S

# Unsorted input decodes the same way.
assert parse_probes(events.sample(frac=1, random_state=0)).equals(
    p.reset_index())

print("ok: parse_probes")
