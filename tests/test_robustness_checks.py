"""Checks for robustness_checks.py's cross-session electrode test (lead 2)
on synthetic data. Run: python tests/test_robustness_checks.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robustness_checks import lead_2  # noqa: E402

rng = np.random.default_rng(3)
CHANNELS = [f"C{i}" for i in range(12)]


def synthetic(signal: float) -> pd.DataFrame:
    rows = []
    for s in range(16):
        for ses in ("ses-01", "ses-02"):
            for p in range(25):
                rating = rng.integers(1, 4)
                ratios = rng.standard_normal(len(CHANNELS))
                ratios[4] += signal * rating  # only C4 carries signal
                rows.append({"subject": f"sub-{s:02d}", "session": ses,
                             "probe_index": p, "mind_wandering_rating": rating,
                             "minimal_ratio": 0.0, "minimal_pac": 0.0,
                             "frontal_alpha_asymmetry": rng.standard_normal(),
                             **{f"ratio_{c}": v for c, v in zip(CHANNELS, ratios)}})
    return pd.DataFrame(rows)


on = lead_2(synthetic(signal=1.0), n_permutations=19)
for row in on:  # picked on one session, works on the other
    assert row["note"].startswith("C4") and row["estimate"] > 0.7, row
    assert row["p_value"] == 0.05, row

off = lead_2(synthetic(signal=0.0), n_permutations=19)
for row in off:  # nothing to find -> the transferred electrode is at chance
    assert abs(row["estimate"] - 0.5) < 0.08, row

print("ok: robustness_checks")
