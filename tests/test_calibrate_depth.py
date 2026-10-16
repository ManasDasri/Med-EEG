"""Checks for calibrate_depth.py. Run: python tests/test_calibrate_depth.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from calibrate_depth import auc, calibrated  # noqa: E402

rng = np.random.default_rng(5)

# Each person carries depth on their OWN feature: other people can't teach the
# model that, but the person's other session can.
rows = []
for p in range(8):
    for ses in ("ses-1", "ses-2"):
        for k in range(30):
            d = float(rng.integers(1, 6))
            x = rng.standard_normal(8)
            x[p] += 1.5 * (d - 3)
            rows.append({"subject": f"s{p}", "session": ses, "recording": f"s{p}_{ses}",
                         "depth_rating": d, **{f"f{i}": x[i] for i in range(8)}})
df = pd.DataFrame(rows)
cols = [f"f{i}" for i in range(8)]
pred = calibrated(df, df, cols, dict.fromkeys(df["subject"].unique(), 10.0), df["depth_rating"])
a0, a20 = auc(df, df["depth_rating"], pred[0.0]), auc(df, df["depth_rating"], pred[20.0])
assert a20 > 0.8 and a20 > a0 + 0.15, (a0, a20)

# The test session's own labels are never used: shuffling them leaves its predictions unchanged.
y = df["depth_rating"].copy()
test = ((df["subject"] == "s0") & (df["session"] == "ses-2")).to_numpy()
y[test] = rng.permutation(y[test].to_numpy())
again = calibrated(df, df, cols, dict.fromkeys(df["subject"].unique(), 10.0), y)
assert np.allclose(again[20.0][test], pred[20.0][test])

print("ok: calibrate_depth")
