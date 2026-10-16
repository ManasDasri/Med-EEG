"""Checks for explore_depth.py. Run: python tests/test_explore_depth.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from explore_depth import fit_predict, loso, previous_average, zrec  # noqa: E402
from model_comparison import score  # noqa: E402

rng = np.random.default_rng(3)

# previous_average: first report keeps itself, later ones average with the one before.
df = pd.DataFrame({"recording": ["a", "a", "a", "b"], "probe_onset_s": [30, 10, 20, 5],
                   "x": [3.0, 1.0, 2.0, 7.0]})
assert previous_average(df, ["x"])["ctx_x"].tolist() == [2.5, 1.0, 1.5, 7.0]

# zrec: zero mean within each recording.
z = zrec([1, 2, 3, 10, 10], pd.Series(["a", "a", "a", "b", "b"]))
assert abs(z[:3].mean()) < 1e-12 and (z[3:] == 0).all()

# Each learner/target mode finds a planted signal, and not an absent one.
def synthetic(signal):
    rows = []
    for s in range(10):
        offset = rng.normal(0, 2)  # people differ in average rating
        for k in range(25):
            d = float(np.clip(round(3 + offset / 2 + rng.normal()), 1, 5))
            rows.append({"subject": f"s{s}", "recording": f"s{s}", "probe_onset_s": k,
                         "depth_rating": d, "f1": signal * d + rng.normal(), "f2": rng.normal()})
    return pd.DataFrame(rows)

sets = {"riemann": ["f1", "f2"], "spectral_plus": ["f1", "f2"]}
for target in ("raw", "z", "binary"):
    for learner in ("ridge", "ensemble"):
        cand = {"features": "riemann", "learner": learner, "target": target}
        on = synthetic(1.0)
        assert score(on, on["depth_rating"], loso(on, cand, sets), "state")["score"] > 0.75, (target, learner)
off = synthetic(0.0)
cand = {"features": "riemann", "learner": "ridge", "target": "z"}
assert abs(score(off, off["depth_rating"], loso(off, cand, sets), "state")["score"] - 0.5) < 0.12

print("ok: explore_depth")
