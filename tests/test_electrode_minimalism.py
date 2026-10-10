"""Checks for electrode_minimalism.py on synthetic data.
Run: python tests/test_electrode_minimalism.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from electrode_minimalism import curve, within_person_auc  # noqa: E402
from model_comparison import score  # noqa: E402

rng = np.random.default_rng(1)
n_subj, n_probes, n_ch = 20, 30, 16
subjects = np.repeat([f"s{i}" for i in range(n_subj)], n_probes)
y = rng.integers(1, 4, len(subjects)).astype(float)

# The fast rank-based AUC equals the reference metric.
x = rng.standard_normal(len(y))
ref = score(pd.DataFrame({"subject": subjects}), pd.Series(y), x, "state")
assert abs(within_person_auc(x[:, None], y, subjects)[0] - ref["score"]) < 1e-12

# Signal in electrodes 0 and 1 only (electrode 1 inverted): small sets find it.
X = rng.standard_normal((len(y), n_ch))
X[:, 0] += 0.8 * y
X[:, 1] -= 0.8 * y
on = curve(X, y, subjects, {"pair": [0, 1]})
assert on["top_2"] > 0.75 and on["top_2"] > on["top_16"], on
assert on["pair"] < 0.6  # averaging +signal and -signal cancels: sign matters

# No signal anywhere -> every set stays near chance (selection doesn't leak).
off = curve(rng.standard_normal((len(y), n_ch)), y, subjects, {"pair": [0, 1]})
assert all(abs(v - 0.5) < 0.06 for v in off.values()), off

print("ok: electrode_minimalism")
