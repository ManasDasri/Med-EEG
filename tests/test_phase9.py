"""Checks for fidelity.py and horizon.py. Run: python tests/test_phase9.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fidelity import paired_sessions, partial_spearman, perm_spearman, session_fidelity  # noqa: E402
from horizon import BINS, _kernel_parts, _predict, bin_covariances, loso_ridge  # noqa: E402
from model_comparison import subject_scores  # noqa: E402

rng = np.random.default_rng(9)

# --- Kernel ridge equals sklearn's ridge (with intercept, standardised) --------
X, Xt = rng.standard_normal((80, 200)), rng.standard_normal((10, 200))
Y = rng.standard_normal((80, 3))
for alpha in (0.5, 50.0):
    ours = _predict(_kernel_parts(X, Xt), Y, alpha)
    for k in range(3):
        ref = make_pipeline(StandardScaler(), Ridge(alpha)).fit(X, Y[:, k]).predict(Xt)
        assert np.allclose(ours[:, k], ref), alpha

# --- loso_ridge finds planted within-person signal, and its columns are independent
rows, feats = [], []
for s in range(12):
    for k in range(30):
        d = float(rng.integers(1, 6))
        rows.append({"subject": f"s{s}", "recording": f"s{s}", "probe_onset_s": k, "depth_rating": d})
        f = rng.standard_normal(40); f[0] += 0.8 * d
        feats.append(f)
df, X = pd.DataFrame(rows), np.array(feats)
noise = rng.permutation(df["depth_rating"].to_numpy())
P = loso_ridge(X, np.column_stack([df["depth_rating"], noise]), df["subject"].to_numpy())
on = subject_scores(df, df["depth_rating"], P[:, 0], "state")["score"].mean()
off = subject_scores(df, noise, P[:, 1], "state")["score"].mean()
assert on > 0.8 and abs(off - 0.5) < 0.1, (on, off)

# --- bin_covariances: right bins, NaN when too few segments -------------------
data = rng.standard_normal((12, 4, 100))
sbp = np.array([2, 4, 6, 8, 12, 14, 16, 40, 70, 90, 110, 118], float)
c = bin_covariances(data, sbp)
assert c.shape == (len(BINS), 4, 4)
assert not np.isnan(c[0]).any() and not np.isnan(c[1]).any()  # 4 and 3 segments
assert np.isnan(c[2]).all() and not np.isnan(c[3]).any()       # 1 and 4 segments

# --- fidelity helpers --------------------------------------------------------
pred = pd.DataFrame({"subject": ["a"] * 8 + ["b"] * 8,
                     "session": (["s1"] * 4 + ["s2"] * 4) * 2,
                     "target": [1, 2, 3, 4, 1, 2, 3, 4, 2, 2, 2, 2, 1, 3, 1, 3],
                     "prediction": [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 0, 1, 0, 1]})
fid = session_fidelity(pred)
assert len(fid) == 3  # b/s1 has all-equal ratings: unscorable
f = fid.set_index(["subject", "session"])["F"]
assert f[("a", "s1")] == 1.0 and f[("a", "s2")] == 0.0 and f[("b", "s2")] == 1.0
assert list(paired_sessions(fid)["subject"]) == ["a"]

x = rng.standard_normal(40); y = x + rng.standard_normal(40) * 0.5
rho, p = perm_spearman(x, y, 2000, rng)
assert rho > 0.6 and p < 0.01
assert perm_spearman(x, rng.standard_normal(40), 2000, rng)[1] > 0.01
z = rng.standard_normal(40)  # a shared driver: partialling it out removes the correlation
assert abs(partial_spearman(z + 0.3 * rng.standard_normal(40), z + 0.3 * rng.standard_normal(40),
                            z[:, None])) < 0.35

print("ok: phase9")
