"""Checks for depth_decoding.py: Riemannian re-centring and the
segment-level model. Run: python tests/test_depth_decoding.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from depth_decoding import (covariance_stack, epoch_predictions,  # noqa: E402
                            tangent_features)
from model_comparison import score  # noqa: E402

rng = np.random.default_rng(11)


def random_spd(n_ch: int) -> np.ndarray:
    a = rng.standard_normal((n_ch, n_ch))
    return a @ a.T + n_ch * np.eye(n_ch)


# --- Tangent space: per-recording re-centring -----------------------------------
n_ch, n_rec, per = 4, 3, 12
covs = np.stack([random_spd(n_ch) for _ in range(n_rec * per)])
recordings = pd.Series(np.repeat([f"r{i}" for i in range(n_rec)], per))
t = tangent_features(covs, recordings)
assert t.shape == (n_rec * per, n_ch * (n_ch + 1) // 2)
for r in recordings.unique():  # each recording is centred on its own mean
    assert np.abs(t[(recordings == r).to_numpy()].mean(axis=0)).max() < 1e-8

scaled = covs.copy()
scaled[(recordings == "r1").to_numpy()] *= 37.0  # e.g. a person with 6x the amplitude
assert np.allclose(tangent_features(scaled, recordings), t, atol=1e-8)

# Columns -> matrices round trip.
names = ["Fz", "Cz", "Pz", "Oz"]
rows, cols = np.triu_indices(n_ch)
frame = pd.DataFrame({f"cov__{names[i]}__{names[j]}": covs[:, i, j] for i, j in zip(rows, cols)})
assert np.allclose(covariance_stack(frame), covs)

# --- Segment model: finds planted signal, not absent signal ----------------------
def synthetic(signal: float):
    probes, epochs = [], []
    for s in range(12):
        for k in range(20):
            depth = float(rng.integers(1, 6))
            probes.append({"subject": f"s{s}", "session": "ses-1", "recording": f"s{s}",
                           "probe_index": k, "depth_rating": depth})
            for _ in range(5):
                epochs.append({"subject": f"s{s}", "session": "ses-1", "recording": f"s{s}",
                               "probe_index": k, "f1": signal * depth + rng.standard_normal(),
                               "f2": rng.standard_normal()})
    return pd.DataFrame(probes), pd.DataFrame(epochs)


probes, epochs = synthetic(0.6)
pred = epoch_predictions(probes, epochs, ["f1", "f2"], probes["depth_rating"])
assert not np.isnan(pred).any()
assert score(probes, probes["depth_rating"], pred, "state")["score"] > 0.75

probes, epochs = synthetic(0.0)
pred = epoch_predictions(probes, epochs, ["f1", "f2"], probes["depth_rating"])
assert abs(score(probes, probes["depth_rating"], pred, "state")["score"] - 0.5) < 0.1

print("ok: depth_decoding")
