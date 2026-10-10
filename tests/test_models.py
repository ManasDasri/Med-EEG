"""Checks for model_comparison.py on synthetic feature tables with a known
answer: real signal must score well above chance, no signal must not.
Run: python tests/test_models.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_comparison import (evaluate, prepare, shuffled_target,  # noqa: E402
                              spectral_columns)

rng = np.random.default_rng(0)
N_SUBJECTS, N_PROBES = 24, 30
FEATURES = ["frontal_alpha_log", "occipital_alpha_rel", "frontal_slope",
            "frontal_alpha_asymmetry", "minimal_ratio", "minimal_pac"]


def synthetic(trait_signal: float, state_signal: float) -> pd.DataFrame:
    rows = []
    for s in range(N_SUBJECTS):
        expert = s % 2
        scale = rng.uniform(1, 6)  # people differ ~6x in raw power
        for p in range(N_PROBES):
            rating = rng.integers(1, 4)
            feats = rng.standard_normal(len(FEATURES))
            feats[0] += state_signal * rating          # state: alpha tracks rating
            feats[1] += trait_signal * expert          # trait: experts differ
            rows.append({"subject": f"sub-{s:03d}", "session": "ses-01",
                         "probe_index": p, "group_binary": expert,
                         "concentration_rating": rating,
                         **dict(zip(FEATURES, feats * scale))})
    return pd.DataFrame(rows)


def run(task: str, df: pd.DataFrame, n_perm: int = 9) -> dict:
    df = prepare(df, task)
    return evaluate(df, task, "linear_spectral", spectral_columns(df),
                    n_perm)[0]


# Real signal -> clearly above chance, and the permutation test agrees.
on = run("expertise", synthetic(trait_signal=1.5, state_signal=0))
assert on["score"] >= 0.9 and on["auc"] >= 0.9 and on["p_value"] == 0.1, on
on = run("concentration", synthetic(trait_signal=0, state_signal=1.0))
assert on["score"] >= 0.75 and on["p_value"] == 0.1, on

# No signal -> not significant. (With 24 people, no-signal accuracy itself
# swings widely -- sd ~13 points, >= 70% in ~8% of runs -- so check the
# p-value, not a fixed range. Before class balancing this scored 0%.)
off = run("expertise", synthetic(0, 0))
assert off["p_value"] > 0.1 and off["chance"] > 0.3, off
off = run("concentration", synthetic(0, 0))
assert abs(off["score"] - 0.5) < 0.08, off

# Shuffles keep the structure: one label per person; each person's ratings.
df = synthetic(0, 0)
y = shuffled_target(df, "group_binary", "trait", rng)
assert (y.groupby(df["subject"]).nunique() == 1).all()
assert y.groupby(df["subject"]).first().sum() == N_SUBJECTS // 2
y = shuffled_target(df, "concentration_rating", "state", rng)
for s, g in df.groupby("subject"):
    assert sorted(y[g.index]) == sorted(g["concentration_rating"])

print("ok: model_comparison")
