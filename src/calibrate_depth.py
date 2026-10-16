"""Phase 8, round 3: does one rated calibration session help decode a
person's next session? As fixed in docs/search_protocol_depth_calibration.md.

Usage:
    python src/calibrate_depth.py --derivatives <phase 7 run dir> --output_dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANDOM_STATE  # noqa: E402
from explore_depth import HOLDOUT, TARGET, TIE, load, ridge, zrec  # noqa: E402
from model_comparison import (mean_interval, score, shuffled_target,  # noqa: E402
                              subject_scores)

WEIGHTS = (0.0, 1.0, 5.0, 20.0)


def person_predictions(pool: pd.DataFrame, person: pd.DataFrame, cols: list[str],
                       alpha: float, y: pd.Series) -> dict[float, np.ndarray]:
    """For each weight w: predict each of the person's sessions from the pool
    plus the person's OTHER session (weighted w). y: the person's ratings."""
    out = {w: np.full(len(person), np.nan) for w in WEIGHTS}
    for s in person["session"].unique():
        test = (person["session"] == s).to_numpy()
        own = person[~test]
        X = pd.concat([pool[cols], own[cols]])
        target = np.concatenate([pool[TARGET].to_numpy(), y[~test].to_numpy()])
        for w in WEIGHTS:
            weight = np.concatenate([np.ones(len(pool)), np.full(len(own), w)])
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha))
            model.fit(X, target, ridge__sample_weight=weight)
            out[w][test] = model.predict(person.loc[test, cols])
    return out


def calibrated(pool: pd.DataFrame, people: pd.DataFrame, cols: list[str],
               alphas: dict[str, float], y: pd.Series) -> dict[float, np.ndarray]:
    """Predictions for everyone in `people`; the pool excludes the person."""
    out = {w: np.full(len(people), np.nan) for w in WEIGHTS}
    for p in people["subject"].unique():
        m = (people["subject"] == p).to_numpy()
        preds = person_predictions(pool[pool["subject"] != p], people[m], cols,
                                   alphas[p], y[m])
        for w in WEIGHTS:
            out[w][m] = preds[w]
    return out


def auc(df: pd.DataFrame, y, pred) -> float:
    return score(df, y, zrec(pred, df["recording"]), "state")["score"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derivatives", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n_shifts", type=int, default=1000)
    parser.add_argument("--n_jobs", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, sets = load(args.derivatives)
    cols = sets["riemann"]
    held = df["subject"].isin(HOLDOUT).to_numpy()
    dev, test = df[~held].reset_index(drop=True), df[held].reset_index(drop=True)
    if not (df.groupby("subject")["session"].nunique() == 2).all():
        raise SystemExit("everyone needs two sessions for this design")

    # Alpha per left-out person, chosen without that person (person-grouped CV).
    def alpha_without(pool, p):
        rest = pool[pool["subject"] != p]
        return p, ridge(rest).fit(rest[cols], rest[TARGET])[-1].alpha_
    alphas = dict(Parallel(n_jobs=args.n_jobs)(
        delayed(alpha_without)(dev, p) for p in dev["subject"].unique()))
    alphas_test = dict.fromkeys(HOLDOUT, ridge(dev).fit(dev[cols], dev[TARGET])[-1].alpha_)

    # Search set.
    preds = calibrated(dev, dev, cols, alphas, dev[TARGET])
    table = pd.DataFrame({"w": WEIGHTS, "auc": [auc(dev, dev[TARGET], preds[w]) for w in WEIGHTS]})
    table.to_csv(args.output_dir / "calibration_search.csv", index=False)
    print(table.to_string(index=False), flush=True)
    best = table.loc[table["auc"] >= table["auc"].max() - TIE, "w"].iloc[0]
    print(f"[ok] chosen w = {best:g}", flush=True)

    # Held-out test, once. The null re-trains with shifted held-out ratings
    # (they are also the calibration labels).
    final = calibrated(dev, test, cols, alphas_test, test[TARGET])
    shifts = [shuffled_target(test, TARGET, "state", np.random.default_rng(RANDOM_STATE + i))
              for i in range(args.n_shifts)]

    def null(y):
        p = calibrated(dev, test, cols, alphas_test, y)
        return [auc(test, y, p[w]) for w in (0.0, best)]
    nulls = np.array(Parallel(n_jobs=args.n_jobs)(delayed(null)(y) for y in shifts))
    rows, per = [], {}
    for k, w in enumerate((0.0, best)):
        a = auc(test, test[TARGET], final[w])
        per[w] = subject_scores(test, test[TARGET], zrec(final[w], test["recording"]),
                                "state").set_index("subject")["score"]
        lo, hi = mean_interval(per[w])
        rows.append({"w": w, "heldout_auc": a, "ci_low": lo, "ci_high": hi,
                     "chance": nulls[:, k].mean(),
                     "p_value": (1 + (nulls[:, k] >= a).sum()) / (1 + len(nulls))})
    held_table = pd.DataFrame(rows).drop_duplicates("w")
    d = (per[best] - per[0.0]).dropna()
    lo, hi = mean_interval(d)
    held_table.to_csv(args.output_dir / "calibration_heldout.csv", index=False)
    print(held_table.to_string(index=False))
    print(f"w={best:g} minus w=0, per person: {d.mean():+.3f} [{lo:.3f}, {hi:.3f}]")
    helps = best > 0 and lo > 0
    print(f"\nRESULT: calibration {'HELPS' if helps else 'is NOT shown to help'} "
          f"on the held-out people")


if __name__ == "__main__":
    main()
