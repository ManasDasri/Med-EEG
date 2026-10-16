"""Phase 8: search for a better depth decoder, as fixed in
docs/search_protocol_depth.md.

Stage 1 scores every candidate on the 26 search-set people (LOSO,
within-person AUC); stage 2 combines the best feature and target options if
both helped. The winner and the reference are then trained on the search
set and tested once on the 13 held-out people. Finally the winner is run
with the phase 7 setup (all 39, LOSO, permutations) for the dashboard.

Usage:
    python src/explore_depth.py --derivatives <phase 7 run dir> --output_dir <dir>
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BOOSTING_PARAMS, RANDOM_STATE  # noqa: E402
from depth_decoding import (PLUS_SUFFIXES, covariance_stack,  # noqa: E402
                            recording_keys, tangent_features)
from model_comparison import (make_model, mean_interval, prepare,  # noqa: E402
                              score, shuffled_target, spectral_columns,
                              subject_scores, zscore_within_recording)

TARGET = "depth_rating"
HOLDOUT = ["sub-005", "sub-006", "sub-008", "sub-011", "sub-019", "sub-024",
           "sub-025", "sub-026", "sub-029", "sub-030", "sub-031", "sub-035",
           "sub-039"]
TIE = 0.005
warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn 1.10 API notices


# --- Features ----------------------------------------------------------------

def tangent_block(covs: np.ndarray, rec: pd.Series, prefix: str, index) -> pd.DataFrame:
    t = tangent_features(covs, rec)
    return pd.DataFrame(t, columns=[f"{prefix}{i}" for i in range(t.shape[1])], index=index)


def previous_average(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Each report's features averaged with the previous report's in the same
    recording (the first report keeps its own). EEG only, no labels."""
    order = df.sort_values(["recording", "probe_onset_s"])
    prev = order.groupby("recording")[cols].shift(1).fillna(order[cols])
    out = (order[cols] + prev) / 2
    return out.rename(columns=lambda c: f"ctx_{c}").loc[df.index]


def load(derivatives: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = prepare(pd.read_parquet(derivatives / "features.parquet"), "depth")
    df["recording"] = recording_keys(df)
    plus = [c for c in df.columns if c.endswith(PLUS_SUFFIXES)]
    df = zscore_within_recording(df, plus)
    spectral = spectral_columns(df)
    rec = df["recording"]
    blocks = [tangent_block(covariance_stack(df), rec, "tan_", df.index)]

    bands = np.load(derivatives / "band_covariances.npz")
    cov_names = list(dict.fromkeys(c.split("__")[1] for c in df.columns if c.startswith("cov__")))
    if list(bands["channels"]) != cov_names:
        raise SystemExit("band covariance channel order differs from cov__ columns")
    where = {(r, int(p)): i for i, (r, p) in
             enumerate(zip(bands["recording"], bands["probe_index"]))}
    rows = [where[(r, int(p))] for r, p in zip(rec, df["probe_index"])]
    for b, band in enumerate(bands["bands"]):
        blocks.append(tangent_block(bands["covs"][rows, b], rec, f"{band}_", df.index))
    df = pd.concat([df.drop(columns=[c for c in df.columns if c.startswith("cov__")])]
                   + blocks, axis=1)
    df["is_tones"] = (df["block_type"] == "sapo").astype(float)
    df["is_prompted"] = (df["variant"] == "probe").astype(float)

    cols = {k: [c for c in df.columns if c.startswith(k)]
            for k in ("tan_", "theta_", "alpha_", "beta_")}
    df = pd.concat([df, previous_average(df, cols["tan_"])], axis=1)
    sets = {
        "riemann": cols["tan_"],
        "riemann_theta": cols["theta_"],
        "riemann_alpha": cols["alpha_"],
        "riemann_beta": cols["beta_"],
        "riemann_filterbank": cols["tan_"] + cols["theta_"] + cols["alpha_"] + cols["beta_"],
        "riemann_plus_spectral": cols["tan_"] + spectral + plus,
        "riemann_context": [f"ctx_{c}" for c in cols["tan_"]],
        "spectral_plus": spectral + plus,
        "baseline_context": ["probe_onset_s", "is_tones", "is_prompted"],
    }
    return df, sets


# --- Models --------------------------------------------------------------------

def zrec(values, rec: pd.Series) -> np.ndarray:
    s = pd.Series(np.asarray(values, float), index=rec.index)
    g = s.groupby(rec)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, 1)).fillna(0).to_numpy()


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cand: dict, sets: dict
                ) -> np.ndarray:
    """cand: {"features": set name, "learner": ridge|boost|ensemble|baseline,
    "target": raw|z|binary}."""
    if cand["learner"] == "ensemble":
        a = fit_predict(train, test, {**cand, "features": "riemann", "learner": "ridge"}, sets)
        b = fit_predict(train, test, {**cand, "features": "spectral_plus", "learner": "boost"}, sets)
        return (zrec(a, test["recording"]) + zrec(b, test["recording"])) / 2
    X, Xt = train[sets[cand["features"]]], test[sets[cand["features"]]]
    y, rec = train[TARGET], train["recording"]
    if cand["target"] == "z":
        y = zrec(y, rec)
    if cand["target"] == "binary":
        y = (y > y.groupby(rec).transform("mean")).astype(int)
        if cand["learner"] == "boost":
            model = HistGradientBoostingClassifier(**BOOSTING_PARAMS, random_state=RANDOM_STATE)
        else:
            folds = list(GroupKFold(5).split(X, y, train["subject"]))
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegressionCV(Cs=np.logspace(-5, 1, 7), cv=folds,
                                                       scoring="roc_auc", max_iter=3000))
        return model.fit(X, y).predict_proba(Xt)[:, 1]
    name = {"ridge": "riemann_ridge", "boost": "boosting_spectral_plus",
            "baseline": "baseline_context"}[cand["learner"]]
    return make_model(name, "state").fit(X, y).predict(Xt)


def loso(df: pd.DataFrame, cand: dict, sets: dict) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    for s in df["subject"].unique():
        test = (df["subject"] == s).to_numpy()
        pred[test] = fit_predict(df[~test], df[test], cand, sets)
    return pred


CANDIDATES = {
    "riemann": {"features": "riemann", "learner": "ridge", "target": "raw"},
    "riemann_theta": {"features": "riemann_theta", "learner": "ridge", "target": "raw"},
    "riemann_alpha": {"features": "riemann_alpha", "learner": "ridge", "target": "raw"},
    "riemann_beta": {"features": "riemann_beta", "learner": "ridge", "target": "raw"},
    "riemann_filterbank": {"features": "riemann_filterbank", "learner": "ridge", "target": "raw"},
    "riemann_plus_spectral": {"features": "riemann_plus_spectral", "learner": "ridge", "target": "raw"},
    "ensemble": {"features": "riemann", "learner": "ensemble", "target": "raw"},
    "riemann_context": {"features": "riemann_context", "learner": "ridge", "target": "raw"},
    "riemann_ztarget": {"features": "riemann", "learner": "ridge", "target": "z"},
    "riemann_logistic": {"features": "riemann", "learner": "ridge", "target": "binary"},
}
FEATURE_OPTIONS = ["riemann_theta", "riemann_alpha", "riemann_beta", "riemann_filterbank",
                   "riemann_plus_spectral", "ensemble", "riemann_context"]
TARGET_OPTIONS = ["riemann_ztarget", "riemann_logistic"]
BASELINE = {"features": "baseline_context", "learner": "baseline", "target": "raw"}


def pick(table: pd.DataFrame, names: list[str]) -> str:
    """Best AUC; within TIE of the best, the earliest-listed (simpler) one."""
    sub = table.loc[names]
    return sub.index[sub["auc"] >= sub["auc"].max() - TIE][0]


# --- Main ------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derivatives", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n_permutations", type=int, default=200)
    parser.add_argument("--n_holdout_shifts", type=int, default=1000)
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, sets = load(args.derivatives)
    held = df["subject"].isin(HOLDOUT).to_numpy()
    if held.sum() == 0 or df.loc[held, "subject"].nunique() != len(HOLDOUT):
        raise SystemExit("held-out people missing from the features")
    dev, test = df[~held].reset_index(drop=True), df[held].reset_index(drop=True)
    print(f"[ok] {len(dev)} search reports ({dev['subject'].nunique()} people), "
          f"{len(test)} held-out reports ({test['subject'].nunique()} people)", flush=True)

    # Stage 1 (+2) on the search set.
    def run(name, cand):
        pred = loso(dev, cand, sets)
        return name, pred, score(dev, dev[TARGET], pred, "state")["score"]
    stage1 = Parallel(n_jobs=args.n_jobs)(delayed(run)(n, c) for n, c in CANDIDATES.items())
    table = pd.DataFrame([{"candidate": n, "auc": a, "stage": 1} for n, _, a in stage1]
                         ).set_index("candidate")
    for n, a in table["auc"].items():
        print(f"  {n:24s} search-set AUC {a:.3f}", flush=True)
    candidates = dict(CANDIDATES)
    ref = table.loc["riemann", "auc"]
    best_f, best_t = pick(table, FEATURE_OPTIONS), pick(table, TARGET_OPTIONS)
    if table.loc[best_f, "auc"] > ref and table.loc[best_t, "auc"] > ref:
        combo = f"{best_f}+{best_t.removeprefix('riemann_')}"
        candidates[combo] = {**CANDIDATES[best_f], "target": CANDIDATES[best_t]["target"]}
        _, _, a = run(combo, candidates[combo])
        table.loc[combo] = {"auc": a, "stage": 2}
        print(f"  {combo:24s} search-set AUC {a:.3f} (stage 2)", flush=True)
    winner = pick(table, list(table.index))
    table.to_csv(args.output_dir / "search_table.csv")
    print(f"[ok] winner on the search set: {winner}", flush=True)

    # Final test, once, on the held-out people.
    final = {}
    for name in dict.fromkeys([winner, "riemann", "baseline_context"]):
        cand = BASELINE if name == "baseline_context" else candidates[name]
        final[name] = fit_predict(dev, test, cand, sets)
    rng = np.random.default_rng(RANDOM_STATE)
    rows, per_person = [], {}
    shifts = [shuffled_target(test, TARGET, "state", rng) for _ in range(args.n_holdout_shifts)]
    for name, pred in final.items():
        auc = score(test, test[TARGET], pred, "state")["score"]
        null = np.array([score(test, y, pred, "state")["score"] for y in shifts])
        per_person[name] = subject_scores(test, test[TARGET], pred, "state").set_index("subject")["score"]
        lo, hi = mean_interval(per_person[name])
        rows.append({"model": name, "heldout_auc": auc, "ci_low": lo, "ci_high": hi,
                     "chance": null.mean(), "p_value": (1 + (null >= auc).sum()) / (1 + len(null))})
    held_table = pd.DataFrame(rows)
    diffs = []
    for other in [m for m in final if m != winner]:
        d = (per_person[winner] - per_person[other]).dropna()
        lo, hi = mean_interval(d)
        diffs.append({"model": winner, "vs": other, "mean_diff": d.mean(), "ci_low": lo, "ci_high": hi})
    diff_table = pd.DataFrame(diffs)
    held_table.to_csv(args.output_dir / "heldout_results.csv", index=False)
    diff_table.to_csv(args.output_dir / "heldout_comparisons.csv", index=False)
    print(held_table.to_string(index=False))
    print(diff_table.to_string(index=False))
    improved = (winner != "riemann" and
                diff_table.loc[diff_table["vs"] == "riemann", "ci_low"].iloc[0] > 0)
    print(f"\nRESULT: {winner} " + ("IMPROVES on riemann on the held-out people"
                                  if improved else "is NOT a confirmed improvement over riemann "
                                  "on the held-out people"), flush=True)

    # Dashboard run: the winner under the phase 7 setup (all 39 people).
    cand = candidates[winner]
    pred = loso(df, cand, sets)
    auc = score(df, df[TARGET], pred, "state")["score"]

    def null_auc(seed):
        y = shuffled_target(df, TARGET, "state", np.random.default_rng(seed))
        return score(df, y, loso(df.assign(**{TARGET: y}), cand, sets), "state")["score"]
    null = np.array(Parallel(n_jobs=args.n_jobs)(
        delayed(null_auc)(RANDOM_STATE + i) for i in range(args.n_permutations)))
    lo, hi = mean_interval(subject_scores(df, df[TARGET], pred, "state")["score"])
    pd.DataFrame([{"model": winner, "score": auc, "ci_low": lo, "ci_high": hi,
                   "chance": null.mean(), "p_value": (1 + (null >= auc).sum()) / (1 + len(null)),
                   "n_permutations": len(null), "n_subjects": df["subject"].nunique(),
                   "n_probes": len(df), "selection": "selected by search; exploratory"}]
                 ).to_csv(args.output_dir / "winner_all39.csv", index=False)
    pd.DataFrame({"subject": df["subject"], "recording": df["recording"],
                  "probe_index": df["probe_index"], "target": df[TARGET],
                  "prediction": pred}).to_csv(args.output_dir / "winner_predictions.csv", index=False)
    print(f"[ok] {winner}, all 39 people: AUC {auc:.3f} (chance {null.mean():.3f}, "
          f"p={(1 + (null >= auc).sum()) / (1 + len(null)):.3f})")


if __name__ == "__main__":
    main()
