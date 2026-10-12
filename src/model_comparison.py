"""Step 5: how well do EEG features predict meditation expertise and
moment-to-moment self-reports -- measured honestly.

Tasks (--task):
  expertise       expert vs novice. A trait: one label per PERSON, so the
                  score is per person (24 cases), not per probe.
  concentration   self-rated depth of meditation (1-3), per probe
  mind_wandering  self-rated mind wandering (1-3), per probe

Every model is scored with leave-one-subject-out cross-validation (the
person being tested was never seen in training), and every score comes with
a permutation p-value: the same pipeline re-run on shuffled labels, which
shows what score luck alone produces.

Models:
  interpretable_ratio_pac  minimal_ratio + minimal_pac, linear model
  linear_spectral          all regional spectral features, L2-regularised
  boosting_spectral        same features, gradient-boosted trees
  baseline_*               non-EEG baselines the EEG models must beat:
                           sex and age (expertise), time-on-task (ratings)
followed by a SHAP audit of boosting_spectral.

Writes model_comparison_<task>.csv, predictions_<task>.csv (out-of-fold
prediction per probe, best model) and shap_importance_<task>.csv.

Usage:
    python src/model_comparison.py --bids_root data/ds001787 --task expertise
    python src/model_comparison.py --bids_root data/ds001787 --task concentration
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (BOOSTING_PARAMS, LINEAR_C, N_PERMUTATIONS,  # noqa: E402
                    RANDOM_STATE, RIDGE_ALPHA, Paths)

TASKS = {  # task -> (target column, trait or state)
    "expertise": ("group_binary", "trait"),
    "concentration": ("concentration_rating", "state"),
    "mind_wandering": ("mind_wandering_rating", "state"),
}
INTERPRETABLE_FEATURES = ["minimal_ratio", "minimal_pac"]
# Non-EEG baselines, scored exactly like the EEG models. An EEG model only
# means something if it beats these, not just chance: in this dataset experts
# are 9/11 male and novices 10/12 female, and ratings drift over a session.
BASELINES = {
    "trait": {"baseline_sex": ["male"], "baseline_age": ["age"]},
    "state": {"baseline_time": ["probe_onset_s"]},
}


def spectral_columns(df: pd.DataFrame) -> list[str]:
    """The regional features from extract_features.py. QC columns
    (n_epochs_*, window_start_s, ...) are deliberately excluded: they track
    drowsiness and probe timing, which would be a shortcut, not EEG."""
    return ([c for c in df.columns if c.endswith(("_log", "_rel", "_slope"))]
            + ["frontal_alpha_asymmetry"])


def make_model(name: str, kind: str):
    """Classifiers balance the classes: leaving one person out of 12 vs 12
    leaves 11 vs 12, and a model with nothing to learn then always predicts
    the larger class -- i.e. always the held-out person's opposite, scoring
    ~0% instead of ~50%."""
    trait = kind == "trait"
    if name == "boosting_spectral":
        if trait:
            return HistGradientBoostingClassifier(
                **BOOSTING_PARAMS, class_weight="balanced",
                random_state=RANDOM_STATE)
        return HistGradientBoostingRegressor(**BOOSTING_PARAMS,
                                             random_state=RANDOM_STATE)
    linear = (LogisticRegression(C=LINEAR_C, class_weight="balanced",
                                 max_iter=2000) if trait
              else Ridge(alpha=RIDGE_ALPHA))
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         linear)


def prepare(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Drop probes without the target. For state tasks, z-score every
    feature within each recording: people differ ~6x in raw power for
    reasons unrelated to their state, and this uses only that recording's
    EEG -- never its labels -- so it is available for a new person too."""
    target, kind = TASKS[task]
    if target not in df.columns:
        raise SystemExit(f"'{target}' missing from features.parquet -- re-run "
                         f"extract_features.py (group labels come from "
                         f"participants.tsv)")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if "gender" in df.columns:
        df["male"] = df["gender"].str.upper().map({"M": 1.0, "F": 0.0})
    if kind == "state":
        df = zscore_within_recording(df, INTERPRETABLE_FEATURES
                                     + spectral_columns(df))
    return df


def zscore_within_recording(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Each column minus its recording's mean, over its recording's std:
    keeps how a feature moves within a session, removes between-person
    scale. Uses only that recording's EEG, never its labels."""
    df = df.copy()
    df[cols] = df.groupby(["recording"] if "recording" in df else ["subject", "session"])[cols].transform(
        lambda x: (x - x.mean()) / x.std())
    return df


def loso_predictions(df: pd.DataFrame, features: list[str], target: pd.Series,
                     name: str, kind: str) -> np.ndarray:
    """Out-of-fold prediction for every probe, each from a model that never
    saw that probe's subject. Trait: probability of 'expert'. State: a score
    where higher = higher rating (only its order within a person matters)."""
    method = "predict_proba" if kind == "trait" else "predict"
    pred = cross_val_predict(make_model(name, kind), df[features], target,
                             groups=df["subject"], cv=LeaveOneGroupOut(),
                             method=method)
    return pred[:, 1] if kind == "trait" else pred


def score(df: pd.DataFrame, target: pd.Series, pred: np.ndarray, kind: str
          ) -> dict:
    """Trait: average each person's probe predictions, then score the 24
    people. State: per person, AUC of telling their above-own-average
    ratings from their below-average ones, averaged over people."""
    frame = pd.DataFrame({"subject": df["subject"], "y": target, "p": pred})
    if kind == "trait":
        per_subject = frame.groupby("subject")[["y", "p"]].mean()
        return {"metric": "subject_accuracy",
                "score": float(((per_subject.p > 0.5) == per_subject.y).mean()),
                "auc": float(roc_auc_score(per_subject.y, per_subject.p))}
    aucs = []
    for _, g in frame.groupby("subject"):
        high = g.y > g.y.mean()
        if high.nunique() == 2:  # people who always gave the same rating can't be scored
            aucs.append(roc_auc_score(high, g.p))
    return {"metric": "within_subject_auc", "score": float(np.mean(aucs)),
            "n_scored_subjects": len(aucs)}


def subject_scores(df, target, pred, kind):
    frame = pd.DataFrame({"subject": df["subject"].to_numpy(),
                          "target": np.asarray(target), "prediction": pred})
    rows = []
    for subject, group in frame.groupby("subject"):
        if kind == "trait":
            value = float((group.prediction.mean() > .5) == group.target.iloc[0])
        else:
            high = group.target > group.target.mean()
            if high.nunique() < 2: continue
            value = roc_auc_score(high, group.prediction)
        rows.append({"subject": subject, "score": value})
    return pd.DataFrame(rows, columns=["subject", "score"])


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values): return np.nan, np.nan
    rng = np.random.default_rng(RANDOM_STATE)
    means = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [.025, .975]))


def paired_comparisons(scores):
    rows = []
    for model in scores.model.unique():
        if model.startswith("baseline_"): continue
        for baseline in scores.model.unique():
            if not baseline.startswith("baseline_"): continue
            paired = scores[scores.model == model].merge(
                scores[scores.model == baseline], on="subject", suffixes=("_eeg", "_baseline"))
            differences = paired.score_eeg - paired.score_baseline
            low, high = mean_interval(differences)
            rows.append({"model": model, "baseline": baseline,
                         "mean_difference": differences.mean(), "ci_low": low,
                         "ci_high": high, "n_subjects": len(paired),
                         "interval_scope": "descriptive subject bootstrap; fixed out-of-fold predictions"})
    return pd.DataFrame(rows)


def shuffled_target(df: pd.DataFrame, target: str, kind: str,
                    rng: np.random.Generator) -> pd.Series:
    """Labels with the real EEG-label link broken but the structure kept:
    trait labels are reassigned between PEOPLE; state ratings are shuffled
    by a circular shift WITHIN each recording (preserves rating order)."""
    if kind == "trait":
        labels = df.groupby("subject")[target].first()
        return df["subject"].map(pd.Series(rng.permutation(labels.to_numpy()),
                                           index=labels.index))
    shuffled = df[target].copy()
    keys = ["recording"] if "recording" in df else ["subject", "session"]
    order = [c for c in ("probe_onset_s", "probe_index") if c in df]
    for _, group in df.groupby(keys, sort=False):
        if order: group = group.sort_values(order, kind="stable")
        # Include zero: identity is part of the circular-shift null group.
        shuffled.loc[group.index] = np.roll(group[target].to_numpy(), rng.integers(len(group)))
    return shuffled


def evaluate(df: pd.DataFrame, task: str, name: str, features: list[str],
             n_permutations: int, n_jobs: int = 2) -> tuple[dict, np.ndarray]:
    target, kind = TASKS[task]
    pred = loso_predictions(df, features, df[target], name, kind)
    result = score(df, df[target], pred, kind)

    def null_score(seed: int) -> float:
        y = shuffled_target(df, target, kind, np.random.default_rng(seed))
        return score(df, y, loso_predictions(df, features, y, name, kind),
                     kind)["score"]

    null = np.array(Parallel(n_jobs=n_jobs)(
        delayed(null_score)(RANDOM_STATE + i) for i in range(n_permutations)))
    ci_low, ci_high = mean_interval(subject_scores(df, df[target], pred, kind)["score"])
    result.update({
        "ci_low": ci_low, "ci_high": ci_high,
        "n_permutations": n_permutations,
        "null_method": "subject_label_shuffle" if kind == "trait" else "recording_circular_shift",
        "model": name,
        "n_features": len(features),
        "chance": float(null.mean()) if n_permutations else np.nan,
        "p_value": float((1 + (null >= result["score"]).sum())
                         / (1 + n_permutations)) if n_permutations else np.nan,
        "n_subjects": int(df["subject"].nunique()),
        "n_probes": len(df),
    })
    return result, pred


def shap_audit(df: pd.DataFrame, features: list[str], task: str,
               out_dir: Path) -> None:
    """Which features the boosted model leans on. Muscle markers (temporal
    gamma, flat slopes) near the top would mean it learned artifact, not
    brain activity."""
    target, kind = TASKS[task]
    model = make_model("boosting_spectral", kind).fit(df[features], df[target])
    values = shap.TreeExplainer(model).shap_values(df[features])
    values = values[..., -1] if np.ndim(values) == 3 else values
    importance = pd.Series(np.abs(values).mean(axis=0), index=features,
                           name="importance").sort_values(ascending=False)
    importance.rename_axis("feature").to_csv(out_dir / f"shap_importance_{task}.csv")
    print("\n=== SHAP: top 10 features (boosting_spectral) ===")
    print(importance.head(10).round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    parser.add_argument("--task", choices=list(TASKS), required=True)
    parser.add_argument("--features", type=Path, default=None,
                        help="Feature table to use (default: "
                             "derivatives/neurodial/features.parquet), e.g. "
                             "a copy made with extract_features.py --window_s")
    parser.add_argument("--n_permutations", type=int, default=N_PERMUTATIONS,
                        help="Shuffled-label repeats for the p-value "
                             "(0 = skip, much faster)")
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="New result directory; defaults to a unique run directory")
    parser.add_argument("--derivatives", type=Path, default=None,
                        help="Explicit intermediate directory (preserve prior runs)")
    args = parser.parse_args()

    if args.n_permutations < 0:
        parser.error("n_permutations must be nonnegative")
    paths = Paths.from_root(args.bids_root, args.derivatives)
    from provenance import analysis_output
    input_path = getattr(args, "features", None) or paths.derivatives / "features.parquet"
    output_dir = analysis_output(paths, args, [input_path])
    feat_path = args.features or paths.derivatives / "features.parquet"
    if not feat_path.exists():
        raise SystemExit(f"{feat_path} not found -- run extract_features.py")
    df = prepare(pd.read_parquet(feat_path), args.task)
    target, kind = TASKS[args.task]
    feature_sets = {"interpretable_ratio_pac": INTERPRETABLE_FEATURES,
                    "linear_spectral": spectral_columns(df),
                    "boosting_spectral": spectral_columns(df)}
    for name, cols in BASELINES[kind].items():
        if all(c in df.columns for c in cols):
            feature_sets[name] = cols
        else:
            print(f"[warn] {name} skipped: {cols} not in the feature table "
                  f"(re-run extract_features.py)")
    print(f"[ok] task={args.task}: {len(df)} probes, "
          f"{df['subject'].nunique()} subjects, "
          f"{args.n_permutations} permutations per model")

    results, preds = [], {}
    for name, cols in feature_sets.items():
        result, preds[name] = evaluate(df, args.task, name, cols,
                                       args.n_permutations, args.n_jobs)
        results.append(result)
        print(f"  {name:24s} {result['metric']}={result['score']:.3f}  "
              f"(chance {result['chance']:.3f}, p={result['p_value']:.3f})")

    from statsmodels.stats.multitest import multipletests
    summary = pd.DataFrame(results)
    summary["p_holm_task"] = np.nan
    tested = summary["p_value"].notna()
    if tested.any():
        summary.loc[tested, "p_holm_task"] = multipletests(
            summary.loc[tested, "p_value"], method="holm")[1]
    summary["multiplicity_family"] = f"all models including baselines for {args.task}"
    summary.to_csv(output_dir / f"model_comparison_{args.task}.csv", index=False)
    all_predictions, per_subject = [], []
    for name, prediction in preds.items():
        frame = df[["subject", "session", "probe_index"]].copy()
        if "recording" in df: frame["recording"] = df["recording"]
        frame["target"], frame["prediction"], frame["model"] = df[target], prediction, name
        all_predictions.append(frame)
        per_subject.append(subject_scores(df, df[target], prediction, kind).assign(model=name))
    pd.concat(all_predictions, ignore_index=True).to_csv(
        output_dir / f"predictions_{args.task}.csv", index=False)
    scores = pd.concat(per_subject, ignore_index=True)
    scores.to_csv(output_dir / f"subject_scores_{args.task}.csv", index=False)
    paired_comparisons(scores).to_csv(output_dir / f"baseline_comparisons_{args.task}.csv", index=False)

    shap_audit(df, feature_sets["boosting_spectral"], args.task,
               output_dir)
    print(f"\n[ok] wrote model_comparison_{args.task}.csv, "
          f"predictions_{args.task}.csv, shap_importance_{args.task}.csv "
          f"to {paths.derivatives}")


if __name__ == "__main__":
    main()
