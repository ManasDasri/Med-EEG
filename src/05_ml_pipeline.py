"""Step 5: does the simple, interpretable biomarker actually compete with a
black-box model? Leave-one-subject-out cross-validation (never random
k-fold -- that leaks subject identity between train/test and inflates
accuracy) comparing:

  A) Interpretable: 2 features (minimal_ratio, minimal_pac) -> logistic
     regression / linear regression
  B) Black box: full 5-band x full-scalp features -> Random Forest
  C) Black box: same features -> small MLP

Then a SHAP audit on (B) to check it's actually leaning on theta/gamma power
and not some artifact.

Usage:
    python src/05_ml_pipeline.py --bids_root data/ds001787 --task classify
    python src/05_ml_pipeline.py --bids_root data/ds001787 --task regress
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import Paths  # noqa: E402

INTERPRETABLE_FEATURES = ["minimal_ratio", "minimal_pac"]
BLACKBOX_FEATURES = [
    "scalp_delta_power", "scalp_theta_power", "scalp_alpha_power",
    "scalp_beta_power", "scalp_gamma_power",
]


def loso_evaluate(df: pd.DataFrame, feature_cols: list[str], outcome: str,
                   task: str) -> dict:
    X = df[feature_cols].to_numpy()
    y = df[outcome].to_numpy()
    groups = df["subject"].to_numpy()
    logo = LeaveOneGroupOut()

    preds, truths = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        if task == "classify":
            model = LogisticRegression(max_iter=1000) if len(feature_cols) <= 2 \
                else RandomForestClassifier(n_estimators=300, random_state=42)
        else:
            model = Ridge(alpha=1.0) if len(feature_cols) <= 2 \
                else RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y[train_idx])
        preds.append(model.predict(X_test))
        truths.append(y[test_idx])

    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    if task == "classify":
        score = accuracy_score(truths, preds)
        metric_name = "accuracy"
    else:
        score = r2_score(truths, preds)
        metric_name = "r2"
    return {"metric": metric_name, "score": score, "n": len(truths)}


def run_mlp_comparison(df: pd.DataFrame, feature_cols: list[str],
                        outcome: str, task: str) -> dict:
    X = df[feature_cols].to_numpy()
    y = df[outcome].to_numpy()
    groups = df["subject"].to_numpy()
    logo = LeaveOneGroupOut()
    preds, truths = [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        model = (MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=2000,
                                random_state=42) if task == "classify"
                 else MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=2000,
                                    random_state=42))
        model.fit(X_train, y[train_idx])
        preds.append(model.predict(X_test))
        truths.append(y[test_idx])
    preds, truths = np.concatenate(preds), np.concatenate(truths)
    if task == "classify":
        return {"metric": "accuracy", "score": accuracy_score(truths, preds)}
    return {"metric": "r2", "score": r2_score(truths, preds)}


def _extract_positive_class_shap(shap_values) -> np.ndarray:
    """Different shap versions return classifier SHAP values differently:
    older versions return a list of (n_samples, n_features) arrays, one per
    class; newer versions return a single (n_samples, n_features, n_classes)
    array. Regression models just return a plain (n_samples, n_features)
    array. Normalize all three to (n_samples, n_features) for the positive
    class / regression output."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        class_idx = 1 if shap_values.shape[2] > 1 else 0
        return shap_values[:, :, class_idx]
    return shap_values


def shap_audit(df: pd.DataFrame, feature_cols: list[str], outcome: str,
               task: str, out_dir: Path) -> None:
    X = df[feature_cols]
    y = df[outcome]
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols)

    model = (RandomForestClassifier(n_estimators=300, random_state=42)
             if task == "classify"
             else RandomForestRegressor(n_estimators=300, random_state=42))
    model.fit(X_scaled, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    mean_abs_shap = np.abs(_extract_positive_class_shap(shap_values)).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=feature_cols).sort_values(
        ascending=False)
    print("\n=== SHAP feature importance (black-box model) ===")
    print(importance)

    top_is_freq_band = importance.index[0] in (
        "scalp_theta_power", "scalp_gamma_power")
    print(f"[{'OK' if top_is_freq_band else 'CHECK'}] top feature is "
          f"{'a theta/gamma band' if top_is_freq_band else 'NOT theta or gamma -- '
           'the black box may be relying on something other than the '
           'biomarker this project is testing'}: {importance.index[0]}")

    importance.to_csv(out_dir / f"shap_importance_{task}.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    parser.add_argument("--task", choices=["classify", "regress"],
                         default="regress",
                         help="classify = novice vs expert group; "
                              "regress = continuous concentration_rating")
    args = parser.parse_args()

    paths = Paths.from_root(args.bids_root)
    feat_path = paths.derivatives / "features.parquet"
    if not feat_path.exists():
        raise SystemExit(f"{feat_path} not found -- run 03_features.py first.")
    df = pd.read_parquet(feat_path)

    outcome = "group_binary" if args.task == "classify" else "concentration_rating"
    if outcome not in df.columns:
        if outcome == "group_binary":
            raise SystemExit(
                "No 'group_binary' column -- merge participants.tsv group "
                "labels in and encode expert=1/novice=0 before running "
                "--task classify.")
        raise SystemExit(
            "concentration_rating missing -- fill it in during "
            "02_preprocess.py first.")

    needed = INTERPRETABLE_FEATURES + BLACKBOX_FEATURES + [outcome, "subject"]
    df = df.dropna(subset=[c for c in needed if c in df.columns])
    if df.empty:
        raise SystemExit("No complete rows after dropping NaNs -- check "
                          "features.parquet and the rating fill-in step.")

    print(f"[ok] {len(df)} rows, {df['subject'].nunique()} subjects, "
          f"task={args.task}")

    interpretable = loso_evaluate(df, INTERPRETABLE_FEATURES, outcome, args.task)
    blackbox_rf = loso_evaluate(df, BLACKBOX_FEATURES, outcome, args.task)
    blackbox_mlp = run_mlp_comparison(df, BLACKBOX_FEATURES, outcome, args.task)

    print("\n=== LOSO-CV shootout ===")
    print(f"Interpretable (ratio+PAC, 2 features): "
          f"{interpretable['metric']}={interpretable['score']:.3f}")
    print(f"Black box (Random Forest, 5-band scalp features): "
          f"{blackbox_rf['metric']}={blackbox_rf['score']:.3f}")
    print(f"Black box (MLP, 5-band scalp features): "
          f"{blackbox_mlp['metric']}={blackbox_mlp['score']:.3f}")
    gap = blackbox_rf["score"] - interpretable["score"]
    print(f"\nGap (RF - interpretable): {gap:.3f} "
          f"-- this gap is the headline number for the project's novelty "
          f"claim, small is a win for the interpretable-biomarker hypothesis.")

    shap_audit(df, BLACKBOX_FEATURES, outcome, args.task, paths.derivatives)

    summary = pd.DataFrame([
        {"model": "interpretable_ratio_pac", **interpretable},
        {"model": "random_forest_5band", **blackbox_rf},
        {"model": "mlp_5band", **blackbox_mlp},
    ])
    out_path = paths.derivatives / f"model_comparison_{args.task}.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
