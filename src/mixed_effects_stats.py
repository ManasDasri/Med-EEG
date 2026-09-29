"""Step 4: within-subject statistics.

Group-level (expert vs novice) contrasts are confounded by trait
differences between the two groups. The stronger, less confounded test is
whether gamma/theta ratio and PAC predict a person's OWN moment-to-moment
concentration rating, treating subject as a random effect.

Usage:
    python src/mixed_effects_stats.py --bids_root data/ds001787
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402


def fit_lme(df: pd.DataFrame, outcome: str, predictors: list[str]
            ) -> None:
    formula = f"{outcome} ~ " + " + ".join(predictors)
    model = smf.mixedlm(formula, df, groups=df["subject"])
    result = model.fit(reml=True)
    print(f"\n=== LME: {formula}  (random intercept by subject) ===")
    print(result.summary())


def group_contrast(df: pd.DataFrame, feature: str, group_col: str = "group"
                    ) -> None:
    if group_col not in df.columns:
        print(f"[skip] no '{group_col}' column -- merge participants.tsv "
              f"group labels into features.parquet first (see "
              f"inspect_dataset.py participants.tsv inspection).")
        return
    means = df.groupby(group_col)[feature].mean()
    print(f"\n{feature} by group:\n{means}")


def electrode_minimalism_curve(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Compare how much variance in the outcome each electrode-set's ratio
    explains, as a quick proxy for the fuller LOSO-CV comparison in
    model_comparison.py."""
    results = []
    for cluster in ["minimal", "frontal_theta", "posterior_gamma"]:
        col = f"{cluster}_ratio"
        if col not in df.columns:
            continue
        sub = df[[col, outcome, "subject"]].dropna()
        if len(sub) < 10:
            continue
        formula = f"{outcome} ~ {col}"
        model = smf.mixedlm(formula, sub, groups=sub["subject"])
        result = model.fit(reml=True)
        results.append({
            "electrode_set": cluster,
            "n_electrodes": {"minimal": 2, "frontal_theta": 3,
                              "posterior_gamma": 3}.get(cluster),
            "coef": result.params.get(col, float("nan")),
            "pvalue": result.pvalues.get(col, float("nan")),
            "n_obs": len(sub),
        })
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    args = parser.parse_args()

    paths = Paths.from_root(args.bids_root)
    feat_path = paths.derivatives / "features.parquet"
    if not feat_path.exists():
        raise SystemExit(f"{feat_path} not found -- run extract_features.py first.")
    df = pd.read_parquet(feat_path)

    if df["concentration_rating"].isna().all():
        raise SystemExit(
            "concentration_rating is entirely NaN -- go back to "
            "preprocess.py and fill in the actual rating values from "
            "events.tsv before running any stats.")

    df = df.dropna(subset=["concentration_rating", "minimal_ratio"])

    fit_lme(df, "concentration_rating",
            ["minimal_ratio", "minimal_pac"])

    group_contrast(df, "frontal_theta_ratio")

    curve = electrode_minimalism_curve(df, "concentration_rating")
    print("\n=== electrode-minimalism curve (ratio -> concentration) ===")
    print(curve)
    out_path = paths.derivatives / "electrode_minimalism_curve.csv"
    curve.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
