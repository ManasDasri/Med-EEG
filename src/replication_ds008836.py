"""Step 9: the pre-registered replication of lead 1 on OpenNeuro ds008836.

Implements docs/preregistration_ds008836.md exactly:
  primary (section 5): SILENT-block depth reports;
      depth ~ minimal_ratio + time, random intercept + slope per person,
      depth, ratio and time z-scored within each recording;
      replicated if the ratio coefficient is negative with one-sided p < 0.05
  secondary (section 6, exploratory): all blocks; probe reports only; gamma
      and theta alone; temple-gamma covariate; in_analysis participants only;
      per-person slope signs

Run after: preprocess.py --dataset ds008836, then extract_features.py.
Writes replication_results.csv.

Usage:
    python src/replication_ds008836.py --bids_root /Volumes/Crucial/EEG_ML/ds008836
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from mixed_effects_stats import TIME, fit_lme, standardise  # noqa: E402
from model_comparison import zscore_within_recording  # noqa: E402

OUTCOME = "depth_rating"
PRIMARY_BLOCKS = ("silent_probe", "silent_emerge")
ALPHA_ONE_SIDED = 0.05


def model(df: pd.DataFrame, predictor: str, covariates: str = "") -> dict:
    """The pre-registered model on any subset/predictor; returns the
    predictor's coefficient with two- and one-sided (negative) p-values."""
    d = standardise(df, OUTCOME, [predictor])
    d = zscore_within_recording(d, [OUTCOME]).dropna(subset=[OUTCOME])
    _, tidy = fit_lme(d, f"{OUTCOME} ~ {predictor} + {TIME}{covariates}",
                      "replication", re_formula=f"~{predictor}")
    row = tidy[tidy["term"] == predictor].iloc[0]
    p = row["p_value"]
    return {"fit_valid": bool(row["fit_valid"]),
            "fit_diagnostics": row["fit_diagnostics"],
            "estimate": row["coef"], "ci_low": row["ci_low"],
            "ci_high": row["ci_high"], "p_two_sided": p,
            "p_one_sided_negative": p / 2 if row["coef"] < 0 else 1 - p / 2,
            "n_reports": len(d), "n_participants": d["subject"].nunique()}


def slope_signs(df: pd.DataFrame) -> dict:
    """Per participant: least-squares slope of within-recording z-scored
    depth on z-scored ratio, pooled over that participant's recordings."""
    d = standardise(df, OUTCOME, ["minimal_ratio"])
    d = zscore_within_recording(d, [OUTCOME]).dropna(subset=[OUTCOME])
    slopes = d.groupby("subject").apply(
        lambda g: np.polyfit(g["minimal_ratio"], g[OUTCOME], 1)[0]
        if len(g) > 2 else np.nan).dropna()
    return {"estimate": float(slopes.median()),
            "n_participants": len(slopes),
            "note": f"{(slopes < 0).sum()} of {len(slopes)} participants "
                    f"negative"}


def analyses(features: pd.DataFrame, in_analysis: set[str]) -> pd.DataFrame:
    features = features.assign(temporal_gamma=features[
        ["left_temporal_gamma_log", "right_temporal_gamma_log"]].mean(axis=1))
    silent = features[features["block"].isin(PRIMARY_BLOCKS)]
    rows = [
        {"analysis": "PRIMARY: SILENT blocks", **model(silent, "minimal_ratio")},
        {"analysis": "S1 all blocks (+ block type, variant)",
         **model(features, "minimal_ratio", " + block_type + variant")},
        {"analysis": "S2 SILENT, prompted reports only",
         **model(silent[silent["variant"] == "probe"], "minimal_ratio")},
        {"analysis": "S3 gamma power alone",
         **model(silent, "minimal_gamma_power")},
        {"analysis": "S3 theta power alone",
         **model(silent, "minimal_theta_power")},
        {"analysis": "S4 ratio + temple gamma covariate",
         **model(silent.dropna(subset=["temporal_gamma"]), "minimal_ratio",
                 " + temporal_gamma")},
        {"analysis": "S5 in_analysis participants only",
         **model(silent[silent["subject"].isin(in_analysis)], "minimal_ratio")},
        {"analysis": "S6 per-person slopes (median)", **slope_signs(silent)},
    ]
    return pd.DataFrame(rows)


def verdict(primary: pd.Series) -> str:
    if not primary.get("fit_valid", False) or not np.isfinite(primary["p_one_sided_negative"]):
        return "INCONCLUSIVE (invalid mixed-model fit; inference withheld)"
    if primary["estimate"] < 0 and primary["p_one_sided_negative"] < ALPHA_ONE_SIDED:
        return "REPLICATED (negative slope, one-sided p < 0.05)"
    return "NOT REPLICATED (pre-registered rule: negative slope with one-sided p < 0.05)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="New result directory; defaults to a unique run directory")
    parser.add_argument("--derivatives", type=Path, default=None,
                        help="Explicit intermediate directory (preserve prior runs)")
    args = parser.parse_args()

    paths = Paths.from_root(args.bids_root, args.derivatives)
    from provenance import analysis_output
    input_path = getattr(args, "features", None) or paths.derivatives / "features.parquet"
    output_dir = analysis_output(paths, args, [input_path])
    features = pd.read_parquet(paths.derivatives / "features.parquet")
    participants = pd.read_csv(paths.bids_root / "participants.tsv", sep="\t")
    in_analysis = set(participants.loc[participants["in_analysis"] == "yes",
                                       "participant_id"])

    out = analyses(features, in_analysis)
    out.to_csv(output_dir / "replication_results.csv", index=False)
    with pd.option_context("display.width", 200):
        print(out.round(4).to_string(index=False))
    print(f"\nds001787 experts (reference): -0.12 per SD of ratio in rating "
          f"units, about -0.21 in within-person SD units.")
    print(f"PRIMARY RESULT: {verdict(out.iloc[0])}")
    print("Everything below PRIMARY is exploratory (pre-registration "
          "section 6).")


if __name__ == "__main__":
    main()
