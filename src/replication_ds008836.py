"""Step 9: the pre-registered replication of lead 1 on OpenNeuro ds008836.

Implements docs/preregistration_ds008836.md as amended by
docs/preregistration_ds008836_amendment.md:
  primary: SILENT-block depth reports; depth, ratio and time z-scored within
      each recording; depth ~ minimal_ratio + time by OLS with standard
      errors clustered by participant (t with participants - 1 df).
      (The pre-registered mixed model is unidentifiable after within-
      recording z-scoring -- every recording mean is 0 -- so it never
      converges; it is still reported, as invalid.)
  cross-check: one slope per participant, one-sided t-test against 0
  verdict: negative slope with one-sided p < 0.05 -> replicated, unless the
      cross-check disagrees, which makes it inconclusive
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
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from mixed_effects_stats import TIME, fit_lme, standardise  # noqa: E402
from model_comparison import zscore_within_recording  # noqa: E402

OUTCOME = "depth_rating"
PRIMARY_BLOCKS = ("silent_probe", "silent_emerge")
ALPHA_ONE_SIDED = 0.05


def prepared(df: pd.DataFrame, predictor: str) -> pd.DataFrame:
    """Reports with outcome, predictor and time z-scored within recording."""
    d = standardise(df, OUTCOME, [predictor])
    return zscore_within_recording(d, [OUTCOME]).dropna(subset=[OUTCOME])


def one_sided(coef: float, p_two_sided: float) -> float:
    return p_two_sided / 2 if coef < 0 else 1 - p_two_sided / 2


def model(df: pd.DataFrame, predictor: str, covariates: str = "") -> dict:
    """Amended primary model: OLS with participant-clustered standard errors
    (t with participants - 1 df). Returns the predictor's coefficient with
    two- and one-sided (negative) p-values."""
    d = prepared(df, predictor)
    fit = smf.ols(f"{OUTCOME} ~ {predictor} + {TIME}{covariates}", d).fit(
        cov_type="cluster", cov_kwds={"groups": d["subject"].astype("category").cat.codes},
        use_t=True)
    coef, p = fit.params[predictor], fit.pvalues[predictor]
    ci = fit.conf_int().loc[predictor]
    return {"fit_valid": bool(np.isfinite(fit.bse[predictor])),
            "method": "OLS, participant-clustered SE",
            "estimate": coef, "ci_low": ci[0], "ci_high": ci[1],
            "p_two_sided": p, "p_one_sided_negative": one_sided(coef, p),
            "n_reports": len(d), "n_participants": d["subject"].nunique()}


def per_person_ttest(df: pd.DataFrame, predictor: str = "minimal_ratio") -> dict:
    """Cross-check: one OLS slope per participant (depth on predictor and
    time), then a one-sided one-sample t-test of the slopes against 0."""
    d = prepared(df, predictor)
    slopes = d.groupby("subject").apply(
        lambda g: smf.ols(f"{OUTCOME} ~ {predictor} + {TIME}", g).fit().params[predictor]
        if len(g) > 3 else np.nan).dropna()
    t = stats.ttest_1samp(slopes, 0.0, alternative="less")
    ci = stats.t.interval(0.95, len(slopes) - 1, slopes.mean(), stats.sem(slopes))
    return {"fit_valid": True, "method": "per-participant slopes, one-sample t",
            "estimate": float(slopes.mean()), "ci_low": ci[0], "ci_high": ci[1],
            "p_two_sided": float(stats.ttest_1samp(slopes, 0.0).pvalue),
            "p_one_sided_negative": float(t.pvalue),
            "n_participants": len(slopes),
            "note": f"{(slopes < 0).sum()} of {len(slopes)} slopes negative"}


def mixed_model(df: pd.DataFrame, predictor: str) -> dict:
    """The model exactly as originally pre-registered (random intercept +
    slope). Reported for transparency; it is unidentifiable here."""
    d = prepared(df, predictor)
    _, tidy = fit_lme(d, f"{OUTCOME} ~ {predictor} + {TIME}",
                      "replication", re_formula=f"~{predictor}")
    row = tidy[tidy["term"] == predictor].iloc[0]
    p = row["p_value"]
    return {"fit_valid": bool(row["fit_valid"]),
            "method": "mixed model as pre-registered",
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
        {"analysis": "CROSS-CHECK: per-participant slopes",
         **per_person_ttest(silent)},
        {"analysis": "ORIGINAL pre-registered mixed model",
         **mixed_model(silent, "minimal_ratio")},
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


def passes(row: pd.Series) -> bool:
    return bool(row["estimate"] < 0 and row["p_one_sided_negative"] < ALPHA_ONE_SIDED)


def verdict(primary: pd.Series, cross_check: pd.Series) -> str:
    """Amendment 1: decided by the primary test; if the cross-check disagrees
    (direction or significance), the result is inconclusive."""
    if not primary.get("fit_valid", False) or not np.isfinite(primary["p_one_sided_negative"]):
        return "INCONCLUSIVE (invalid fit; inference withheld)"
    agree = (passes(primary) == passes(cross_check)
             and np.sign(primary["estimate"]) == np.sign(cross_check["estimate"]))
    if not agree:
        return "INCONCLUSIVE (primary test and per-participant cross-check disagree)"
    if passes(primary):
        return "REPLICATED (negative slope, one-sided p < 0.05; cross-check agrees)"
    return "NOT REPLICATED (negative slope with one-sided p < 0.05 not met)"


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
    print(f"PRIMARY RESULT: {verdict(out.iloc[0], out.iloc[1])}")
    print("Rows after the ORIGINAL mixed model are exploratory "
          "(pre-registration section 6).")


if __name__ == "__main__":
    main()
