"""Step 7 (optional, follow-up): four extensions to the core LME result,
all using features already computed in extract_features.py -- no need to re-run
preprocessing. Each answers a distinct question that came up from the main
findings:

  A) Delta power test -- model_comparison.py's SHAP audit found the black-box
     model actually leans on scalp_delta_power, not theta/gamma. This
     formalizes that as its own hypothesis test instead of just an
     incidental finding.
  B) Tiredness outcome -- does the ratio (or delta) predict how tired
     someone reported feeling, even though it doesn't predict concentration?
  C) Individual differences (random slopes) -- the main LME only tested
     whether ratio predicts concentration ON AVERAGE across everyone. This
     tests whether the ratio-concentration relationship's *strength* varies
     meaningfully from person to person -- i.e. does it work for some
     people even if not on average?
  D) Expertise moderation -- does being an experienced meditator change
     whether the ratio predicts concentration (an interaction test)?

Usage:
    python src/followup_analysis.py --bids_root data/ds001787
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def test_delta_power(df: pd.DataFrame) -> None:
    section("A) Does delta power predict concentration better than the ratio?")
    sub = df.dropna(subset=["concentration_rating", "scalp_delta_power",
                            "minimal_ratio"])
    model = smf.mixedlm(
        "concentration_rating ~ scalp_delta_power + minimal_ratio",
        sub, groups=sub["subject"])
    result = model.fit(reml=True)
    print(result.summary())
    delta_p = result.pvalues.get("scalp_delta_power", np.nan)
    ratio_p = result.pvalues.get("minimal_ratio", np.nan)
    print(f"\n[interpretation] delta_power p={delta_p:.3f}, "
          f"ratio p={ratio_p:.3f}. "
          f"{'Delta is the stronger predictor here.' if delta_p < ratio_p else 'Ratio holds up better than delta here.'}")


def test_tiredness_outcome(df: pd.DataFrame) -> None:
    section("B) Does the ratio (or delta) predict TIREDNESS instead of concentration?")
    sub = df.dropna(subset=["tiredness_rating", "minimal_ratio",
                            "scalp_delta_power"])
    if sub.empty:
        print("[skip] no non-missing tiredness_rating rows -- check "
              "preprocess.py decoded it correctly.")
        return
    model = smf.mixedlm(
        "tiredness_rating ~ minimal_ratio + scalp_delta_power",
        sub, groups=sub["subject"])
    result = model.fit(reml=True)
    print(result.summary())
    print(f"\n[interpretation] if scalp_delta_power significantly predicts "
          f"tiredness (p<.05) but not concentration, that's a coherent "
          f"story: the black-box model in model_comparison.py may have been "
          f"partly tracking drowsiness rather than meditation depth.")


def test_individual_differences(df: pd.DataFrame) -> pd.DataFrame:
    section("C) Does the ratio-concentration link vary person to person?")
    sub = df.dropna(subset=["concentration_rating", "minimal_ratio"])
    # Random slope model: each subject gets their own estimated ratio effect,
    # not just their own baseline (random intercept, the main LME's model).
    model = smf.mixedlm(
        "concentration_rating ~ minimal_ratio", sub,
        groups=sub["subject"], re_formula="~minimal_ratio")
    result = model.fit(reml=True, method="lbfgs")
    print(result.summary())

    slope_var = result.cov_re.iloc[1, 1] if result.cov_re.shape[0] > 1 else np.nan
    print(f"\n[interpretation] the random-slope variance ({slope_var:.4f}) "
          f"tells you how much the ratio's effect differs across people. "
          f"A near-zero value means everyone is equally unaffected by "
          f"ratio (consistent with the main null finding); a large value "
          f"would mean some people show a real effect that averages out "
          f"against others who show the opposite.")

    # Per-subject random effects (empirical Bayes estimates) -- who, if
    # anyone, looks like a "responder"?
    re_df = pd.DataFrame(result.random_effects).T
    re_df.columns = ["intercept_deviation", "ratio_slope_deviation"] \
        if re_df.shape[1] > 1 else ["intercept_deviation"]
    if "ratio_slope_deviation" in re_df.columns:
        ranked = re_df.sort_values("ratio_slope_deviation", ascending=False)
        print("\nTop 5 subjects by estimated individual ratio effect "
              "(most positive -- ratio going up most associated with "
              "concentration going up, for that person specifically):")
        print(ranked.head(5)[["ratio_slope_deviation"]])
        print("\nBottom 5 (most negative):")
        print(ranked.tail(5)[["ratio_slope_deviation"]])
        return ranked
    return re_df


def test_expertise_moderation(df: pd.DataFrame) -> None:
    section("D) Does expertise change whether the ratio predicts concentration?")
    if "group" not in df.columns:
        print("[skip] no 'group' column -- run extract_features.py's "
              "participants.tsv merge first.")
        return
    sub = df.dropna(subset=["concentration_rating", "minimal_ratio", "group"])
    sub = sub.copy()
    sub["group"] = sub["group"].astype(str)
    model = smf.mixedlm(
        "concentration_rating ~ minimal_ratio * group", sub,
        groups=sub["subject"])
    result = model.fit(reml=True)
    print(result.summary())
    interaction_terms = [p for p in result.pvalues.index if ":" in p]
    if interaction_terms:
        p = result.pvalues[interaction_terms[0]]
        print(f"\n[interpretation] interaction term p={p:.3f}. "
              f"{'Expertise DOES change the ratio-concentration relationship.' if p < 0.05 else 'No evidence expertise changes the relationship -- the null result holds for both groups.'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    args = parser.parse_args()

    paths = Paths.from_root(args.bids_root)
    feat_path = paths.derivatives / "features.parquet"
    if not feat_path.exists():
        raise SystemExit(f"{feat_path} not found -- run extract_features.py first.")
    df = pd.read_parquet(feat_path)

    test_delta_power(df)
    test_tiredness_outcome(df)
    responder_ranking = test_individual_differences(df)
    test_expertise_moderation(df)

    if isinstance(responder_ranking, pd.DataFrame) and not responder_ranking.empty:
        out_path = paths.derivatives / "individual_ratio_effects.csv"
        responder_ranking.to_csv(out_path)
        print(f"\n[ok] wrote per-subject ratio-effect ranking to {out_path}")

    print("\n" + "="*70)
    print("All follow-up tests complete. These are exploratory (not part "
          "of the pre-registered core hypothesis) -- report them as such "
          "in your writeup, e.g. 'exploratory follow-up analyses "
          "suggested...' rather than as confirmatory findings.")


if __name__ == "__main__":
    main()
