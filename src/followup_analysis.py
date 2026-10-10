"""Step 7 (optional): four EXPLORATORY follow-ups to the core mixed-effects
test in mixed_effects_stats.py. Same safeguards: predictors and time-on-task
z-scored within each recording, random intercept per subject. Report these
as exploratory ("follow-up analyses suggested..."), not as confirmed.

  A) Delta vs ratio -- does scalp delta power (slow waves, drowsiness)
     track concentration better than the gamma/theta ratio?
  B) Tiredness -- do the ratio or delta track how tired people say they are?
  C) Individual differences -- random slopes: does the ratio's link with
     each rating differ from person to person? (Per-person estimates are
     written to individual_ratio_effects.csv.)
  D) Expertise -- does the ratio's link with each rating differ between
     experts and novices? Caution: in this dataset group is nearly the same
     split as sex (experts 9/11 male, novices 10/12 female).

Writes lme_followup_results.csv (one row per term) and
individual_ratio_effects.csv.

Usage:
    python src/followup_analysis.py --bids_root data/ds001787
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from mixed_effects_stats import RATINGS, TIME, fit_lme, standardise  # noqa: E402

ALPHA = 0.05


def report(title: str, tidy: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print(tidy[["outcome", "term", "coef", "ci_low", "ci_high", "p_value"]]
          .round(4).to_string(index=False))
    hits = tidy[(tidy["p_value"] < ALPHA) & (tidy["term"] != TIME)]
    if hits.empty:
        print(f"[interpretation] no EEG term reaches p < {ALPHA}.")
    for t in hits.itertuples():
        direction = "higher" if t.coef > 0 else "lower"
        print(f"[interpretation] {t.term}: {direction} {t.outcome} "
              f"(coef {t.coef:+.3f}, p = {t.p_value:.3f}) -- exploratory.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    args = parser.parse_args()

    paths = Paths.from_root(args.bids_root)
    feat_path = paths.derivatives / "features.parquet"
    if not feat_path.exists():
        raise SystemExit(f"{feat_path} not found -- run extract_features.py first.")
    features = pd.read_parquet(feat_path)
    tables = []

    # A) delta vs ratio for concentration
    df = standardise(features, "concentration_rating",
                     ["scalp_delta_power", "minimal_ratio"])
    _, tidy = fit_lme(df, f"concentration_rating ~ scalp_delta_power + "
                          f"minimal_ratio + {TIME}", "A_delta_vs_ratio")
    report("A) Delta power vs gamma/theta ratio -> concentration", tidy)
    tables.append(tidy)

    # B) tiredness
    df = standardise(features, "tiredness_rating",
                     ["scalp_delta_power", "minimal_ratio"])
    _, tidy = fit_lme(df, f"tiredness_rating ~ scalp_delta_power + "
                          f"minimal_ratio + {TIME}", "B_tiredness")
    report("B) Delta power and ratio -> tiredness", tidy)
    tables.append(tidy)

    # C) random slopes: does the ratio effect differ between people?
    slopes = []
    for outcome in RATINGS:
        df = standardise(features, outcome, ["minimal_ratio"])
        result, tidy = fit_lme(df, f"{outcome} ~ minimal_ratio + {TIME}",
                               "C_random_slopes", re_formula="~minimal_ratio")
        report(f"C) Person-to-person differences: ratio -> {outcome}", tidy)
        slope_sd = result.cov_re.loc["minimal_ratio", "minimal_ratio"] ** 0.5
        print(f"[interpretation] the ratio's effect varies between people "
              f"with SD {slope_sd:.3f} around the average "
              f"{tidy.loc[tidy.term == 'minimal_ratio', 'coef'].iloc[0]:+.3f}.")
        per_person = pd.DataFrame(result.random_effects).T
        slopes.append(pd.DataFrame({
            "subject": per_person.index, "outcome": outcome,
            "ratio_slope": (per_person["minimal_ratio"].to_numpy()
                            + result.fe_params["minimal_ratio"])}))
        tables.append(tidy)

    # D) expertise moderation
    for outcome in RATINGS:
        df = standardise(features.dropna(subset=["group"]), outcome,
                         ["minimal_ratio"])
        _, tidy = fit_lme(df, f"{outcome} ~ minimal_ratio * group + {TIME}",
                          "D_expertise")
        report(f"D) Expertise x ratio -> {outcome} (group ~ sex here!)", tidy)
        tables.append(tidy)

    out = pd.concat(tables, ignore_index=True)
    out.to_csv(paths.derivatives / "lme_followup_results.csv", index=False)
    pd.concat(slopes, ignore_index=True).sort_values(
        ["outcome", "ratio_slope"]).to_csv(
        paths.derivatives / "individual_ratio_effects.csv", index=False)
    print(f"\n[ok] wrote lme_followup_results.csv and "
          f"individual_ratio_effects.csv to {paths.derivatives}")
    print("These are exploratory: report them as follow-ups, not confirmed "
          "findings, and remember the 18 decoding tests in phase 3.")


if __name__ == "__main__":
    main()
