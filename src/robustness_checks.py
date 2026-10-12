"""Step 8: stress-test the two leads from phases 3-4, using data already on
disk. A real effect should survive all of these; an artifact or a fluke
usually fails at least one.

Lead 1 -- expertise x gamma/theta ratio -> concentration (followup D):
  a) session split: refit on first sessions only, then on later sessions only
  b) leave-one-person-out: refit once per person with that person removed
  c) components: gamma power alone, theta power alone, instead of the ratio
  d) muscle control: temple-electrode gamma (mostly muscle) instead of the
     ratio, then midline gamma with temple gamma in the same model -- if
     the midline effect survives, muscle can't explain it
Lead 2 -- one electrode detects mind wandering (electrode_minimalism):
  e) cross-session: choose the best electrode on first sessions only and
     score it on later sessions only (and the reverse), with permutations

Writes robustness_checks.csv.

Usage:
    python src/robustness_checks.py --bids_root data/ds001787
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import N_PERMUTATIONS, RANDOM_STATE, Paths  # noqa: E402
from electrode_minimalism import within_person_auc  # noqa: E402
from mixed_effects_stats import TIME, fit_lme, standardise  # noqa: E402
from model_comparison import prepare, zscore_within_recording  # noqa: E402

OUTCOME = "concentration_rating"
FIRST_SESSION = "ses-01"


def interaction(features: pd.DataFrame, predictor: str,
                control: str | None = None) -> dict:
    """Follow-up D's model (random slope per person, sex-adjusted) with any
    predictor; returns the expertise x predictor interaction. `control`
    adds a second predictor with its own expertise interaction, so the
    first is tested over and above it."""
    extra = [control] if control else []
    df = standardise(features.dropna(subset=["group", "gender"]), OUTCOME,
                     [predictor, *extra])
    control_terms = f" + {control} * group" if control else ""
    try:
        _, tidy = fit_lme(df, f"{OUTCOME} ~ {predictor} * group + "
                              f"{predictor} * gender{control_terms} + "
                              f"{TIME}", "check",
                          re_formula=f"~{predictor}")
        row = tidy[tidy["term"] == f"{predictor}:group[T.novice]"].iloc[0]
        return {"fit_valid": bool(row["fit_valid"]),
            "fit_diagnostics": row["fit_diagnostics"],
            "estimate": row["coef"], "ci_low": row["ci_low"],
                "ci_high": row["ci_high"], "p_value": row["p_value"],
                "n_subjects": df["subject"].nunique()}
    except (np.linalg.LinAlgError, ValueError, IndexError) as e:
        # e.g. too few novices with later sessions to estimate the model
        return {"estimate": np.nan, "p_value": np.nan,
                "n_subjects": df["subject"].nunique(), "note": str(e)[:80]}


def lead_1(features: pd.DataFrame) -> list[dict]:
    features = features.assign(temporal_gamma=features[
        ["left_temporal_gamma_log", "right_temporal_gamma_log"]].mean(axis=1))
    first = features["session"] == FIRST_SESSION
    rows = [{"check": "1 reference: all data", **interaction(features, "minimal_ratio")},
            {"check": "1a first sessions only", **interaction(features[first], "minimal_ratio")},
            {"check": "1a later sessions only", **interaction(features[~first], "minimal_ratio")}]

    jack = pd.DataFrame([{"left_out": s, **interaction(
        features[features["subject"] != s], "minimal_ratio")}
        for s in sorted(features["subject"].unique())])
    valid_jack = jack.dropna(subset=["p_value"])
    worst = valid_jack.loc[valid_jack["p_value"].idxmax()] if len(valid_jack) else jack.iloc[0]
    rows.append({"check": "1b leave-one-person-out: weakest refit",
                 "estimate": worst["estimate"], "p_value": worst["p_value"],
                 "n_subjects": worst["n_subjects"],
                 "note": f"without {worst['left_out']}; estimate range "
                         f"{jack['estimate'].min():+.3f} to "
                         f"{jack['estimate'].max():+.3f}"})

    for label, col in [("1c gamma power alone", "minimal_gamma_power"),
                       ("1c theta power alone", "minimal_theta_power"),
                       ("1d muscle control: temple gamma", "temporal_gamma")]:
        rows.append({"check": label, **interaction(features, col)})
    rows.append({"check": "1d midline gamma, temple gamma controlled",
                 **interaction(features, "minimal_gamma_power",
                               control="temporal_gamma")})
    rows.append({"check": "1d ratio, temple gamma controlled",
                 **interaction(features, "minimal_ratio",
                               control="temporal_gamma")})
    return rows


def lead_2(features: pd.DataFrame, n_permutations: int) -> list[dict]:
    df = prepare(features, "mind_wandering")
    ratio_cols = [c for c in df.columns if c.startswith("ratio_")]
    df = zscore_within_recording(df, ratio_cols)
    X = df[ratio_cols].fillna(0).to_numpy()
    y = df["mind_wandering_rating"].to_numpy()
    subjects = df["subject"].to_numpy()
    first = (df["session"] == FIRST_SESSION).to_numpy()
    rng = np.random.default_rng(RANDOM_STATE)

    rows = []
    for pick, test, label in [(first, ~first, "first -> later"),
                              (~first, first, "later -> first")]:
        auc = within_person_auc(X[pick], y[pick], subjects[pick])
        best = int(np.argmax(np.abs(auc - 0.5)))
        x_test = X[test][:, [best]] * np.sign(auc[best] - 0.5)
        score = within_person_auc(x_test, y[test], subjects[test])[0]
        null = []
        for _ in range(n_permutations):
            y_perm = pd.Series(y[test]).groupby(subjects[test]).transform(
                lambda r: rng.permutation(r.to_numpy())).to_numpy()
            null.append(within_person_auc(x_test, y_perm, subjects[test])[0])
        rows.append({
            "check": f"2e best electrode, chosen {label}",
            "estimate": score,
            "p_value": (1 + np.sum(np.array(null) >= score)) / (1 + n_permutations),
            "n_subjects": len(np.unique(subjects[test])),
            "note": f"{ratio_cols[best][len('ratio_'):]} (AUC {auc[best]:.3f} "
                    f"where chosen); chance ~{np.mean(null):.3f}"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    parser.add_argument("--n_permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="New result directory; defaults to a unique run directory")
    args = parser.parse_args()

    paths = Paths.from_root(args.bids_root)
    from provenance import analysis_output
    input_path = getattr(args, "features", None) or paths.derivatives / "features.parquet"
    output_dir = analysis_output(paths, args, [input_path])
    features = pd.read_parquet(paths.derivatives / "features.parquet")
    out = pd.DataFrame(lead_1(features) + lead_2(features, args.n_permutations))
    cols = ["check", "estimate", "ci_low", "ci_high", "p_value", "n_subjects", "fit_valid", "fit_diagnostics", "note"]
    out = out.reindex(columns=cols)
    out.to_csv(output_dir / "robustness_checks.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(out.round(4).to_string(index=False))
    print(f"\n[ok] wrote {paths.derivatives / 'robustness_checks.csv'}")


if __name__ == "__main__":
    main()
