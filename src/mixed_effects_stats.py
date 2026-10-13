"""Step 4: within-person mixed-effects models -- the project's core
statistical test.

Does a person's OWN gamma/theta ratio (and PAC) track their OWN
moment-to-moment ratings? Two safeguards:
  - predictors are z-scored within each recording, so a coefficient means
    "rating change per 1 SD change within a session", and between-person
    differences in raw power (~6x) can't pose as an effect
  - time-on-task (probe onset, also z-scored within recording) is a
    covariate: ratings drift over a session, and anything else that drifts
    would otherwise look like an effect
Random intercept per subject (each person's own average rating).

Writes lme_results.csv: one row per term, with 95% CI and p-value.

Usage:
    python src/mixed_effects_stats.py --bids_root data/ds001787
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths  # noqa: E402
from model_comparison import zscore_within_recording  # noqa: E402

CORE_PREDICTORS = ["minimal_ratio", "minimal_pac"]
TIME = "probe_onset_s"
RATINGS = ["concentration_rating", "mind_wandering_rating"]


def standardise(df: pd.DataFrame, outcome: str, predictors: list[str]
                ) -> pd.DataFrame:
    """Rows with the outcome and all predictors present; predictors and
    time-on-task z-scored within each recording. Recordings with too few
    probes (or no variation) to z-score are dropped -- e.g. tiredness was
    answered only once in some recordings."""
    df = df.dropna(subset=[outcome, *predictors]).reset_index(drop=True)
    df = zscore_within_recording(df, [*predictors, TIME])
    return df.dropna(subset=[*predictors, TIME]).reset_index(drop=True)


def fit_lme(df: pd.DataFrame, formula: str, analysis: str,
            re_formula: str | None = None):
    """Fit a random-intercept (optionally random-slope) model per subject.
    Returns (statsmodels result, tidy table of the fixed effects)."""
    # Fixed order chosen for numerical validity, never coefficient or p-value.
    attempts = []
    result = None
    valid = False
    for optimizer in ("lbfgs", "bfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                candidate = smf.mixedlm(formula, df, groups=df["subject"],
                    re_formula=re_formula).fit(reml=True, method=optimizer,
                                              maxiter=2000)
            messages = [str(w.message) for w in caught]
            result = candidate
            covariance = np.asarray(result.cov_params())
            valid = bool(result.converged and np.isfinite(covariance).all()
                         and np.linalg.eigvalsh(covariance).min() > 0
                         and not any("not positive definite" in m for m in messages))
            attempts.append({"optimizer": optimizer, "valid": valid,
                             "warnings": messages})
            if valid:
                break
        except (np.linalg.LinAlgError, ValueError) as exc:
            attempts.append({"optimizer": optimizer, "valid": False,
                             "warnings": [str(exc)]})
    if result is None:
        raise ValueError(f"All mixed-model fits failed: {attempts}")
    import json
    diagnostics = json.dumps(attempts)
    if not valid:
        warnings.warn(f"Invalid mixed-model fit; inference withheld: {diagnostics}",
                      RuntimeWarning)
    result.audit_valid = valid

    ci = result.conf_int()
    terms = [t for t in result.fe_params.index if t != "Intercept"]
    tidy = pd.DataFrame({
        "analysis": analysis,
        "outcome": formula.split("~")[0].strip(),
        "term": terms,
        "coef": result.fe_params[terms].to_numpy(),
        "ci_low": ci.loc[terms, 0].to_numpy(),
        "ci_high": ci.loc[terms, 1].to_numpy(),
        "p_value": result.pvalues[terms].to_numpy() if valid else np.nan,
        "fit_valid": valid,
        "converged": bool(result.converged),
        "optimizer": attempts[-1]["optimizer"],
        "fit_diagnostics": diagnostics,
        "n_obs": int(result.nobs),
        "n_subjects": df["subject"].nunique(),
    })
    return result, tidy


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
    feat_path = paths.derivatives / "features.parquet"
    if not feat_path.exists():
        raise SystemExit(f"{feat_path} not found -- run extract_features.py first.")
    features = pd.read_parquet(feat_path)

    tables = []
    for outcome in RATINGS:
        df = standardise(features, outcome, CORE_PREDICTORS)
        formula = f"{outcome} ~ " + " + ".join([*CORE_PREDICTORS, TIME])
        _, tidy = fit_lme(df, formula, "core")
        tables.append(tidy)
        print(f"\n=== {formula}  (random intercept per subject) ===")
        print(tidy[["term", "coef", "ci_low", "ci_high", "p_value"]]
              .round(4).to_string(index=False))
        print(f"    {tidy['n_obs'].iloc[0]} probes, "
              f"{tidy['n_subjects'].iloc[0]} subjects. Coefficients: rating "
              f"change per 1 SD within-session change.")

    out = pd.concat(tables, ignore_index=True)
    out_path = output_dir / "lme_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
