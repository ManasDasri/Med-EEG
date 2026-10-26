"""Phase 9, part A: introspective fidelity, as pre-registered in
docs/preregistration_phase9.md.

A person's fidelity = the within-session AUC of a model that never saw them
(phase 7's leave-one-person-out riemann_ridge predictions): how well their
self-reported depth lines up with what their EEG shows. Tests whether it is
stable across a person's two sessions (A1, A1b), whether it grows with
practice (A2), and compares it with a person-specific model (A3).

Usage:
    python src/fidelity.py --bids_root ds008836 \\
        --phase7 <phase 7 run dir> --output_dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANDOM_STATE  # noqa: E402
from model_comparison import shuffled_target  # noqa: E402

MODEL = "riemann_ridge"
N_PAIRING_PERMS = 10_000
N_RATING_SHIFTS = 1_000


def session_fidelity(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per scorable recording: subject, session, F, rating_sd, n_distinct."""
    rows = []
    for (subject, session), g in pred.groupby(["subject", "session"]):
        high = g["target"] > g["target"].mean()
        if high.nunique() < 2:
            continue  # every rating equal: no AUC
        rows.append({"subject": subject, "session": session,
                     "F": roc_auc_score(high, g["prediction"]),
                     "rating_sd": g["target"].std(), "n_distinct": g["target"].nunique()})
    return pd.DataFrame(rows)


def paired_sessions(fid: pd.DataFrame) -> pd.DataFrame:
    """People with two scorable sessions, one row each: <col>_1, <col>_2."""
    wide = fid.pivot(index="subject", columns="session")
    wide = wide.dropna()
    sessions = sorted(fid["session"].unique())
    wide.columns = [f"{c}_{sessions.index(s) + 1}" for c, s in wide.columns]
    return wide.reset_index()


def perm_spearman(x, y, n: int, rng: np.random.Generator) -> tuple[float, float]:
    """Spearman rho and its one-sided (rho > 0) permutation p."""
    x, y = rankdata(x), rankdata(y)
    rho = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n)])
    return float(rho), float((1 + (null >= rho).sum()) / (1 + n))


def partial_spearman(x, y, controls: np.ndarray) -> float:
    """Spearman correlation of x and y after regressing all ranks on the controls."""
    z = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in controls.T])
    resid = lambda v: rankdata(v) - z @ np.linalg.lstsq(z, rankdata(v), rcond=None)[0]
    return float(np.corrcoef(resid(x), resid(y))[0, 1])


def self_model_fidelity(df: pd.DataFrame, cols: list[str], alpha: float) -> pd.Series:
    """A3: per person, train on one session only, score the other; mean of both
    directions. Within-session AUC as for F."""
    out = {}
    for subject, g in df.groupby("subject"):
        if g["session"].nunique() < 2:
            continue
        aucs = []
        for s in g["session"].unique():
            train, test = g[g["session"] != s], g[g["session"] == s]
            high = test["depth_rating"] > test["depth_rating"].mean()
            if high.nunique() < 2 or train["depth_rating"].nunique() < 2:
                continue
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha))
            model.fit(train[cols], train["depth_rating"])
            aucs.append(roc_auc_score(high, model.predict(test[cols])))
        if aucs:
            out[subject] = float(np.mean(aucs))
    return pd.Series(out, name="F_self")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    parser.add_argument("--phase7", required=True, type=Path,
                        help="phase 7 run dir (features.parquet, band_covariances.npz, decoding/)")
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    pred = pd.read_csv(args.phase7 / "decoding" / "predictions_depth.csv")
    pred = pred[pred["model"] == MODEL].reset_index(drop=True)
    fid = session_fidelity(pred)
    pairs = paired_sessions(fid)
    fid.to_csv(args.output_dir / "fidelity_by_session.csv", index=False)
    rows = []

    # A1: stability across sessions.
    rho, p = perm_spearman(pairs["F_1"], pairs["F_2"], N_PAIRING_PERMS, rng)
    shifted = []
    for i in range(N_RATING_SHIFTS):
        y = shuffled_target(pred, "target", "state", np.random.default_rng(RANDOM_STATE + i))
        sp = paired_sessions(session_fidelity(pred.assign(target=y)))
        shifted.append(spearmanr(sp["F_1"], sp["F_2"])[0])
    shifted = np.array(shifted)
    rows.append({"test": "A1 stability: rho(F session 1, F session 2)", "estimate": rho,
                 "p_one_sided": p, "n": len(pairs),
                 "note": f"rating-shift null: mean {shifted.mean():+.3f}, "
                         f"95% {np.percentile(shifted, 2.5):+.3f} to {np.percentile(shifted, 97.5):+.3f}"})

    # A1b: controlling for rating style in both sessions.
    controls = pairs[["rating_sd_1", "n_distinct_1", "rating_sd_2", "n_distinct_2"]].to_numpy(float)
    rows.append({"test": "A1b stability, controlling rating SD and distinct ratings",
                 "estimate": partial_spearman(pairs["F_1"], pairs["F_2"], controls),
                 "n": len(pairs)})

    # A2: practice.
    traits = pd.read_csv(args.bids_root / "participants.tsv", sep="\t").rename(
        columns={"participant_id": "subject"})
    pooled = fid.groupby("subject")["F"].mean().rename("F_pooled").reset_index()
    person = pooled.merge(traits, on="subject", how="left")
    person["log_retreat_days"] = np.log1p(person["retreat_days"])
    a2 = [perm_spearman(person.dropna(subset=[c])["F_pooled"], person.dropna(subset=[c])[c],
                        N_PAIRING_PERMS, rng) + (c,)
          for c in ("log_retreat_days", "meditation_years_experience")]
    holm = multipletests([p for _, p, _ in a2], method="holm")[1]
    for (rho, p, c), ph in zip(a2, holm):
        rows.append({"test": f"A2 practice: rho(F, {c})", "estimate": rho, "p_one_sided": p,
                     "p_holm": ph, "n": int(person[c].notna().sum())})
    person["male"] = (person["sex"] == "male").astype(float)
    for c in ("log_retreat_days", "meditation_years_experience"):
        d = person.dropna(subset=[c, "age"])
        rows.append({"test": f"A2 exploratory: partial rho(F, {c} | age, sex)",
                     "estimate": partial_spearman(d["F_pooled"], d[c], d[["age", "male"]].to_numpy(float)),
                     "n": len(d)})
    for c in ("meditation_frequency", "meditation_length"):
        d = person.dropna(subset=[c])
        rows.append({"test": f"A2 exploratory: rho(F, {c})",
                     "estimate": spearmanr(d["F_pooled"], d[c])[0], "n": len(d)})

    # A3: a person-specific model, for interpretation.
    from explore_depth import load, ridge
    df, sets = load(args.phase7)
    cols = sets["riemann"]
    alpha = ridge(df).fit(df[cols], df["depth_rating"])[-1].alpha_
    f_self = self_model_fidelity(df, cols, alpha)
    person = person.merge(f_self, left_on="subject", right_index=True, how="left")
    d = person.dropna(subset=["F_self"])
    rows.append({"test": "A3 rho(F group model, F own-session model)",
                 "estimate": spearmanr(d["F_pooled"], d["F_self"])[0], "n": len(d),
                 "note": f"own-session model mean F {d['F_self'].mean():.3f}, alpha {alpha:g}"})
    for c in ("log_retreat_days", "meditation_years_experience"):
        e = d.dropna(subset=[c])
        rows.append({"test": f"A3 rho(F own-session model, {c})",
                     "estimate": spearmanr(e["F_self"], e[c])[0], "n": len(e)})

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "fidelity_results.csv", index=False)
    person.to_csv(args.output_dir / "fidelity_by_person.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    a1 = table.iloc[0]
    style = table.iloc[1]["estimate"] < a1["estimate"] / 2
    if shifted.mean() > 0.1 or shifted.mean() < -0.1:
        verdict = "A1 UNINTERPRETABLE: the rating-shift null is not centred on 0"
    elif a1["p_one_sided"] < 0.05 and not style:
        verdict = "A1 SUPPORTED: fidelity is stable across sessions"
    elif a1["p_one_sided"] < 0.05:
        verdict = "A1 passes, but A1b reads it as RATING STYLE, not introspection"
    else:
        verdict = "A1 NOT SUPPORTED: fidelity is not shown to be stable across sessions"
    print(f"\nRESULT: {verdict}")


if __name__ == "__main__":
    main()
