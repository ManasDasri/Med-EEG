"""Step 11 (phase 7): decode self-rated meditation depth in ds008836.

Implements docs/analysis_plan_depth_decoding.md exactly: five EEG models
(primary: riemann_ridge) and two non-EEG baselines, each scored by
within-person AUC under leave-one-subject-out, with recording-aware
permutations and Holm correction across all of them. Scoring, permutations
and paired comparisons are the same code as model_comparison.py.

EEG features are normalised within each recording using only that
recording's EEG (never labels): z-scores for the spectral features, and
for the covariance matrices the matrix equivalent -- re-centring on the
recording's mean covariance before mapping to the tangent space.

Writes, in the run directory: model_comparison_depth.csv,
predictions_depth.csv, subject_scores_depth.csv, baseline_comparisons_depth.csv.

Usage:
    python src/depth_decoding.py --bids_root ds008836 \\
        --derivatives <dir with features.parquet + features_epochs.parquet>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANDOM_STATE, Paths  # noqa: E402
from model_comparison import (TASKS, evaluate, make_model,  # noqa: E402
                              mean_interval, paired_comparisons, prepare,
                              score, shuffled_target, spectral_columns,
                              subject_scores, zscore_within_recording)

TASK = "depth"
PRIMARY = "riemann_ridge"
PLUS_SUFFIXES = ("_aperiodic", "_iaf", "_alphapeak")
ALPHA = 0.05


def recording_keys(df: pd.DataFrame) -> pd.Series:
    if "recording" in df:
        return df["recording"].astype(str)
    return df["subject"].astype(str) + "/" + df["session"].astype(str)


# --- Riemannian features ----------------------------------------------------

def covariance_stack(df: pd.DataFrame) -> np.ndarray:
    """Rebuild each probe's symmetric covariance from its cov__a__b columns."""
    cols = [c for c in df.columns if c.startswith("cov__")]
    if not cols:
        raise SystemExit("no cov__ columns -- re-run extract_features.py")
    pairs = [c.split("__")[1:] for c in cols]
    names = list(dict.fromkeys(a for a, _ in pairs))
    index = {n: i for i, n in enumerate(names)}
    values = df[cols].to_numpy(dtype=float)
    covs = np.zeros((len(df), len(names), len(names)))
    for k, (a, b) in enumerate(pairs):
        covs[:, index[a], index[b]] = covs[:, index[b], index[a]] = values[:, k]
    return covs


def spd_function(mats: np.ndarray, fn) -> np.ndarray:
    """Apply fn to the eigenvalues of symmetric positive-definite matrices."""
    w, v = np.linalg.eigh(mats)
    return (v * fn(w)[..., None, :]) @ np.swapaxes(v, -1, -2)


def riemannian_mean(covs: np.ndarray, tol: float = 1e-10, max_iter: int = 50
                    ) -> np.ndarray:
    """Karcher (affine-invariant Riemannian) mean: the point where the
    tangent vectors of all matrices average to zero. Fixed-point iteration
    from the log-Euclidean mean, as in pyriemann's mean_riemann."""
    mean = spd_function(spd_function(covs, np.log).mean(axis=0), np.exp)
    for _ in range(max_iter):
        root = spd_function(mean, np.sqrt)
        inv_root = spd_function(mean, lambda w: w ** -0.5)
        step = spd_function(inv_root @ covs @ inv_root, np.log).mean(axis=0)
        mean = root @ spd_function(step, np.exp) @ root
        if np.linalg.norm(step) < tol:
            break
    return mean


def tangent_features(covs: np.ndarray, recordings: pd.Series) -> np.ndarray:
    """Re-centre each recording's covariances on that recording's Riemannian
    mean M (C -> M^-1/2 C M^-1/2), then take the matrix log: the tangent
    space at the identity, where each recording's vectors now average to 0.
    Uses only the recording's own EEG. Upper triangle, off-diagonal x sqrt(2)
    so Euclidean distance between vectors matches the matrix distance."""
    n = covs.shape[-1]
    rows, cols = np.triu_indices(n)
    weight = np.where(rows == cols, 1.0, np.sqrt(2.0))
    out = np.empty((len(covs), len(rows)))
    for rec in recordings.unique():
        m = (recordings == rec).to_numpy()
        whiten = spd_function(riemannian_mean(covs[m]), lambda w: w ** -0.5)
        logs = spd_function(whiten @ covs[m] @ whiten, np.log)
        out[m] = logs[:, rows, cols] * weight
    return out


# --- Segment-level model -------------------------------------------------------

def probe_key(df: pd.DataFrame) -> pd.Series:
    return recording_keys(df) + "#" + df["probe_index"].astype(str)


def epoch_predictions(probes: pd.DataFrame, epochs: pd.DataFrame,
                      features: list[str], target: pd.Series) -> np.ndarray:
    """Train on 2-s segments (each carrying its report's rating), leave one
    subject out, and average a report's segment predictions."""
    y = probe_key(epochs).map(pd.Series(target.to_numpy(), index=probe_key(probes)))
    keep = y.notna().to_numpy()
    X, y, subj = epochs.loc[keep, features], y[keep].to_numpy(), epochs.loc[keep, "subject"]
    keys = probe_key(epochs.loc[keep])
    pred = pd.Series(np.nan, index=keys.index)
    for s in subj.unique():
        test = (subj == s).to_numpy()
        model = make_model("epoch_ridge", "state").fit(X[~test], y[~test])
        pred[keys.index[test]] = model.predict(X[test])
    per_probe = pred.groupby(keys).mean()
    return probe_key(probes).map(per_probe).to_numpy()


def evaluate_epoch(probes: pd.DataFrame, epochs: pd.DataFrame,
                   features: list[str], n_permutations: int, n_jobs: int
                   ) -> tuple[dict, np.ndarray]:
    """Same outputs as model_comparison.evaluate, for the segment model."""
    target = TASKS[TASK][0]
    pred = epoch_predictions(probes, epochs, features, probes[target])
    result = score(probes, probes[target], pred, "state")

    def null_score(seed: int) -> float:
        y = shuffled_target(probes, target, "state", np.random.default_rng(seed))
        return score(probes, y, epoch_predictions(probes, epochs, features, y),
                     "state")["score"]

    null = np.array(Parallel(n_jobs=n_jobs)(
        delayed(null_score)(RANDOM_STATE + i) for i in range(n_permutations)))
    ci_low, ci_high = mean_interval(subject_scores(probes, probes[target], pred, "state")["score"])
    result.update({"ci_low": ci_low, "ci_high": ci_high, "model": "epoch_ridge",
                   "n_permutations": n_permutations, "n_features": len(features),
                   "null_method": "recording_circular_shift",
                   "chance": float(null.mean()) if n_permutations else np.nan,
                   "p_value": float((1 + (null >= result["score"]).sum())
                                    / (1 + n_permutations)) if n_permutations else np.nan,
                   "n_subjects": int(probes["subject"].nunique()),
                   "n_probes": len(probes)})
    return result, pred


# --- Main ------------------------------------------------------------------------

def verdict(summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    """The plan's success criterion, applied to the primary model only."""
    row = summary.set_index("model").loc[PRIMARY]
    baselines = summary[summary["model"].str.startswith("baseline_")]
    best_base = baselines.loc[baselines["score"].idxmax(), "model"]
    vs = paired[(paired["model"] == PRIMARY) & (paired["baseline"] == best_base)]
    beats_chance = bool(np.isfinite(row["p_holm_task"]) and row["p_holm_task"] < ALPHA)
    beats_base = bool(len(vs) and vs["ci_low"].iloc[0] > 0)
    if beats_chance and beats_base:
        return f"DECODED: {PRIMARY} beats chance (Holm p < {ALPHA}) and {best_base}"
    return (f"NOT DECODED by the plan's criterion: {PRIMARY} "
            f"{'beats' if beats_chance else 'does not beat'} chance after Holm, "
            f"{'beats' if beats_base else 'does not beat'} {best_base}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    parser.add_argument("--derivatives", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--n_permutations", type=int, default=200)
    parser.add_argument("--n_jobs", type=int, default=2)
    args = parser.parse_args()
    if args.n_permutations < 0:
        parser.error("n_permutations must be nonnegative")

    paths = Paths.from_root(args.bids_root, args.derivatives)
    feat_path = paths.derivatives / "features.parquet"
    epoch_path = paths.derivatives / "features_epochs.parquet"
    from provenance import analysis_output
    output_dir = analysis_output(paths, args, [feat_path, epoch_path])

    target = TASKS[TASK][0]
    df = prepare(pd.read_parquet(feat_path), TASK)
    plus = [c for c in df.columns if c.endswith(PLUS_SUFFIXES)]
    df = zscore_within_recording(df, plus)
    tangent = tangent_features(covariance_stack(df), recording_keys(df))
    tan_cols = [f"tan_{i}" for i in range(tangent.shape[1])]
    df = pd.concat([df.drop(columns=[c for c in df.columns if c.startswith("cov__")]),
                    pd.DataFrame(tangent, columns=tan_cols, index=df.index)], axis=1)
    df["is_tones"] = (df["block_type"] == "sapo").astype(float)
    df["is_prompted"] = (df["variant"] == "probe").astype(float)

    epochs = pd.read_parquet(epoch_path)
    epochs = epochs[probe_key(epochs).isin(set(probe_key(df)))].reset_index(drop=True)
    spectral = spectral_columns(df)
    epochs = zscore_within_recording(epochs, spectral)

    feature_sets = {
        "linear_spectral": spectral,
        "linear_spectral_plus": spectral + plus,
        "boosting_spectral_plus": spectral + plus,
        "riemann_ridge": tan_cols,
        "baseline_time": ["probe_onset_s"],
        "baseline_context": ["probe_onset_s", "is_tones", "is_prompted"],
    }
    print(f"[ok] depth: {len(df)} reports, {df['subject'].nunique()} people, "
          f"{len(epochs)} segments, {len(tan_cols)} tangent features, "
          f"{args.n_permutations} permutations per model")

    results, preds = [], {}
    for name, cols in feature_sets.items():
        result, preds[name] = evaluate(df, TASK, name, cols, args.n_permutations, args.n_jobs)
        results.append(result)
        print(f"  {name:24s} AUC {result['score']:.3f}  (chance {result['chance']:.3f}, "
              f"p={result['p_value']:.3f})", flush=True)
    result, preds["epoch_ridge"] = evaluate_epoch(df, epochs, spectral,
                                                  args.n_permutations, args.n_jobs)
    results.append(result)
    print(f"  {'epoch_ridge':24s} AUC {result['score']:.3f}  (chance {result['chance']:.3f}, "
          f"p={result['p_value']:.3f})", flush=True)

    summary = pd.DataFrame(results)
    summary["p_holm_task"] = np.nan
    tested = summary["p_value"].notna()
    if tested.any():
        summary.loc[tested, "p_holm_task"] = multipletests(
            summary.loc[tested, "p_value"], method="holm")[1]
    summary["multiplicity_family"] = "all models including baselines for depth"
    summary["evaluation_scope"] = ("retrospective LOSO; full-recording EEG "
                                   "normalization and preprocessing")
    summary["primary"] = summary["model"] == PRIMARY

    frames, per_subject = [], []
    for name, prediction in preds.items():
        frame = df[["subject", "session", "probe_index"]].copy()
        frame["recording"] = recording_keys(df)
        frame["target"], frame["prediction"], frame["model"] = df[target], prediction, name
        frames.append(frame)
        per_subject.append(subject_scores(df, df[target], prediction, "state").assign(model=name))
    scores = pd.concat(per_subject, ignore_index=True)
    paired = paired_comparisons(scores)

    summary.to_csv(output_dir / "model_comparison_depth.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(output_dir / "predictions_depth.csv", index=False)
    scores.to_csv(output_dir / "subject_scores_depth.csv", index=False)
    paired.to_csv(output_dir / "baseline_comparisons_depth.csv", index=False)
    print(f"\nRESULT: {verdict(summary, paired)}")
    print(f"[ok] wrote results to {output_dir}")


if __name__ == "__main__":
    main()
