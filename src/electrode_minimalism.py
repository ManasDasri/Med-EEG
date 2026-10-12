"""Electrode minimalism: how few electrodes keep the full-scalp signal?

Uses the gamma/theta ratio of each electrode (ratio_<channel> from
extract_features.py), z-scored within each recording. For every held-out
person (leave-one-subject-out), electrodes are ranked ON THE OTHER PEOPLE
ONLY by how well their ratio tracks the rating within each person; the top-k
(sign-aligned) are averaged into one score for the held-out person.
Ranking electrodes on everyone would leak the answer into the test.

The predefined clusters (minimal = Fz+Pz, frontal-theta, posterior-gamma)
need no selection and are scored the same way. Every score gets a chance
level and p-value from ratings shuffled within each person.

Writes electrode_minimalism_<task>.csv (used by the dashboard).

Usage:
    python src/electrode_minimalism.py --bids_root data/ds001787 --task mind_wandering
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import N_PERMUTATIONS, PAC_CLUSTERS, RANDOM_STATE, Paths  # noqa: E402
from model_comparison import TASKS, prepare, zscore_within_recording  # noqa: E402

N_ELECTRODES = [1, 2, 4, 8, 16, 32, 64]


def within_person_auc(X: np.ndarray, y: np.ndarray, subjects: np.ndarray
                      ) -> np.ndarray:
    """Same metric as model_comparison.score, for every column of X at once:
    per person, the AUC of each column for separating their above-own-
    average ratings from the rest (via ranks), averaged over people."""
    aucs = []
    for s in np.unique(subjects):
        m = subjects == s
        high = y[m] > y[m].mean()
        n1, n0 = high.sum(), (~high).sum()
        if n1 == 0 or n0 == 0:
            continue
        ranks = rankdata(X[m], axis=0)
        aucs.append((ranks[high].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0))
    return np.mean(aucs, axis=0)


def loso_top_k(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
               ks: list[int]) -> dict[int, np.ndarray]:
    """Out-of-fold scores for each k: electrodes ranked and sign-aligned on
    the training people only, then the top-k averaged for the test person."""
    preds = {k: np.zeros(len(y)) for k in ks}
    for s in np.unique(subjects):
        test = subjects == s
        auc = within_person_auc(X[~test], y[~test], subjects[~test])
        order = np.argsort(-np.abs(auc - 0.5))
        sign = np.sign(auc - 0.5)
        for k in ks:
            top = order[:k]
            preds[k][test] = (X[test][:, top] * sign[top]).mean(axis=1)
    return preds


def curve(X: np.ndarray, y: np.ndarray, subjects: np.ndarray,
          clusters: dict[str, list[int]]) -> dict[str, float]:
    """Score of every top-k set and every predefined cluster."""
    ks = [k for k in N_ELECTRODES if k <= X.shape[1]]
    scores = {f"top_{k}": within_person_auc(p[:, None], y, subjects)[0]
              for k, p in loso_top_k(X, y, subjects, ks).items()}
    for name, cols in clusters.items():
        cluster_x = X[:, cols].mean(axis=1, keepdims=True)
        p = loso_top_k(cluster_x, y, subjects, [1])[1]
        scores[name] = within_person_auc(p[:, None], y, subjects)[0]
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    parser.add_argument("--task", choices=[t for t, (_, kind) in TASKS.items()
                                           if kind == "state"], required=True)
    parser.add_argument("--n_permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="New result directory; defaults to a unique run directory")
    parser.add_argument("--derivatives", type=Path, default=None,
                        help="Explicit intermediate directory (preserve prior runs)")
    args = parser.parse_args()

    paths = Paths.from_root(args.bids_root, args.derivatives)
    from provenance import analysis_output
    input_path = getattr(args, "features", None) or paths.derivatives / "features.parquet"
    output_dir = analysis_output(paths, args, [input_path])
    df = prepare(pd.read_parquet(paths.derivatives / "features.parquet"),
                 args.task)
    channels = [c[len("ratio_"):] for c in df.columns if c.startswith("ratio_")]
    if not channels:
        raise SystemExit("no ratio_<channel> columns -- re-run "
                         "extract_features.py")
    df = zscore_within_recording(df, [f"ratio_{c}" for c in channels])
    X = df[[f"ratio_{c}" for c in channels]].fillna(0).to_numpy()
    y = df[TASKS[args.task][0]].to_numpy()
    subjects = df["subject"].to_numpy()
    clusters = {name: [channels.index(c) for c in chs if c in channels]
                for name, chs in PAC_CLUSTERS.items()}

    real = curve(X, y, subjects, clusters)

    def shuffled(seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        y_perm = pd.Series(y).groupby(subjects).transform(
            lambda r: rng.permutation(r.to_numpy())).to_numpy()
        return curve(X, y_perm, subjects, clusters)

    null = pd.DataFrame(Parallel(n_jobs=-1)(
        delayed(shuffled)(RANDOM_STATE + i) for i in range(args.n_permutations)))

    # Which electrodes the top sets pick when ranked on everyone. Descriptive
    # only -- the scores above never see the test person during ranking.
    overall = within_person_auc(X, y, subjects)
    ranked = [channels[i] for i in np.argsort(-np.abs(overall - 0.5))]
    full = real["top_64"] - 0.5
    rows = []
    for name, score in real.items():
        n = int(name[4:]) if name.startswith("top_") else len(clusters[name])
        rows.append({
            "electrode_set": name,
            "n_electrodes": n,
            "score": score,
            "chance": null[name].mean(),
            "p_value": (1 + (null[name] >= score).sum()) / (1 + len(null)),
            "share_of_full_scalp": (score - 0.5) / full if full > 0 else np.nan,
            "electrodes": " ".join(ranked[:n]) if name.startswith("top_")
            else " ".join(PAC_CLUSTERS[name]),
        })
    out = pd.DataFrame(rows)
    out_path = output_dir / f"electrode_minimalism_{args.task}.csv"
    out.to_csv(out_path, index=False)
    print(f"=== electrode minimalism: {args.task}, within-person AUC, "
          f"{args.n_permutations} permutations ===")
    print(out.drop(columns="electrodes").round(3).to_string(index=False))
    print(f"\n[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
