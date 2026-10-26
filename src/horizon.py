"""Phase 9, part B: the lead-up horizon, as pre-registered in
docs/preregistration_phase9.md.

How far before a report does the EEG carry the depth that will be
reported? Each report's window is split into bins by how long before the
report each 2-s segment starts; a covariance model is trained and scored
per bin (B1, B2), and separately for spontaneous vs prompted reports (B3).

Step 1 (needs the epoch files) writes <phase7>/bin_covariances.npz;
step 2 decodes. Usage:
    python src/horizon.py --phase7 <phase 7 run dir> --output_dir <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp, wilcoxon
from sklearn.covariance import oas
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANDOM_STATE  # noqa: E402
from depth_decoding import recording_keys, tangent_features  # noqa: E402
from explore_depth import ALPHA_GRID  # noqa: E402
from model_comparison import prepare, shuffled_target, subject_scores  # noqa: E402

BINS = {"B1": (0, 10), "B2": (10, 30), "B3": (30, 60), "B4": (60, 120)}  # seconds before report
MIN_SEGMENTS = 3
TARGET = "depth_rating"


# --- Step 1: one covariance per report per bin --------------------------------

def bin_covariances(data: np.ndarray, seconds_before: np.ndarray) -> np.ndarray:
    """(n_epochs, n_ch, n_times) -> (n_bins, n_ch, n_ch); NaN where a bin has
    fewer than MIN_SEGMENTS segments."""
    n = data.shape[1]
    out = np.full((len(BINS), n, n), np.nan)
    for b, (lo, hi) in enumerate(BINS.values()):
        m = (seconds_before > lo) & (seconds_before <= hi)
        if m.sum() >= MIN_SEGMENTS:
            out[b] = oas(data[m].transpose(1, 0, 2).reshape(n, -1).T)[0]
    return out


def one_probe(path: str) -> tuple[np.ndarray, list[str]]:
    epochs = mne.read_epochs(path, preload=True, verbose="error")
    picks = mne.pick_types(epochs.info, eeg=True)
    return (bin_covariances(epochs.get_data(picks="eeg"),
                            epochs.metadata["seconds_before_probe"].to_numpy()),
            [epochs.ch_names[i] for i in picks])


def extract(phase7: Path, n_jobs: int) -> Path:
    out = phase7 / "bin_covariances.npz"
    if out.exists():
        return out
    labels = pd.read_csv(phase7 / "probe_labels.csv")
    results = Parallel(n_jobs=n_jobs)(delayed(one_probe)(p) for p in labels["epochs_file"])
    if any(r[1] != results[0][1] for r in results):
        raise SystemExit("channel order differs between probes")
    np.savez(out, covs=np.stack([r[0] for r in results]),
             recording=labels["recording"].to_numpy(str),
             probe_index=labels["probe_index"].to_numpy(),
             channels=np.array(results[0][1]))
    return out


# --- Fast exact LOSO ridge for many targets at once --------------------------

def _kernel_parts(X_tr: np.ndarray, X_te: np.ndarray):
    """Standardise on the training rows; return eigendecomposition of the
    training kernel and the test-by-train kernel projected onto it."""
    mu, sd = X_tr.mean(0), X_tr.std(0)
    sd[sd == 0] = 1
    A, B = (X_tr - mu) / sd, (X_te - mu) / sd
    lam, U = np.linalg.eigh(A @ A.T)
    return lam, U, (B @ A.T) @ U


def _predict(parts, Y_tr: np.ndarray, alpha: float) -> np.ndarray:
    lam, U, KU = parts
    mean = Y_tr.mean(0)
    return mean + KU @ ((U.T @ (Y_tr - mean)) / (np.maximum(lam, 0) + alpha)[:, None])


def loso_ridge(X: np.ndarray, Y: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Leave-one-subject-out ridge predictions for every column of Y at once.
    Alpha per outer fold and per column, by 5-fold subject-grouped CV (mean
    R^2) over ALPHA_GRID -- the phase 8 rule. Exact ridge with intercept,
    computed in kernel form: each fold is decomposed once for all columns."""
    pred = np.full(Y.shape, np.nan)
    for tr, te in LeaveOneGroupOut().split(X, groups=subjects):
        r2 = np.zeros((len(ALPHA_GRID), Y.shape[1]))
        for itr, iva in GroupKFold(5).split(X[tr], groups=subjects[tr]):
            parts = _kernel_parts(X[tr][itr], X[tr][iva])
            yv = Y[tr][iva]
            ss = ((yv - yv.mean(0)) ** 2).sum(0)
            ss[ss == 0] = 1
            for a, alpha in enumerate(ALPHA_GRID):
                r2[a] += 1 - ((yv - _predict(parts, Y[tr][itr], alpha)) ** 2).sum(0) / ss
        best = np.array(ALPHA_GRID)[r2.argmax(0)]
        parts = _kernel_parts(X[tr], X[te])
        for alpha in np.unique(best):
            cols = best == alpha
            pred[np.ix_(te, cols)] = _predict(parts, Y[tr][:, cols], alpha)
    return pred


# --- Step 2: decoding ------------------------------------------------------------

def within_auc(df: pd.DataFrame, y, pred) -> float:
    s = subject_scores(df, y, pred, "state")["score"]
    return float(s.mean())


def decode(df: pd.DataFrame, X: np.ndarray, n_perm: int) -> dict:
    """Real + permuted LOSO predictions in one pass; per-person AUCs."""
    shifts = [shuffled_target(df, TARGET, "state", np.random.default_rng(RANDOM_STATE + i))
              for i in range(n_perm)]
    Y = np.column_stack([df[TARGET].to_numpy()] + [s.to_numpy() for s in shifts])
    P = loso_ridge(X, Y, df["subject"].to_numpy())
    auc = within_auc(df, df[TARGET], P[:, 0])
    null = np.array([within_auc(df, Y[:, k], P[:, k]) for k in range(1, Y.shape[1])])
    per = subject_scores(df, df[TARGET], P[:, 0], "state").set_index("subject")["score"]
    return {"auc": auc, "chance": float(null.mean()) if n_perm else np.nan,
            "p": float((1 + (null >= auc).sum()) / (1 + n_perm)) if n_perm else np.nan,
            "per_person": per, "n_reports": len(df), "n_people": df["subject"].nunique()}


def bin_frames(df: pd.DataFrame, covs: np.ndarray) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Per bin: the reports that have it, and their re-centred tangent vectors."""
    out = {}
    for b, name in enumerate(BINS):
        ok = ~np.isnan(covs[:, b, 0, 0])
        sub = df[ok].reset_index(drop=True)
        out[name] = (sub, tangent_features(covs[ok, b], recording_keys(sub)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase7", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n_permutations", type=int, default=200)
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(extract(args.phase7, args.n_jobs))
    df = prepare(pd.read_parquet(args.phase7 / "features.parquet",
                                 columns=None), "depth")
    df["recording"] = recording_keys(df)
    df = df.drop(columns=[c for c in df.columns if c.startswith("cov__")])
    where = {(r, int(p)): i for i, (r, p) in enumerate(zip(npz["recording"], npz["probe_index"]))}
    covs = npz["covs"][[where[(r, int(p))] for r, p in zip(df["recording"], df["probe_index"])]]
    print(f"[ok] {len(df)} reports; per bin: " + ", ".join(
        f"{n} {(~np.isnan(covs[:, b, 0, 0])).sum()}" for b, n in enumerate(BINS)), flush=True)

    # B1 + B2: all reports.
    rows, per = [], {}
    for name, (sub, X) in bin_frames(df, covs).items():
        r = decode(sub, X, args.n_permutations)
        per[name] = r.pop("per_person")
        rows.append({"analysis": "all reports", "bin": name, **r})
        print(f"  {name} {BINS[name]} s: AUC {r['auc']:.3f} (chance {r['chance']:.3f}, "
              f"p={r['p']:.3f}, {r['n_reports']} reports)", flush=True)
    table = pd.DataFrame(rows)
    table["p_holm"] = multipletests(table["p"], method="holm")[1]

    wide = pd.DataFrame(per).dropna()
    slopes = np.array([np.polyfit(np.arange(1, 5), row.to_numpy(), 1)[0]
                       for _, row in wide[list(BINS)].iterrows()])
    t = ttest_1samp(slopes, 0, alternative="less")

    # B3: spontaneous vs prompted, separate models per type.
    gain = {}
    for variant, label in (("emerge", "spontaneous"), ("probe", "prompted")):
        sub_per = {}
        for name, (sub, X) in bin_frames(df[df["variant"] == variant].reset_index(drop=True),
                                         covs[(df["variant"] == variant).to_numpy()]).items():
            if name not in ("B1", "B4"):
                continue
            r = decode(sub, X, 0)
            sub_per[name] = r["per_person"]
            rows.append({"analysis": label, "bin": name, "auc": r["auc"],
                         "n_reports": r["n_reports"], "n_people": r["n_people"]})
        both = pd.DataFrame(sub_per).dropna()
        gain[label] = both["B1"] - both["B4"]
    g = pd.DataFrame(gain).dropna()
    w = wilcoxon(g["spontaneous"], g["prompted"], alternative="greater")

    table = pd.DataFrame(rows).merge(table[["analysis", "bin", "p_holm"]], how="left",
                                     on=["analysis", "bin"])
    table.to_csv(args.output_dir / "horizon_results.csv", index=False)
    wide.assign(slope=slopes).to_csv(args.output_dir / "horizon_per_person.csv")
    tests = pd.DataFrame([
        {"test": "B2 per-person AUC slope over bins (one-sided < 0)", "estimate": slopes.mean(),
         "p": t.pvalue, "n": len(slopes)},
        {"test": "B3 near-minus-far gain, spontaneous minus prompted (one-sided > 0)",
         "estimate": float((g["spontaneous"] - g["prompted"]).median()), "p": w.pvalue, "n": len(g)},
    ])
    tests.to_csv(args.output_dir / "horizon_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(tests.to_string(index=False))
    b4 = table[(table["analysis"] == "all reports") & (table["bin"] == "B4")].iloc[0]
    print(f"\nRESULT: B1 {'SUPPORTED' if b4['p_holm'] < 0.05 else 'NOT SUPPORTED'}: depth "
          f"{'is' if b4['p_holm'] < 0.05 else 'is not shown to be'} decodable 60-120 s before "
          f"the report (AUC {b4['auc']:.3f}, Holm p {b4['p_holm']:.3f})")


if __name__ == "__main__":
    main()
