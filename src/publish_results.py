"""Step 10: bundle finished analysis runs into one small JSON file that the
public results dashboard fetches live (results/latest/results.json).

Only summary tables go in: model scores with chance levels and p-values,
per-person scores, paired baseline comparisons, the replication table with
its verdict, and recording QC summaries -- plus each source run's
provenance (code revision, time, permutation count). No raw EEG, no
per-probe data, no participant attributes beyond anonymised subject IDs.

Usage:
    python src/publish_results.py \\
        --models_run   ../audit-work/evaluation/dashboard-run-20261013 \\
        --replication_run ../audit-work/evaluation/ds008836-amended-replication \\
        --qc ds001787=../audit-work/evaluation/ds001787/preprocessing_qc.csv \\
        --qc ds008836=../audit-work/evaluation/ds008836/preprocessing_qc.csv \\
        --depth_run ../audit-work/evaluation/ds008836-phase7/decoding
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_comparison import TASKS  # noqa: E402
from replication_ds008836 import verdict  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ALPHA = 0.05
LABELS = {
    "interpretable_ratio_pac": "Interpretable (γ/θ ratio + PAC)",
    "linear_spectral": "Linear, 67 spectral features",
    "boosting_spectral": "Gradient boosting, 67 spectral features",
    "baseline_sex": "Sex only (no EEG)",
    "baseline_age": "Age only (no EEG)",
    "baseline_time": "Time into session (no EEG)",
    "linear_spectral_plus": "Linear, spectral + 1/f + alpha peak",
    "boosting_spectral_plus": "Gradient boosting, spectral + 1/f + alpha peak",
    "riemann_ridge": "Riemannian, 64-channel covariance",
    "epoch_ridge": "Linear, per 2-s segment",
    "baseline_context": "Time + block + report type (no EEG)",
}
MANIFESTS = {"depth": "depth_decoding_manifest.json"}  # default: model_comparison_<task>


def clean(value):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {pattern} under {root}, "
                         f"found {len(matches)}")
    return matches[0]


def provenance(manifest_path: Path) -> dict:
    m = json.loads(manifest_path.read_text())
    return {"stage": m.get("stage"), "created_at": m.get("created_at"),
            "code_revision": m.get("code_revision"),
            "dirty_checkout": m.get("dirty_checkout"),
            "n_permutations": (m.get("arguments") or {}).get("n_permutations"),
            "packages": m.get("packages")}


def task_bundle(root: Path, task: str) -> dict:
    comparison = pd.read_csv(find_one(root, f"model_comparison_{task}.csv"))
    subjects = pd.read_csv(find_one(root, f"subject_scores_{task}.csv"))
    paired = pd.read_csv(find_one(root, f"baseline_comparisons_{task}.csv"))
    source = provenance(find_one(root, MANIFESTS.get(task, f"model_comparison_{task}_manifest.json")))

    p_col = "p_holm_task" if comparison["p_holm_task"].notna().any() else "p_value"
    models = [{
        "model": r.model, "label": LABELS.get(r.model, r.model),
        "kind": "baseline" if r.model.startswith("baseline_") else "eeg",
        "score": r.score, "ci_low": r.ci_low, "ci_high": r.ci_high,
        "chance": r.chance, "p_value": r.p_value, "p_corrected": getattr(r, p_col),
        "n_subjects": r.n_subjects, "n_probes": r.n_probes,
    } for r in comparison.itertuples()]

    eeg = [m for m in models if m["kind"] == "eeg"]
    baselines = [m for m in models if m["kind"] == "baseline"]
    best = max(eeg, key=lambda m: m["score"])
    best_base = max(baselines, key=lambda m: m["score"]) if baselines else None
    vs_base = paired[(paired["model"] == best["model"])
                     & (paired["baseline"] == (best_base or {}).get("model"))]
    beats_chance = (best["p_corrected"] is not None
                    and pd.notna(best["p_corrected"]) and best["p_corrected"] < ALPHA)
    beats_baseline = bool(len(vs_base) and vs_base["ci_low"].iloc[0] > 0)
    return {
        "task": task, "kind": TASKS[task][1], "metric": comparison["metric"].iloc[0],
        "chance_reference": 0.5, "p_column": p_col,
        "multiplicity": comparison.get("multiplicity_family", pd.Series([None])).iloc[0],
        "scope": comparison.get("evaluation_scope", pd.Series([None])).iloc[0],
        "models": models,
        "subject_scores": {m: g[["subject", "score"]].to_dict("records")
                           for m, g in subjects.groupby("model")},
        "baseline_comparisons": paired.drop(columns=["interval_scope"],
                                            errors="ignore").to_dict("records"),
        "headline": {"best_model": best["model"], "best_score": best["score"],
                     "best_baseline": (best_base or {}).get("model"),
                     "best_baseline_score": (best_base or {}).get("score"),
                     "beats_chance": bool(beats_chance),
                     "beats_best_baseline": beats_baseline},
        "source": source,
    }


def replication_bundle(root: Path) -> dict:
    table = pd.read_csv(find_one(root, "replication_results.csv"))
    rows = table.drop(columns=["fit_diagnostics"], errors="ignore")
    return {"verdict": verdict(table.iloc[0], table.iloc[1]),
            "reference_estimate": -0.21, "alpha_one_sided": ALPHA,
            "rows": rows.to_dict("records"),
            "source": provenance(find_one(root, "replication_ds008836_manifest.json"))}


def qc_bundle(path: Path) -> dict:
    qc = pd.read_csv(path)
    key = "recording" if "recording" in qc else ["subject", "session"]
    if qc.duplicated(key).any():
        before = len(qc)
        qc = qc.drop_duplicates(key, keep="last")
        print(f"[warn] {path}: {before - len(qc)} duplicate QC rows "
              f"dropped (same recording listed more than once)")
    n_eeg = qc["n_eeg_channels"] if "n_eeg_channels" in qc else 64
    excluded = (qc["n_bad_channels"] / n_eeg > 0.25) | (qc["epoch_reject_rate"] > 0.5)
    return {"recordings": len(qc), "subjects": qc["subject"].nunique(),
            "probes_kept": int(qc["n_probes_kept"].sum()),
            "epochs_kept": int(qc["n_epochs_kept"].sum()),
            "median_reject_rate": float(qc["epoch_reject_rate"].median()),
            "median_bad_channels": float(qc["n_bad_channels"].median()),
            "excluded_recordings": qc.loc[excluded, "recording"].tolist()
            if "recording" in qc else int(excluded.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_run", required=True, type=Path)
    parser.add_argument("--replication_run", required=True, type=Path)
    parser.add_argument("--qc", action="append", default=[],
                        help="name=path/to/preprocessing_qc.csv (repeatable)")
    parser.add_argument("--depth_run", type=Path, default=None,
                        help="depth_decoding.py run dir (ds008836); depth is skipped without it")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "latest")
    args = parser.parse_args()

    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    bundle = {
        "schema": 1,
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "publisher_revision": revision,
        "tasks": {t: task_bundle(args.depth_run if t == "depth" else args.models_run, t)
                  for t in TASKS if t != "depth" or args.depth_run},
        "replication": replication_bundle(args.replication_run),
        "qc": {name: qc_bundle(Path(path))
               for name, path in (q.split("=", 1) for q in args.qc)},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "results.json"
    out.write_text(json.dumps(clean(bundle), indent=1, allow_nan=False))
    print(f"[ok] wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    for t, b in bundle["tasks"].items():
        h = b["headline"]
        print(f"  {t:15s} best {h['best_model']} {h['best_score']:.3f}; "
              f"beats chance: {h['beats_chance']}; beats {h['best_baseline']}: "
              f"{h['beats_best_baseline']}")
    print(f"  replication: {bundle['replication']['verdict']}")


if __name__ == "__main__":
    main()
