"""Step 6: bundle every result the pipeline produced into a single JSON file
that dashboard.html can load with a file picker (no server needed).

Usage:
    python src/06_export_dashboard_data.py --bids_root data/ds001787
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import Paths  # noqa: E402


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] {path.name} not found -- that section of the "
              f"dashboard will show 'no data' instead of erroring.")
        return None
    return pd.read_csv(path)


def clean_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe records (NaN -> None)."""
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


def _clean_scalar(v):
    """Convert a single raw pandas/numpy scalar to a JSON-safe value.
    pandas' NaN (float('nan')) is technically writable by Python's json
    module but is NOT valid JSON -- browsers reject it. Anything that
    reads a raw value straight from a DataFrame (not through
    clean_records) must pass through this first."""
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _sanitize_for_json(obj):
    """Recursively walk the final payload and replace any remaining NaN
    with None. This is a last-resort safety net -- individual builders
    above should already be clean, but a single missed NaN anywhere in
    the tree breaks the entire JSON file for the browser, so this makes
    the export fail-safe rather than fail-silent."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (obj != obj):  # NaN != NaN is always True
        return None
    return obj


def build_subjects_payload(features: pd.DataFrame,
                            participants: pd.DataFrame | None) -> list[dict]:
    if participants is not None and "participant_id" in participants.columns:
        # 03_features.py may already have merged 'group' in -- only pull
        # columns that aren't already present, so we never end up with a
        # group_x/group_y collision.
        wanted = ["group", "age", "gender"]
        to_merge = ["participant_id"] + [c for c in wanted
                                         if c in participants.columns
                                         and c not in features.columns]
        if len(to_merge) > 1:
            features = features.merge(
                participants[to_merge],
                left_on="subject", right_on="participant_id", how="left")

    subjects = []
    for sub, g in features.groupby("subject"):
        probes = g.sort_values("probe_onset_s") if "probe_onset_s" in g else g
        subjects.append({
            "subject": sub,
            "group": g["group"].iloc[0] if "group" in g.columns else None,
            "age": _clean_scalar(g["age"].iloc[0]) if "age" in g.columns else None,
            "gender": _clean_scalar(g["gender"].iloc[0]) if "gender" in g.columns else None,
            "mean_ratio": _safe_mean(g.get("minimal_ratio")),
            "mean_pac": _safe_mean(g.get("minimal_pac")),
            "mean_concentration": _safe_mean(g.get("concentration_rating")),
            "mean_mind_wandering": _safe_mean(g.get("mind_wandering_rating")),
            "probes": clean_records(probes[[
                c for c in ["probe_index", "session", "probe_onset_s",
                            "minimal_ratio", "minimal_pac",
                            "concentration_rating", "mind_wandering_rating",
                            "tiredness_rating"]
                if c in probes.columns
            ]]),
        })
    return subjects


def _safe_mean(series) -> float | None:
    if series is None:
        return None
    val = pd.to_numeric(series, errors="coerce").mean()
    return None if pd.isna(val) else round(float(val), 4)


def build_group_summary(features: pd.DataFrame,
                         participants: pd.DataFrame | None) -> dict:
    if "group" in features.columns:
        merged = features  # 03_features.py already merged it in
    elif participants is not None and "group" in participants.columns:
        merged = features.merge(
            participants[["participant_id", "group"]],
            left_on="subject", right_on="participant_id", how="left")
    else:
        return {}
    summary = {}
    for group_name, g in merged.groupby("group"):
        summary[str(group_name)] = {
            "n_probes": int(len(g)),
            "n_subjects": int(g["subject"].nunique()),
            "mean_ratio": _safe_mean(g.get("minimal_ratio")),
            "mean_pac": _safe_mean(g.get("minimal_pac")),
            "mean_concentration": _safe_mean(g.get("concentration_rating")),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None,
                         help="Output JSON path (default: "
                              "derivatives/dashboard_data.json)")
    args = parser.parse_args()

    paths = Paths.from_root(args.bids_root)
    feat_path = paths.derivatives / "features.parquet"
    if not feat_path.exists():
        raise SystemExit(f"{feat_path} not found -- run 03_features.py "
                          f"(and ideally 04/05) first.")
    features = pd.read_parquet(feat_path)

    participants_path = paths.bids_root / "participants.tsv"
    participants = (pd.read_csv(participants_path, sep="\t")
                     if participants_path.exists() else None)
    if participants is None:
        print("[warn] no participants.tsv found -- group/age/gender and "
              "the group-comparison panel will be empty in the dashboard.")

    # 05_ml_pipeline.py now writes task-specific filenames
    # (model_comparison_regress.csv / model_comparison_classify.csv) so a
    # --task regress run and a --task classify run don't overwrite each
    # other. Prefer the regress variant for the dashboard's headline
    # numbers (concentration prediction is this project's core question),
    # falling back to the old undifferentiated filename for compatibility
    # with earlier runs.
    model_comparison = safe_read_csv(
        paths.derivatives / "model_comparison_regress.csv")
    if model_comparison is None:
        model_comparison = safe_read_csv(
            paths.derivatives / "model_comparison_classify.csv")
    if model_comparison is None:
        model_comparison = safe_read_csv(
            paths.derivatives / "model_comparison.csv")
    electrode_curve = safe_read_csv(
        paths.derivatives / "electrode_minimalism_curve.csv")
    shap_path = paths.derivatives / "shap_importance_regress.csv"
    if not shap_path.exists():
        shap_path = paths.derivatives / "shap_importance_classify.csv"
    if not shap_path.exists():
        shap_path = paths.derivatives / "shap_importance.csv"
    shap_df = None
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path)
        shap_df.columns = ["feature", "importance"]

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "n_subjects": int(features["subject"].nunique()),
        "n_probes": int(len(features)),
        "subjects": build_subjects_payload(features, participants),
        "group_summary": build_group_summary(features, participants),
        "model_comparison": clean_records(model_comparison)
                            if model_comparison is not None else [],
        "electrode_minimalism": clean_records(electrode_curve)
                                if electrode_curve is not None else [],
        "shap_importance": clean_records(shap_df)
                           if shap_df is not None else [],
    }

    out_path = args.out or (paths.derivatives / "dashboard_data.json")
    out_path.write_text(json.dumps(_sanitize_for_json(payload), indent=2,
                                    allow_nan=False))
    print(f"[ok] wrote {out_path}")
    print("Open dashboard.html in a browser and load this file to view "
          "your results.")


if __name__ == "__main__":
    main()
