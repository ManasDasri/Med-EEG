"""Sequential, fail-fast pipeline. Outputs and full logs live in one run directory."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

SRC = Path(__file__).resolve().parents[1] / "src"


def run_stage(command, log):
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["ds001787", "ds008836"], default="ds001787")
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--derivatives", type=Path)
    parser.add_argument("--stages", default="preprocess,features,models,stats,export")
    parser.add_argument("--subjects")
    parser.add_argument("--n_permutations", type=int, default=100)
    args = parser.parse_args()
    if args.n_permutations < 0: parser.error("n_permutations must be nonnegative")
    run = args.run_dir or args.bids_root / "derivatives" / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True, exist_ok=True)
    intermediate = args.derivatives or run / "intermediates"
    common = ["--bids_root", str(args.bids_root.resolve()), "--derivatives", str(intermediate.resolve())]
    commands = {"preprocess": [("preprocess", ["--dataset", args.dataset] + (["--subjects", args.subjects] if args.subjects else []))],
                "features": [("extract_features", [])],
                "models": [("model_comparison", ["--task", t, "--n_permutations", str(args.n_permutations)]) for t in ("expertise", "concentration", "mind_wandering")],
                "stats": [("replication_ds008836", [])] if args.dataset == "ds008836" else [("mixed_effects_stats", []), ("followup_analysis", [])],
                "export": [("export_dashboard", ["--results_dir", str(run.resolve())])]}
    for stage in args.stages.split(","):
        if stage not in commands: parser.error(f"Unknown stage: {stage}")
        for name, extra in commands[stage]:
            key = name + ("_" + extra[1] if "--task" in extra else "")
            marker = run / (key + ".done")
            if marker.exists(): raise SystemExit(f"Refusing to overwrite completed stage: {key}")
            command = [sys.executable, str(SRC / (name + ".py")), *common, *extra]
            if stage in ("models", "stats"): command += ["--output_dir", str(run.resolve())]
            print(f"Running {key}; log: {run / (key + '.log')}", flush=True)
            run_stage(command, run / (key + ".log"))
            marker.write_text("success\n")
    print(f"PIPELINE FINISHED: {run}")


if __name__ == "__main__": main()
