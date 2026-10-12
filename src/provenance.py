"""Small immutable analysis manifests; raw data stays outside Git."""
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def analysis_output(paths, args, inputs):
    stage = Path(__import__("sys").argv[0]).stem
    task = getattr(args, "task", None)
    name = stage + ("_" + task if task else "")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output_dir or paths.derivatives / "runs" / (name + "_" + stamp)
    out.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
            cwd=Path(__file__).parent, text=True).strip())
    except subprocess.CalledProcessError:
        revision, dirty = None, None
    import config
    payload = {"configuration": {k: v for k, v in vars(config).items() if k.isupper()},"stage": name, "created_at": stamp, "code_revision": revision,
        "dirty_checkout": dirty, "arguments": vars(args),
        "inputs": {str(Path(p).resolve()): sha256(p) for p in inputs},
        "dataset_description": (paths.bids_root / "dataset_description.json").read_text()
            if (paths.bids_root / "dataset_description.json").exists() else None,
        "packages": {p: importlib.metadata.version(p) for p in
            ("mne", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels")}}
    # Refuse accidental reuse, including smoke runs overwriting final results.
    with (out / (name + "_manifest.json")).open("x") as stream:
        json.dump(payload, stream, default=str, indent=2)
    return out
