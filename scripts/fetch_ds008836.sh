#!/usr/bin/env bash
set -euo pipefail
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
WORKSPACE=$(dirname -- "$REPO")
PYTHON=${EEG_PYTHON:-"$WORKSPACE/venv/bin/python"}
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi
exec "$PYTHON" "$REPO/scripts/download_dataset.py" --dataset ds008836 --bids_root "${EEG_BIDS_ROOT:-$WORKSPACE/ds008836}" "$@"
