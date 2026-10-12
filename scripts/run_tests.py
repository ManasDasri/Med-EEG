"""Run every standalone regression check; fail at the first failed script."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="medeeg-tests-") as cache:
    env = dict(os.environ, MPLCONFIGDIR=cache, NUMBA_CACHE_DIR=cache,
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", LOKY_MAX_CPU_COUNT="2")
    for test in sorted((root / "tests").glob("test_*.py")):
        print(f"Running {test.name}", flush=True)
        subprocess.run([sys.executable, str(test)], cwd=root, env=env, check=True)
print("All regression scripts passed")
