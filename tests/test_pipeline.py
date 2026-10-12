import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
p = Path(__file__).resolve().parents[1] / "scripts/run_pipeline.py"
spec = importlib.util.spec_from_file_location("runner", p)
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
with TemporaryDirectory() as tmp:
    log = Path(tmp) / "stage.log"
    try: runner.run_stage([sys.executable, "-c", "print('failure detail'); raise SystemExit(3)"], log)
    except subprocess.CalledProcessError as exc: assert exc.returncode == 3
    else: raise AssertionError("Failure swallowed")
    assert "failure detail" in log.read_text()
print("ok: stage failures propagate with complete logs")
