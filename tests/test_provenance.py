import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from argparse import Namespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import Paths
from provenance import analysis_output, sha256
with TemporaryDirectory() as tmp:
    root = Path(tmp); f = root / "input"; f.write_text("data")
    args = Namespace(output_dir=root / "run")
    paths = Paths.from_root(root)
    out = analysis_output(paths, args, [f])
    assert sha256(f) in next(out.glob("*manifest.json")).read_text()
    try: analysis_output(paths, args, [f])
    except FileExistsError: pass
    else: raise AssertionError("Run overwritten")
print("ok: immutable provenance")
