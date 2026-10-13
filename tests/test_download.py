import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
p = Path(__file__).resolve().parents[1] / "scripts/download_dataset.py"
spec = importlib.util.spec_from_file_location("download", p)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with TemporaryDirectory() as tmp:
    root = Path(tmp); target = root / "file"; target.write_bytes(b"old")
    with patch.object(m, "urlopen", side_effect=OSError("network failed")):
        try: m.download(root, "ds008836", {"path":"file", "size":4, "etag":"x"}, attempts=1)
        except OSError: pass
        else: raise AssertionError("Download failure swallowed")
    assert target.read_bytes() == b"old"
    try: m.download(root, "ds008836", {"path":"../escape", "size":4, "etag":"x"})
    except ValueError: pass
    else: raise AssertionError("Path traversal accepted")
print("ok: failed downloads preserve files and reject unsafe paths")
