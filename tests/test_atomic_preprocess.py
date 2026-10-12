import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocess import atomic_text, replace_recording
old = [{"subject":"s", "session":"1", "recording":"run-a", "value":1},
       {"subject":"s", "session":"1", "recording":"run-b", "value":2}]
assert replace_recording(old, [], old[0]) == [old[1]]
with TemporaryDirectory() as tmp:
    p = Path(tmp) / "manifest.csv"; p.write_text("good")
    with patch.object(Path, "replace", side_effect=OSError("disk error")):
        try: atomic_text(p, "bad")
        except OSError: pass
    assert p.read_text() == "good"
    atomic_text(p, "new"); assert p.read_text() == "new"
print("ok: recording replacement and atomic writes")
