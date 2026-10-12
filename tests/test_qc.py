import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extract_features import exclude_poor_recordings
with TemporaryDirectory() as tmp:
    p = Path(tmp) / "qc.csv"
    manifest = pd.DataFrame({"recording":["a", "b"]})
    try: exclude_poor_recordings(manifest, p)
    except FileNotFoundError: pass
    else: raise AssertionError("Missing QC accepted")
    pd.DataFrame({"recording":["a", "b"], "n_eeg_channels":[16,64], "n_bad_channels":[5,5], "epoch_reject_rate":[0.1,0.1]}).to_csv(p,index=False)
    assert exclude_poor_recordings(manifest,p).recording.tolist() == ["b"]
print("ok: QC uses actual cap size and requires coverage")
