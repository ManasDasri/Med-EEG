import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inspect_dataset import event_inventory
first = event_inventory(pd.DataFrame({"trial_type":["STATUS","stimulus","response"],"value":[254,128,2]}),"ds001787")
assert first["probes"] == 1
second = event_inventory(pd.DataFrame({"trial_type":["silent_probe_begin","prompt_window_begin","depth_rating"],"depth":[None,None,4]}),"ds008836")
assert second["probes"] == 1 and second["report_markers"] == 1
print("ok: dataset-aware inspection")
