import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from export_dashboard import build_subjects_payload
f=pd.DataFrame({"subject":["s"]*4,"session":["2","1","2","1"],"probe_index":[0,0,1,1],"probe_onset_s":[10,10,20,20],"depth_rating":[1,2,3,4],"block":["silent"]*4})
p=build_subjects_payload(f,None)[0]["probes"]
assert [r["session"] for r in p]==["1","1","2","2"]
assert all("depth_rating" in r and "block" in r for r in p)
print("ok: recording order and replication metadata")
