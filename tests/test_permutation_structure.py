import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_comparison import shuffled_target
frame=pd.DataFrame({"subject":["s"]*10,"session":["a"]*5+["b"]*5,"probe_index":list(range(5))*2,"rating":list(range(5))+list(range(10,15))})
y=shuffled_target(frame,"rating","state",np.random.default_rng(4))
for _,g in frame.groupby("session"):
    original=g.rating.to_numpy(); actual=y.loc[g.index].to_numpy()
    assert any(np.array_equal(actual,np.roll(original,k)) for k in range(len(g)))
print("ok: shuffles preserve recording boundaries and circular order")
