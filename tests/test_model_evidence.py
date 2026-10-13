import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_comparison import mean_interval, paired_comparisons
assert mean_interval([1,1,1]) == (1,1)
scores=pd.DataFrame({"subject":["a","b"]*2,"model":["eeg"]*2+["baseline_time"]*2,"score":[.8,.7,.6,.5]})
r=paired_comparisons(scores).iloc[0]
assert np.isclose(r.mean_difference,.2) and r.n_subjects==2
assert np.isclose(r.ci_low,.2) and np.isclose(r.ci_high,.2)
print("ok: paired subject comparisons and intervals")
