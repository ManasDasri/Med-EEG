import sys
from pathlib import Path
import warnings
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from extract_features import spectral_features, pac_with_surrogates
from electrode_minimalism import curve
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    f = spectral_features(np.zeros((1,89)), np.arange(1,45.5,.5), ["Fz"])
assert np.isnan(f["frontal_alpha_asymmetry"])
assert not any(np.isinf(v).any() for v in f.values())
_, z = pac_with_surrogates(np.zeros(100), np.ones(100), np.random.default_rng(0))
assert np.isnan(z)
rng=np.random.default_rng(1); y=np.tile([1,2,3,1,2,3],4)
result=curve(rng.normal(size=(24,4)),y,np.repeat(range(4),6),{"missing":[]})
assert "top_4" in result and np.isnan(result["missing"])
print("ok: missing channels, zero spectra, degenerate PAC and empty clusters")
