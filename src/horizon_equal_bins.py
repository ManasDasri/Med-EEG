"""Phase 9, exploratory (written after the pre-registered results were seen): horizon with equal 10-s bins,
to separate distance from bin length.

Usage: python src/horizon_equal_bins.py <phase 7 run dir> <output dir>"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import horizon as h
from depth_decoding import recording_keys
from model_comparison import prepare

h.BINS = {"0-10": (0, 10), "20-30": (20, 30), "50-60": (50, 60), "80-90": (80, 90), "110-120": (110, 120)}
P7, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "equal_bins.npz"
labels = pd.read_csv(P7 / "probe_labels.csv")
def one(path, bins):
    import mne
    from sklearn.covariance import oas
    ep = mne.read_epochs(path, preload=True, verbose="error")
    data, sbp = ep.get_data(picks="eeg"), ep.metadata["seconds_before_probe"].to_numpy()
    n = data.shape[1]; c = np.full((len(bins), n, n), np.nan)
    for b, (lo, hi) in enumerate(bins.values()):
        m = (sbp > lo) & (sbp <= hi)
        if m.sum() >= h.MIN_SEGMENTS:
            c[b] = oas(data[m].transpose(1, 0, 2).reshape(n, -1).T)[0]
    return c, None
if not out.exists():
    res = Parallel(n_jobs=5)(delayed(one)(p, h.BINS) for p in labels["epochs_file"])
    np.savez(out, covs=np.stack([r[0] for r in res]), recording=labels["recording"].to_numpy(str),
             probe_index=labels["probe_index"].to_numpy())
npz = np.load(out)
df = prepare(pd.read_parquet(P7 / "features.parquet"), "depth")
df["recording"] = recording_keys(df)
df = df.drop(columns=[c for c in df.columns if c.startswith("cov__")])
where = {(r, int(p)): i for i, (r, p) in enumerate(zip(npz["recording"], npz["probe_index"]))}
covs = npz["covs"][[where[(r, int(p))] for r, p in zip(df["recording"], df["probe_index"])]]
from depth_decoding import riemannian_mean, tangent_features
def frames(df, covs):
    out = {}
    for b, name in enumerate(h.BINS):
        ok = ~np.isnan(covs[:, b, 0, 0])
        for r in df.loc[ok, "recording"].unique():  # drop recordings whose mean diverges
            m = ok & (df["recording"] == r).to_numpy()
            try: riemannian_mean(covs[m, b])
            except np.linalg.LinAlgError:
                ok &= ~m; print(f"[note] {name}: dropped {r} ({m.sum()} reports): mean did not converge")
        sub = df[ok].reset_index(drop=True)
        out[name] = (sub, tangent_features(covs[ok, b], sub["recording"]))
    return out
rows = []
for name, (sub, X) in frames(df, covs).items():
    r = h.decode(sub, X, 200); r.pop("per_person")
    rows.append({"bin_s": name, **r}); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(OUT / "equal_bins_exploratory.csv", index=False)
