"""Phase 8: per-probe channel covariance within one frequency band
(theta, alpha, beta), for the depth search (explore_depth.py).

A band covariance is the real part of the cross-spectrum summed over the
band's frequency bins (Hann-tapered FFT of each 2-s epoch, all epochs of the
probe pooled) -- the covariance the signal would have after an ideal
band-pass, without filter edge effects on 2-s epochs. Shrunk toward a scaled
identity because average referencing leaves it rank-deficient.

Writes <derivatives>/band_covariances.npz: covs (n_probes, n_bands, n_ch,
n_ch), recording, probe_index, bands, channels.

Usage:
    python src/band_covariances.py --derivatives <dir with probe_labels.csv>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
SHRINKAGE = 0.05  # fixed before any run, not tuned


def band_covariances(data: np.ndarray, sfreq: float) -> np.ndarray:
    """data (n_epochs, n_ch, n_times) -> (n_bands, n_ch, n_ch)."""
    spec = np.fft.rfft(data * np.hanning(data.shape[-1]), axis=-1)
    freqs = np.fft.rfftfreq(data.shape[-1], 1 / sfreq)
    n = data.shape[1]
    out = []
    for lo, hi in BANDS.values():
        x = spec[..., (freqs >= lo) & (freqs < hi)]
        cov = np.einsum("eif,ejf->ij", x, x.conj()).real / (x.shape[0] * x.shape[-1])
        out.append((1 - SHRINKAGE) * cov + SHRINKAGE * np.trace(cov) / n * np.eye(n))
    return np.stack(out)


def one_probe(path: str) -> tuple[np.ndarray, list[str]]:
    epochs = mne.read_epochs(path, preload=True, verbose="error")
    picks = mne.pick_types(epochs.info, eeg=True)
    return (band_covariances(epochs.get_data(picks="eeg"), epochs.info["sfreq"]),
            [epochs.ch_names[i] for i in picks])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derivatives", required=True, type=Path)
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()
    labels = pd.read_csv(args.derivatives / "probe_labels.csv")
    results = Parallel(n_jobs=args.n_jobs)(delayed(one_probe)(p) for p in labels["epochs_file"])
    channels = results[0][1]
    if any(r[1] != channels for r in results):
        raise SystemExit("channel order differs between probes")
    out = args.derivatives / "band_covariances.npz"
    np.savez(out, covs=np.stack([r[0] for r in results]),
             recording=labels["recording"].to_numpy(str),
             probe_index=labels["probe_index"].to_numpy(),
             bands=np.array(list(BANDS)), channels=np.array(channels))
    print(f"[ok] wrote {len(results)} probes x {len(BANDS)} bands to {out}")


if __name__ == "__main__":
    # Self-check: a 10 Hz source shared by channels 0 and 1 shows up as
    # their covariance in alpha only.
    rng = np.random.default_rng(0)
    t = np.arange(512) / 256
    data = rng.standard_normal((20, 4, 512)) * 0.1
    data[:, :2] += np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6, (20, 1, 1)))
    c = band_covariances(data, 256.0)
    assert c[1, 0, 1] > 10 * abs(c[0, 0, 1]) and c[1, 0, 1] > 10 * abs(c[2, 0, 1])
    assert all(np.linalg.eigvalsh(m).min() > 0 for m in c)
    main()
