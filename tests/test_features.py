"""Checks for extract_features.py on synthetic signals with known answers.
Run: python tests/test_features.py"""
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import (ALL_BANDS, EPOCH_LENGTH_S, EPOCH_OVERLAP_S,  # noqa: E402
                    PAC_CLUSTERS, REGIONS)
from extract_features import pac_features, spectral_features  # noqa: E402

mne.set_log_level("ERROR")
SFREQ = 256.0
CHANNELS = [ch for chs in REGIONS.values() for ch in chs]


def overlapping_epochs(signal: np.ndarray) -> mne.EpochsArray:
    """Cut a (n_channels, n_times) signal the way preprocess.py does."""
    n, step = int(EPOCH_LENGTH_S * SFREQ), int((EPOCH_LENGTH_S - EPOCH_OVERLAP_S) * SFREQ)
    starts = range(0, signal.shape[1] - n + 1, step)
    data = np.stack([signal[:, s:s + n] for s in starts])
    info = mne.create_info(CHANNELS, SFREQ, "eeg")
    return mne.EpochsArray(data, info)


# --- PAC: coupled signal scores high, uncoupled doesn't -----------------------
rng = np.random.default_rng(0)
t = np.arange(int(60 * SFREQ)) / SFREQ
# Theta as band-limited noise, like real EEG. A pure sine would be perfectly
# periodic, so a time-shifted surrogate would still be coupled (just at a
# different phase) and the z-score would be meaningless.
theta = mne.filter.filter_data(rng.standard_normal(len(t)), SFREQ, 4, 8)
theta /= theta.std()
gamma = np.sin(2 * np.pi * 40 * t)
noise = 0.5 * rng.standard_normal((len(CHANNELS), len(t)))
coupled = theta + (1 + 0.9 * theta) * 0.3 * gamma + noise
uncoupled = theta + 0.3 * gamma + noise

pac_on = pac_features(overlapping_epochs(coupled * 1e-6))
pac_off = pac_features(overlapping_epochs(uncoupled * 1e-6))
for cluster in PAC_CLUSTERS:
    assert pac_on[f"{cluster}_pac_z"] > 10, pac_on
    assert abs(pac_off[f"{cluster}_pac_z"]) < 4, pac_off
    assert pac_on[f"{cluster}_pac"] > 10 * abs(pac_off[f"{cluster}_pac"])

# Too little signal -> NaN, not a number that looks real.
short = pac_features(overlapping_epochs(coupled[:, :int(4 * SFREQ)] * 1e-6))
assert np.isnan(short["minimal_pac"]) and np.isnan(short["minimal_pac_z"])

# --- Spectral features on a known spectrum --------------------------------------
freqs = np.arange(1.0, 45.5, 0.5)
psd = np.tile(freqs ** -2.0, (len(CHANNELS), 1))  # pure 1/f^2 everywhere
psd[CHANNELS.index("F3")] *= 2.0                  # more power on the left
f = spectral_features(psd, freqs, CHANNELS)

for region in REGIONS:
    assert abs(f[f"{region}_slope"] - (-2.0)) < 1e-9          # exponent recovered
    total_rel = sum(f[f"{region}_{b}_rel"] for b in ALL_BANDS)
    assert abs(total_rel - 1.0) < 1e-9                        # shares add to 1
assert f["frontal_alpha_asymmetry"] < 0                       # right < left
assert f["minimal_ratio"] == f["minimal_gamma_power"] / f["minimal_theta_power"]

# Per-epoch input gives one value per epoch.
stacked = spectral_features(np.stack([psd, psd * 3]), freqs, CHANNELS)
assert stacked["scalp_alpha_power"].shape == (2,)
assert np.allclose(stacked["frontal_alpha_rel"], f["frontal_alpha_rel"])

print("ok: extract_features")
