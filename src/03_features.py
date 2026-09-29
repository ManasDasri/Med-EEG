"""Step 3: turn preprocessed epochs into a feature table.

For every probe-epoch (from 02_preprocess.py) and every electrode, compute:
  - band power in delta/theta/alpha/beta/gamma (Welch PSD)
  - gamma/theta power ratio
  - theta-phase to gamma-amplitude coupling (Tort Modulation Index)

Then aggregate to (a) full-scalp average, (b) frontal-midline-theta /
posterior-gamma cluster average, and (c) the single minimal 2-electrode set,
so 05_ml_pipeline.py can run the "electrode minimalism" comparison.

Usage:
    python src/03_features.py --bids_root data/ds001787
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (ALL_BANDS, FRONTAL_MIDLINE_THETA_CHS, GAMMA_BAND,
                    MINIMAL_CH_SET, POSTERIOR_GAMMA_CHS, THETA_BAND,
                    Paths)  # noqa: E402

mne.set_log_level("WARNING")


def band_power(epochs: mne.Epochs, picks: list[str],
               fmin: float, fmax: float) -> np.ndarray:
    """Welch PSD band power per epoch, averaged over the given channels.
    Returns shape (n_epochs,)."""
    psds, freqs = epochs.compute_psd(
        method="welch", picks=picks, fmin=fmin, fmax=fmax,
        n_fft=int(epochs.info["sfreq"] * 2)).get_data(return_freqs=True)
    # psds: (n_epochs, n_channels, n_freqs) -> mean over channels and freqs
    return psds.mean(axis=(1, 2))


def tort_modulation_index(theta_phase: np.ndarray, gamma_amp: np.ndarray,
                           n_bins: int = 18) -> float:
    """Tort et al. (2010) Modulation Index for phase-amplitude coupling.
    theta_phase, gamma_amp: 1-D arrays, same length, already filtered/Hilbert
    transformed (phase of theta, amplitude envelope of gamma)."""
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(theta_phase, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    mean_amp = np.array([
        gamma_amp[bin_idx == b].mean() if np.any(bin_idx == b) else 0.0
        for b in range(n_bins)
    ])
    if mean_amp.sum() == 0:
        return 0.0
    p = mean_amp / mean_amp.sum()
    p_safe = np.clip(p, 1e-12, None)
    entropy = -np.sum(p_safe * np.log(p_safe))
    kl_max = np.log(n_bins)
    mi = (kl_max - entropy) / kl_max  # normalized 0-1
    return float(mi)


def epoch_pac(epochs: mne.Epochs, picks: list[str]) -> float:
    """Average Tort MI across the given channels and all epochs, computed on
    the concatenated, filtered signal (PAC needs continuous phase, not
    per-epoch fragments shorter than a few theta cycles)."""
    data = epochs.get_data(picks=picks)  # (n_epochs, n_ch, n_times)
    concat = data.transpose(1, 0, 2).reshape(len(picks), -1)  # (ch, time)
    sfreq = epochs.info["sfreq"]

    theta_filt = mne.filter.filter_data(
        concat, sfreq, THETA_BAND[0], THETA_BAND[1], verbose=False)
    gamma_filt = mne.filter.filter_data(
        concat, sfreq, GAMMA_BAND[0], GAMMA_BAND[1], verbose=False)

    mis = []
    for ch in range(len(picks)):
        phase = np.angle(hilbert(theta_filt[ch]))
        amp = np.abs(hilbert(gamma_filt[ch]))
        mis.append(tort_modulation_index(phase, amp))
    return float(np.mean(mis)) if mis else np.nan


def channel_set_features(epochs: mne.Epochs, cluster_name: str,
                          picks: list[str]) -> dict:
    picks = [p for p in picks if p in epochs.ch_names]
    if not picks:
        return {f"{cluster_name}_theta_power": np.nan,
                f"{cluster_name}_gamma_power": np.nan,
                f"{cluster_name}_ratio": np.nan,
                f"{cluster_name}_pac": np.nan}
    theta_p = band_power(epochs, picks, *THETA_BAND).mean()
    gamma_p = band_power(epochs, picks, *GAMMA_BAND).mean()
    ratio = gamma_p / theta_p if theta_p > 0 else np.nan
    pac = epoch_pac(epochs, picks)
    return {
        f"{cluster_name}_theta_power": theta_p,
        f"{cluster_name}_gamma_power": gamma_p,
        f"{cluster_name}_ratio": ratio,
        f"{cluster_name}_pac": pac,
    }


def full_scalp_band_features(epochs: mne.Epochs) -> dict:
    """All 5 canonical bands, full-scalp average -- this is the feature set
    the 'black box' models in 05_ml_pipeline.py get to use, so they have a
    fair information advantage over the 2-electrode interpretable model."""
    out = {}
    for band_name, (fmin, fmax) in ALL_BANDS.items():
        out[f"scalp_{band_name}_power"] = band_power(
            epochs, epochs.ch_names, fmin, fmax).mean()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    args = parser.parse_args()

    paths = Paths.from_root(args.bids_root)
    manifest_path = paths.derivatives / "probe_labels.csv"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found -- run "
                          f"02_preprocess.py first.")
    manifest = pd.read_csv(manifest_path)

    rows = []
    for _, row in manifest.iterrows():
        epo_path = Path(row["epochs_file"])
        if not epo_path.exists():
            continue
        epochs = mne.read_epochs(epo_path, preload=True)

        feat = dict(row)
        feat.update(channel_set_features(
            epochs, "frontal_theta", FRONTAL_MIDLINE_THETA_CHS))
        feat.update(channel_set_features(
            epochs, "posterior_gamma", POSTERIOR_GAMMA_CHS))
        feat.update(channel_set_features(
            epochs, "minimal", MINIMAL_CH_SET))
        feat.update(full_scalp_band_features(epochs))
        rows.append(feat)
        print(f"[ok] features for {row['subject']} probe {row['probe_index']}")

    feat_df = pd.DataFrame(rows)

    participants_path = paths.bids_root / "participants.tsv"
    if participants_path.exists():
        participants = pd.read_csv(participants_path, sep="\t")
        if "group" in participants.columns:
            feat_df = feat_df.merge(
                participants[["participant_id", "group"]],
                left_on="subject", right_on="participant_id", how="left")
            feat_df = feat_df.drop(columns=["participant_id"])
            # expert -> 1, novice -> 0 for the classify task in
            # 05_ml_pipeline.py; anything else (typos, missing) -> NaN so
            # it's dropped rather than silently misclassified.
            feat_df["group_binary"] = feat_df["group"].str.lower().map(
                {"expert": 1, "experienced": 1, "novice": 0})
            n_unmapped = feat_df["group_binary"].isna().sum()
            if n_unmapped:
                print(f"[warn] {n_unmapped} rows have a 'group' value that "
                      f"didn't map to expert/novice -- check participants."
                      f"tsv for typos or extra categories.")
        else:
            print("[warn] participants.tsv has no 'group' column -- "
                  "group_binary not created, --task classify won't work "
                  "until this is fixed.")
    else:
        print("[warn] no participants.tsv found at bids_root -- "
              "group_binary not created, --task classify won't work.")

    out_path = paths.derivatives / "features.parquet"
    feat_df.to_parquet(out_path, index=False)
    print(f"\n[ok] wrote {len(feat_df)} rows x {feat_df.shape[1]} columns "
          f"to {out_path}")


if __name__ == "__main__":
    main()
