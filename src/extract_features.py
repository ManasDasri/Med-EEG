"""Step 3: turn each probe's clean epochs (from preprocess.py) into features.

Every feature comes from ONE power spectrum per epoch (Welch, 1-45 Hz):
  - band power, delta..gamma: absolute (log10) and relative (share of the
    1-45 Hz total), per scalp region
  - the columns the stats / model / dashboard scripts already use:
    scalp_<band>_power, <cluster>_theta_power / _gamma_power / _ratio
  - aperiodic slope (30-45 Hz) per region
  - frontal alpha asymmetry (F4 vs F3)
plus, per probe, theta-gamma phase-amplitude coupling for each cluster:
the Tort Modulation Index corrected for its small-sample bias
(<cluster>_pac) and its z-score against time-shifted surrogates
(<cluster>_pac_z).

Writes:
  - features.parquet         one row per probe (spectra averaged over epochs)
  - features_epochs.parquet  one row per epoch, spectral features only (PAC
                             needs more than one 2 s epoch of signal)

Usage:
    python src/extract_features.py --bids_root data/ds001787
    python src/extract_features.py --bids_root data/ds001787 --window_s 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ALL_BANDS, ASYMMETRY_PAIR, BANDPASS_HIGH_HZ, GAMMA_BAND,
                    PAC_CLUSTERS, PAC_EDGE_S, PAC_MIN_SECONDS, PAC_N_BINS,
                    PAC_N_SURROGATES, PAC_RANDOM_STATE, REGIONS,
                    SLOPE_RANGE_HZ, THETA_BAND, Paths)  # noqa: E402

mne.set_log_level("WARNING")

TOTAL_BAND = (min(lo for lo, _ in ALL_BANDS.values()), BANDPASS_HIGH_HZ)
LABEL_COLUMNS = ["subject", "session", "probe_index", "concentration_rating",
                 "mind_wandering_rating", "tiredness_rating"]


# --- Spectral features ------------------------------------------------------------

def band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Half-open [lo, hi) so a bin on a band edge (e.g. 8 Hz) is counted in
    exactly one band and relative powers add up to 1."""
    return (freqs >= band[0]) & (freqs < band[1])


def loglog_slope(spectrum: np.ndarray, freqs: np.ndarray,
                 band: tuple[float, float]) -> np.ndarray:
    """Least-squares slope of log10(power) vs log10(frequency) inside band,
    over the last axis. A steeper (more negative) slope means relatively more
    slow than fast activity; a slope near 0 at high frequencies usually
    means muscle noise."""
    mask = (freqs >= band[0]) & (freqs <= band[1])
    x = np.log10(freqs[mask])
    y = np.log10(spectrum[..., mask])
    xc = x - x.mean()
    return (y * xc).sum(axis=-1) / (xc ** 2).sum()


def spectral_features(psd: np.ndarray, freqs: np.ndarray,
                      ch_names: list[str]) -> dict[str, np.ndarray]:
    """All spectrum-based features from psd of shape (..., n_channels,
    n_freqs). Works for one averaged spectrum (-> scalars) and for a stack
    of per-epoch spectra (-> one value per epoch)."""
    idx = {ch: i for i, ch in enumerate(ch_names)}

    def chans(names: list[str]) -> list[int]:
        return [idx[n] for n in names if n in idx]

    # Mean power density per band and channel, shape (..., n_channels).
    band_power = {b: psd[..., band_mask(freqs, rng)].mean(axis=-1)
                  for b, rng in ALL_BANDS.items()}
    band_sum = {b: psd[..., band_mask(freqs, rng)].sum(axis=-1)
                for b, rng in ALL_BANDS.items()}
    total_sum = psd[..., band_mask(freqs, TOTAL_BAND)].sum(axis=-1)

    feats: dict[str, np.ndarray] = {}
    # Columns downstream scripts rely on (same definitions as before).
    for b in ALL_BANDS:
        feats[f"scalp_{b}_power"] = band_power[b].mean(axis=-1)
    for cluster, names in PAC_CLUSTERS.items():
        c = chans(names)
        theta = band_power["theta"][..., c].mean(axis=-1)
        gamma = band_power["gamma"][..., c].mean(axis=-1)
        feats[f"{cluster}_theta_power"] = theta
        feats[f"{cluster}_gamma_power"] = gamma
        feats[f"{cluster}_ratio"] = gamma / theta

    for region, names in REGIONS.items():
        c = chans(names)
        region_total = total_sum[..., c].sum(axis=-1)
        for b in ALL_BANDS:
            feats[f"{region}_{b}_log"] = np.log10(
                band_power[b][..., c].mean(axis=-1))
            feats[f"{region}_{b}_rel"] = (band_sum[b][..., c].sum(axis=-1)
                                         / region_total)
        feats[f"{region}_slope"] = loglog_slope(
            psd[..., c, :].mean(axis=-2), freqs, SLOPE_RANGE_HZ)

    right, left = chans(list(ASYMMETRY_PAIR))
    feats["frontal_alpha_asymmetry"] = (np.log10(band_power["alpha"][..., right])
                                        - np.log10(band_power["alpha"][..., left]))
    return feats


# --- Phase-amplitude coupling --------------------------------------------------------

def tort_modulation_index(theta_phase: np.ndarray, gamma_amp: np.ndarray,
                          n_bins: int = PAC_N_BINS) -> float:
    """Tort et al. (2010) Modulation Index for phase-amplitude coupling:
    how far the mean gamma amplitude per theta-phase bin is from flat,
    normalised to 0 (no coupling) - 1. Inputs are 1-D (phase, amplitude)
    samples; they don't need to be continuous in time."""
    bin_idx = np.clip(((theta_phase + np.pi) / (2 * np.pi) * n_bins)
                      .astype(int), 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins)
    sums = np.bincount(bin_idx, weights=gamma_amp, minlength=n_bins)
    mean_amp = np.divide(sums, counts, out=np.zeros(n_bins), where=counts > 0)
    if mean_amp.sum() == 0:
        return 0.0
    p = np.clip(mean_amp / mean_amp.sum(), 1e-12, None)
    entropy = -np.sum(p * np.log(p))
    return float((np.log(n_bins) - entropy) / np.log(n_bins))


def pac_with_surrogates(theta_phase: np.ndarray, gamma_amp: np.ndarray,
                        rng: np.random.Generator) -> tuple[float, float]:
    """Compare the real MI with PAC_N_SURROGATES versions of the same data
    whose amplitude is circularly shifted in time (coupling destroyed,
    everything else kept). Returns:
      - bias-corrected MI = MI - mean surrogate MI: coupling STRENGTH, ~0
        when there is none, whatever the window length (raw MI is inflated
        in short windows)
      - z-score vs the surrogates: EVIDENCE for coupling, > ~2 means real;
        grows with window length when coupling exists."""
    mi = tort_modulation_index(theta_phase, gamma_amp)
    n = len(gamma_amp)
    shifts = rng.integers(n // 10, n - n // 10, PAC_N_SURROGATES)
    null = np.array([tort_modulation_index(theta_phase, np.roll(gamma_amp, s))
                     for s in shifts])
    return mi - null.mean(), float((mi - null.mean()) / null.std())


def pac_features(epochs: mne.Epochs) -> dict[str, float]:
    """PAC per cluster: filter each epoch separately, trim the distorted
    edges, pool the (phase, amplitude) samples of all epochs, then compute
    MI per channel and average over the cluster's channels."""
    sfreq = epochs.info["sfreq"]
    edge = int(round(PAC_EDGE_S * sfreq))
    n_kept = len(epochs) * (len(epochs.times) - 2 * edge)
    feats = {}
    for cluster, names in PAC_CLUSTERS.items():
        picks = [n for n in names if n in epochs.ch_names]
        if not picks or n_kept < PAC_MIN_SECONDS * sfreq:
            feats[f"{cluster}_pac"] = feats[f"{cluster}_pac_z"] = np.nan
            continue
        data = epochs.get_data(picks=picks)  # (n_epochs, n_ch, n_times)
        # IIR (Butterworth, zero-phase) settles within a few theta cycles --
        # an FIR filter sharp enough for 4-8 Hz would be longer than the epoch.
        theta = mne.filter.filter_data(data, sfreq, *THETA_BAND,
                                       method="iir", verbose=False)
        gamma = mne.filter.filter_data(data, sfreq, *GAMMA_BAND,
                                       method="iir", verbose=False)
        phase = np.angle(hilbert(theta, axis=-1))[..., edge:-edge]
        amp = np.abs(hilbert(gamma, axis=-1))[..., edge:-edge]
        # (n_epochs, n_ch, n_kept) -> (n_ch, n_epochs * n_kept)
        phase = phase.transpose(1, 0, 2).reshape(len(picks), -1)
        amp = amp.transpose(1, 0, 2).reshape(len(picks), -1)
        rng = np.random.default_rng(PAC_RANDOM_STATE)
        results = [pac_with_surrogates(phase[i], amp[i], rng)
                   for i in range(len(picks))]
        feats[f"{cluster}_pac"] = float(np.mean([mi for mi, _ in results]))
        feats[f"{cluster}_pac_z"] = float(np.mean([z for _, z in results]))
    return feats


# --- Per probe -------------------------------------------------------------------------

def probe_features(row: dict, window_s: float | None
                   ) -> tuple[dict, pd.DataFrame] | None:
    """Features for one probe's epochs file: (probe-level row, epoch-level
    table), or None if nothing is left to compute from."""
    epochs = mne.read_epochs(row["epochs_file"], preload=True)
    if epochs.metadata is None or "seconds_before_probe" not in epochs.metadata:
        raise ValueError(f"{row['epochs_file']} has no epoch metadata -- "
                         f"re-run preprocess.py to regenerate it")
    if window_s is not None:
        epochs = epochs[f"seconds_before_probe <= {window_s}"]
    if len(epochs) == 0:
        return None

    spectrum = epochs.compute_psd(method="welch", fmin=TOTAL_BAND[0],
                                  fmax=TOTAL_BAND[1], n_fft=len(epochs.times))
    psd, freqs = spectrum.get_data(return_freqs=True)

    # Probe level: average the spectra first (a Welch estimate over the whole
    # window), then compute features -- more stable than averaging features.
    probe_row = dict(row)
    probe_row.update({k: float(v) for k, v in
                      spectral_features(psd.mean(axis=0), freqs,
                                        epochs.ch_names).items()})
    probe_row.update(pac_features(epochs))
    probe_row["n_epochs_used"] = len(epochs)

    epoch_table = pd.DataFrame(spectral_features(psd, freqs, epochs.ch_names))
    epoch_table.insert(0, "seconds_before_probe",
                       epochs.metadata["seconds_before_probe"].to_numpy())
    for col in reversed(LABEL_COLUMNS):
        epoch_table.insert(0, col, row[col])
    return probe_row, epoch_table


def add_group_labels(df: pd.DataFrame, bids_root: Path) -> pd.DataFrame:
    """Merge expert/novice from participants.tsv; expert -> 1, novice -> 0
    as group_binary for model_comparison.py --task classify."""
    participants_path = bids_root / "participants.tsv"
    if not participants_path.exists():
        print("[warn] no participants.tsv found at bids_root -- "
              "group_binary not created, --task classify won't work.")
        return df
    participants = pd.read_csv(participants_path, sep="\t")
    if "group" not in participants.columns:
        print("[warn] participants.tsv has no 'group' column -- "
              "group_binary not created, --task classify won't work.")
        return df
    df = df.merge(participants[["participant_id", "group"]],
                  left_on="subject", right_on="participant_id", how="left")
    df = df.drop(columns=["participant_id"])
    # Anything other than expert/novice (typos, missing) -> NaN so it's
    # dropped rather than silently misclassified.
    df["group_binary"] = df["group"].str.lower().map(
        {"expert": 1, "experienced": 1, "novice": 0})
    n_unmapped = df["group_binary"].isna().sum()
    if n_unmapped:
        print(f"[warn] {n_unmapped} rows have a 'group' value that didn't "
              f"map to expert/novice -- check participants.tsv.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", required=True, type=Path)
    parser.add_argument("--window_s", type=float, default=None,
                        help="Only use epochs starting within this many "
                             "seconds before each probe (default: the whole "
                             "window preprocess.py kept, up to 120 s).")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Probes processed in parallel (-1 = all CPU "
                             "cores, 1 = sequential, easier to debug).")
    args = parser.parse_args()

    paths = Paths.from_root(args.bids_root)
    manifest_path = paths.derivatives / "probe_labels.csv"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found -- run preprocess.py "
                         f"first.")
    manifest = pd.read_csv(manifest_path)
    rows = [r for r in manifest.to_dict("records")
            if Path(r["epochs_file"]).exists()]
    if len(rows) < len(manifest):
        print(f"[warn] {len(manifest) - len(rows)} epochs files listed in "
              f"probe_labels.csv are missing -- re-run preprocess.py")
    print(f"[..] extracting features for {len(rows)} probes"
          + (f", last {args.window_s:g} s before each" if args.window_s else ""))

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(probe_features)(r, args.window_s) for r in rows)
    results = [r for r in results if r is not None]
    if not results:
        raise SystemExit("[ERROR] no probe had epochs left to use")

    probe_df = add_group_labels(pd.DataFrame([r for r, _ in results]),
                                paths.bids_root)
    epoch_df = add_group_labels(pd.concat([e for _, e in results],
                                          ignore_index=True), paths.bids_root)

    probe_path = paths.derivatives / "features.parquet"
    epoch_path = paths.derivatives / "features_epochs.parquet"
    probe_df.to_parquet(probe_path, index=False)
    epoch_df.to_parquet(epoch_path, index=False)
    print(f"[ok] wrote {len(probe_df)} probes x {probe_df.shape[1]} columns "
          f"to {probe_path}")
    print(f"[ok] wrote {len(epoch_df)} epochs x {epoch_df.shape[1]} columns "
          f"to {epoch_path}")
    n_no_pac = probe_df["minimal_pac"].isna().sum()
    if n_no_pac:
        print(f"[note] {n_no_pac} probes had < {PAC_MIN_SECONDS:g} s of "
              f"signal, so their PAC is NaN")


if __name__ == "__main__":
    main()
