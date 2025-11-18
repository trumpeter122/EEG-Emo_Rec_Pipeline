"""Power spectral density (PSD) features."""

from __future__ import annotations

import numpy as np
from scipy import signal  # type: ignore[import-untyped]

from config import GENEVA_32, SFREQ, FeatureOption

__all__ = ["_psd"]

# Frequency bins follow the original DEAP feature script where Welch spectra
# are averaged over integer-Hz bins with inclusive upper bounds.
PSD_BANDS: dict[str, tuple[int, int]] = {
    "theta": (4, 7),
    "slow_alpha": (8, 10),
    "alpha": (8, 13),
    "beta": (14, 29),
    "gamma": (30, 45),
}
_EPS = 1e-12


def _extract_psd(trial_data: np.ndarray, channel_pick: list[str]) -> np.ndarray:
    """
    Compute band-power (log-PSD) features for one segment.
    """
    trial_data = np.asarray(trial_data)
    if trial_data.ndim == 1:
        trial_data = trial_data[np.newaxis, :]

    ch_indices = [GENEVA_32.index(ch) for ch in channel_pick]
    picked = trial_data[ch_indices, :]

    n_ch, n_samples = picked.shape
    feats = np.zeros((n_ch, len(PSD_BANDS)), dtype=np.float32)

    for ch_idx in range(n_ch):
        nperseg = min(SFREQ, n_samples)
        freqs, psd_vals = signal.welch(x=picked[ch_idx], fs=SFREQ, nperseg=nperseg)
        psd_vals = psd_vals * (1e6**2)
        psd_vals = 10.0 * np.log10(psd_vals + _EPS)

        band_vals = []
        for fmin, fmax in PSD_BANDS.values():
            mask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(mask):
                raise RuntimeError(
                    f"No frequency bins found for PSD band {fmin}-{fmax} Hz "
                    f"(nperseg={nperseg}, available up to {freqs.max():.2f} Hz)",
                )
            band_vals.append(float(np.mean(psd_vals[mask])))
        feats[ch_idx, :] = np.asarray(band_vals, dtype=np.float32)

    return feats


_psd = FeatureOption(name="psd", feature_channel_extraction_method=_extract_psd)
