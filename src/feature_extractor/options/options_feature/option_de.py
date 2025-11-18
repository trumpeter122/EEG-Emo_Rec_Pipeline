"""Differential entropy (DE) features."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal  # type: ignore[import-untyped]

from config import GENEVA_32, SFREQ, FeatureOption

__all__ = ["_de"]

# Match the reference DEAP pipeline bandpass configuration.
DE_BANDS: dict[str, tuple[int, int]] = {
    "theta": (4, 8),
    "slow_alpha": (8, 10),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}
_EPS = 1e-12


def _extract_de(trial_data: np.ndarray, channel_pick: list[str]) -> np.ndarray:
    """Compute differential entropy features for one segment."""
    trial_data = np.asarray(trial_data)
    if trial_data.ndim == 1:
        trial_data = trial_data[np.newaxis, :]

    ch_indices = [GENEVA_32.index(ch) for ch in channel_pick]
    picked = trial_data[ch_indices, :]

    n_ch, _ = picked.shape
    feats = np.zeros((n_ch, len(DE_BANDS)), dtype=np.float32)

    nyquist = 0.5 * SFREQ
    for band_idx, (fmin, fmax) in enumerate(DE_BANDS.values()):
        low, high = fmin / nyquist, fmax / nyquist
        butter_result = signal.butter(
            N=4,
            Wn=[low, high],
            btype="band",
            output="ba",
        )
        if not isinstance(butter_result, tuple) or len(butter_result) != 2:
            msg = "Butter filter coefficients must return a (b, a) tuple."
            raise RuntimeError(msg)
        b_coef, a_coef = butter_result

        band_sig = signal.filtfilt(b=b_coef, a=a_coef, x=picked, axis=1)
        band_sig *= 1e6  # convert to microvolts to match reference scaling

        stds = np.std(band_sig, axis=1) + _EPS
        de = 0.5 * np.log10(2 * math.pi * math.e * (stds**2))
        feats[:, band_idx] = de.astype(np.float32)

    return feats


_de = FeatureOption(name="de", feature_channel_extraction_method=_extract_de)
