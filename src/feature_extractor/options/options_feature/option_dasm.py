"""Differential asymmetry (DASM) features built on PSD values."""

import numpy as np

from config import FeatureOption

from .option_psd import _extract_psd
from .utils import _available_pairs

__all__ = ["_dasm"]


def _extract_dasm(trial_data: np.ndarray, channel_pick: list[str]) -> np.ndarray:
    """Compute left-right PSD differences."""
    psd_feats = _extract_psd(trial_data, channel_pick)
    pairs = _available_pairs(channel_pick)
    out = np.zeros((len(pairs), psd_feats.shape[1]), dtype=np.float32)
    for pair_idx, (left_idx, right_idx, _) in enumerate(pairs):
        out[pair_idx] = psd_feats[left_idx] - psd_feats[right_idx]
    return out


_dasm = FeatureOption(name="dasm", feature_channel_extraction_method=_extract_dasm)
