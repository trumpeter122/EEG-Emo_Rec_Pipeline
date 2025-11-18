"""Differential entropy asymmetry (DEASM) features."""

import numpy as np

from config import FeatureOption

from .option_de import _extract_de
from .utils import _available_pairs

__all__ = ["_deasm"]


def _extract_deasm(trial_data: np.ndarray, channel_pick: list[str]) -> np.ndarray:
    """Compute left-right differential entropy differences."""
    de_feats = _extract_de(trial_data, channel_pick)
    pairs = _available_pairs(channel_pick)
    out = np.zeros((len(pairs), de_feats.shape[1]), dtype=np.float32)
    for pair_idx, (left_idx, right_idx, _) in enumerate(pairs):
        out[pair_idx] = de_feats[left_idx] - de_feats[right_idx]
    return out


_deasm = FeatureOption(name="deasm", feature_channel_extraction_method=_extract_deasm)
