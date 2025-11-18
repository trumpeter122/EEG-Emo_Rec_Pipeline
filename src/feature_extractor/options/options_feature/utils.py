"""Shared helpers for asymmetry-based features."""

from config import (
    ASYM_PAIRS,
)

__all__ = ["_available_pairs"]


def _available_pairs(ch_names: list[str]) -> list[tuple[int, int, str]]:
    """
    Map configured asymmetric channel pairs to indices in the current pick.
    """
    name_to_idx = {nm: idx for idx, nm in enumerate(ch_names)}
    pairs: list[tuple[int, int, str]] = []
    for left, right in ASYM_PAIRS:
        if left in name_to_idx and right in name_to_idx:
            pairs.append((name_to_idx[left], name_to_idx[right], f"{left}-{right}"))
    return pairs
