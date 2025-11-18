"""Utility helpers used by preprocessing pipelines."""

from pathlib import Path

import mne  # type: ignore[import-untyped]

from config import (
    DEAP_ORIGINAL,
)

__all__ = ["_bdf_path", "_subject_npy_path", "_load_raw_subject"]


def _bdf_path(subject_id: int) -> Path:
    """Return the absolute path to the raw BDF file for ``subject_id``."""
    return DEAP_ORIGINAL / f"s{subject_id:02d}.bdf"


def _subject_npy_path(folder: Path, subject_id: int) -> Path:
    """
    Create ``folder`` if needed and return the derived ``.npy`` file path.
    """
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"s{subject_id:02}.npy"


def _load_raw_subject(subject_id: int) -> mne.io.BaseRaw:
    """Load a raw DEAP recording (preloaded) for ``subject_id``."""
    return mne.io.read_raw_bdf(
        _bdf_path(subject_id=subject_id),
        preload=True,
        verbose=False,
    ).load_data()
