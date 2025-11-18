"""Cleaning pipeline that applies filtering and referencing."""

import mne  # type: ignore[import-untyped]
import numpy as np

from config import SFREQ_TARGET, PreprocessingOption

from .utils import (
    _apply_filter_reference,
    _base_bdf_process,
    _epoch_and_resample,
    _get_events,
    _prepare_channels,
)

__all__ = ["_option_clean"]


def _clean_bdf(
    raw: mne.io.BaseRaw,
    subject_id: int,
) -> np.ndarray:
    """Apply filtering/reference + resampling while preserving ordering."""
    eeg_channels, stim_ch, raw_stim, raw_eeg = _prepare_channels(raw=raw)
    events = _get_events(
        raw_stim=raw_stim,
        stim_ch_name=stim_ch,
        subject_id=subject_id,
    )

    _apply_filter_reference(raw=raw_eeg)
    data_down = _epoch_and_resample(
        raw=raw_eeg,
        events=events,
        eeg_channels=eeg_channels,
        sfreq_target=SFREQ_TARGET,
    )

    return _base_bdf_process(
        data_down=data_down,
        eeg_channels=eeg_channels,
        subject_id=subject_id,
    )


_option_clean = PreprocessingOption(
    name="clean",
    root_dir="cleaned",
    preprocessing_method=_clean_bdf,
)
