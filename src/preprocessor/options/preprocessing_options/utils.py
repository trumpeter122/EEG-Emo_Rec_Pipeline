"""Helper utilities shared by preprocessing option implementations."""

import warnings

import mne  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from config import (
    DEAP_CHANNELS_XLSX,
    DEAP_RATINGS_CSV,
    EEG_ELECTRODES_NUM,
    EPOCH_TMAX,
    EPOCH_TMIN,
    TRIALS_NUM,
)

# -----------------------------
# Warning Suppression
# -----------------------------
warnings.filterwarnings("ignore", message="Channels contain different highpass filters")
warnings.filterwarnings("ignore", message="Channels contain different lowpass filters")
warnings.filterwarnings("ignore", message="Channel names are not unique")


def _apply_filter_reference(raw: mne.io.BaseRaw) -> None:
    """Apply DEAP-specific filtering and referencing to ``raw`` in-place."""
    montage = mne.channels.make_standard_montage(
        kind="biosemi32",
        head_size=0.095,  # type: ignore[arg-type]
    )
    raw.set_montage(montage=montage, verbose=False)
    raw.notch_filter(
        freqs=np.arange(start=50, stop=251, step=50),
        fir_design="firwin",
        verbose=False,
    )
    raw.filter(
        l_freq=4,
        h_freq=45,
        fir_design="firwin",
        verbose=False,
    )
    raw.set_eeg_reference(verbose=False)


def _epoch_and_resample(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    eeg_channels: list[str],
    sfreq_target: float,
) -> np.ndarray:
    """Epoch ``raw`` using ``events`` then resample to ``sfreq_target``."""
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=4,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        picks=eeg_channels,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs.copy().resample(sfreq=sfreq_target).get_data()


def _get_events(
    raw_stim: mne.io.BaseRaw,
    stim_ch_name: str,
    subject_id: int,
) -> np.ndarray:
    """Extract event markers for ``subject_id`` with DEAP-specific fixups."""
    events = mne.find_events(
        raw=raw_stim,
        stim_channel=stim_ch_name,
        verbose=False,
        initial_event=True,
    )
    if subject_id > 23:  # DEAP quirk after s23
        events[:, 2] -= 1703680
        events[:, 2] %= 65536
    return events[np.where(events[:, 2] == 4)[0], :]


def _prepare_channels(
    raw: mne.io.BaseRaw,
) -> tuple[list[str], str, mne.io.BaseRaw, mne.io.BaseRaw]:
    """
    Split ``raw`` into EEG and stimulation channels.

    Returns the EEG channel names, the stim channel name, and copies of the raw
    objects filtered to each subset.
    """
    ch_names = raw.ch_names
    eeg_channels = ch_names[:EEG_ELECTRODES_NUM]
    stim_ch_name = ch_names[-1]
    raw_stim = raw.copy().pick(picks=[stim_ch_name])
    raw_eeg = raw.copy().pick(picks=eeg_channels, verbose=False)
    return eeg_channels, stim_ch_name, raw_stim, raw_eeg


def _reorder_channels(eeg_channels: list[str]) -> list[int]:
    """Map the recording order to the canonical Geneva ordering."""
    df = pd.read_excel(io=DEAP_CHANNELS_XLSX)
    target_order = df["Channel_name_Geneva"].values
    return [eeg_channels.index(ch) for ch in target_order]


def _base_bdf_process(
    data_down: np.ndarray,
    eeg_channels: list[str],
    subject_id: int,
) -> np.ndarray:
    """Reorder trials and channels to canonical DEAP positions."""
    ch_idx = _reorder_channels(eeg_channels=eeg_channels)
    trial_idx = _reorder_trials(subject_id=subject_id)
    epoch_len = data_down.shape[-1]
    data_out = np.zeros(
        shape=(TRIALS_NUM, EEG_ELECTRODES_NUM, epoch_len),
        dtype=data_down.dtype,
    )
    for target_idx, source_idx in enumerate(trial_idx):
        data_out[target_idx] = data_down[source_idx][ch_idx, :].copy()

    return data_out


def _reorder_trials(subject_id: int) -> list[int]:
    """Return the canonical trial ordering for ``subject_id``."""
    ratings = pd.read_csv(filepath_or_buffer=DEAP_RATINGS_CSV)
    subj = ratings[ratings["Participant_id"] == subject_id]
    return [
        int(subj.loc[subj["Experiment_id"] == (i + 1), "Trial"].iloc[0]) - 1
        for i in range(TRIALS_NUM)
    ]
