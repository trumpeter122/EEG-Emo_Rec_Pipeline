import warnings

import mne
import numpy as np
import pandas as pd

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
    montage = mne.channels.make_standard_montage("biosemi32", head_size=0.095)
    raw.set_montage(montage, verbose=False)
    raw.notch_filter(np.arange(50, 251, 50), fir_design="firwin", verbose=False)
    raw.filter(4, 45, fir_design="firwin", verbose=False)
    raw.set_eeg_reference(verbose=False)


def _epoch_and_resample(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    eeg_channels: list[str],
    sfreq_target: float,
) -> np.ndarray:
    epochs = mne.Epochs(
        raw,
        events,
        event_id=4,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        picks=eeg_channels,
        baseline=None,
        preload=True,
        verbose=False,
    )
    return epochs.copy().resample(sfreq_target).get_data()


def _get_events(
    raw_stim: mne.io.BaseRaw,
    stim_ch_name: str,
    subject_id: int,
) -> np.ndarray:
    events = mne.find_events(
        raw_stim,
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
    ch_names = raw.ch_names
    eeg_channels = ch_names[:EEG_ELECTRODES_NUM]
    stim_ch_name = ch_names[-1]
    raw_stim = raw.copy().pick([stim_ch_name])
    raw_eeg = raw.copy().pick(eeg_channels, verbose=False)
    return eeg_channels, stim_ch_name, raw_stim, raw_eeg


def _reorder_channels(eeg_channels: list[str]) -> list[int]:
    df = pd.read_excel(DEAP_CHANNELS_XLSX)
    target_order = df["Channel_name_Geneva"].values
    return [eeg_channels.index(ch) for ch in target_order]


def _base_bdf_process(
    data_down: np.ndarray,
    eeg_channels: list[str],
    subject_id: int,
) -> np.ndarray:
    ch_idx = _reorder_channels(eeg_channels)
    trial_idx = _reorder_trials(subject_id)
    epoch_len = data_down.shape[-1]
    data_out = np.zeros(
        (TRIALS_NUM, EEG_ELECTRODES_NUM, epoch_len),
        dtype=data_down.dtype,
    )
    for src, tgt in zip(trial_idx, range(TRIALS_NUM)):
        data_out[tgt] = data_down[src][ch_idx, :].copy()

    return data_out


def _reorder_trials(subject_id: int) -> list[int]:
    ratings = pd.read_csv(DEAP_RATINGS_CSV)
    subj = ratings[ratings["Participant_id"] == subject_id]
    return [
        int(subj.loc[subj["Experiment_id"] == (i + 1), "Trial"].iloc[0]) - 1
        for i in range(TRIALS_NUM)
    ]
