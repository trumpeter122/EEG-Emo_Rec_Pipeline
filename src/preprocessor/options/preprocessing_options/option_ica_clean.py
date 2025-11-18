"""Cleaning pipeline that removes ICA components linked to artifacts."""

import mne  # type: ignore[import-untyped]
import numpy as np

from config import (
    EEG_ELECTRODES_NUM,
    EPOCH_TMAX,
    EPOCH_TMIN,
    SFREQ_TARGET,
    PreprocessingOption,
)

from .utils import (
    _apply_filter_reference,
    _base_bdf_process,
    _get_events,
    _prepare_channels,
)

__all__ = ["_option_ica_clean"]


def _ica_clean_bdf(
    raw: mne.io.BaseRaw,
    subject_id: int,
) -> np.ndarray:
    """Apply ICA artefact removal followed by canonical ordering."""
    eeg_channels, stim_ch, raw_stim, raw_eeg = _prepare_channels(raw=raw)
    events = _get_events(
        raw_stim=raw_stim,
        stim_ch_name=stim_ch,
        subject_id=subject_id,
    )

    _apply_filter_reference(raw=raw_eeg)

    epochs = mne.Epochs(
        raw=raw_eeg,
        events=events,
        event_id=4,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        picks=eeg_channels,
        baseline=None,
        preload=True,
        verbose=False,
    )

    n_ica = EEG_ELECTRODES_NUM - 1
    ica = mne.preprocessing.ICA(
        n_components=n_ica,
        method="fastica",
        random_state=23,
        max_iter="auto",
    )
    ica.fit(inst=epochs, verbose=False)

    eog_inds: list[int] = []
    ecg_inds: list[int] = []
    eog_proxy_names = [ch for ch in ("Fp1", "Fp2") if ch in raw.ch_names]
    if eog_proxy_names:
        raw_eog = raw.copy()
        raw_eog.set_channel_types(
            mapping=dict.fromkeys(eog_proxy_names, "eog"),
            verbose=False,
        )
        try:
            eog_inds, _ = ica.find_bads_eog(
                inst=raw_eog,
                ch_name=eog_proxy_names,
                verbose=False,
            )
        except RuntimeError as error:
            if not (error.args and "EOG channel" in error.args[0]):
                raise error

    comp_var = np.var(a=ica.get_sources(inst=raw_eeg).get_data(), axis=1)
    highpower_inds = np.where(comp_var > np.percentile(comp_var, 99))[0].tolist()

    ica.exclude = sorted({*eog_inds, *ecg_inds, *highpower_inds})

    cleaned = ica.apply(inst=epochs.copy(), verbose=False)
    data_down = cleaned.resample(sfreq=SFREQ_TARGET).get_data()

    return _base_bdf_process(
        data_down=data_down,
        eeg_channels=eeg_channels,
        subject_id=subject_id,
    )


_option_ica_clean = PreprocessingOption(
    name="ica_clean",
    root_dir="ica_cleaned",
    preprocessing_method=_ica_clean_bdf,
)
