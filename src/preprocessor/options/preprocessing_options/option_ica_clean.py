import mne
import numpy as np

from config import (
    EEG_ELECTRODES_NUM,
    EPOCH_TMAX,
    EPOCH_TMIN,
    SFREQ_TARGET,
)

from .utils import (
    _apply_filter_reference,
    _base_bdf_process,
    _get_events,
    _prepare_channels,
)


def _ica_clean_bdf(
    raw: mne.io.BaseRaw,
    subject_id: int,
) -> np.ndarray:
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

    eog_inds, ecg_inds = [], []
    # Use frontal channels as EOG proxies so ICA can flag blink artifacts.
    eog_proxy_names = [ch for ch in ("Fp1", "Fp2") if ch in raw.ch_names]
    if eog_proxy_names:
        raw_eog = raw.copy()
        raw_eog.set_channel_types(
            dict.fromkeys(eog_proxy_names, "eog"),
            verbose=False,
        )
        try:
            eog_inds, _ = ica.find_bads_eog(
                raw_eog, ch_name=eog_proxy_names, verbose=False
            )
        except RuntimeError as e:
            # If MNE still complains about missing EOG picks, fall back silently.
            if not (e.args and "EOG channel" in e.args[0]):
                raise e

    comp_var = np.var(ica.get_sources(inst=raw_eeg).get_data(), axis=1)
    highpower_inds = np.where(comp_var > np.percentile(comp_var, 99))[0].tolist()

    ica.exclude = sorted(set(eog_inds + ecg_inds + highpower_inds))

    cleaned = ica.apply(inst=epochs.copy(), verbose=False)
    data_down = cleaned.resample(sfreq=SFREQ_TARGET).get_data()

    return _base_bdf_process(
        data_down=data_down,
        eeg_channels=eeg_channels,
        subject_id=subject_id,
    )
