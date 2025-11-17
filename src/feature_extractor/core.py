import joblib
import numpy as np
import pandas as pd

from config import (
    BASELINE_SEC,
    SFREQ,
    FeatureExtractionOption,
)
from utils import track


def _extract_feature(
    trial_df: pd.DataFrame,
    feature_extraction_option: FeatureExtractionOption,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Slice a single trial into overlapping segments and compute features
    for each segment.
    ...
    """
    trial = trial_df.iloc[0]
    fe_method = feature_extraction_option.extraction_method

    segmentation_option = feature_extraction_option.segmentation_option
    window = int(segmentation_option.time_window * SFREQ)
    step = int(segmentation_option.time_step * SFREQ)

    trial_data = trial.get("data")  # (32 channels * sampling points)
    trial_data = np.asarray(trial_data)
    if trial_data.ndim == 1:
        trial_data = trial_data[np.newaxis, :]

    _, n_samples = trial_data.shape
    if n_samples < window:
        raise ValueError("time_window is longer than the trial length.")

    baseline_samples = int(BASELINE_SEC * SFREQ)
    if window > baseline_samples:
        raise ValueError("time_window exceeds available baseline duration.")

    # Baseline segment: last window within the baseline portion
    baseline_start = max(baseline_samples - window, 0)
    baseline_segment = trial_data[:, baseline_start : baseline_start + window]
    baseline_feature = fe_method(baseline_segment)

    # Trial segments: start immediately after baseline
    trial_start = baseline_samples
    if trial_start >= n_samples:
        raise ValueError("trial segment starts beyond the available samples.")

    trial_samples = n_samples - baseline_samples
    if trial_samples < window:
        raise ValueError(
            "time_window is longer than the available trial duration after baseline."
        )
    relative_starts = np.arange(
        0,
        trial_samples - window + 1,
        step,
        dtype=int,
    )
    trial_starts = trial_start + relative_starts
    trial_features: list[np.ndarray] = []
    for s in trial_starts:
        segment = trial_data[:, s : s + window]
        trial_features.append(fe_method(segment))

    if not trial_features:
        raise ValueError(
            "No trial segments were generated; check time_window and time_step."
        )

    out_df = pd.DataFrame(
        {
            "data": [trial_features],
            **{col: trial[col] for col in trial.index if col != "data"},
        }
    )

    # Explode so each segment slice becomes its own row
    out_df = out_df.explode(["data"], ignore_index=True)

    return baseline_feature, out_df


def run_feature_extractor(
    feature_extraction_option: FeatureExtractionOption,
):
    preprocessing_option = feature_extraction_option.preprocessing_option

    trials_path = preprocessing_option.get_trial_path()
    features_path = preprocessing_option.get_feature_path()
    trial_files = sorted(trials_path.glob("*.joblib"))
    for trial_file in track(
        iterable=trial_files,
        description="Extracting feature with"
        f"option {{{feature_extraction_option.name}}} for "
        f"option {{{preprocessing_option.name}}}",
        context="Feature Extractor",
    ):
        out_dir_path = features_path / feature_extraction_option.name
        out_dir_path.mkdir(exist_ok=True)
        out_name = f"{trial_file.stem}.joblib"
        out_path = out_dir_path / out_name

        baseline_out_dir_path = out_dir_path / "baseline"
        baseline_out_dir_path.mkdir(exist_ok=True)
        baseline_out_name = f"{trial_file.stem}.npy"
        baseline_out_path = baseline_out_dir_path / baseline_out_name

        if not (out_path.exists() and baseline_out_path.exists()):
            trial_df = joblib.load(trial_file)

            baseline_array, feature_df = _extract_feature(
                trial_df=trial_df, feature_extraction_option=feature_extraction_option
            )

            np.save(baseline_out_path, baseline_array)

            joblib.dump(feature_df, out_path, compress=3)
