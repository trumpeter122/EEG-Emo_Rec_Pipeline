"""Feature extraction pipeline for DEAP trials."""

from __future__ import annotations

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from config import (
    BASELINE_SEC,
    SFREQ,
    FeatureExtractionOption,
)
from utils import track

__all__ = ["run_feature_extractor"]


def _extract_feature(
    trial_df: pd.DataFrame,
    feature_extraction_option: FeatureExtractionOption,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Segment a trial into overlapping windows and compute features for each slice.

    The baseline window always corresponds to the most recent segment that fits
    within the annotated baseline period. All subsequent segments start after
    the baseline.
    """
    trial = trial_df.iloc[0]
    fe_method = feature_extraction_option.extraction_method

    segmentation_option = feature_extraction_option.segmentation_option
    window = int(segmentation_option.time_window * SFREQ)
    step = int(segmentation_option.time_step * SFREQ)

    trial_data = np.asarray(trial.get("data"))
    if trial_data.ndim == 1:
        trial_data = trial_data[np.newaxis, :]

    _, n_samples = trial_data.shape
    if n_samples < window:
        raise ValueError("time_window is longer than the trial length.")

    baseline_samples = int(BASELINE_SEC * SFREQ)
    if window > baseline_samples:
        raise ValueError("time_window exceeds available baseline duration.")

    baseline_start = max(baseline_samples - window, 0)
    baseline_segment = trial_data[:, baseline_start : baseline_start + window]
    baseline_feature = fe_method(baseline_segment)

    trial_start = baseline_samples
    if trial_start >= n_samples:
        raise ValueError("trial segment starts beyond the available samples.")

    trial_samples = n_samples - baseline_samples
    if trial_samples < window:
        raise ValueError(
            "time_window is longer than the available trial duration after baseline.",
        )

    relative_starts = np.arange(
        start=0,
        stop=trial_samples - window + 1,
        step=step,
        dtype=int,
    )
    trial_starts = trial_start + relative_starts
    trial_features: list[np.ndarray] = []
    for start in trial_starts:
        segment = trial_data[:, start : start + window]
        trial_features.append(fe_method(segment))

    if not trial_features:
        raise ValueError(
            "No trial segments were generated; check time_window and time_step.",
        )

    out_df = pd.DataFrame(
        data={
            "data": [trial_features],
            **{col: trial[col] for col in trial.index if col != "data"},
        },
    ).explode(column=["data"], ignore_index=True)

    return baseline_feature, out_df


def run_feature_extractor(
    feature_extraction_option: FeatureExtractionOption,
) -> None:
    """Extract baseline arrays and per-segment features for all trials."""
    preprocessing_option = feature_extraction_option.preprocessing_option

    trials_path = preprocessing_option.get_trial_path()
    features_path = preprocessing_option.get_feature_path()
    trial_files = sorted(trials_path.glob("*.joblib"))
    for trial_file in track(
        iterable=trial_files,
        description="Extracting feature with "
        f"option {{{feature_extraction_option.name}}} for "
        f"option {{{preprocessing_option.name}}}",
        context="Feature Extractor",
    ):
        out_dir_path = features_path / feature_extraction_option.name
        out_dir_path.mkdir(exist_ok=True)
        out_path = out_dir_path / f"{trial_file.stem}.joblib"

        baseline_out_dir_path = out_dir_path / "baseline"
        baseline_out_dir_path.mkdir(exist_ok=True)
        baseline_out_path = baseline_out_dir_path / f"{trial_file.stem}.npy"

        if out_path.exists() and baseline_out_path.exists():
            continue

        trial_df = joblib.load(filename=trial_file)
        baseline_array, feature_df = _extract_feature(
            trial_df=trial_df,
            feature_extraction_option=feature_extraction_option,
        )

        np.save(file=baseline_out_path, arr=baseline_array)
        joblib.dump(value=feature_df, filename=out_path, compress=3)
