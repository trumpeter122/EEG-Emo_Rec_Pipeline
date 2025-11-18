"""Preprocessing entry points for the DEAP pipeline."""

from __future__ import annotations

from typing import cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from config import DEAP_RATINGS_CSV, TRIALS_NUM, PreprocessingOption
from utils import track

from .utils import (
    _load_raw_subject,
    _subject_npy_path,
)

__all__ = ["run_preprocessor"]


def _preprocess_subjects(preprocessing_option: PreprocessingOption) -> None:
    """
    Convert raw BDF files into subject-level numpy arrays.

    Each subject is processed only once; if the destination file already exists
    the subject is skipped to avoid redundant computation.
    """
    out_folder = preprocessing_option.get_subject_path()

    for subject_id in track(
        iterable=range(1, 33),
        description=f"Preprocessing with option {{{preprocessing_option.name}}}",
        context="Preprocessor",
    ):
        out_path = _subject_npy_path(folder=out_folder, subject_id=subject_id)
        if out_path.exists():
            continue

        raw = _load_raw_subject(subject_id=subject_id)
        data_out = preprocessing_option.preprocessing_method(raw, subject_id)

        np.save(file=out_path, arr=data_out)


def _split_trials(preprocessing_option: PreprocessingOption) -> None:
    """
    Split subject-level arrays into per-trial joblib files with metadata.
    """
    source_folder = preprocessing_option.get_subject_path()
    target_folder = preprocessing_option.get_trial_path()

    npy_files = sorted(
        file_path for file_path in source_folder.iterdir() if file_path.suffix == ".npy"
    )
    ratings = pd.read_csv(filepath_or_buffer=DEAP_RATINGS_CSV)

    trial_counter = 0
    for file_path in track(
        iterable=npy_files,
        description="Splitting subject into trials for "
        f"option {{{preprocessing_option.name}}}",
        context="Preprocessor",
    ):
        subject_id = int(file_path.stem[1:3])
        data = np.load(file=file_path)
        subj_mask = ratings["Participant_id"] == subject_id
        subj_ratings = cast("pd.DataFrame", ratings.loc[subj_mask]).sort_values(
            by="Experiment_id",
        )

        for trial_idx in range(TRIALS_NUM):
            trial_data = np.squeeze(a=data[trial_idx])
            row = subj_ratings.iloc[trial_idx]

            trial_df = pd.DataFrame(
                data=[
                    {
                        "data": trial_data,
                        "subject": int(row["Participant_id"]),
                        "trial": int(row["Trial"]),
                        "experiment_id": int(row["Experiment_id"]),
                        "valence": float(row["Valence"]),
                        "arousal": float(row["Arousal"]),
                        "dominance": float(row["Dominance"]),
                        "liking": float(row["Liking"]),
                    },
                ],
            )

            trial_counter += 1
            out_name = f"t{trial_counter:04}.joblib"
            out_path = target_folder / out_name
            if not out_path.exists():
                joblib.dump(value=trial_df, filename=out_path, compress=3)


def run_preprocessor(preprocessing_option: PreprocessingOption) -> None:
    """Execute both preprocessing stages for ``preprocessing_option``."""
    _preprocess_subjects(preprocessing_option=preprocessing_option)
    _split_trials(preprocessing_option=preprocessing_option)
