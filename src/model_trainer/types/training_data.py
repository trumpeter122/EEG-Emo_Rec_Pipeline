"""Options describing how extracted features become datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.preprocessing import (  # type: ignore[import-untyped]
    MinMaxScaler,
    StandardScaler,
)

from .dataset import SegmentDataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.utils.data import Dataset

    from feature_extractor.types import FeatureExtractionOption

__all__ = ["TrainingDataOption"]


@dataclass(slots=True)
class TrainingDataOption:
    """
    Recipe for transforming extracted feature frames into train/test datasets.

    - This option is responsible for loading the joblib frames, selecting the
      requested target column, encoding classification labels, scaling the
      feature tensors, and generating reproducible train/test segment splits.
    - The resulting torch ``Dataset`` objects are attached for downstream
      dataloader construction, and metadata helpers expose the exact label set
      so metrics can be mapped back to user-friendly values.
    """

    feature_extraction_option: FeatureExtractionOption
    target: str
    random_seed: int
    use_size: float
    test_size: float
    target_kind: Literal["regression", "classification"]
    feature_scaler: Literal["none", "standard", "minmax"]
    class_labels_expected: Sequence[float] | None = None

    name: str = field(init=False)
    train_dataset: Dataset[tuple[np.ndarray, float]] = field(init=False)
    test_dataset: Dataset[tuple[np.ndarray, float]] = field(init=False)
    segment_splits: dict[str, list[int]] = field(init=False)
    class_labels: list[float] | None = field(init=False, default=None)
    _target_dtype: np.dtype[Any] = field(init=False)
    _frame: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_sizes()
        self.name = self._build_name()

        frame = self._load_feature_frame()
        trimmed = cast(
            "pd.DataFrame",
            frame.loc[:, ["data", self.target]].reset_index(drop=True),
        )
        trimmed = self._encode_targets(frame=trimmed)
        trimmed = self._scale_feature_column(frame=trimmed)
        self._frame = trimmed

        splits = self._generate_segment_splits(total=len(trimmed))
        self.segment_splits = splits
        self.train_dataset = self._build_dataset(
            frame=self._frame,
            indices=splits["train-segments"],
        )
        self.test_dataset = self._build_dataset(
            frame=self._frame,
            indices=splits["test-segments"],
        )

    def _validate_sizes(self) -> None:
        if not 0 < self.use_size <= 1:
            raise ValueError("use_size must fall within the interval (0, 1].")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must fall within the interval (0, 1).")

    def _build_name(self) -> str:
        return (
            f"{self.feature_extraction_option.name}"
            f"|target={self.target}"
            f"|use={self.use_size:.2f}"
            f"|test={self.test_size:.2f}"
            f"|seed={self.random_seed}"
        )

    def _load_feature_frame(self) -> pd.DataFrame:
        file_paths = self.feature_extraction_option.get_file_paths()
        frames: list[pd.DataFrame] = []
        for filename in file_paths:
            frame_obj = joblib.load(filename=filename)
            if not isinstance(frame_obj, pd.DataFrame):
                raise TypeError(
                    f"Feature file {filename} did not contain a pandas DataFrame.",
                )
            frames.append(frame_obj)

        if not frames:
            raise ValueError(
                "No feature data found for option "
                f"{self.feature_extraction_option.name}.",
            )

        frame = cast("pd.DataFrame", pd.concat(frames, axis=0, ignore_index=True))
        if self.target not in frame.columns:
            raise KeyError(
                f'Target column "{self.target}" is missing from the features frame.',
            )
        if "data" not in frame.columns:
            raise KeyError('Column "data" is missing from the features frame.')

        return frame

    def _encode_targets(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame.loc[:, self.target].to_numpy(copy=True)
        if self.target_kind == "classification":
            if self.class_labels_expected is not None:
                unique_values = [float(value) for value in self.class_labels_expected]
            else:
                unique_values = sorted({float(value) for value in values})
            if len(unique_values) < 2:
                raise ValueError(
                    "classification targets must contain at least two classes.",
                )
            if self.class_labels_expected is not None:
                label_array = np.asarray(unique_values, dtype=np.float32)
                value_array = values.astype(np.float32)
                encoded = (
                    np.abs(value_array[:, None] - label_array[None, :])
                    .argmin(
                        axis=1,
                    )
                    .astype(np.int64)
                )
            else:
                mapping = {label: index for index, label in enumerate(unique_values)}
                encoded = np.asarray(
                    [mapping[float(value)] for value in values],
                    dtype=np.int64,
                )
            self.class_labels = unique_values
            self._target_dtype = np.dtype(np.int64)
            frame = frame.copy()
            frame.loc[:, self.target] = encoded
        elif self.target_kind == "regression":
            frame = frame.copy()
            frame.loc[:, self.target] = values.astype(np.float32)
            self.class_labels = None
            self._target_dtype = np.dtype(np.float32)
        else:
            raise ValueError(f"Unsupported target_kind: {self.target_kind}")
        return frame

    def _scale_feature_column(self, frame: pd.DataFrame) -> pd.DataFrame:
        arrays = [
            np.asarray(feature, dtype=np.float32) for feature in frame["data"].tolist()
        ]
        if not arrays:
            raise ValueError("No feature arrays were found to scale.")

        if self.feature_scaler == "none":
            frame = frame.copy()
            frame.loc[:, "data"] = arrays
            return frame

        flattened = np.stack([array.reshape(-1) for array in arrays], axis=0)
        if self.feature_scaler == "standard":
            scaler = StandardScaler()
        elif self.feature_scaler == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported feature_scaler: {self.feature_scaler}")
        scaled = scaler.fit_transform(flattened)
        reshaped = [
            scaled[row_index].reshape(arrays[row_index].shape)
            for row_index in range(len(arrays))
        ]
        frame = frame.copy()
        frame.loc[:, "data"] = reshaped
        return frame

    def _generate_segment_splits(self, total: int) -> dict[str, list[int]]:
        if total < 2:
            raise ValueError("At least two segments are required for training.")

        usable = max(1, int(total * self.use_size))
        if usable < 2:
            raise ValueError(
                "use_size selects fewer than two segments; increase use_size.",
            )

        rng = random.Random(self.random_seed)
        indices = list(range(total))
        rng.shuffle(indices)
        used_indices = indices[:usable]

        test_count = max(1, int(round(len(used_indices) * self.test_size)))
        if test_count >= len(used_indices):
            raise ValueError(
                "test_size allocates all usable segments to testing; "
                "decrease test_size or increase use_size.",
            )

        test_indices = sorted(used_indices[:test_count])
        train_indices = sorted(used_indices[test_count:])

        return {
            "train-segments": train_indices,
            "test-segments": test_indices,
        }

    def _build_dataset(
        self,
        *,
        frame: pd.DataFrame,
        indices: list[int],
    ) -> Dataset[tuple[np.ndarray, float]]:
        subset = frame.iloc[indices].reset_index(drop=True)
        features = subset["data"].tolist()
        targets = subset[self.target].tolist()
        return SegmentDataset(
            features=features,
            targets=targets,
            target_dtype=self._target_dtype,
        )

    def _arrays_from_indices(self, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Build feature/target arrays for sklearn-compatible training flows.

        Features are flattened per-segment to a 2D matrix while preserving the
        target dtype set during encoding.
        """
        subset = self._frame.iloc[indices].reset_index(drop=True)
        features = [
            np.asarray(feature, dtype=np.float32).reshape(-1)
            for feature in subset["data"].tolist()
        ]
        targets = subset[self.target].to_numpy(dtype=self._target_dtype, copy=True)
        return np.stack(features, axis=0), targets

    def get_numpy_splits(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Return ``(x_train, y_train), (x_test, y_test)`` for sklearn flows."""
        train_indices = self.segment_splits["train-segments"]
        test_indices = self.segment_splits["test-segments"]
        return (
            self._arrays_from_indices(indices=train_indices),
            self._arrays_from_indices(indices=test_indices),
        )

    def to_params(self) -> dict[str, Any]:
        """Serialize training configuration metadata."""
        return {
            "name": self.name,
            "feature_extraction_option": self.feature_extraction_option.to_params(),
            "target": self.target,
            "random_seed": self.random_seed,
            "use_size": self.use_size,
            "test_size": self.test_size,
            "target_kind": self.target_kind,
            "feature_scaler": self.feature_scaler,
            "class_labels": list(self.class_labels) if self.class_labels else None,
            "class_labels_expected": (
                list(self.class_labels_expected)
                if self.class_labels_expected is not None
                else None
            ),
        }

    def get_class_values(self) -> np.ndarray | None:
        if not self.class_labels:
            return None
        return np.asarray(self.class_labels, dtype=np.float32)
