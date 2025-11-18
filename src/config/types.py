"""Typed configuration objects shared across the preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, overload

from .constants import BASELINE_SEC, DEAP_ROOT, GENEVA_32

if TYPE_CHECKING:
    import mne  # type: ignore[import-untyped]
    import numpy as np

__all__ = [
    "ChannelPickOption",
    "FeatureExtractionOption",
    "FeatureOption",
    "ModelOption",
    "ModelTrainingOption",
    "OptionList",
    "PreprocessingOption",
    "PreprocessingMethod",
    "SegmentationOption",
    "TrainingOption",
    "FeatureChannelExtractionMethod",
]


class _NamedOption(Protocol):
    """Protocol describing option objects with a ``name`` attribute."""

    name: str


_OptionType = TypeVar("_OptionType", bound=_NamedOption)


def _callable_path(callable_obj: Callable[..., Any]) -> str:
    """Return a dotted path for serializing callable references."""
    qualname = getattr(callable_obj, "__qualname__", callable_obj.__name__)
    return f"{callable_obj.__module__}.{qualname}"


class PreprocessingMethod(Protocol):
    """
    Callable signature for preprocessing functions.

    Implementations should ingest an MNE Raw object plus a 1-based DEAP
    subject identifier and return a numpy array shaped
    ``(TRIALS_NUM, EEG_ELECTRODES_NUM, samples)`` that can flow directly into
    trial-splitting.
    """

    def __call__(self, raw: mne.io.BaseRaw, subject_id: int) -> np.ndarray: ...


class FeatureChannelExtractionMethod(Protocol):
    """
    Callable signature for feature extraction per segment.

    ``trial_data`` must contain the full channel set; implementations should
    respect ``channel_pick`` before computing features so the output aligns
    with downstream expectations.
    """

    def __call__(
        self,
        trial_data: np.ndarray,
        channel_pick: list[str],
    ) -> np.ndarray: ...


class OptionList(Sequence[_OptionType]):
    """
    Ordered registry of option objects.

    The class behaves like a regular ``Sequence`` (supporting iteration,
    indexing, and ``len``) while also providing helpers for retrieving options
    by name.  This makes it straightforward to dynamically compose pipelines
    in orchestration scripts.
    """

    def __init__(self, options: Iterable[_OptionType]):
        self._options = list(options)

    @overload
    def __getitem__(self, index: int) -> _OptionType: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[_OptionType]: ...

    def __getitem__(self, index: int | slice) -> _OptionType | Sequence[_OptionType]:
        return self._options[index]

    def __len__(self) -> int:
        return len(self._options)

    def __iter__(self) -> Iterator[_OptionType]:
        return iter(self._options)

    def get_name(self, name: str) -> _OptionType:
        """
        Retrieve a single option by name.

        Raises
        ------
        KeyError
            If the requested name does not exist in the collection.
        """
        for option in self._options:
            if option.name == name:
                return option

        raise KeyError(f'Name "{name}" does not exist.')

    def get_names(self, names: Sequence[str]) -> list[_OptionType]:
        """
        Return a list containing the options that match ``names``.

        The options are returned in the same order as ``names``.
        """
        return [self.get_name(name=name) for name in names]


@dataclass(slots=True)
class PreprocessingOption:
    """
    Container describing a specific preprocessing configuration.

    When adding a new preprocessing routine, instantiate this class inside
    ``PREPROCESSING_OPTIONS`` with a unique ``name`` and ``root_dir``.  The
    ``preprocessing_method`` should handle all signal conditioning before
    the data are serialized via ``run_preprocessor``.
    """

    name: str
    root_dir: str | Path
    preprocessing_method: PreprocessingMethod
    root_path: Path = field(init=False)

    def __post_init__(self) -> None:
        root_dir_path = Path(self.root_dir)
        self.root_path = DEAP_ROOT / "generated" / root_dir_path
        self.root_path.mkdir(parents=True, exist_ok=True)

    def get_subject_path(self) -> Path:
        """
        Directory for subject-level numpy arrays.

        Use this helper instead of hard-coding paths so new options inherit
        the standard ``generated/<root>/subject`` layout.
        """
        path = self.root_path / "subject"
        path.mkdir(exist_ok=True)
        return path

    def get_trial_path(self) -> Path:
        """
        Directory for per-trial joblib data frames produced by splitting.
        """
        path = self.root_path / "trial"
        path.mkdir(exist_ok=True)
        return path

    def get_feature_path(self) -> Path:
        """Directory for storing extracted feature files (joblib + baseline)."""
        path = self.root_path / "feature"
        path.mkdir(exist_ok=True)
        return path

    def to_params(self) -> dict[str, dict[str, str]]:
        """Serialize the option metadata into a JSON-friendly dictionary."""
        return {
            "preprocessing option": {
                "name": self.name,
                "root_path": str(self.root_path),
                "subject_path": str(self.get_subject_path()),
                "trial_path": str(self.get_trial_path()),
                "feature_path": str(self.get_feature_path()),
            },
        }


@dataclass(slots=True)
class ChannelPickOption:
    """
    Subset of EEG channels retained for feature extraction.

    Contributors can register new picks in
    ``feature_extractor/options/options_channel_pick/__init__.py`` as long as
    the channel names exist in the canonical ``GENEVA_32`` order.
    """

    name: str
    channel_pick: list[str]

    def __post_init__(self) -> None:
        invalid = [
            channel_name
            for channel_name in self.channel_pick
            if channel_name not in GENEVA_32
        ]
        if invalid:
            raise ValueError(f"Invalid channel names: {invalid}")

    def to_params(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the option."""
        return {
            "name": self.name,
            "channel_pick": list(self.channel_pick),
        }


@dataclass(slots=True)
class FeatureOption:
    """
    Feature extraction method wrapper used to instantiate pipeline variants.

    Register new feature extractors in
    ``feature_extractor/options/options_feature/__init__.py`` by creating a
    ``FeatureOption`` whose ``feature_channel_extraction_method`` implements
    the ``FeatureChannelExtractionMethod`` protocol.
    """

    name: str
    feature_channel_extraction_method: FeatureChannelExtractionMethod

    def to_params(self) -> dict[str, Any]:
        """Serialize the feature option metadata."""
        return {
            "name": self.name,
            "feature_channel_extraction_method": _callable_path(
                self.feature_channel_extraction_method,
            ),
        }


@dataclass(slots=True)
class SegmentationOption:
    """
    Sliding-window configuration for slicing each trial.

    Time-window constraints are validated up-front, so invalid configurations
    surface during option definition rather than mid-execution.
    """

    time_window: float
    time_step: float
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.time_window <= 0:
            raise ValueError("time_window must be positive.")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive.")
        if self.time_step > self.time_window:
            raise ValueError("time_step cannot exceed time_window.")
        if self.time_window > BASELINE_SEC:
            raise ValueError("time_window cannot exceed the baseline duration.")

        self.name = f"{self.time_window:.2f}s_{self.time_step:.2f}s"

    def to_params(self) -> dict[str, Any]:
        """Serialize the segmentation configuration."""
        return {
            "name": self.name,
            "time_window": self.time_window,
            "time_step": self.time_step,
        }


@dataclass(slots=True)
class FeatureExtractionOption:
    """
    Fully-qualified feature extraction pipeline configuration.

    Combines preprocessing, feature selection, channel-pick, and segmentation
    parameters into a single object.  ``run_feature_extractor`` consumes this
    class and relies on ``extraction_method`` to slice segment data.
    """

    preprocessing_option: PreprocessingOption
    feature_option: FeatureOption
    channel_pick_option: ChannelPickOption
    segmentation_option: SegmentationOption
    name: str = field(init=False)
    extraction_method: Callable[[np.ndarray], np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.name = "+".join(
            [
                self.preprocessing_option.name,
                self.feature_option.name,
                self.channel_pick_option.name,
                self.segmentation_option.name,
            ],
        )
        self.extraction_method = self._build_extraction_method()

    def _build_extraction_method(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Partially apply the feature extractor with the configured channels.

        The returned callable expects a numpy array containing all channels
        for a single segment and handles the channel picking internally.
        """

        def _extract(trial_data: np.ndarray) -> np.ndarray:
            return self.feature_option.feature_channel_extraction_method(
                trial_data=trial_data,
                channel_pick=self.channel_pick_option.channel_pick,
            )

        return _extract

    def to_params(self) -> dict[str, Any]:
        """Serialize the aggregation of the underlying option metadata."""
        return {
            "name": self.name,
            "preprocessing_option": self.preprocessing_option.to_params(),
            "feature_option": self.feature_option.to_params(),
            "channel_pick_option": self.channel_pick_option.to_params(),
            "segmentation_option": self.segmentation_option.to_params(),
        }


@dataclass(slots=True)
class ModelOption:
    """Placeholder describing a concrete model choice."""

    name: str
    model: Any


@dataclass(slots=True)
class TrainingOption:
    """Placeholder describing a set of training hyperparameters."""

    feature_extraction_option: FeatureExtractionOption
    name: str


@dataclass(slots=True)
class ModelTrainingOption:
    """Aggregated configuration used for end-to-end training experiments."""

    model_option: ModelOption
    training_option: TrainingOption
