from importlib import import_module
from typing import Any

from .constants import *  # noqa: F403
from .option_utils import (  # noqa: F401
    CriterionBuilder,
    FeatureChannelExtractionMethod,
    ModelBuilder,
    OptimizerBuilder,
    OptionList,
    PreprocessingMethod,
)

__all__ = [
    "PreprocessingOption",
    "ChannelPickOption",
    "FeatureOption",
    "SegmentationOption",
    "FeatureExtractionOption",
    "BuildDatasetOption",
    "TrainingDataOption",
    "TrainingMethodOption",
    "TrainingOption",
    "ModelOption",
    "ModelTrainingOption",
    "OptionList",
    "PreprocessingMethod",
    "FeatureChannelExtractionMethod",
    "ModelBuilder",
    "OptimizerBuilder",
    "CriterionBuilder",
]

PreprocessingOption: Any
ChannelPickOption: Any
FeatureOption: Any
SegmentationOption: Any
FeatureExtractionOption: Any
BuildDatasetOption: Any
TrainingDataOption: Any
TrainingMethodOption: Any
TrainingOption: Any
ModelOption: Any
ModelTrainingOption: Any

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "PreprocessingOption": ("preprocessor.types", "PreprocessingOption"),
    "ChannelPickOption": ("feature_extractor.types", "ChannelPickOption"),
    "FeatureOption": ("feature_extractor.types", "FeatureOption"),
    "SegmentationOption": ("feature_extractor.types", "SegmentationOption"),
    "FeatureExtractionOption": ("feature_extractor.types", "FeatureExtractionOption"),
    "BuildDatasetOption": ("model_trainer.types", "BuildDatasetOption"),
    "TrainingDataOption": ("model_trainer.types", "TrainingDataOption"),
    "TrainingMethodOption": ("model_trainer.types", "TrainingMethodOption"),
    "TrainingOption": ("model_trainer.types", "TrainingOption"),
    "ModelOption": ("model_trainer.types", "ModelOption"),
    "ModelTrainingOption": ("model_trainer.types", "ModelTrainingOption"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(name)
