from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import TYPE_CHECKING, TypeVar

from config.option_utils import OptionList, _NamedOption
from feature_extractor import run_feature_extractor
from feature_extractor.types import FeatureExtractionOption
from model_trainer import run_model_trainer
from model_trainer.types import ModelTrainingOption, TrainingDataOption, TrainingOption
from preprocessor import run_preprocessor

if TYPE_CHECKING:
    from feature_extractor.types import (
        ChannelPickOption,
        FeatureOption,
        SegmentationOption,
    )
    from model_trainer.types import (
        BuildDatasetOption,
        ModelOption,
        TrainingMethodOption,
    )
    from preprocessor.types import PreprocessingOption

_Opt = TypeVar("_Opt", bound=_NamedOption)


def _ensure_option_list(
    options: OptionList[_Opt] | Sequence[_Opt] | _Opt,
) -> OptionList[_Opt]:
    """
    Normalize inputs to an OptionList so downstream loops stay unchanged.
    """
    if isinstance(options, OptionList):
        return options
    if isinstance(options, Sequence):
        return OptionList(options)
    return OptionList([options])


def run_pipeline(
    preprocessing_options: OptionList[PreprocessingOption]
    | Sequence[PreprocessingOption]
    | PreprocessingOption,
    channel_pick_options: OptionList[ChannelPickOption]
    | Sequence[ChannelPickOption]
    | ChannelPickOption,
    feature_options: OptionList[FeatureOption]
    | Sequence[FeatureOption]
    | FeatureOption,
    segmentation_options: OptionList[SegmentationOption]
    | Sequence[SegmentationOption]
    | SegmentationOption,
    model_options: OptionList[ModelOption] | Sequence[ModelOption] | ModelOption,
    build_dataset_options: OptionList[BuildDatasetOption]
    | Sequence[BuildDatasetOption]
    | BuildDatasetOption,
    training_method_options: OptionList[TrainingMethodOption]
    | Sequence[TrainingMethodOption]
    | TrainingMethodOption,
) -> None:
    preprocessing_options_list = _ensure_option_list(preprocessing_options)
    channel_pick_options_list = _ensure_option_list(channel_pick_options)
    feature_options_list = _ensure_option_list(feature_options)
    segmentation_options_list = _ensure_option_list(segmentation_options)
    model_options_list = _ensure_option_list(model_options)
    build_dataset_options_list = _ensure_option_list(build_dataset_options)
    training_method_options_list = _ensure_option_list(training_method_options)

    for preprocessing_option in preprocessing_options_list:
        run_preprocessor(preprocessing_option=preprocessing_option)

        feature_extraction_options = OptionList(
            [
                FeatureExtractionOption(
                    preprocessing_option=preprocessing_option,
                    channel_pick_option=channel_pick_option,
                    feature_option=feature_option,
                    segmentation_option=segmentation_option,
                )
                for (
                    channel_pick_option,
                    feature_option,
                    segmentation_option,
                ) in product(
                    channel_pick_options_list,
                    feature_options_list,
                    segmentation_options_list,
                )
            ]
        )

        for feature_extraction_option in feature_extraction_options:
            run_feature_extractor(feature_extraction_option=feature_extraction_option)

            training_data_options = OptionList(
                [
                    TrainingDataOption(
                        feature_extraction_option=feature_extraction_option,
                        build_dataset_option=build_dataset_option,
                    )
                    for build_dataset_option in build_dataset_options_list
                ]
            )

            _tmp_training_options = []
            for training_data_option, training_method_option in product(
                training_data_options, training_method_options_list
            ):
                try:
                    _tmp_training_options.append(
                        TrainingOption(
                            training_data_option=training_data_option,
                            training_method_option=training_method_option,
                        )
                    )
                except ValueError:
                    continue
            training_options = OptionList(_tmp_training_options)

            _tmp_model_training_options = []
            for training_option, model_option in product(
                training_options, model_options_list
            ):
                try:
                    _tmp_model_training_options.append(
                        ModelTrainingOption(
                            training_option=training_option,
                            model_option=model_option,
                        )
                    )
                except ValueError:
                    continue
            model_training_options = OptionList(_tmp_model_training_options)

            for model_training_option in model_training_options:
                run_model_trainer(model_training_option=model_training_option)


# Backwards compatibility with existing imports
run_pipeline_runner = run_pipeline
