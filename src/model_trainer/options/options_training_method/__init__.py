"""Selectable optimizer/criterion configurations for model training."""

from __future__ import annotations

from config.option_utils import OptionList
from model_trainer.types import TrainingMethodOption  # noqa: TCH001

from .option_adam_classification import _adam_classification
from .option_adam_regression import _adam_regression
from .option_sklearn import _sklearn_classification, _sklearn_regression

__all__ = ["TRAINING_METHOD_OPTIONS"]

TRAINING_METHOD_OPTIONS: OptionList[TrainingMethodOption] = OptionList(
    options=[
        _adam_regression,
        _adam_classification,
        _sklearn_classification,
        _sklearn_regression,
    ],
)
