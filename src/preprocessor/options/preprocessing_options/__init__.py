"""Available preprocessing strategies for DEAP EEG data."""

from config import (
    OptionList,
)

from .option_clean import _option_clean
from .option_ica_clean import _option_ica_clean
from .option_unclean import _option_unclean

__all__ = ["PREPROCESSING_OPTIONS"]

PREPROCESSING_OPTIONS: OptionList = OptionList(
    options=[_option_clean, _option_ica_clean, _option_unclean],
)
