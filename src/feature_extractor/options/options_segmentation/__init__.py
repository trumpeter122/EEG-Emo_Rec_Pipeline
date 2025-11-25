"""Available segmentation strategies for the sliding-window extractor."""

from config.option_utils import OptionList
from feature_extractor.types import SegmentationOption

__all__ = ["SEGMENTATION_OPTIONS"]

SEGMENTATION_OPTIONS: OptionList[SegmentationOption] = OptionList(
    options=[
        SegmentationOption(time_window=2.0, time_step=0.25),
        SegmentationOption(time_window=3.0, time_step=0.15),
    ],
)
