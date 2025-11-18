"""Development entry point for running preprocessing + extraction locally."""

from __future__ import annotations

from itertools import product

from config import FeatureExtractionOption
from feature_extractor import (
    CHANNEL_PICK_OPTIONS,
    FEATURE_OPTIONS,
    SEGMENTATION_OPTIONS,
    run_feature_extractor,
)
from preprocessor import run_preprocessor
from preprocessor.options import PREPROCESSING_OPTIONS
import json


def _run_extraction_examples() -> None:
    """Execute a small subset of preprocessing + feature extraction variants."""
    run_preprocessor(preprocessing_option=PREPROCESSING_OPTIONS.get_name(name="clean"))
    run_preprocessor(
        preprocessing_option=PREPROCESSING_OPTIONS.get_name(name="ica_clean"),
    )

    feop_keys = [
        "preprocessing_option",
        "feature_option",
        "channel_pick_option",
        "segmentation_option",
    ]
    feop_values = [
        PREPROCESSING_OPTIONS.get_names(names=["clean"]),
        FEATURE_OPTIONS,
        CHANNEL_PICK_OPTIONS.get_names(names=["standard_32"]),
        SEGMENTATION_OPTIONS,
    ]
    feop_combos = [
        {key: value for key, value in zip(feop_keys, feop_combo)}
        for feop_combo in product(*feop_values)
    ]

    for combo in feop_combos:
        fe_option = FeatureExtractionOption(**combo)
        print(json.dumps(fe_option.to_params(), indent=2))
        run_feature_extractor(feature_extraction_option=fe_option)


if __name__ == "__main__":
    _run_extraction_examples()
