"""Development entry point for running preprocessing + extraction locally."""

from __future__ import annotations

from feature_extractor.options import (
    CHANNEL_PICK_OPTIONS,
    FEATURE_OPTIONS,
    SEGMENTATION_OPTIONS,
)
from model_trainer.options import (
    BUILD_DATASET_OPTIONS,
    MODEL_OPTIONS,
    TRAINING_METHOD_OPTIONS,
)
from pipeline_runner import run_pipeline
from preprocessor.options import PREPROCESSING_OPTIONS

print(
    "\n".join(
        [
            "PREPROCESSING_OPTIONS:",
            f"{PREPROCESSING_OPTIONS}",
            "",
            "CHANNEL_PICK_OPTIONS:",
            f"{CHANNEL_PICK_OPTIONS}",
            "",
            "FEATURE_OPTIONS:",
            f"{FEATURE_OPTIONS}",
            "",
            "SEGMENTATION_OPTIONS:",
            f"{SEGMENTATION_OPTIONS}",
            "",
            "BUILD_DATASET_OPTIONS:",
            f"{BUILD_DATASET_OPTIONS}",
            "",
            "MODEL_OPTIONS:",
            f"{MODEL_OPTIONS}",
            "",
            "TRAINING_METHOD_OPTIONS:",
            f"{TRAINING_METHOD_OPTIONS}",
            "",
        ]
    )
)

# Run for some combinations
# run_pipeline(
#     preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean", "unclean"]),
#     channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(
#         [
#             "minimal_frontal_parietal",
#             "balanced_classic_6",
#             "optimized_gold_standard_8",
#             "standard_32",
#         ]
#     ),
#     feature_options=FEATURE_OPTIONS.get_names(["psd", "deasm", "dasm", "de"]),
#     segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
#     model_options=MODEL_OPTIONS.get_names(
#         # ["cnn1d_n1_regression", "cnn1d_n1_classification"]
#         ["sgd_classifier_sklearn"]
#     ),
#     build_dataset_options=BUILD_DATASET_OPTIONS,
#     training_method_options=TRAINING_METHOD_OPTIONS.get_names(
#         [
#             "adam_regression",
#             "adam_classification",
#             "sklearn_default_classification",
#             "sklearn_default_regression",
#         ]
#     ),
# )

# Add more here
# run_pipeline()
