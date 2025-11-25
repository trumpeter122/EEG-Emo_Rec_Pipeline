"""Development entry point for running preprocessing + extraction locally."""

from __future__ import annotations

from feature_extractor import (
    CHANNEL_PICK_OPTIONS,
    FEATURE_OPTIONS,
    SEGMENTATION_OPTIONS,
)
from model_trainer.options import MODEL_OPTIONS, TRAINING_METHOD_OPTIONS
from pipeline_runner import TrainingExperiment, run_pipeline
from preprocessor.options import PREPROCESSING_OPTIONS


def main() -> None:
    """Run the default preprocessing + training suite used during development."""
    experiments = [
        # TrainingExperiment(
        #     target="valence",
        #     target_kind="regression",
        #     feature_scaler="minmax",
        #     training_method_name="adam_regression",
        #     model_name="cnn1d_n1_regression",
        #     random_seed=42,
        #     use_size=0.3,
        #     test_size=0.2,
        # ),
        # TrainingExperiment(
        #     target="valence",
        #     target_kind="classification",
        #     feature_scaler="minmax",
        #     training_method_name="adam_classification",
        #     model_name="cnn1d_n1_classification",
        #     random_seed=42,
        #     use_size=0.3,
        #     test_size=0.2,
        #     class_labels_expected=[float(index) for index in range(1, 10)],
        # ),
        TrainingExperiment(
            target="valence",
            target_kind="classification",
            feature_scaler="standard",
            training_method_name="sklearn_default_classification",
            model_name="linear_svc_sklearn",
            random_seed=42,
            use_size=0.3,
            test_size=0.2,
            class_labels_expected=[float(index) for index in range(1, 10)],
        ),
        TrainingExperiment(
            target="valence",
            target_kind="regression",
            feature_scaler="standard",
            training_method_name="sklearn_default_regression",
            model_name="gb_regression_sklearn",
            random_seed=42,
            use_size=0.3,
            test_size=0.2,
        ),
    ]

    run_pipeline(
        preprocessing_options=PREPROCESSING_OPTIONS.get_names(
            names=["clean", "unclean"],
        ),
        feature_options=list(
            FEATURE_OPTIONS.get_names(names=["psd", "de", "deasm", "deasm"]),
        ),
        channel_pick_options=list(
            CHANNEL_PICK_OPTIONS.get_names(
                names=[
                    "minimal_frontal_parietal",
                    "balanced_classic_6",
                    "optimized_gold_standard_8",
                    "standard_32",
                ],
            ),
        ),
        segmentation_options=list(SEGMENTATION_OPTIONS.get_names(["2.00s_0.25s"])),
        experiments=experiments,
        model_options=MODEL_OPTIONS,
        training_method_options=TRAINING_METHOD_OPTIONS,
    )


if __name__ == "__main__":
    main()
