from itertools import product

from config import BuildDatasetOption, OptionList

__all__: list[str] = ["BUILD_DATASET_OPTIONS"]

BUILD_DATASET_OPTIONS = OptionList(
    [
        BuildDatasetOption(
            target=target,
            random_seed=42,
            use_size=1.0,
            test_size=0.2,
            target_kind=target_kind,
            feature_scaler=feature_scaler,
        )
        for target, use_size, target_kind, feature_scaler in product(
            ["valence", "arousal"],
            [1.0, 0.3],
            ["classification", "regression"],
            ["standard", "minmax"],
        )
    ]
)
