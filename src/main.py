if __name__ == "__main__":
    # Currently for test runs only
    # So, ignore the codes here at present

    from preprocessor import run_preprocessor
    from preprocessor.options import PREPROCESSING_OPTIONS

    run_preprocessor(PREPROCESSING_OPTIONS.get_name("clean"))
    run_preprocessor(PREPROCESSING_OPTIONS.get_name("ica_clean"))

    # import joblib

    # df = joblib.load("./data/DEAP/generated/cleaned/trial/t01.joblib")
    # print(df)

    # print("data shape: ", df.iloc[0].get("data").shape)

    from config import FeatureExtractionOption
    from feature_extractor import (
        CHANNEL_PICK_OPTIONS,
        FEATURE_OPTIONS,
        SEGMENTATION_OPTIONS,
        run_feature_extractor,
    )

    from itertools import product
    import random

    feop_keys = [
        "preprocessing_option",
        "feature_option",
        "channel_pick_option",
        "segmentation_option",
    ]
    feop_values = [
        PREPROCESSING_OPTIONS.get_names(["clean", "ica_clean"]),
        FEATURE_OPTIONS,
        CHANNEL_PICK_OPTIONS.get_names(["balanced_classic_6"]),
        SEGMENTATION_OPTIONS,
    ]

    feop_combos = [
        dict(zip(feop_keys, feop_combo)) for feop_combo in product(*feop_values)
    ]

    for combo in feop_combos:
        feop = FeatureExtractionOption(**combo)

        run_feature_extractor(feop)
