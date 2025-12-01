# Development Guide: Building and Extending the Option-Driven Pipeline

This codebase is designed around *options*—small, immutable configuration objects that describe every choice in the EEG pipeline. Each module exposes an `OptionList` registry so combinations can be enumerated automatically by `pipeline_runner.run_pipeline`. Use this guide whenever you add a preprocessing recipe, feature extractor, dataset recipe, model, or training loop.

## Core Principles
- **Single source of truth**: Each option lives beside the code it configures and is exported through that module’s `OptionList` in `__init__.py`. Names must be unique and stable because they drive file-system layout (`data/generated/*`, `results/*`) and appear in metadata JSON.
- **Explicit interfaces**: Every callable is typed via protocols in `config.option_utils` (e.g., `PreprocessingMethod`, `FeatureChannelExtractionMethod`, `ModelBuilder`). Match the required signature exactly.
- **Composable stages**: `run_pipeline` iterates over all registered options and builds Cartesian products (preprocess → feature extraction → dataset → training → model). Keep options compatible across stages (target kind, backend, expected shapes).
- **Determinism and safety**: Use provided helpers (atomic writes, shape guards, reproducible seeds) instead of ad-hoc IO or randomization. Avoid changing existing option names to preserve reproducibility.

## Repository Map
- `src/preprocessor`: Loads DEAP BDF files, filters/re-references/epochs them, splits into trials, and writes artifacts under `data/DEAP/generated/<root_dir>/`.
- `src/feature_extractor`: Segments trials, computes baseline + per-window features, and stores them under the preprocessing root in `feature/<pre+feat+ch+seg>/`.
- `src/model_trainer`: Turns features into train/test datasets, builds dataloaders or sklearn matrices, trains models, and writes runs to `results/<target>/<target_kind>/<timestamp>/`.
- `src/pipeline_runner`: Builds combinations from option registries and invokes each stage in order.
- `src/config/option_utils.py`: Protocols (`PreprocessingMethod`, `FeatureChannelExtractionMethod`, `ModelBuilder`, `OptimizerBuilder`, `CriterionBuilder`) and the `OptionList` registry wrapper.

## How to Add or Modify Options

### 1) Preprocessing Options
Path: `src/preprocessor/options/options_preprocessing/option_<label>.py`  
Type: `PreprocessingOption(name: str, root_dir: str | Path, preprocessing_method: PreprocessingMethod)`

Interface for `preprocessing_method`: `(raw: mne.io.BaseRaw, subject_id: int) -> np.ndarray` returning an array shaped `(TRIALS_NUM, EEG_ELECTRODES_NUM, samples)`.

Steps:
- Use helpers from `options_preprocessing/utils.py`:
  - `_prepare_channels` → split EEG vs stim channels.
  - `_apply_filter_reference` → notch + band-pass + re-reference.
  - `_get_events` → event extraction with DEAP fixups.
  - `_epoch_and_resample` → epoch and resample to `SFREQ_TARGET`.
  - `_base_bdf_process` → reorder trials/channels into canonical DEAP order.
- Choose a **unique `name`** and a **`root_dir`** (folder under `data/DEAP/generated/`). Changing either moves the output.
- Implement the method, ensuring **consistent shapes across subjects** (the preprocessor asserts uniformity). Preserve dtype (`float32` recommended).
- Instantiate and export a module-level option (e.g., `_option_myvariant = PreprocessingOption(...)`).
- Register it in `src/preprocessor/options/options_preprocessing/__init__.py` inside `PREPROCESSING_OPTIONS` and include in `__all__`.

### 2) Feature Extraction Options
#### a) Feature methods
Path: `src/feature_extractor/options/options_feature/option_<label>.py`  
Type: `FeatureOption(name: str, feature_channel_extraction_method: FeatureChannelExtractionMethod)`

Interface: `(trial_data: np.ndarray, channel_pick: list[str]) -> np.ndarray` where `trial_data` is a single segment `(n_channels, n_samples)` and the return is a feature array with **consistent shape across segments**.

Steps:
- Map channel indices using the provided `channel_pick`; do **not** assume the full 32-channel order unless you explicitly index via `GENEVA_32`.
- Reuse helpers in `options_feature/utils.py` (e.g., `_available_pairs` for asymmetry).
- Keep outputs `np.float32` and deterministic.
- Create the option instance and append it to `FEATURE_OPTIONS` in `options_feature/__init__.py`.

#### b) Channel picks
Path: `src/feature_extractor/options/options_channel_pick/__init__.py`  
Type: `ChannelPickOption(name: str, channel_pick: list[str])`

Rules:
- Channel names must come from `GENEVA_32`; the dataclass will raise otherwise.
- Names should describe the region/logic (e.g., `frontal_full_10`).
- Append to `CHANNEL_PICK_OPTIONS`.

#### c) Segmentation
Path: `src/feature_extractor/options/options_segmentation/__init__.py`  
Type: `SegmentationOption(time_window: float, time_step: float)`

Rules:
- `time_window > 0`, `time_step > 0`, `time_step <= time_window`, and `time_window <= BASELINE_SEC`.
- The option name is auto-derived (`w{window}_s{step}`) in `__post_init__`.
- Append to `SEGMENTATION_OPTIONS`.

### 3) Dataset-Building Options
Path: `src/model_trainer/options/options_training_data/options_build_dataset/__init__.py`  
Type: `BuildDatasetOption(target: str, random_seed: int, use_size: float, test_size: float, target_kind: Literal["regression","classification"], feature_scaler: Literal["none","standard","minmax"])`

Behavior:
- `use_size` and `test_size` must be within `(0,1]` and `(0,1)` respectively; the splits are reproducible via `random_seed`.
- Classification defaults expect class labels `1..9`; the option encodes labels to indices and records the mapping.
- `feature_scaler` is applied per-segment: `standard` and `minmax` flatten/reshape automatically; `none` passes through raw values.

Steps:
- Add new `BuildDatasetOption` entries to the list comprehension inside `BUILD_DATASET_OPTIONS`. Keep names stable (`target+use+test+seed+kind+scaler`).
- `TrainingDataOption` objects are constructed dynamically in `pipeline_runner`; you should not pre-register them.

### 4) Training Method Options
Path: `src/model_trainer/options/options_training_method/option_<label>.py`  
Type: `TrainingMethodOption(...)`

Interfaces required for `backend="torch"`:
- `optimizer_builder(model: nn.Module) -> Optimizer`
- `criterion_builder() -> nn.Module`
- `batch_formatter(features, targets, device, target_kind) -> tuple[Tensor, Tensor]`
- `train_epoch_fn(model, loader, optimizer, criterion, device, target_kind, class_values, batch_formatter)`
- `evaluate_epoch_fn(model, loader, criterion, device, target_kind, class_values, batch_formatter)`
- `prediction_collector(model, loader, device, target_kind, class_values, batch_formatter)`

Rules:
- Supply `epochs`, `batch_size`, `num_workers`, `pin_memory`, `drop_last`, `device`, and `target_kind`.
- The dataclass validates that all required callables are present for torch backends.
- For sklearn backends, set `backend="sklearn"` and only `name` + `target_kind` are needed; dataloaders/optimizers are skipped.

Steps:
- Reuse helpers in `options_training_method/utils.py` (`format_conv1d_batch`, `train_epoch_conv1d`, etc.) when working with Conv1d-friendly tensors.
- Append the new option to `TRAINING_METHOD_OPTIONS` and export via `__all__`.

### 5) Model Options
Path: `src/model_trainer/options/options_model/option_<label>.py`  
Type: `ModelOption(name: str, model_builder: ModelBuilder | SklearnBuilder, target_kind: Literal["regression","classification"], backend: Literal["torch","sklearn"], output_size: int | None = None)`

Rules:
- **Torch backends**: `model_builder` must accept `output_size` as a keyword argument. Use shape-flexible layers (`nn.Flatten`, `nn.LazyLinear`) instead of hardcoding dimensions. Set `output_size` on the option.
- **Sklearn backends**: `model_builder` returns an initialized estimator and does not take `output_size`. Set `backend="sklearn"` and leave `output_size=None`.
- Names should reflect both architecture and backend (e.g., `cnn1d_n1_regression`, `svc_rbf_sklearn`).
- Register new options in `MODEL_OPTIONS`.

### 6) Registering and Exporting
- Every new option file should define a module-level instance (prefixed with `_` to avoid auto-import noise) and include it in the module’s list (`PREPROCESSING_OPTIONS`, `FEATURE_OPTIONS`, `CHANNEL_PICK_OPTIONS`, `SEGMENTATION_OPTIONS`, `BUILD_DATASET_OPTIONS`, `TRAINING_METHOD_OPTIONS`, `MODEL_OPTIONS`).
- Update `__all__` when new symbols are introduced so downstream imports remain explicit.
- Keep names unique across a registry; `OptionList.get_name` will throw if duplicates occur.

## Running Pipelines and Experiments
- Inspect available options: `uv run python src/main.py` prints each registry.
- Run a combination manually:
```bash
uv run python - <<'PY'
from pipeline_runner import run_pipeline
from preprocessor.options import PREPROCESSING_OPTIONS
from feature_extractor.options import CHANNEL_PICK_OPTIONS, FEATURE_OPTIONS, SEGMENTATION_OPTIONS
from model_trainer.options import BUILD_DATASET_OPTIONS, MODEL_OPTIONS, TRAINING_METHOD_OPTIONS

run_pipeline(
    preprocessing_options=PREPROCESSING_OPTIONS.get_names(["clean"]),
    channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["balanced_classic_6"]),
    feature_options=FEATURE_OPTIONS.get_names(["psd"]),
    segmentation_options=SEGMENTATION_OPTIONS.get_names(["w2.00_s0.25"]),
    model_options=MODEL_OPTIONS.get_names(["logreg_sklearn"]),
    build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
        ["valence+use1.00+test0.20+seed42+classification+standard"]
    ),
    training_method_options=TRAINING_METHOD_OPTIONS.get_names(
        ["sklearn_default_classification"]
    ),
)
PY
```
- Artifacts:
  - Preprocessing → `data/DEAP/generated/<root_dir>/subject|trial|metadata`.
  - Features → `data/DEAP/generated/<root_dir>/feature/<pre+feat+ch+seg>/`.
  - Training runs → `results/<target>/<target_kind>/<timestamp>/` containing params, metrics, splits, and model weights/joblib.

## Example Walkthrough: Replicating a Paper via a New Pipeline Config
Scenario: A paper reports valence classification using ICA-cleaned EEG, an 8-channel frontal/parietal montage, 4 s windows with 1 s step, PSD features, 70/30 split, standard scaling, and a logistic regression classifier. Here’s how to reproduce it:

1) **Preprocessing option**  
   - Copy `src/preprocessor/options/options_preprocessing/option_ica_clean.py` to `option_paper_psd.py`.  
   - Keep the same pipeline (`_prepare_channels` → `_apply_filter_reference` → `_get_events` → `_epoch_and_resample` → `_base_bdf_process`), but set a distinct root for isolation:
   ```python
   from preprocessor.types import PreprocessingOption
   from .utils import (
       _apply_filter_reference,
       _base_bdf_process,
       _epoch_and_resample,
       _get_events,
       _prepare_channels,
   )

   def _paper_psd(raw, subject_id):
       eeg_channels, stim_ch_name, raw_stim, raw_eeg = _prepare_channels(raw)
       _apply_filter_reference(raw_eeg)
       events = _get_events(raw_stim, stim_ch_name, subject_id)
       data_down = _epoch_and_resample(raw_eeg, events, eeg_channels, sfreq_target=128.0)
       return _base_bdf_process(data_down=data_down, eeg_channels=eeg_channels, subject_id=subject_id)

   _option_paper_psd = PreprocessingOption(
       name="paper_psd",
       root_dir="paper_psd",
       preprocessing_method=_paper_psd,
   )
   ```
   - Register `_option_paper_psd` in `options_preprocessing/__init__.py`.

2) **Channel pick**  
   - In `src/feature_extractor/options/options_channel_pick/__init__.py`, append:
   ```python
   ChannelPickOption(
       name="paper_frontal_parietal_8",
       channel_pick=["Fp1", "Fp2", "F3", "F4", "F7", "F8", "P3", "P4"],
   ),
   ```

3) **Segmentation**  
   - In `options_segmentation/__init__.py`, add `SegmentationOption(time_window=4.0, time_step=1.0)`; the name will be `w4.00_s1.00`.

4) **Feature option**  
   - Reuse the existing PSD extractor by referencing the `psd` option (no code change needed) or add a new variant if the paper requires different bands (follow the Feature method steps above).

5) **Dataset split/scaling**  
   - In `options_training_data/options_build_dataset/__init__.py`, add:
   ```python
   BuildDatasetOption(
       target="valence",
       random_seed=23,
       use_size=1.0,
       test_size=0.30,
       target_kind="classification",
       feature_scaler="standard",
   ),
   ```
   - The generated name will be `valence+use1.00+test0.30+seed23+classification+standard`.

6) **Training method and model**  
   - Reuse the sklearn defaults:
     - Training method: `sklearn_default_classification`.
     - Model: `logreg_sklearn` (already registered).
   - Ensure the backend (`sklearn`) and `target_kind` (`classification`) align.

7) **Run the pipeline**  
   ```bash
   uv run python - <<'PY'
   from pipeline_runner import run_pipeline
   from preprocessor.options import PREPROCESSING_OPTIONS
   from feature_extractor.options import CHANNEL_PICK_OPTIONS, FEATURE_OPTIONS, SEGMENTATION_OPTIONS
   from model_trainer.options import BUILD_DATASET_OPTIONS, MODEL_OPTIONS, TRAINING_METHOD_OPTIONS

   run_pipeline(
       preprocessing_options=PREPROCESSING_OPTIONS.get_names(["paper_psd"]),
       channel_pick_options=CHANNEL_PICK_OPTIONS.get_names(["paper_frontal_parietal_8"]),
       feature_options=FEATURE_OPTIONS.get_names(["psd"]),
       segmentation_options=SEGMENTATION_OPTIONS.get_names(["w4.00_s1.00"]),
       model_options=MODEL_OPTIONS.get_names(["logreg_sklearn"]),
       build_dataset_options=BUILD_DATASET_OPTIONS.get_names(
           ["valence+use1.00+test0.30+seed23+classification+standard"]
       ),
       training_method_options=TRAINING_METHOD_OPTIONS.get_names(
           ["sklearn_default_classification"]
       ),
   )
   PY
   ```
   - Check `data/DEAP/generated/paper_psd/feature/*/metadata` for shapes and `results/valence/classification/<timestamp>/metrics.json` for performance. The params hash in `params.sha256` ensures reruns are reproducible.

## Validation and Hygiene Checklist
- [ ] New option file created in the correct directory with a unique `name`.
- [ ] Option registered in the module’s `OptionList` and exported via `__all__`.
- [ ] Callable signatures match the required protocol for that stage.
- [ ] Shapes and dtypes are handled generically (no hardcoded channel counts or segment lengths unless enforced by validation).
- [ ] Runs complete without skipping combinations due to mismatched `target_kind` or `backend`.
- [ ] Format and lint before submitting:
  ```bash
  uv run ruff format src
  uv run ruff check --fix src
  uv run mypy src
  uv run pyright src
  ```
- [ ] Confirm new artifacts land in the expected `data/` or `results/` paths and metadata JSON files serialize without errors.
