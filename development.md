# Development Playbook: Adding Options to the EEG Pipeline

This project wires every pipeline phase through small option objects registered in module-specific `OptionList` collections. Additions should slot into the existing structure so `pipeline_runner.run_pipeline` can materialize combinations automatically.

## Registry Pattern
- Keep option dataclasses alongside their module under `src/<module>/types.py` and register concrete instances in `src/<module>/options/`.
- Expose new options from the package `__init__.py` via `OptionList` so callers can look them up by `name` (see `config.option_utils.OptionList`).
- Validate names stay unique and descriptive; pipeline lookups in `main.py` use `get_names([...])` and experiment descriptors reference names directly (e.g., `training_method_name`).

## Preprocessor Options
- Add a new file under `src/preprocessor/options/options_preprocessing/option_<label>.py`.
- Implement a preprocessing function with signature `(raw: mne.io.BaseRaw, subject_id: int) -> np.ndarray`. Reuse helpers from `options_preprocessing/utils.py` for channel prep, event extraction, filtering, and resampling to keep behavior consistent.
- Instantiate `PreprocessingOption(name=..., root_dir=..., preprocessing_method=...)`. `root_dir` becomes the subfolder under `data/generated/` used for subject/trial outputs.
- Register the instance in `src/preprocessor/options/options_preprocessing/__init__.py` inside `PREPROCESSING_OPTIONS`, and export it via `__all__` so `src/preprocessor/options/__init__.py` can re-export it.

## Feature Extraction Options
- **Feature method**: Create `src/feature_extractor/options/options_feature/option_<label>.py` with a function `(trial_data: np.ndarray, channel_pick: list[str]) -> np.ndarray` that returns a consistent shape per segment. Wrap it in `FeatureOption(name=..., feature_channel_extraction_method=...)` and append to `FEATURE_OPTIONS` in `options_feature/__init__.py`.
- **Channel picks**: Add `ChannelPickOption(name=..., channel_pick=[...])` entries in `options_channel_pick/__init__.py`. The dataclass validates picks against `GENEVA_32`, so keep values from that list.
- **Segmentation**: Append `SegmentationOption(time_window=..., time_step=...)` to `options_segmentation/__init__.py`. The dataclass auto-builds the `name` and enforces `time_step <= time_window <= BASELINE_SEC`.
- The pipeline mixes every preprocessing/feature/channel/segmentation combination via `pipeline_runner.build_feature_extraction_options`, so ensure new options are compatible (e.g., feature methods handle any channel count and segment length).

## Model + Training Options
- **Models**: Define architectures in `src/model_trainer/options/options_model/option_<label>.py` and expose a builder `def _build_*(output_size: int) -> nn.Module`. Wrap it in `ModelOption(name=..., model_builder=_build_*, output_size=..., target_kind=...)` and register in `options_model/__init__.py`. Builders must accept arbitrary segment shapes (use lazy layers or derived shapes instead of hardcoded dimensions).
- **Training methods**: Add `option_<label>.py` under `options_training_method/` defining optimizer/criterion builders plus batch/epoch functions (see `option_adam_*` and `options_training_method/utils.py`). Create `TrainingMethodOption(...)` with the proper `target_kind`, then register it in `options_training_method/__init__.py`.
- **Training data**: `TrainingDataOption` is instantiated on demand within scripts (see `pipeline_runner._build_model_training_option`); if you need a preset, add a factory near your orchestration script rather than pre-registering it in `options_training_data`.

### Sklearn backends
- Set `backend="sklearn"` on `ModelOption` and use a builder returning a scikit-learn estimator (no `output_size` required).
- Pair it with a `TrainingMethodOption(..., backend="sklearn")` entry; dataloaders/optimizers/criterions are skipped and training runs directly on numpy arrays (see the `sklearn_default_*` options for reference).

## Wiring Experiments
- Reference new option names in `src/main.py` (or other scripts) when passing `preprocessing_options`, `feature_options`, `channel_pick_options`, `segmentation_options`, and within `TrainingExperiment` entries (`training_method_name`, `model_name`, `feature_scaler`, etc.).
- `run_pipeline` will generate artifacts under `data/generated/*` and `results/*` using the option names and `to_params()` metadata, so keep naming stable for reproducibility.

## Quick Checklist
- [ ] New option file lives under the appropriate `options/` subdirectory with a unique `name`.
- [ ] The option is added to the module’s `OptionList` and exported through `__init__.py`.
- [ ] Signatures match the module protocol (`PreprocessingMethod`, `FeatureChannelExtractionMethod`, `ModelBuilder`, `OptimizerBuilder`, etc.).
- [ ] Option logic handles arbitrary segment shapes/channel selections where applicable.
- [ ] `uv run python src/main.py` (or your script) completes with the new option included; follow with `uv run ruff format src && uv run ruff check --fix src && uv run mypy src && uv run pyright src` before submitting.
