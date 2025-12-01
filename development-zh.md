# 开发指南：构建和扩展基于 Option 的流水线 

该代码库围绕 *options* 设计——即描述 EEG 流水线中每一个选择的小型不可变配置对象。每个模块都会通过自身的 `OptionList` 注册表导出这些选项，这样 `pipeline_runner.run_pipeline` 就可以自动枚举所有组合。每当你添加预处理配方、特征提取器、数据集配方、模型或训练循环时，请参考本指南。

## 核心原则

* **单一事实来源（Single source of truth）**：每个 option 都应与其配置的代码放在一起，并通过该模块的 `__init__.py` 中的 `OptionList` 导出。名称必须唯一且稳定，因为它们会驱动文件系统布局（`data/generated/*`，`results/*`），并出现在元数据 JSON 中。
* **显式接口**：所有可调用对象都通过 `config.option_utils` 中的 protocol 进行类型标注（例如 `PreprocessingMethod`、`FeatureChannelExtractionMethod`、`ModelBuilder`）。必须严格匹配要求的函数签名。
* **可组合阶段**：`run_pipeline` 会遍历所有已注册的选项，并构建笛卡尔积组合（预处理 → 特征提取 → 数据集 → 训练 → 模型）。要确保各阶段之间的 option 互相兼容（target 类型、backend、期望的张量形状）。
* **确定性与安全性**：使用提供的辅助函数（原子写入、shape 校验、可复现随机种子），而不是临时的 IO 或随机化方式。避免更改已有 option 的名称，以保持结果可复现。

## 仓库结构概览

* `src/preprocessor`：加载 DEAP BDF 文件，完成滤波 / 重参考 / 划分 epoch，将其切分为 trial，并写入 `data/DEAP/generated/<root_dir>/`。
* `src/feature_extractor`：对 trial 进行分段，计算基线与每个窗口的特征，并将其存储在预处理根路径下的 `feature/<pre+feat+ch+seg>/` 中。
* `src/model_trainer`：将特征变为训练 / 测试数据集，构建 dataloader 或 sklearn 矩阵，训练模型，并将运行结果写入 `results/<target>/<target_kind>/<timestamp>/`。
* `src/pipeline_runner`：从各个 option 注册表构建组合，并按顺序调用每个阶段。
* `src/config/option_utils.py`：定义 protocol（`PreprocessingMethod`、`FeatureChannelExtractionMethod`、`ModelBuilder`、`OptimizerBuilder`、`CriterionBuilder`）以及 `OptionList` 注册表封装器。

## 如何添加或修改 Options

### 1) 预处理选项（Preprocessing Options）

路径：`src/preprocessor/options/options_preprocessing/option_<label>.py`
类型：`PreprocessingOption(name: str, root_dir: str | Path, preprocessing_method: PreprocessingMethod)`

`preprocessing_method` 的接口：`(raw: mne.io.BaseRaw, subject_id: int) -> np.ndarray`，返回形状为 `(TRIALS_NUM, EEG_ELECTRODES_NUM, samples)` 的数组。

步骤：

* 使用 `options_preprocessing/utils.py` 中的辅助函数：

  * `_prepare_channels` → 拆分 EEG 与 stim 通道。
  * `_apply_filter_reference` → notch 滤波 + 带通滤波 + 重参考。
  * `_get_events` → 提取事件，并对 DEAP 的特殊情况进行修正。
  * `_epoch_and_resample` → 划分 epoch 并重采样到 `SFREQ_TARGET`。
  * `_base_bdf_process` → 将 trial / 通道重排为标准的 DEAP 顺序。
* 选择一个**唯一的 `name`** 和一个 **`root_dir`**（位于 `data/DEAP/generated/` 下的文件夹）。更改任一字段都会导致输出路径变动。
* 实现该方法，确保在所有受试者之间 **输出形状一致**（预处理模块会断言这一点）。保持 dtype（推荐使用 `float32`）。
* 实例化并导出一个模块级的 option（例如 `_option_myvariant = PreprocessingOption(...)`）。
* 将其注册到 `src/preprocessor/options/options_preprocessing/__init__.py` 中的 `PREPROCESSING_OPTIONS`，并加入 `__all__`。

### 2) 特征提取选项（Feature Extraction Options）

#### a) 特征方法

路径：`src/feature_extractor/options/options_feature/option_<label>.py`
类型：`FeatureOption(name: str, feature_channel_extraction_method: FeatureChannelExtractionMethod)`

接口：`(trial_data: np.ndarray, channel_pick: list[str]) -> np.ndarray`，其中 `trial_data` 是单个分段的数据，形状为 `(n_channels, n_samples)`，返回值是一个在所有分段间 **形状一致** 的特征数组。

步骤：

* 使用传入的 `channel_pick` 来映射通道索引；**不要** 假定始终存在完整的 32 通道顺序，除非你显式地通过 `GENEVA_32` 进行索引。
* 重用 `options_feature/utils.py` 中的辅助函数（例如 `_available_pairs` 用于不对称性特征）。
* 保持输出为 `np.float32` 且具备确定性。
* 创建 option 实例，并将其追加到 `options_feature/__init__.py` 中的 `FEATURE_OPTIONS`。

#### b) 通道选择（Channel picks）

路径：`src/feature_extractor/options/options_channel_pick/__init__.py`
类型：`ChannelPickOption(name: str, channel_pick: list[str])`

规则：

* 通道名必须来自 `GENEVA_32`；否则 dataclass 会抛出异常。
* 名称应描述区域 / 逻辑（例如 `frontal_full_10`）。
* 将新建对象追加到 `CHANNEL_PICK_OPTIONS` 中。

#### c) 分段（Segmentation）

路径：`src/feature_extractor/options/options_segmentation/__init__.py`
类型：`SegmentationOption(time_window: float, time_step: float)`

规则：

* `time_window > 0`，`time_step > 0`，`time_step <= time_window`，且 `time_window <= BASELINE_SEC`。
* option 名称在 `__post_init__` 中自动生成（例如 `w{window}_s{step}`）。
* 将新建对象追加到 `SEGMENTATION_OPTIONS` 中。

### 3) 数据集构建选项（Dataset-Building Options）

路径：`src/model_trainer/options/options_training_data/options_build_dataset/__init__.py`
类型：
`BuildDatasetOption(target: str, random_seed: int, use_size: float, test_size: float, target_kind: Literal["regression","classification"], feature_scaler: Literal["none","standard","minmax"])`

行为：

* `use_size` 和 `test_size` 必须分别属于 `(0,1]` 和 `(0,1)`；通过 `random_seed` 确保划分可复现。
* 分类任务默认假定类别标签为 `1..9`；option 会将标签编码为索引并记录映射。
* `feature_scaler` 按 segment 进行缩放：`standard` 与 `minmax` 会在内部自动进行 flatten / reshape；`none` 则直接传递原始值。

步骤：

* 在 `BUILD_DATASET_OPTIONS` 的列表推导中添加新的 `BuildDatasetOption` 条目。保持名称稳定（`target+use+test+seed+kind+scaler`）。
* `TrainingDataOption` 对象由 `pipeline_runner` 动态构建，不需要预先注册。

### 4) 训练方法选项（Training Method Options）

路径：`src/model_trainer/options/options_training_method/option_<label>.py`
类型：`TrainingMethodOption(...)`

当 `backend="torch"` 时所需的接口：

* `optimizer_builder(model: nn.Module) -> Optimizer`
* `criterion_builder() -> nn.Module`
* `batch_formatter(features, targets, device, target_kind) -> tuple[Tensor, Tensor]`
* `train_epoch_fn(model, loader, optimizer, criterion, device, target_kind, class_values, batch_formatter)`
* `evaluate_epoch_fn(model, loader, criterion, device, target_kind, class_values, batch_formatter)`
* `prediction_collector(model, loader, device, target_kind, class_values, batch_formatter)`

规则：

* 需要提供 `epochs`、`batch_size`、`num_workers`、`pin_memory`、`drop_last`、`device` 和 `target_kind`。
* 对于 torch backend，dataclass 会校验上述所有必需的可调用对象是否存在。
* 对于 sklearn backend，设置 `backend="sklearn"`，只需要 `name` 和 `target_kind`；不会涉及 dataloader / optimizer。

步骤：

* 在处理 Conv1d 兼容张量时，重用 `options_training_method/utils.py` 中的辅助函数（例如 `format_conv1d_batch`、`train_epoch_conv1d` 等）。
* 将新建 option 追加到 `TRAINING_METHOD_OPTIONS`，并通过 `__all__` 导出。

### 5) 模型选项（Model Options）

路径：`src/model_trainer/options/options_model/option_<label>.py`
类型：
`ModelOption(name: str, model_builder: ModelBuilder | SklearnBuilder, target_kind: Literal["regression","classification"], backend: Literal["torch","sklearn"], output_size: int | None = None)`

规则：

* **Torch backend**：`model_builder` 必须接受 `output_size` 这个关键字参数。应使用对形状友好的层（如 `nn.Flatten`、`nn.LazyLinear`），避免硬编码维度。在 option 上设置 `output_size`。
* **Sklearn backend**：`model_builder` 返回一个已初始化的 estimator，并且不接受 `output_size` 参数。此时设置 `backend="sklearn"`，并将 `output_size=None`。
* 名称应同时体现架构和 backend（例如 `cnn1d_n1_regression`、`svc_rbf_sklearn`）。
* 在 `MODEL_OPTIONS` 中注册新的选项。

### 6) 注册与导出（Registering and Exporting）

* 每个新的 option 文件应定义一个模块级实例（通常以下划线前缀，例如 `_option_xxx`，以避免自动导入时的噪声），并将其包含在模块的列表中（`PREPROCESSING_OPTIONS`、`FEATURE_OPTIONS`、`CHANNEL_PICK_OPTIONS`、`SEGMENTATION_OPTIONS`、`BUILD_DATASET_OPTIONS`、`TRAINING_METHOD_OPTIONS`、`MODEL_OPTIONS`）。
* 当引入新的符号时，更新 `__all__`，以保持下游导入的显式性。
* 在同一个注册表中保持名称唯一；如果发生重复，`OptionList.get_name` 会抛出异常。

## 运行流水线和实验

* 查看可用的 options：运行 `uv run python src/main.py`，它会打印每个注册表中的内容。
* 手动运行某个组合：

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

* 产物路径：

  * 预处理 → `data/DEAP/generated/<root_dir>/subject|trial|metadata`。
  * 特征 → `data/DEAP/generated/<root_dir>/feature/<pre+feat+ch+seg>/`。
  * 训练运行 → `results/<target>/<target_kind>/<timestamp>/`，其中包含参数、指标、划分信息以及模型权重 / joblib。

## 示例演练：通过新的流水线配置复现论文

场景：某篇论文报告了使用 ICA 清理后的 EEG、8 通道额叶 / 顶叶 montage、4 s 窗口、1 s 步长、PSD 特征、70/30 划分、标准化缩放以及 logistic regression 分类器进行情感（valence）分类。下面演示如何复现该配置：

1. **预处理选项（Preprocessing option）**

   * 将 `src/preprocessor/options/options_preprocessing/option_ica_clean.py` 复制为 `option_paper_psd.py`。
   * 保持相同的处理流程（`_prepare_channels` → `_apply_filter_reference` → `_get_events` → `_epoch_and_resample` → `_base_bdf_process`），但为隔离起见启用一个独立的根目录：

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

   * 在 `options_preprocessing/__init__.py` 中注册 `_option_paper_psd`。

2. **通道选择（Channel pick）**

   * 在 `src/feature_extractor/options/options_channel_pick/__init__.py` 中追加：

   ```python
   ChannelPickOption(
       name="paper_frontal_parietal_8",
       channel_pick=["Fp1", "Fp2", "F3", "F4", "F7", "F8", "P3", "P4"],
   ),
   ```

3. **分段（Segmentation）**

   * 在 `options_segmentation/__init__.py` 中添加 `SegmentationOption(time_window=4.0, time_step=1.0)`；其名称将为 `w4.00_s1.00`。

4. **特征选项（Feature option）**

   * 重用已有的 PSD 提取器：直接引用 `psd` option（无需修改代码）；如果论文要求不同频段，可以按照前述“特征方法”步骤添加新变体。

5. **数据集划分 / 缩放（Dataset split / scaling）**

   * 在 `options_training_data/options_build_dataset/__init__.py` 中添加：

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

   * 生成的名称将为 `valence+use1.00+test0.30+seed23+classification+standard`。

6. **训练方法与模型（Training method and model）**

   * 重用 sklearn 默认配置：

     * 训练方法：`sklearn_default_classification`。
     * 模型：`logreg_sklearn`（已注册）。
   * 确保 backend（`sklearn`）与 `target_kind`（`classification`）一致。

7. **运行流水线**

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

   * 在 `data/DEAP/generated/paper_psd/feature/*/metadata` 中检查 shape，在 `results/valence/classification/<timestamp>/metrics.json` 中查看性能。`params.sha256` 中的参数哈希可以保证重复运行的可复现性。

## 校验与规范检查清单（Validation and Hygiene Checklist）

* [ ] 在正确的目录下创建了新的 option 文件，且 `name` 唯一。
* [ ] 已将 option 注册到对应模块的 `OptionList` 中，并通过 `__all__` 导出。
* [ ] 可调用对象的签名与该阶段所需的 protocol 严格匹配。
* [ ] 通用地处理 shape 与 dtype（不要硬编码通道数量或分段长度，除非通过校验显式强制）。
* [ ] 运行可以顺利完成，不会因为 `target_kind` 或 `backend` 不匹配而跳过组合。
* [ ] 在提交前完成格式化与静态检查：

  ```bash
  uv run ruff format src
  uv run ruff check --fix src
  uv run mypy src
  uv run pyright src
  ```
* [ ] 确认新的产物写入预期的 `data/` 或 `results/` 路径，并且元数据 JSON 文件可以无错误地序列化。

