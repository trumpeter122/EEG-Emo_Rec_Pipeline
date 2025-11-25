"""Model training orchestration for DEAP feature datasets."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import torch
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

from config.constants import RESULTS_ROOT
from utils import message, prompt_confirm, spinner, track
from utils.fs import atomic_directory

if TYPE_CHECKING:
    from pathlib import Path

    from torch.utils.data import DataLoader

    from model_trainer.types import (
        ModelTrainingOption,
        TrainingMethodOption,
        TrainingOption,
    )
    from model_trainer.types.training_method import (
        BatchFormatter,
        EvaluateEpochFn,
        PredictionCollector,
        TrainEpochFn,
    )

__all__ = ["run_model_trainer"]
ZERO_DIVISION: Any = 0
TargetKind = Literal["regression", "classification"]


def _hash_params(params: dict[str, Any]) -> str:
    """Return a deterministic SHA256 hash for the params payload."""
    serialized = json.dumps(
        params,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_existing_param_hashes(results_root: Path) -> dict[str, list[Path]]:
    """Return mapping of param hashes to existing run directories."""
    hashes: dict[str, list[Path]] = {}
    if not results_root.exists():
        return hashes

    for target_dir in results_root.iterdir():
        if not target_dir.is_dir():
            continue
        for run_dir in target_dir.iterdir():
            if not run_dir.is_dir():
                continue
            hash_path = run_dir / "params.sha256"
            if not hash_path.exists():
                continue
            try:
                hash_value = hash_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not hash_value:
                continue
            hashes.setdefault(hash_value, []).append(run_dir)
    return hashes


def _confirm_overwrite_existing_runs(
    *,
    params_hash: str,
    existing_runs: list[Path],
) -> bool:
    """
    Prompt the user before overwriting previously generated run directories.

    Returns ``True`` when the user confirms, otherwise ``False``.
    """
    run_dirs = ", ".join(str(path) for path in existing_runs)
    return prompt_confirm(
        prompt=(
            f"Identical params hash {params_hash} already exists at {run_dirs}. "
            "Overwrite existing results?"
        ),
        default=False,
        timeout_seconds=3,
    )


def _remove_existing_runs(existing_runs: list[Path]) -> None:
    """Delete stale run directories so re-runs do not accumulate duplicates."""
    for run_dir in existing_runs:
        try:
            shutil.rmtree(run_dir)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to remove existing run directory {run_dir}",
            ) from exc


def _collect_existing_runs_for_hash(
    *,
    params_hash: str,
    existing_hashes: dict[str, list[Path]],
) -> list[Path]:
    """Return existing run directories matching the provided params hash."""
    return existing_hashes.get(params_hash, [])


class _EstimatorProtocol(Protocol):
    """Minimal sklearn-like estimator interface."""

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> Any: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class _EpochMetrics:
    """Summary statistics captured for a single epoch."""

    epoch: int
    train_loss: float
    train_mae: float
    val_loss: float
    val_mae: float
    seconds: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable representation."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_mae": self.train_mae,
            "val_loss": self.val_loss,
            "val_mae": self.val_mae,
            "seconds": self.seconds,
        }


def _regression_metrics(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """
    Compute a suite of regression metrics for the provided arrays.

    - The returned mapping includes MAE/MSE-derived statistics plus r2 and
      explained variance so experiment summaries capture both absolute and
      relative performance.
    - When no predictions are available an empty dict is returned to avoid
      polluting the metrics JSON with NaNs.
    """
    if predictions.size == 0:
        return {}

    mse = mean_squared_error(y_true=targets, y_pred=predictions)
    mae = mean_absolute_error(y_true=targets, y_pred=predictions)
    return {
        "mae": float(mae),
        "median_absolute_error": float(
            median_absolute_error(y_true=targets, y_pred=predictions),
        ),
        "mape": float(
            mean_absolute_percentage_error(y_true=targets, y_pred=predictions),
        ),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true=targets, y_pred=predictions)),
        "explained_variance": float(
            explained_variance_score(y_true=targets, y_pred=predictions),
        ),
        "max_error": float(np.max(np.abs(predictions - targets))),
    }


def _classification_metrics(
    *,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    """
    Compute classification metrics after mapping labels onto integer indices.

    - Predictions and targets are converted into deterministic index arrays so
      non-integer labels (e.g., continuous class encodings) are handled without
      lossy rounding.
    - Macro/micro/weighted flavors of precision, recall, and F1 are included to
      diagnose class-imbalance behavior.
    - The confusion matrix is serialized for downstream visualization.
    """
    if predictions.size == 0:
        return {}

    label_set = np.unique(np.concatenate([targets, predictions]))
    pred_labels = np.searchsorted(label_set, predictions)
    true_labels = np.searchsorted(label_set, targets)
    label_indices = list(range(len(label_set)))

    accuracy = accuracy_score(y_true=true_labels, y_pred=pred_labels)
    precision_macro = precision_score(
        y_true=true_labels,
        y_pred=pred_labels,
        average="macro",
        zero_division=ZERO_DIVISION,
    )
    recall_macro = recall_score(
        y_true=true_labels,
        y_pred=pred_labels,
        average="macro",
        zero_division=ZERO_DIVISION,
    )
    f1_macro = f1_score(
        y_true=true_labels,
        y_pred=pred_labels,
        average="macro",
        zero_division=ZERO_DIVISION,
    )

    metrics: dict[str, Any] = {
        "labels": label_set.tolist(),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true=true_labels, y_pred=pred_labels),
        ),
        "precision_macro": float(precision_macro),
        "precision_micro": float(
            precision_score(
                y_true=true_labels,
                y_pred=pred_labels,
                average="micro",
                zero_division=ZERO_DIVISION,
            ),
        ),
        "precision_weighted": float(
            precision_score(
                y_true=true_labels,
                y_pred=pred_labels,
                average="weighted",
                zero_division=ZERO_DIVISION,
            ),
        ),
        "recall_macro": float(recall_macro),
        "recall_micro": float(
            recall_score(
                y_true=true_labels,
                y_pred=pred_labels,
                average="micro",
                zero_division=ZERO_DIVISION,
            ),
        ),
        "recall_weighted": float(
            recall_score(
                y_true=true_labels,
                y_pred=pred_labels,
                average="weighted",
                zero_division=ZERO_DIVISION,
            ),
        ),
        "f1_macro": float(f1_macro),
        "f1_micro": float(
            f1_score(
                y_true=true_labels,
                y_pred=pred_labels,
                average="micro",
                zero_division=ZERO_DIVISION,
            ),
        ),
        "f1_weighted": float(
            f1_score(
                y_true=true_labels,
                y_pred=pred_labels,
                average="weighted",
                zero_division=ZERO_DIVISION,
            ),
        ),
        "confusion_matrix": confusion_matrix(
            y_true=true_labels,
            y_pred=pred_labels,
            labels=label_indices,
        ).tolist(),
    }
    return metrics


def _dump_json(path: Path, payload: Any) -> None:
    """
    Serialize ``payload`` as indented UTF-8 JSON at ``path``.

    - The parent directory is created if necessary so callers can operate on
      fresh staging areas during atomic writes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="w", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2)


def _build_train_test_metrics(
    *,
    target_kind: TargetKind,
    train_preds: np.ndarray,
    train_targets: np.ndarray,
    test_preds: np.ndarray,
    test_targets: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return train/test metrics organized under the target kind key."""
    metric_key = "regression" if target_kind == "regression" else "classification"
    metric_fn = (
        _regression_metrics if target_kind == "regression" else _classification_metrics
    )
    train_metrics = {
        metric_key: metric_fn(predictions=train_preds, targets=train_targets),
    }
    test_metrics = {
        metric_key: metric_fn(predictions=test_preds, targets=test_targets),
    }
    return train_metrics, test_metrics


def _seed_torch(*, random_seed: int) -> None:
    """Seed PyTorch (and CUDA when available) for deterministic runs."""
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def _select_device(*, device_name: Literal["cpu", "cuda"]) -> torch.device:
    """Return a valid ``torch.device`` honoring CUDA availability."""
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")
    return torch.device(device_name)


def _ensure_dataloaders(
    training_option: TrainingOption,
) -> tuple[DataLoader[tuple[np.ndarray, float]], DataLoader[tuple[np.ndarray, float]]]:
    """Guarantee that dataloaders are attached when using the torch backend."""
    train_loader = training_option.train_loader
    test_loader = training_option.test_loader
    if train_loader is None or test_loader is None:
        raise RuntimeError("Torch backend requires dataloaders to be present.")
    return train_loader, test_loader


def _resolve_torch_callables(
    method_option: TrainingMethodOption,
) -> tuple[BatchFormatter, TrainEpochFn, EvaluateEpochFn, PredictionCollector]:
    """Unpack torch-specific callables, ensuring none are missing."""
    batch_formatter = method_option.batch_formatter
    train_epoch_fn = method_option.train_epoch_fn
    evaluate_epoch_fn = method_option.evaluate_epoch_fn
    prediction_collector = method_option.prediction_collector
    if (
        batch_formatter is None
        or train_epoch_fn is None
        or evaluate_epoch_fn is None
        or prediction_collector is None
    ):
        raise RuntimeError(
            "Torch backend requires batch_formatter, train/evaluate functions, "
            "and a prediction_collector.",
        )
    return batch_formatter, train_epoch_fn, evaluate_epoch_fn, prediction_collector


def _run_torch_epochs(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader[tuple[np.ndarray, float]],
    test_loader: DataLoader[tuple[np.ndarray, float]],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    target_kind: TargetKind,
    class_values_tensor: torch.Tensor | None,
    batch_formatter: BatchFormatter,
    train_epoch_fn: TrainEpochFn,
    evaluate_epoch_fn: EvaluateEpochFn,
    total_epochs: int,
    model_name: str,
) -> tuple[list[_EpochMetrics], OrderedDict[str, torch.Tensor], int, float]:
    """
    Execute the torch training loop and track best-performing checkpoints.
    """
    history: list[_EpochMetrics] = []
    best_state: OrderedDict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_loss = float("inf")

    for epoch in track(
        iterable=range(1, total_epochs + 1),
        description=f"Training {{{model_name}}}",
        context="Model Trainer",
    ):
        start = time.perf_counter()
        train_loss, train_mae = train_epoch_fn(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            target_kind,
            class_values_tensor,
            batch_formatter,
        )
        val_loss, val_mae = evaluate_epoch_fn(
            model,
            test_loader,
            criterion,
            device,
            target_kind,
            class_values_tensor,
            batch_formatter,
        )
        elapsed = time.perf_counter() - start

        history.append(
            _EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_mae=train_mae,
                val_loss=val_loss,
                val_mae=val_mae,
                seconds=elapsed,
            ),
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = OrderedDict(
                (key, value.detach().cpu().clone())
                for key, value in model.state_dict().items()
            )

    if best_state is None:
        raise RuntimeError("Model training did not produce any checkpoints.")

    return history, best_state, best_epoch, best_val_loss


def _collect_torch_predictions(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader[tuple[np.ndarray, float]],
    test_loader: DataLoader[tuple[np.ndarray, float]],
    device: torch.device,
    target_kind: TargetKind,
    class_values_array: np.ndarray | None,
    batch_formatter: BatchFormatter,
    prediction_collector: PredictionCollector,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Gather predictions/targets for both train and test splits."""
    train_preds, train_targets = prediction_collector(
        model,
        train_loader,
        device,
        target_kind,
        class_values_array,
        batch_formatter,
    )
    test_preds, test_targets = prediction_collector(
        model,
        test_loader,
        device,
        target_kind,
        class_values_array,
        batch_formatter,
    )
    return (train_preds, train_targets), (test_preds, test_targets)


def _train_torch_model(
    *,
    model_training_option: ModelTrainingOption,
    class_values_array: np.ndarray | None,
) -> tuple[dict[str, Any], OrderedDict[str, torch.Tensor]]:
    """Train a torch model end-to-end and return metrics + best checkpoint."""
    training_option = model_training_option.training_option
    data_option = training_option.training_data_option
    method_option = training_option.training_method_option
    target_kind: TargetKind = data_option.target_kind

    _seed_torch(random_seed=data_option.random_seed)
    device = _select_device(device_name=method_option.device)

    model_option = model_training_option.model_option
    if target_kind == "classification":
        if class_values_array is None:
            raise RuntimeError("classification targets are missing class labels.")
        if (
            model_option.output_size is None
            or len(class_values_array) != model_option.output_size
        ):
            raise ValueError(
                "Model output size must match the number of classification labels.",
            )
    elif model_option.output_size != 1:
        raise ValueError("Regression models must use output_size=1.")

    raw_model = model_option.build_model()
    torch_model = cast("torch.nn.Module", raw_model)
    model = torch_model.to(device=device)
    optimizer = method_option.build_optimizer(model=model)
    criterion = method_option.build_criterion().to(device=device)
    class_values_tensor: torch.Tensor | None = None
    if target_kind == "classification" and class_values_array is not None:
        class_values_tensor = torch.tensor(
            class_values_array,
            dtype=torch.float32,
            device=device,
        )

    train_loader, test_loader = _ensure_dataloaders(training_option=training_option)
    (
        batch_formatter,
        train_epoch_fn,
        evaluate_epoch_fn,
        prediction_collector,
    ) = _resolve_torch_callables(method_option=method_option)
    if method_option.epochs is None:
        raise RuntimeError("epochs must be configured for torch backend.")

    history, best_state, best_epoch, best_val_loss = _run_torch_epochs(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        target_kind=target_kind,
        class_values_tensor=class_values_tensor,
        batch_formatter=batch_formatter,
        train_epoch_fn=train_epoch_fn,
        evaluate_epoch_fn=evaluate_epoch_fn,
        total_epochs=method_option.epochs,
        model_name=model_option.name,
    )

    model.load_state_dict(best_state)
    (train_preds, train_targets), (test_preds, test_targets) = (
        _collect_torch_predictions(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            target_kind=target_kind,
            class_values_array=class_values_array,
            batch_formatter=batch_formatter,
            prediction_collector=prediction_collector,
        )
    )

    total_seconds = float(sum(metric.seconds for metric in history))
    train_metrics, test_metrics = _build_train_test_metrics(
        target_kind=target_kind,
        train_preds=train_preds,
        train_targets=train_targets,
        test_preds=test_preds,
        test_targets=test_targets,
    )

    metrics_payload: dict[str, Any] = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_seconds": total_seconds,
        "epochs": [metric.to_dict() for metric in history],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    return metrics_payload, best_state


def _train_sklearn_model(
    *,
    model_training_option: ModelTrainingOption,
) -> tuple[dict[str, Any], Any]:
    """Fit a sklearn estimator and compute train/test metrics."""
    from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

    training_option = model_training_option.training_option
    method_option = training_option.training_method_option
    data_option = training_option.training_data_option
    target_kind: TargetKind = data_option.target_kind

    (train_arrays, train_targets), (test_arrays, test_targets) = (
        data_option.get_numpy_splits()
    )
    model = model_training_option.model_option.build_model()
    if not isinstance(model, BaseEstimator):
        raise RuntimeError("Sklearn backend requires a scikit-learn estimator.")
    estimator = cast("_EstimatorProtocol", model)

    fit_kwargs = (
        method_option.fit_kwargs if method_option.fit_kwargs is not None else {}
    )
    status_text = (
        "[bold blue]Training sklearn model "
        f"{{{model_training_option.model_option.name}}}[/bold blue]"
    )
    with spinner(description=status_text, context="Model Trainer"):
        start = time.perf_counter()
        estimator.fit(
            X=train_arrays,
            y=train_targets,
            **fit_kwargs,
        )
        total_seconds = time.perf_counter() - start

    train_preds = estimator.predict(X=train_arrays)
    test_preds = estimator.predict(X=test_arrays)

    train_metrics, test_metrics = _build_train_test_metrics(
        target_kind=target_kind,
        train_preds=train_preds,
        train_targets=train_targets,
        test_preds=test_preds,
        test_targets=test_targets,
    )
    metrics_payload: dict[str, Any] = {
        "best_epoch": 0,
        "best_val_loss": None,
        "total_seconds": total_seconds,
        "epochs": [],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    return metrics_payload, estimator


def _persist_run_artifacts(
    *,
    model_training_option: ModelTrainingOption,
    run_timestamp: str,
    params_payload: dict[str, Any],
    params_hash: str,
    metrics_payload: dict[str, Any],
    splits_payload: dict[str, list[int]],
    model_artifact: Any,
) -> None:
    """Atomically write params, metrics, splits, and model artifacts to disk."""
    backend = model_training_option.training_option.training_method_option.backend
    target_dir = model_training_option.get_run_dir(run_timestamp=run_timestamp)
    params_path = model_training_option.get_params_path(run_timestamp=run_timestamp)
    params_hash_path = model_training_option.get_params_hash_path(
        run_timestamp=run_timestamp,
    )
    metrics_path = model_training_option.get_metrics_path(run_timestamp=run_timestamp)
    splits_path = model_training_option.get_splits_path(run_timestamp=run_timestamp)
    artifact_path = model_training_option.get_model_artifact_path(
        run_timestamp=run_timestamp,
    )

    with atomic_directory(target_dir=target_dir) as staging_dir:
        _dump_json(staging_dir / params_path.name, params_payload)
        (staging_dir / params_hash_path.name).write_text(
            data=params_hash,
            encoding="utf-8",
        )
        _dump_json(staging_dir / metrics_path.name, metrics_payload)
        _dump_json(staging_dir / splits_path.name, splits_payload)

        full_artifact_path = staging_dir / artifact_path.name
        if backend == "torch":
            torch.save(obj=model_artifact, f=full_artifact_path)
        else:
            joblib.dump(value=model_artifact, filename=full_artifact_path, compress=3)


def run_model_trainer(model_training_option: ModelTrainingOption) -> None:
    """
    Train the supplied option end-to-end and persist metrics atomically.

    - The routine seeds PyTorch for determinism, validates that the selected
      backend and configuration match the dataset target kind, and executes
      either the PyTorch training loop or a scikit-learn estimator flow. PyTorch
      runs track the best validation loss, reload the top checkpoint, and then
      compute regression/classification diagnostics on train & test splits.
      sklearn runs fit once and evaluate directly on train/test arrays.
    - All metadata (params, metrics, splits, weights) is written via
      ``atomic_directory`` to guarantee that partially written runs never
    corrupt previous artifacts.
    """
    message(
        description=f"Training model with option {{{model_training_option.name}}}",
        context="Model Trainer",
    )

    training_option = model_training_option.training_option
    params_payload = model_training_option.to_params()
    params_hash = _hash_params(params=params_payload)
    existing_hashes = _load_existing_param_hashes(results_root=RESULTS_ROOT)
    existing_runs = _collect_existing_runs_for_hash(
        params_hash=params_hash,
        existing_hashes=existing_hashes,
    )
    if existing_runs:
        if not _confirm_overwrite_existing_runs(
            params_hash=params_hash,
            existing_runs=existing_runs,
        ):
            message(
                description="Skipping run; existing results were kept.",
                context="Model Trainer",
            )
            return
        _remove_existing_runs(existing_runs=existing_runs)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    data_option = training_option.training_data_option
    method_option = training_option.training_method_option
    class_values_array = data_option.get_class_values()

    if method_option.backend == "torch":
        metrics_payload, model_artifact = _train_torch_model(
            model_training_option=model_training_option,
            class_values_array=class_values_array,
        )
    elif method_option.backend == "sklearn":
        metrics_payload, model_artifact = _train_sklearn_model(
            model_training_option=model_training_option,
        )
    else:
        raise ValueError(f"Unsupported backend: {method_option.backend}")

    metrics_payload["run_timestamp"] = run_timestamp
    metrics_payload["params_hash"] = params_hash
    _persist_run_artifacts(
        model_training_option=model_training_option,
        run_timestamp=run_timestamp,
        params_payload=params_payload,
        params_hash=params_hash,
        metrics_payload=metrics_payload,
        splits_payload=data_option.segment_splits,
        model_artifact=model_artifact,
    )
