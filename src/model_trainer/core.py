"""Model training orchestration for DEAP feature datasets."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

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

from utils import message, prompt_confirm, spinner, track
from utils.fs import atomic_directory, directory_is_populated

if TYPE_CHECKING:
    from pathlib import Path

    from model_trainer.types import ModelTrainingOption

__all__ = ["run_model_trainer"]
ZERO_DIVISION: Any = 0


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
    Compute classification metrics by rounding scores to integer labels.

    - Predictions / targets are first mapped to integer labels so they can feed
      the scikit-learn metrics.
    - Macro/micro/weighted flavors of precision, recall, and F1 are included to
      diagnose class-imbalance behavior.
    - The confusion matrix is serialized for downstream visualization.
    """
    if predictions.size == 0:
        return {}

    pred_labels = np.rint(predictions).astype(int)
    true_labels = np.rint(targets).astype(int)
    label_set = np.unique(np.concatenate([true_labels, pred_labels]))

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
            labels=label_set,
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
    training_option = model_training_option.training_option
    data_option = training_option.training_data_option
    method_option = training_option.training_method_option
    backend = method_option.backend

    target_dir = model_training_option.get_path()
    if directory_is_populated(target_dir):
        description = (
            f'Results already exist at "{target_dir}". Overwrite with the new run?'
        )
        if not prompt_confirm(description, timeout_seconds=3, default=False):
            skip_description = (
                f"Skipping training for {model_training_option.model_option.name}"
            )
            message(description=skip_description, context="Model Trainer")
            return

    target_kind = data_option.target_kind
    class_values_array = data_option.get_class_values()

    model_artifact: Any
    if backend == "torch":
        seed = data_option.random_seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device_name = method_option.device
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine")
        device = torch.device(device_name)

        if target_kind == "classification":
            if class_values_array is None:
                raise RuntimeError("classification targets are missing class labels.")
            if (
                model_training_option.model_option.output_size is None
                or len(class_values_array)
                != model_training_option.model_option.output_size
            ):
                raise ValueError(
                    "Model output size must match the number of classification labels.",
                )
        elif model_training_option.model_option.output_size != 1:
            raise ValueError("Regression models must use output_size=1.")
        raw_model = model_training_option.model_option.build_model()
        torch_model = cast("torch.nn.Module", raw_model)
        model = torch_model.to(device)
        optimizer = method_option.build_optimizer(model=model)
        criterion = method_option.build_criterion().to(device=device)
        class_values_tensor: torch.Tensor | None = None
        if target_kind == "classification" and class_values_array is not None:
            class_values_tensor = torch.tensor(
                class_values_array,
                dtype=torch.float32,
                device=device,
            )

        if training_option.train_loader is None or training_option.test_loader is None:
            raise RuntimeError("Torch backend requires dataloaders to be present.")

        history: list[_EpochMetrics] = []
        best_state: OrderedDict[str, torch.Tensor] | None = None
        best_epoch = 0
        best_val_loss = float("inf")
        if method_option.epochs is None:
            raise RuntimeError("epochs must be configured for torch backend.")
        if method_option.batch_formatter is None:
            raise RuntimeError("batch_formatter must be configured for torch backend.")

        for epoch in track(
            iterable=range(1, method_option.epochs + 1),
            description=f"Training {{{model_training_option.model_option.name}}}",
            context="Model Trainer",
        ):
            start = time.perf_counter()
            if (
                method_option.train_epoch_fn is None
                or method_option.evaluate_epoch_fn is None
            ):
                raise RuntimeError(
                    "Torch backend requires training/evaluation functions."
                )
            train_loss, train_mae = method_option.train_epoch_fn(
                model,
                training_option.train_loader,
                optimizer,
                criterion,
                device,
                target_kind,
                class_values_tensor,
                method_option.batch_formatter,
            )
            val_loss, val_mae = method_option.evaluate_epoch_fn(
                model,
                training_option.test_loader,
                criterion,
                device,
                target_kind,
                class_values_tensor,
                method_option.batch_formatter,
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

        model.load_state_dict(best_state)
        if method_option.prediction_collector is None:
            raise RuntimeError("Torch backend requires a prediction_collector.")
        train_preds, train_targets = method_option.prediction_collector(
            model,
            training_option.train_loader,
            device,
            target_kind,
            class_values_array,
            method_option.batch_formatter,  # type: ignore[arg-type]
        )
        test_preds, test_targets = method_option.prediction_collector(
            model,
            training_option.test_loader,
            device,
            target_kind,
            class_values_array,
            method_option.batch_formatter,  # type: ignore[arg-type]
        )

        total_seconds = float(sum(metric.seconds for metric in history))
        if target_kind == "regression":
            train_metrics = {
                "regression": _regression_metrics(
                    predictions=train_preds,
                    targets=train_targets,
                ),
            }
            test_metrics = {
                "regression": _regression_metrics(
                    predictions=test_preds,
                    targets=test_targets,
                ),
            }
        else:
            train_metrics = {
                "classification": _classification_metrics(
                    predictions=train_preds,
                    targets=train_targets,
                ),
            }
            test_metrics = {
                "classification": _classification_metrics(
                    predictions=test_preds,
                    targets=test_targets,
                ),
            }
        metrics_payload = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "total_seconds": total_seconds,
            "epochs": [metric.to_dict() for metric in history],
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        if best_state is None:
            raise RuntimeError("best_state unexpectedly None prior to serialization.")
        model_artifact = best_state
    else:
        from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

        (train_arrays, train_targets), (test_arrays, test_targets) = (
            data_option.get_numpy_splits()
        )
        model = model_training_option.model_option.build_model()
        if not isinstance(model, BaseEstimator):
            raise RuntimeError("Sklearn backend requires a scikit-learn estimator.")
        estimator = cast("_EstimatorProtocol", model)

        status_text = (
            "[bold blue]Training sklearn model "
            f"{{{model_training_option.model_option.name}}}[/bold blue]"
        )
        with spinner(description=status_text, context="Model Trainer"):
            start = time.perf_counter()
            estimator.fit(
                train_arrays,
                train_targets,
                **(method_option.fit_kwargs or {}),
            )
            total_seconds = time.perf_counter() - start

        train_preds = estimator.predict(train_arrays)
        test_preds = estimator.predict(test_arrays)

        if target_kind == "regression":
            train_metrics = {
                "regression": _regression_metrics(
                    predictions=train_preds,
                    targets=train_targets,
                ),
            }
            test_metrics = {
                "regression": _regression_metrics(
                    predictions=test_preds,
                    targets=test_targets,
                ),
            }
        else:
            train_metrics = {
                "classification": _classification_metrics(
                    predictions=train_preds,
                    targets=train_targets,
                ),
            }
            test_metrics = {
                "classification": _classification_metrics(
                    predictions=test_preds,
                    targets=test_targets,
                ),
            }
        metrics_payload = {
            "best_epoch": 0,
            "best_val_loss": None,
            "total_seconds": total_seconds,
            "epochs": [],
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        model_artifact = model

    params_payload = {
        "time stamp": str(datetime.now()),
        "model_training_option": model_training_option.to_params(),
    }
    splits_payload = training_option.training_data_option.segment_splits

    with atomic_directory(target_dir=target_dir) as staging_dir:
        _dump_json(
            staging_dir / model_training_option.get_params_path().name,
            params_payload,
        )
        _dump_json(
            staging_dir / model_training_option.get_metrics_path().name,
            metrics_payload,
        )
        _dump_json(
            staging_dir / model_training_option.get_splits_path().name,
            splits_payload,
        )
        artifact_path = (
            staging_dir / model_training_option.get_model_artifact_path().name
        )
        if backend == "torch":
            torch.save(obj=model_artifact, f=artifact_path)
        else:
            joblib.dump(value=model_artifact, filename=artifact_path, compress=3)
