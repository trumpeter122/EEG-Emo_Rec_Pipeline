"""Option definitions describing how datasets are consumed during training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from config.option_utils import (
    CriterionBuilder,
    OptimizerBuilder,
    _callable_path,
)

BatchFormatter = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.device,
        Literal["regression", "classification"],
    ],
    tuple[torch.Tensor, torch.Tensor],
]
TrainEpochFn = Callable[
    [
        nn.Module,
        "DataLoader[tuple[np.ndarray, float]]",
        Optimizer,
        nn.Module,
        torch.device,
        Literal["regression", "classification"],
        torch.Tensor | None,
        BatchFormatter,
    ],
    tuple[float, float],
]
EvaluateEpochFn = Callable[
    [
        nn.Module,
        "DataLoader[tuple[np.ndarray, float]]",
        nn.Module,
        torch.device,
        Literal["regression", "classification"],
        torch.Tensor | None,
        BatchFormatter,
    ],
    tuple[float, float],
]
PredictionCollector = Callable[
    [
        nn.Module,
        "DataLoader[tuple[np.ndarray, float]]",
        torch.device,
        Literal["regression", "classification"],
        np.ndarray | None,
        BatchFormatter,
    ],
    tuple[np.ndarray, np.ndarray],
]

__all__ = [
    "BatchFormatter",
    "EvaluateEpochFn",
    "PredictionCollector",
    "TrainEpochFn",
    "TrainingMethodOption",
]


@dataclass(slots=True)
class TrainingMethodOption:
    """
    Optimizer/criterion configuration describing how datasets are consumed.

    - The dataclass bundles epoch count, batch sizing, DataLoader knobs, and
      builders that instantiate the actual optimizer + loss modules.
    - ``backend`` selects between the PyTorch training loop (default) and a
      scikit-learn estimator flow. Each option is tied to a ``target_kind`` so
      incompatible dataset variants are rejected before training begins.
    """

    name: str
    target_kind: Literal["regression", "classification"]
    backend: Literal["torch", "sklearn"] = "torch"
    epochs: int | None = None
    batch_size: int | None = None
    optimizer_builder: OptimizerBuilder | None = None
    criterion_builder: CriterionBuilder | None = None
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    device: Literal["cpu", "cuda"] = "cpu"
    batch_formatter: BatchFormatter | None = None
    train_epoch_fn: TrainEpochFn | None = None
    evaluate_epoch_fn: EvaluateEpochFn | None = None
    prediction_collector: PredictionCollector | None = None
    fit_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate input parameters."""
        if self.backend == "torch":
            if self.epochs is None or self.epochs <= 0:
                raise ValueError("epochs must be a positive integer.")
            if self.batch_size is None or self.batch_size <= 0:
                raise ValueError("batch_size must be a positive integer.")
            if self.num_workers < 0:
                raise ValueError("num_workers cannot be negative.")
            if self.optimizer_builder is None:
                raise ValueError("optimizer_builder is required for torch backend.")
            if self.criterion_builder is None:
                raise ValueError("criterion_builder is required for torch backend.")
            if self.batch_formatter is None:
                raise ValueError("batch_formatter is required for torch backend.")
            if self.train_epoch_fn is None:
                raise ValueError("train_epoch_fn is required for torch backend.")
            if self.evaluate_epoch_fn is None:
                raise ValueError("evaluate_epoch_fn is required for torch backend.")
            if self.prediction_collector is None:
                raise ValueError("prediction_collector is required for torch backend.")
        elif self.backend != "sklearn":
            raise ValueError(f"Unsupported backend: {self.backend}")

    def build_optimizer(self, *, model: nn.Module) -> Optimizer:
        """
        Instantiate the optimizer for ``model``.

        Args:
        ----
            model: The neural network model.

        Returns:
        -------
            An instantiated optimizer.
        """
        if self.optimizer_builder is None:
            raise RuntimeError("optimizer_builder is not configured for this method.")
        return self.optimizer_builder(model=model)

    def build_criterion(self) -> nn.Module:
        """
        Instantiate the configured loss function.

        Returns:
        -------
            An instantiated loss function.
        """
        if self.criterion_builder is None:
            raise RuntimeError("criterion_builder is not configured for this method.")
        return self.criterion_builder()

    def build_dataloader(
        self,
        *,
        dataset: Dataset[tuple[np.ndarray, float]],
        shuffle: bool,
    ) -> DataLoader[tuple[np.ndarray, float]]:
        """
        Create a ``DataLoader`` suitable for the configured training regime.

        Args:
        ----
            dataset: The dataset to load.
            shuffle: Whether to shuffle the data.

        Returns:
        -------
            A DataLoader instance.
        """
        if self.backend != "torch":
            raise RuntimeError("build_dataloader is only valid for torch backend.")
        if self.batch_size is None:
            raise RuntimeError("batch_size must be set for torch backend.")
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def to_params(self) -> dict[str, Any]:
        """Serialize method hyperparameters."""
        return {
            "name": self.name,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer_builder": (
                _callable_path(self.optimizer_builder)
                if self.optimizer_builder is not None
                else None
            ),
            "criterion_builder": (
                _callable_path(self.criterion_builder)
                if self.criterion_builder is not None
                else None
            ),
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "drop_last": self.drop_last,
            "target_kind": self.target_kind,
            "backend": self.backend,
            "batch_formatter": (
                _callable_path(self.batch_formatter)
                if self.batch_formatter is not None
                else None
            ),
            "train_epoch_fn": (
                _callable_path(self.train_epoch_fn)
                if self.train_epoch_fn is not None
                else None
            ),
            "evaluate_epoch_fn": (
                _callable_path(self.evaluate_epoch_fn)
                if self.evaluate_epoch_fn is not None
                else None
            ),
            "prediction_collector": (
                _callable_path(self.prediction_collector)
                if self.prediction_collector is not None
                else None
            ),
            "fit_kwargs": self.fit_kwargs or {},
        }
