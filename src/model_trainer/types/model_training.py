"""Aggregated configuration for executing a full training run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from config.constants import RESULTS_ROOT

if TYPE_CHECKING:
    from pathlib import Path

    from .model import ModelOption
    from .training import TrainingOption

__all__ = ["ModelTrainingOption"]


@dataclass(slots=True)
class ModelTrainingOption:
    """
    Aggregated configuration used for end-to-end training experiments.

    - Couples a ``ModelOption`` with the dataloader-producing ``TrainingOption``.
    - Exposes filesystem helpers for recording params, metrics, splits, and
      checkpoint state dictionaries.
    - Construction time validation ensures the model’s ``target_kind`` matches
      the dataset so incompatibilities are caught long before GPU cycles are
      spent.
    - Run artifacts are rooted at ``results/{timestamp}/``; helpers accept the
      run timestamp to construct file paths without relying on option names.
    """

    model_option: ModelOption
    training_option: TrainingOption
    name: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate that the model and dataset target kinds agree."""
        data_option = self.training_option.training_data_option
        if self.model_option.target_kind != data_option.target_kind:
            raise ValueError(
                "model_option target_kind must match training_data_option target_kind.",
            )
        if (
            self.model_option.backend == "torch"
            and self.model_option.output_size is None
        ):
            raise ValueError("output_size must be set for torch backends.")
        self.name = "+".join(
            [
                self.training_option.name,
                self.model_option.name,
            ],
        )

    def get_run_dir(self, *, run_timestamp: str) -> Path:
        """Directory where model artifacts (weights + metrics) are written."""
        target_root = RESULTS_ROOT / self.model_option.target_kind
        return target_root / run_timestamp

    def get_params_path(self, *, run_timestamp: str) -> Path:
        """Return the file used to persist ``to_params`` metadata."""
        return self.get_run_dir(run_timestamp=run_timestamp) / "params.json"

    def get_params_hash_path(self, *, run_timestamp: str) -> Path:
        """Return the file used to persist the params hash."""
        return self.get_run_dir(run_timestamp=run_timestamp) / "params.sha256"

    def get_metrics_path(self, *, run_timestamp: str) -> Path:
        """Return the metrics JSON path."""
        return self.get_run_dir(run_timestamp=run_timestamp) / "metrics.json"

    def get_model_artifact_path(self, *, run_timestamp: str) -> Path:
        """
        Return the path for the serialized model/estimator weights.

        Torch models use ``best_model.pt`` while sklearn estimators use
        ``best_model.joblib`` to keep artifacts distinct.
        """
        filename = "best_model.pt"
        if self.model_option.backend == "sklearn":
            filename = "best_model.joblib"
        return self.get_run_dir(run_timestamp=run_timestamp) / filename

    def get_splits_path(self, *, run_timestamp: str) -> Path:
        """Return the path for persisting train/test segment identifiers."""
        return self.get_run_dir(run_timestamp=run_timestamp) / "splits.json"

    def to_params(self) -> dict[str, Any]:
        """Serialize the aggregated model + training configuration."""
        return {
            "name": self.name,
            "model_option": self.model_option.to_params(),
            "training_option": self.training_option.to_params(),
        }
