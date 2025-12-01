"""Combination of dataset options and optimization method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from .training_data import TrainingDataOption
    from .training_method import TrainingMethodOption

__all__ = ["TrainingOption"]


@dataclass(slots=True)
class TrainingOption:
    """
    Consolidated configuration describing dataset usage + optimization strategy.

    - When initialized it verifies that the data/method agree on ``target_kind``,
      builds deterministic names for metadata storage, and creates the train/test
      dataloaders by delegating to the selected ``TrainingMethodOption``.
    - This object is therefore the single source of truth for experiment inputs
      fed into ``run_model_trainer``.
    """

    training_data_option: TrainingDataOption
    training_method_option: TrainingMethodOption

    name: str = field(init=False)
    train_loader: DataLoader[Any] | None = field(init=False)
    test_loader: DataLoader[Any] | None = field(init=False)

    def __post_init__(self) -> None:
        """
        Post-initialization to validate target kinds and build data loaders.
        """
        if (
            self.training_method_option.target_kind
            != self.training_data_option.build_dataset_option.target_kind
        ):
            raise ValueError(
                "training_method_option target_kind must match"
                "training_data_option target_kind.",
            )
        self.name = "+".join(
            [
                self.training_data_option.name,
                self.training_method_option.name,
            ],
        )
        if self.training_method_option.backend == "torch":
            self.train_loader = self.training_method_option.build_dataloader(
                dataset=self.training_data_option.train_dataset,
                shuffle=True,
            )
            self.test_loader = self.training_method_option.build_dataloader(
                dataset=self.training_data_option.test_dataset,
                shuffle=False,
            )
        else:
            self.train_loader = None
            self.test_loader = None

    def to_params(self) -> dict[str, Any]:
        """Serialize the combined training configuration."""
        return {
            "name": self.name,
            "training_data_option": self.training_data_option.to_params(),
            "training_method_option": self.training_method_option.to_params(),
        }
