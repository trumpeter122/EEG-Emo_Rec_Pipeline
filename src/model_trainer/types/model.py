"""Model option definitions for torch and sklearn backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from config.option_utils import ModelBuilder, _callable_path

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
    from torch import nn

__all__ = ["ModelOption"]


@runtime_checkable
class _SklearnModelBuilder(Protocol):
    """Callable protocol for constructing scikit-learn style estimators."""

    def __call__(self) -> BaseEstimator: ...


@dataclass(slots=True)
class ModelOption:
    """
    Descriptor for constructing a specific model configuration.

    - Options store a human-readable name, the callable ``model_builder`` that
      produces initialized models, the required ``target_kind``, and optionally
      an ``output_size`` for torch architectures.
    - ``backend`` chooses between PyTorch (default) and scikit-learn; PyTorch
      builders must support arbitrary segment lengths (e.g., use ``nn.Flatten``
      + ``nn.LazyLinear`` instead of hardcoded dimensions), while sklearn
      builders should return ready-to-fit estimators that adhere to the
      scikit-learn API.
    - Serialization helpers expose this data so metrics/artifacts can be traced
      back to the exact model configuration, regardless of backend.
    """

    name: str
    model_builder: ModelBuilder | _SklearnModelBuilder
    target_kind: Literal["regression", "classification"]
    backend: Literal["torch", "sklearn"]
    output_size: int | None = None

    def build_model(self) -> nn.Module | BaseEstimator:
        """
        Instantiate the configured model/estimator.

        Returns
        -------
            A PyTorch ``nn.Module`` when ``backend`` is ``torch`` or a
            scikit-learn estimator when ``backend`` is ``sklearn``.
        """
        if self.backend == "torch":
            if self.output_size is None:
                raise ValueError("output_size must be set for torch models.")
            torch_builder = cast("ModelBuilder", self.model_builder)
            return torch_builder(output_size=self.output_size)
        sklearn_builder = cast("_SklearnModelBuilder", self.model_builder)
        return sklearn_builder()

    def to_params(self) -> dict[str, Any]:
        """Serialize the model descriptor."""
        return {
            "name": self.name,
            "model_builder": _callable_path(self.model_builder),
            "output_size": self.output_size,
            "target_kind": self.target_kind,
            "backend": self.backend,
        }
