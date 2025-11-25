"""Sklearn-friendly training method placeholders."""

from __future__ import annotations

from model_trainer.types import TrainingMethodOption

__all__ = [
    "_sklearn_classification",
    "_sklearn_regression",
]

_sklearn_classification = TrainingMethodOption(
    name="sklearn_default_classification",
    target_kind="classification",
    backend="sklearn",
)

_sklearn_regression = TrainingMethodOption(
    name="sklearn_default_regression",
    target_kind="regression",
    backend="sklearn",
)
