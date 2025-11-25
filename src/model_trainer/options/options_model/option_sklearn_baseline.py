"""Baseline sklearn estimators for non-deep-learning experiments."""

from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingRegressor,  # type: ignore[import-untyped]
    RandomForestClassifier,  # type: ignore[import-untyped]
    RandomForestRegressor,  # type: ignore[import-untyped]
)
from sklearn.linear_model import (
    ElasticNet,
    LogisticRegression,
    Ridge,
    SGDClassifier,
)  # type: ignore[import-untyped]
from sklearn.svm import SVC, SVR, LinearSVC  # type: ignore[import-untyped]

from model_trainer.types import ModelOption

__all__ = [
    "_sklearn_logreg",
    "_sklearn_rf_regression",
    "_sklearn_rf_classification",
    "_sklearn_svc_rbf",
    "_sklearn_svr_rbf",
    "_sklearn_gb_regression",
    "_sklearn_linear_svc",
    "_sklearn_sgd_classifier",
    "_sklearn_ridge_regression",
    "_sklearn_elasticnet_regression",
]


def _build_logreg() -> LogisticRegression:
    return LogisticRegression(
        max_iter=500,
        n_jobs=-1,
        multi_class="auto",
    )


def _build_rf_regression() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=200,
        random_state=23,
        n_jobs=-1,
    )


def _build_rf_classification() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        random_state=23,
        n_jobs=-1,
        class_weight="balanced",
    )


def _build_svc_rbf() -> SVC:
    return SVC(
        kernel="rbf",
        gamma="scale",
        C=10.0,
        probability=True,
    )


def _build_linear_svc() -> LinearSVC:
    return LinearSVC(
        C=1.0,
        class_weight="balanced",
    )


def _build_svr_rbf() -> SVR:
    return SVR(
        kernel="rbf",
        gamma="scale",
        C=10.0,
        epsilon=0.1,
    )


def _build_gb_regression() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=23)


def _build_sgd_classifier() -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        max_iter=1000,
        tol=1e-3,
        n_jobs=-1,
    )


def _build_ridge_regression() -> Ridge:
    return Ridge(alpha=1.0, random_state=23)


def _build_elasticnet_regression() -> ElasticNet:
    return ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=23)


_sklearn_logreg = ModelOption(
    name="logreg_sklearn",
    model_builder=_build_logreg,
    target_kind="classification",
    backend="sklearn",
)

_sklearn_rf_regression = ModelOption(
    name="rf_regression_sklearn",
    model_builder=_build_rf_regression,
    target_kind="regression",
    backend="sklearn",
)

_sklearn_rf_classification = ModelOption(
    name="rf_classification_sklearn",
    model_builder=_build_rf_classification,
    target_kind="classification",
    backend="sklearn",
)

_sklearn_svc_rbf = ModelOption(
    name="svc_rbf_sklearn",
    model_builder=_build_svc_rbf,
    target_kind="classification",
    backend="sklearn",
)

_sklearn_linear_svc = ModelOption(
    name="linear_svc_sklearn",
    model_builder=_build_linear_svc,
    target_kind="classification",
    backend="sklearn",
)

_sklearn_svr_rbf = ModelOption(
    name="svr_rbf_sklearn",
    model_builder=_build_svr_rbf,
    target_kind="regression",
    backend="sklearn",
)

_sklearn_gb_regression = ModelOption(
    name="gb_regression_sklearn",
    model_builder=_build_gb_regression,
    target_kind="regression",
    backend="sklearn",
)

_sklearn_sgd_classifier = ModelOption(
    name="sgd_classifier_sklearn",
    model_builder=_build_sgd_classifier,
    target_kind="classification",
    backend="sklearn",
)

_sklearn_ridge_regression = ModelOption(
    name="ridge_regression_sklearn",
    model_builder=_build_ridge_regression,
    target_kind="regression",
    backend="sklearn",
)

_sklearn_elasticnet_regression = ModelOption(
    name="elasticnet_regression_sklearn",
    model_builder=_build_elasticnet_regression,
    target_kind="regression",
    backend="sklearn",
)
"""Baseline sklearn estimators for non-deep-learning experiments."""

# mypy: disable-error-code=import-untyped
