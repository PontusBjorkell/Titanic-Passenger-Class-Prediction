"""Model evaluation utilities for passenger-class prediction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)

from titanic_passenger_class_prediction.config import (
    CV_FOLDS,
    N_JOBS,
    RANDOM_STATE,
)


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def build_cross_validator() -> StratifiedKFold:
    """
    Create the shared stratified cross-validation strategy.

    Returns
    -------
    sklearn.model_selection.StratifiedKFold
        Configured cross-validation splitter.
    """
    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def evaluate_model_cv(
    model_name: str,
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float | str]:
    """
    Evaluate one estimator using stratified cross-validation.

    Parameters
    ----------
    model_name
        Stable model identifier used in reports.
    estimator
        Unfitted scikit-learn estimator or pipeline.
    features
        Modeling feature matrix.
    target
        Target labels.

    Returns
    -------
    dict[str, float | str]
        Mean and standard deviation for each metric, plus timing data.
    """
    _validate_features_and_target(
        features=features,
        target=target,
    )

    scores = cross_validate(
        estimator=estimator,
        X=features,
        y=target,
        scoring=SCORING,
        cv=build_cross_validator(),
        n_jobs=N_JOBS,
        return_train_score=False,
        error_score="raise",
    )

    return {
        "model": model_name,
        "cv_accuracy_mean": float(
            np.mean(scores["test_accuracy"])
        ),
        "cv_accuracy_std": float(
            np.std(scores["test_accuracy"])
        ),
        "cv_balanced_accuracy_mean": float(
            np.mean(scores["test_balanced_accuracy"])
        ),
        "cv_balanced_accuracy_std": float(
            np.std(scores["test_balanced_accuracy"])
        ),
        "cv_f1_macro_mean": float(
            np.mean(scores["test_f1_macro"])
        ),
        "cv_f1_macro_std": float(
            np.std(scores["test_f1_macro"])
        ),
        "fit_time_mean": float(
            np.mean(scores["fit_time"])
        ),
        "score_time_mean": float(
            np.mean(scores["score_time"])
        ),
    }


def compare_models_cv(
    models: Mapping[str, BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Compare candidate models using identical cross-validation folds.

    Models are ranked by:

    1. Macro F1
    2. Balanced accuracy
    3. Accuracy

    Parameters
    ----------
    models
        Mapping of model names to unfitted estimators.
    features
        Modeling feature matrix.
    target
        Target labels.

    Returns
    -------
    pandas.DataFrame
        Sorted model-comparison table.
    """
    if not models:
        raise ValueError(
            "At least one candidate model is required."
        )

    results = [
        evaluate_model_cv(
            model_name=model_name,
            estimator=estimator,
            features=features,
            target=target,
        )
        for model_name, estimator in models.items()
    ]

    comparison = pd.DataFrame(results)

    return comparison.sort_values(
        by=[
            "cv_f1_macro_mean",
            "cv_balanced_accuracy_mean",
            "cv_accuracy_mean",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)


def select_best_model_name(
    comparison: pd.DataFrame,
) -> str:
    """
    Return the name of the highest-ranked model.

    Parameters
    ----------
    comparison
        Sorted model-comparison table.

    Returns
    -------
    str
        Selected model name.
    """
    if comparison.empty:
        raise ValueError(
            "Cannot select a model from an empty comparison table."
        )

    if "model" not in comparison.columns:
        raise ValueError(
            "Comparison table must contain a 'model' column."
        )

    return str(comparison.iloc[0]["model"])


# ---------------------------------------------------------------------------
# Holdout-test evaluation
# ---------------------------------------------------------------------------

def evaluate_holdout(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a fitted estimator on unseen holdout data.

    Parameters
    ----------
    estimator
        Fitted estimator or pipeline.
    features
        Holdout feature matrix.
    target
        Holdout target labels.
    labels
        Optional ordered class labels. When omitted, labels are inferred
        from the target and predictions.

    Returns
    -------
    dict[str, Any]
        Overall metrics, confusion matrix, and classification report.
    """
    _validate_features_and_target(
        features=features,
        target=target,
    )

    predictions = estimator.predict(features)

    resolved_labels = (
        list(labels)
        if labels is not None
        else sorted(
            set(target.tolist())
            | set(predictions.tolist())
        )
    )

    matrix = confusion_matrix(
        target,
        predictions,
        labels=resolved_labels,
    )

    report = classification_report(
        target,
        predictions,
        labels=resolved_labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(
            accuracy_score(
                target,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                target,
                predictions,
            )
        ),
        "f1_macro": float(
            f1_score(
                target,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "labels": [
            int(label)
            for label in resolved_labels
        ],
        "confusion_matrix": matrix.tolist(),
        "classification_report": (
            _convert_report_values(report)
        ),
        "test_rows": int(len(target)),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_features_and_target(
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """
    Validate feature and target inputs used during evaluation.
    """
    if features.empty:
        raise ValueError(
            "Cannot evaluate a model with an empty feature matrix."
        )

    if target.empty:
        raise ValueError(
            "Cannot evaluate a model with an empty target."
        )

    if len(features) != len(target):
        raise ValueError(
            "Features and target must contain the same number of rows."
        )


def _convert_report_values(
    value: Any,
) -> Any:
    """
    Convert NumPy values in a classification report to native Python.
    """
    if isinstance(value, dict):
        return {
            str(key): _convert_report_values(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _convert_report_values(item)
            for item in value
        ]

    if isinstance(value, np.generic):
        return value.item()

    return value