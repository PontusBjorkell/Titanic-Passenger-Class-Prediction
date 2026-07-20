"""Model evaluation utilities for passenger-class prediction."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
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
# Evaluation metrics
# ---------------------------------------------------------------------------

SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
}


# ---------------------------------------------------------------------------
# Cross-validation configuration
# ---------------------------------------------------------------------------

def build_cross_validator() -> StratifiedKFold:
    """
    Create the shared stratified cross-validation strategy.

    Stratification preserves the approximate target-class distribution
    inside every fold. Shuffling prevents the original row order from
    determining the fold composition.

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


# ---------------------------------------------------------------------------
# Individual model evaluation
# ---------------------------------------------------------------------------

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
        Stable name used to identify the model in reports.
    estimator
        Unfitted scikit-learn estimator or pipeline.
    features
        Modeling feature matrix.
    target
        Target class labels.

    Returns
    -------
    dict[str, float | str]
        Cross-validation summary containing the mean and standard
        deviation of each metric, plus average fit and scoring times.

    Raises
    ------
    ValueError
        If the feature matrix or target is empty, or if their lengths
        do not match.
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

    cross_validator = build_cross_validator()

    scores = cross_validate(
        estimator=estimator,
        X=features,
        y=target,
        scoring=SCORING,
        cv=cross_validator,
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


# ---------------------------------------------------------------------------
# Candidate-model comparison
# ---------------------------------------------------------------------------

def compare_models_cv(
    models: Mapping[str, BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Compare candidate models using the same cross-validation strategy.

    Every candidate receives the same folds and evaluation metrics,
    making their results directly comparable.

    Models are ranked primarily by macro F1, then by balanced accuracy,
    and finally by ordinary accuracy.

    Parameters
    ----------
    models
        Mapping between stable model names and unfitted estimators.
    features
        Modeling feature matrix.
    target
        Target class labels.

    Returns
    -------
    pandas.DataFrame
        Candidate-model comparison table sorted from strongest to
        weakest.

    Raises
    ------
    ValueError
        If no candidate models are supplied.
    """
    if not models:
        raise ValueError(
            "At least one candidate model is required."
        )

    results: list[dict[str, float | str]] = []

    for model_name, estimator in models.items():
        result = evaluate_model_cv(
            model_name=model_name,
            estimator=estimator,
            features=features,
            target=target,
        )
        results.append(result)

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
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

    return comparison


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_best_model_name(
    comparison: pd.DataFrame,
) -> str:
    """
    Select the highest-ranked model from a comparison table.

    The comparison table is expected to have already been sorted by
    ``compare_models_cv``.

    Parameters
    ----------
    comparison
        Sorted cross-validation model-comparison table.

    Returns
    -------
    str
        Stable name of the highest-ranked candidate.

    Raises
    ------
    ValueError
        If the table is empty or does not contain a ``model`` column.
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