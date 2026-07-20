"""Hyperparameter tuning utilities for passenger-class models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV

from titanic_passenger_class_prediction.config import (
    N_JOBS,
    RANDOM_STATE,
)
from titanic_passenger_class_prediction.evaluation import (
    build_cross_validator,
)


DEFAULT_SEARCH_ITERATIONS = 20
TUNING_SCORING = "f1_macro"


@dataclass(frozen=True)
class TuningResult:
    """
    Store the outputs of a completed randomized hyperparameter search.

    Attributes
    ----------
    model_name
        Registry name of the tuned model.
    best_estimator
        Fitted estimator using the best sampled parameters.
    best_parameters
        Parameter values selected by cross-validation.
    best_cv_score
        Best mean cross-validation Macro F1 score.
    search_results
        Ranked table containing every evaluated parameter combination.
    """

    model_name: str
    best_estimator: BaseEstimator
    best_parameters: dict[str, Any]
    best_cv_score: float
    search_results: pd.DataFrame


def get_parameter_distributions(
    model_name: str,
) -> dict[str, Sequence[Any]]:
    """
    Return randomized-search parameter distributions for a model.

    Parameters
    ----------
    model_name
        Candidate-model registry key.

    Returns
    -------
    dict[str, Sequence[Any]]
        Pipeline-compatible hyperparameter distributions.

    Raises
    ------
    ValueError
        If the model does not have a supported tuning configuration.
    """
    parameter_spaces: dict[
        str,
        dict[str, Sequence[Any]],
    ] = {
        "logistic_regression": {
            "model__C": [
                0.001,
                0.01,
                0.1,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                50.0,
                100.0,
            ],
            "model__class_weight": [
                None,
                "balanced",
            ],
            "model__solver": [
                "lbfgs",
                "liblinear",
            ],
        },
        "random_forest": {
            "model__n_estimators": [
                100,
                200,
                300,
                500,
                750,
            ],
            "model__max_depth": [
                None,
                5,
                10,
                15,
                20,
                30,
            ],
            "model__min_samples_split": [
                2,
                5,
                10,
                15,
            ],
            "model__min_samples_leaf": [
                1,
                2,
                4,
                8,
            ],
            "model__max_features": [
                "sqrt",
                "log2",
                None,
            ],
            "model__class_weight": [
                None,
                "balanced",
                "balanced_subsample",
            ],
            "model__bootstrap": [
                True,
                False,
            ],
        },
    }

    if not model_name.strip():
        raise ValueError(
            "Model name cannot be empty."
        )

    if model_name not in parameter_spaces:
        supported_models = sorted(parameter_spaces)

        raise ValueError(
            "No tuning parameter distributions are configured for "
            f"model '{model_name}'. Supported models: "
            f"{supported_models}"
        )

    return parameter_spaces[model_name]


def validate_tuning_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> None:
    """
    Validate feature and target data before hyperparameter tuning.

    Parameters
    ----------
    features
        Training feature matrix.
    target
        Training target vector.

    Raises
    ------
    ValueError
        If the data is empty, row counts differ, the target contains
        missing values, or fewer than two target classes are present.
    """
    if features.empty:
        raise ValueError(
            "Cannot tune a model using an empty feature matrix."
        )

    if target.empty:
        raise ValueError(
            "Cannot tune a model using an empty target vector."
        )

    if len(features) != len(target):
        raise ValueError(
            "Features and target must contain the same number of rows."
        )

    if target.isna().any():
        raise ValueError(
            "The tuning target contains missing values."
        )

    if target.nunique() < 2:
        raise ValueError(
            "Hyperparameter tuning requires at least two target "
            "classes."
        )


def build_randomized_search(
    estimator: BaseEstimator,
    parameter_distributions: Mapping[
        str,
        Sequence[Any],
    ],
    n_iter: int = DEFAULT_SEARCH_ITERATIONS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
) -> RandomizedSearchCV:
    """
    Build a configured randomized hyperparameter search.

    Parameters
    ----------
    estimator
        Unfitted estimator or pipeline to tune.
    parameter_distributions
        Hyperparameter names and candidate values.
    n_iter
        Number of parameter combinations to sample.
    random_state
        Seed controlling randomized parameter sampling.
    n_jobs
        Number of parallel jobs used during cross-validation.

    Returns
    -------
    sklearn.model_selection.RandomizedSearchCV
        Configured randomized-search object.

    Raises
    ------
    ValueError
        If the parameter mapping is empty or the iteration count is
        not positive.
    """
    if not parameter_distributions:
        raise ValueError(
            "At least one parameter distribution is required."
        )

    if n_iter <= 0:
        raise ValueError(
            "Randomized-search iterations must be greater than zero."
        )

    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=dict(
            parameter_distributions
        ),
        n_iter=n_iter,
        scoring=TUNING_SCORING,
        n_jobs=n_jobs,
        cv=build_cross_validator(),
        refit=True,
        random_state=random_state,
        return_train_score=True,
        error_score="raise",
    )


def build_search_results_dataframe(
    search: RandomizedSearchCV,
) -> pd.DataFrame:
    """
    Convert fitted randomized-search results into a ranked table.

    Parameters
    ----------
    search
        Fitted randomized hyperparameter search.

    Returns
    -------
    pandas.DataFrame
        Ranked search results with parameters, scores, and timings.

    Raises
    ------
    ValueError
        If the search has not been fitted.
    """
    if not hasattr(search, "cv_results_"):
        raise ValueError(
            "Randomized search must be fitted before its results "
            "can be converted."
        )

    results = pd.DataFrame(
        search.cv_results_
    ).copy()

    preferred_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
        "params",
    ]

    parameter_columns = sorted(
        column
        for column in results.columns
        if column.startswith("param_")
    )

    selected_columns = [
        column
        for column in (
            preferred_columns[:-1]
            + parameter_columns
            + ["params"]
        )
        if column in results.columns
    ]

    ranked_results = (
        results.loc[:, selected_columns]
        .sort_values(
            by=[
                "rank_test_score",
                "mean_test_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return ranked_results


def tune_model(
    model_name: str,
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    parameter_distributions: Mapping[
        str,
        Sequence[Any],
    ] | None = None,
    n_iter: int = DEFAULT_SEARCH_ITERATIONS,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
) -> TuningResult:
    """
    Tune a model using randomized stratified cross-validation.

    Parameters
    ----------
    model_name
        Candidate-model registry key.
    estimator
        Unfitted estimator or pipeline.
    features
        Training feature matrix.
    target
        Training target vector.
    parameter_distributions
        Optional custom parameter space. When omitted, the configured
        parameter space for ``model_name`` is used.
    n_iter
        Number of sampled parameter combinations.
    random_state
        Seed controlling randomized parameter sampling.
    n_jobs
        Number of parallel jobs.

    Returns
    -------
    TuningResult
        Fitted best estimator, selected parameters, best CV score, and
        complete ranked search results.
    """
    if not model_name.strip():
        raise ValueError(
            "Model name cannot be empty."
        )

    validate_tuning_data(
        features=features,
        target=target,
    )

    distributions = (
        dict(parameter_distributions)
        if parameter_distributions is not None
        else get_parameter_distributions(model_name)
    )

    search = build_randomized_search(
        estimator=estimator,
        parameter_distributions=distributions,
        n_iter=n_iter,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    search.fit(
        features,
        target,
    )

    search_results = build_search_results_dataframe(
        search
    )

    return TuningResult(
        model_name=model_name,
        best_estimator=search.best_estimator_,
        best_parameters=dict(search.best_params_),
        best_cv_score=float(search.best_score_),
        search_results=search_results,
    )