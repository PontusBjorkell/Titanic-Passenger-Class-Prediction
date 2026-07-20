"""Tests for randomized hyperparameter-tuning utilities."""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from titanic_passenger_class_prediction.modeling import (
    build_random_forest_pipeline,
)
from titanic_passenger_class_prediction.tuning import (
    TUNING_SCORING,
    TuningResult,
    build_randomized_search,
    build_search_results_dataframe,
    get_parameter_distributions,
    tune_model,
    validate_tuning_data,
)


def make_tuning_dataframe() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Create a compact three-class dataset for tuning tests.

    Each target class contains five rows so the production five-fold
    stratified cross-validator can operate successfully.
    """
    features = pd.DataFrame(
        {
            "Age": [
                40.0,
                38.0,
                50.0,
                35.0,
                45.0,
                30.0,
                28.0,
                25.0,
                32.0,
                27.0,
                22.0,
                20.0,
                18.0,
                24.0,
                21.0,
            ],
            "SibSp": [
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                2,
                0,
                1,
            ],
            "Parch": [
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
            ],
            "Fare": [
                80.0,
                72.0,
                90.0,
                65.0,
                75.0,
                30.0,
                25.0,
                35.0,
                28.0,
                32.0,
                8.0,
                7.5,
                12.0,
                9.0,
                10.0,
            ],
            "FamilySize": [
                1,
                2,
                2,
                2,
                1,
                1,
                3,
                1,
                2,
                2,
                2,
                1,
                4,
                1,
                2,
            ],
            "IsAlone": [
                1,
                0,
                0,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
            ],
            "HasCabin": [
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "CabinCount": [
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "FareLog": [
                4.39,
                4.29,
                4.51,
                4.19,
                4.33,
                3.43,
                3.26,
                3.58,
                3.37,
                3.50,
                2.20,
                2.14,
                2.56,
                2.30,
                2.40,
            ],
            "Sex": [
                "female",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "male",
                "female",
            ],
            "Embarked": [
                "C",
                "C",
                "S",
                "S",
                "C",
                "S",
                "C",
                "S",
                "S",
                "C",
                "S",
                "S",
                "Q",
                "S",
                "Q",
            ],
            "Title": [
                "Mrs",
                "Miss",
                "Mr",
                "Mrs",
                "Mr",
                "Miss",
                "Mr",
                "Mrs",
                "Mr",
                "Miss",
                "Mr",
                "Miss",
                "Master",
                "Mr",
                "Miss",
            ],
            "FamilySizeGroup": [
                "Alone",
                "Small",
                "Small",
                "Small",
                "Alone",
                "Alone",
                "Small",
                "Alone",
                "Small",
                "Small",
                "Small",
                "Alone",
                "Small",
                "Alone",
                "Small",
            ],
            "CabinDeck": [
                "B",
                "C",
                "B",
                "C",
                "D",
                "Unknown",
                "E",
                "Unknown",
                "D",
                "Unknown",
                "Unknown",
                "Unknown",
                "Unknown",
                "Unknown",
                "Unknown",
            ],
            "TicketPrefix": [
                "PC",
                "PC",
                "NONE",
                "PC",
                "NONE",
                "NONE",
                "NONE",
                "NONE",
                "NONE",
                "NONE",
                "A_5",
                "NONE",
                "STON_O2",
                "NONE",
                "NONE",
            ],
        }
    )

    target = pd.Series(
        [
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
        ],
        name="Pclass",
    )

    return features, target


def test_get_random_forest_parameter_distributions() -> None:
    """Random Forest should expose pipeline-compatible parameters."""
    distributions = get_parameter_distributions(
        "random_forest"
    )

    assert "model__n_estimators" in distributions
    assert "model__max_depth" in distributions
    assert "model__min_samples_split" in distributions
    assert "model__min_samples_leaf" in distributions
    assert "model__class_weight" in distributions


def test_get_logistic_parameter_distributions() -> None:
    """Logistic Regression should expose supported parameters."""
    distributions = get_parameter_distributions(
        "logistic_regression"
    )

    assert "model__C" in distributions
    assert "model__class_weight" in distributions
    assert "model__solver" in distributions


def test_get_parameter_distributions_rejects_unknown_model() -> None:
    """Unsupported models should be rejected explicitly."""
    with pytest.raises(
        ValueError,
        match="No tuning parameter distributions",
    ):
        get_parameter_distributions("dummy")


def test_get_parameter_distributions_rejects_empty_name() -> None:
    """An empty model name should not be accepted."""
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        get_parameter_distributions("")


def test_validate_tuning_data_accepts_valid_data() -> None:
    """Valid feature and target data should pass validation."""
    features, target = make_tuning_dataframe()

    validate_tuning_data(
        features=features,
        target=target,
    )


def test_validate_tuning_data_rejects_empty_features() -> None:
    """An empty feature matrix cannot support tuning."""
    _, target = make_tuning_dataframe()

    with pytest.raises(
        ValueError,
        match="empty feature matrix",
    ):
        validate_tuning_data(
            features=pd.DataFrame(),
            target=target,
        )


def test_validate_tuning_data_rejects_length_mismatch() -> None:
    """Feature and target row counts must match."""
    features, target = make_tuning_dataframe()

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        validate_tuning_data(
            features=features.iloc[:-1],
            target=target,
        )


def test_validate_tuning_data_rejects_missing_target() -> None:
    """Missing target values should be rejected."""
    features, target = make_tuning_dataframe()
    target = target.astype(float)
    target.iloc[0] = None

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        validate_tuning_data(
            features=features,
            target=target,
        )


def test_validate_tuning_data_requires_multiple_classes() -> None:
    """Tuning requires at least two target classes."""
    features, target = make_tuning_dataframe()
    target = pd.Series(
        [1] * len(target),
        name="Pclass",
    )

    with pytest.raises(
        ValueError,
        match="at least two target classes",
    ):
        validate_tuning_data(
            features=features,
            target=target,
        )


def test_build_randomized_search_uses_expected_settings() -> None:
    """The randomized search should use Macro F1 and project CV."""
    estimator = Pipeline(
        steps=[
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                ),
            ),
        ]
    )

    search = build_randomized_search(
        estimator=estimator,
        parameter_distributions={
            "model__n_estimators": [
                5,
                10,
            ],
        },
        n_iter=2,
        random_state=42,
        n_jobs=1,
    )

    assert isinstance(
        search,
        RandomizedSearchCV,
    )
    assert search.n_iter == 2
    assert search.scoring == TUNING_SCORING
    assert search.random_state == 42
    assert search.n_jobs == 1
    assert search.refit is True
    assert search.cv.n_splits == 5


def test_build_randomized_search_rejects_empty_space() -> None:
    """At least one hyperparameter must be configured."""
    with pytest.raises(
        ValueError,
        match="At least one parameter distribution",
    ):
        build_randomized_search(
            estimator=DummyClassifier(),
            parameter_distributions={},
        )


def test_build_randomized_search_rejects_invalid_iterations() -> None:
    """The number of random samples must be positive."""
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        build_randomized_search(
            estimator=DummyClassifier(),
            parameter_distributions={
                "strategy": [
                    "most_frequent",
                ],
            },
            n_iter=0,
        )


def test_build_search_results_requires_fitted_search() -> None:
    """Search results should only be available after fitting."""
    search = build_randomized_search(
        estimator=DummyClassifier(),
        parameter_distributions={
            "strategy": [
                "most_frequent",
            ],
        },
        n_iter=1,
        n_jobs=1,
    )

    with pytest.raises(
        ValueError,
        match="must be fitted",
    ):
        build_search_results_dataframe(search)


def test_tune_model_returns_expected_result() -> None:
    """Tuning should return a fitted estimator and ranked results."""
    features, target = make_tuning_dataframe()

    estimator = build_random_forest_pipeline()

    result = tune_model(
        model_name="random_forest",
        estimator=estimator,
        features=features,
        target=target,
        parameter_distributions={
            "model__n_estimators": [
                5,
                10,
            ],
            "model__max_depth": [
                2,
                4,
            ],
        },
        n_iter=2,
        random_state=42,
        n_jobs=1,
    )

    assert isinstance(
        result,
        TuningResult,
    )
    assert result.model_name == "random_forest"
    assert result.best_parameters
    assert 0.0 <= result.best_cv_score <= 1.0

    assert not result.search_results.empty
    assert "rank_test_score" in result.search_results
    assert "mean_test_score" in result.search_results
    assert "params" in result.search_results

    assert result.search_results[
        "rank_test_score"
    ].is_monotonic_increasing

    predictions = result.best_estimator.predict(
        features
    )

    assert len(predictions) == len(features)


def test_tune_model_supports_custom_parameter_space() -> None:
    """A caller-provided parameter space should override defaults."""
    features, target = make_tuning_dataframe()

    estimator = Pipeline(
        steps=[
            (
                "model",
                DummyClassifier(),
            ),
        ]
    )

    result = tune_model(
        model_name="custom_dummy",
        estimator=estimator,
        features=features,
        target=target,
        parameter_distributions={
            "model__strategy": [
                "most_frequent",
                "stratified",
            ],
        },
        n_iter=2,
        random_state=42,
        n_jobs=1,
    )

    assert result.model_name == "custom_dummy"
    assert "model__strategy" in result.best_parameters
    assert len(result.search_results) == 2