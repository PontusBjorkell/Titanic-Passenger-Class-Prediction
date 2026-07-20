import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from titanic_passenger_class_prediction.evaluation import (
    build_cross_validator,
    compare_models_cv,
    evaluate_model_cv,
    select_best_model_name,
)


def make_evaluation_dataframe() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Create a compact three-class dataset for evaluation tests.

    Each passenger class has five rows. This is important because the
    production configuration uses five-fold stratified cross-validation,
    which requires at least five observations in every target class.
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


def test_build_cross_validator_uses_expected_settings() -> None:
    cross_validator = build_cross_validator()

    assert cross_validator.n_splits == 5
    assert cross_validator.shuffle is True
    assert cross_validator.random_state == 42


def test_evaluate_model_cv_returns_expected_metrics() -> None:
    features, target = make_evaluation_dataframe()

    estimator = DummyClassifier(
        strategy="most_frequent",
    )

    result = evaluate_model_cv(
        model_name="dummy",
        estimator=estimator,
        features=features,
        target=target,
    )

    expected_keys = {
        "model",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_balanced_accuracy_mean",
        "cv_balanced_accuracy_std",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
        "fit_time_mean",
        "score_time_mean",
    }

    assert set(result) == expected_keys
    assert result["model"] == "dummy"

    assert 0.0 <= result["cv_accuracy_mean"] <= 1.0
    assert 0.0 <= result["cv_balanced_accuracy_mean"] <= 1.0
    assert 0.0 <= result["cv_f1_macro_mean"] <= 1.0

    assert result["cv_accuracy_std"] >= 0.0
    assert result["cv_balanced_accuracy_std"] >= 0.0
    assert result["cv_f1_macro_std"] >= 0.0

    assert result["fit_time_mean"] >= 0.0
    assert result["score_time_mean"] >= 0.0


def test_evaluate_model_cv_rejects_empty_features() -> None:
    _, target = make_evaluation_dataframe()

    with pytest.raises(
        ValueError,
        match="empty feature matrix",
    ):
        evaluate_model_cv(
            model_name="dummy",
            estimator=DummyClassifier(),
            features=pd.DataFrame(),
            target=target,
        )


def test_evaluate_model_cv_rejects_length_mismatch() -> None:
    features, target = make_evaluation_dataframe()

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        evaluate_model_cv(
            model_name="dummy",
            estimator=DummyClassifier(),
            features=features.iloc[:-1],
            target=target,
        )


def test_compare_models_cv_returns_sorted_dataframe() -> None:
    features, target = make_evaluation_dataframe()

    models = {
        "most_frequent": DummyClassifier(
            strategy="most_frequent",
        ),
        "stratified": DummyClassifier(
            strategy="stratified",
            random_state=42,
        ),
    }

    comparison = compare_models_cv(
        models=models,
        features=features,
        target=target,
    )

    assert len(comparison) == len(models)
    assert set(comparison["model"]) == set(models)

    assert comparison[
        "cv_f1_macro_mean"
    ].is_monotonic_decreasing


def test_compare_models_cv_rejects_empty_registry() -> None:
    features, target = make_evaluation_dataframe()

    with pytest.raises(
        ValueError,
        match="At least one candidate model",
    ):
        compare_models_cv(
            models={},
            features=features,
            target=target,
        )


def test_select_best_model_name_returns_first_model() -> None:
    comparison = pd.DataFrame(
        {
            "model": [
                "random_forest",
                "logistic_regression",
                "dummy",
            ],
            "cv_f1_macro_mean": [
                0.70,
                0.65,
                0.25,
            ],
        }
    )

    selected = select_best_model_name(comparison)

    assert selected == "random_forest"


def test_select_best_model_name_rejects_empty_table() -> None:
    with pytest.raises(
        ValueError,
        match="empty comparison table",
    ):
        select_best_model_name(pd.DataFrame())


def test_select_best_model_name_requires_model_column() -> None:
    comparison = pd.DataFrame(
        {
            "cv_f1_macro_mean": [
                0.70,
                0.65,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="'model' column",
    ):
        select_best_model_name(comparison)