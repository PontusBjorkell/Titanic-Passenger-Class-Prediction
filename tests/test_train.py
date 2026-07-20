"""Tests for the model-training orchestration script."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier

from scripts.train import (
    BEST_PARAMETERS_FILENAME,
    TUNING_RESULTS_FILENAME,
    save_tuning_results,
    select_model_variant,
    split_training_data,
    train_selected_model,
    validate_training_dataframe,
)
from titanic_passenger_class_prediction.modeling import MODEL_FEATURES
from titanic_passenger_class_prediction.tuning import TuningResult


def make_training_dataframe(
    rows_per_class: int = 10,
) -> pd.DataFrame:
    """Create a compact prepared dataset with all modeling columns."""
    total_rows = rows_per_class * 3

    dataframe = pd.DataFrame(
        {
            feature: [0] * total_rows
            for feature in MODEL_FEATURES
        }
    )

    numeric_features = [
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
        "HasCabin",
        "CabinCount",
        "FareLog",
    ]

    for feature in numeric_features:
        dataframe[feature] = [
            float(index + 1)
            for index in range(total_rows)
        ]

    dataframe["Sex"] = [
        "female" if index % 2 == 0 else "male"
        for index in range(total_rows)
    ]
    dataframe["Embarked"] = ["S"] * total_rows
    dataframe["Title"] = ["Mr"] * total_rows
    dataframe["FamilySizeGroup"] = ["Alone"] * total_rows
    dataframe["CabinDeck"] = ["Unknown"] * total_rows
    dataframe["TicketPrefix"] = ["NONE"] * total_rows

    dataframe["Pclass"] = (
        ([1] * rows_per_class)
        + ([2] * rows_per_class)
        + ([3] * rows_per_class)
    )

    return dataframe


def make_tuning_result(
    *,
    best_estimator: BaseEstimator | None = None,
    best_cv_score: float = 0.75,
) -> TuningResult:
    """Create a reusable tuning result for orchestration tests."""
    estimator = best_estimator or DummyClassifier(
        strategy="stratified",
        random_state=42,
    )

    return TuningResult(
        model_name="dummy",
        best_estimator=estimator,
        best_parameters={
            "model__strategy": "stratified",
        },
        best_cv_score=best_cv_score,
        search_results=pd.DataFrame(
            {
                "rank_test_score": [1, 2],
                "mean_test_score": [
                    best_cv_score,
                    best_cv_score - 0.05,
                ],
                "params": [
                    {"model__strategy": "stratified"},
                    {"model__strategy": "most_frequent"},
                ],
            }
        ),
    )


def test_validate_training_dataframe_accepts_valid_data() -> None:
    validate_training_dataframe(make_training_dataframe())


def test_validate_training_dataframe_rejects_empty_data() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_training_dataframe(pd.DataFrame())


def test_validate_training_dataframe_rejects_missing_columns() -> None:
    dataframe = make_training_dataframe().drop(columns=["FareLog"])

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_training_dataframe(dataframe)


def test_validate_training_dataframe_rejects_missing_target() -> None:
    dataframe = make_training_dataframe()
    dataframe.loc[0, "Pclass"] = None

    with pytest.raises(
        ValueError,
        match="contains missing values",
    ):
        validate_training_dataframe(dataframe)


def test_validate_training_dataframe_requires_multiple_classes() -> None:
    dataframe = make_training_dataframe()
    dataframe["Pclass"] = 1

    with pytest.raises(
        ValueError,
        match="at least two target classes",
    ):
        validate_training_dataframe(dataframe)


def test_split_training_data_preserves_all_rows() -> None:
    dataframe = make_training_dataframe()

    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = split_training_data(dataframe)

    assert len(features_train) + len(features_test) == len(dataframe)
    assert len(target_train) + len(target_test) == len(dataframe)
    assert list(features_train.columns) == MODEL_FEATURES
    assert list(features_test.columns) == MODEL_FEATURES


def test_split_training_data_is_stratified() -> None:
    dataframe = make_training_dataframe(rows_per_class=20)

    (
        _,
        _,
        target_train,
        target_test,
    ) = split_training_data(dataframe)

    assert set(target_train.unique()) == {1, 2, 3}
    assert set(target_test.unique()) == {1, 2, 3}

    train_proportions = (
        target_train.value_counts(normalize=True).sort_index()
    )
    test_proportions = (
        target_test.value_counts(normalize=True).sort_index()
    )

    pd.testing.assert_series_equal(
        train_proportions,
        test_proportions,
        check_names=False,
        atol=0.05,
    )


def test_split_training_data_is_reproducible() -> None:
    dataframe = make_training_dataframe(rows_per_class=20)

    first_split = split_training_data(dataframe)
    second_split = split_training_data(dataframe)

    for first, second in zip(first_split, second_split, strict=True):
        pd.testing.assert_frame_equal(
            first,
            second,
        ) if isinstance(first, pd.DataFrame) else (
            pd.testing.assert_series_equal(first, second)
        )


def test_train_selected_model_fits_clone() -> None:
    dataframe = make_training_dataframe()
    features = dataframe.loc[:, MODEL_FEATURES]
    target = dataframe["Pclass"]

    original = DummyClassifier(strategy="most_frequent")
    models: dict[str, BaseEstimator] = {"dummy": original}

    fitted = train_selected_model(
        model_name="dummy",
        models=models,
        features=features,
        target=target,
    )

    predictions = fitted.predict(features)

    assert fitted is not original
    assert len(predictions) == len(features)
    assert not hasattr(original, "classes_")
    assert hasattr(fitted, "classes_")


def test_train_selected_model_rejects_unknown_name() -> None:
    dataframe = make_training_dataframe()

    with pytest.raises(ValueError, match="Unknown model name"):
        train_selected_model(
            model_name="missing",
            models={"dummy": DummyClassifier()},
            features=dataframe[MODEL_FEATURES],
            target=dataframe["Pclass"],
        )


def test_select_model_variant_chooses_tuned_model() -> None:
    baseline = DummyClassifier(strategy="most_frequent")
    tuned = DummyClassifier(
        strategy="stratified",
        random_state=42,
    )
    tuning_result = make_tuning_result(
        best_estimator=tuned,
        best_cv_score=0.75,
    )

    selected, variant, selected_score = select_model_variant(
        baseline_estimator=baseline,
        baseline_cv_score=0.70,
        tuning_result=tuning_result,
    )

    assert selected is tuned
    assert variant == "tuned"
    assert selected_score == pytest.approx(0.75)


def test_select_model_variant_keeps_baseline_when_worse() -> None:
    baseline = DummyClassifier(strategy="most_frequent")
    tuning_result = make_tuning_result(best_cv_score=0.65)

    selected, variant, selected_score = select_model_variant(
        baseline_estimator=baseline,
        baseline_cv_score=0.70,
        tuning_result=tuning_result,
    )

    assert selected is baseline
    assert variant == "baseline"
    assert selected_score == pytest.approx(0.70)


def test_select_model_variant_keeps_baseline_on_tie() -> None:
    baseline = DummyClassifier(strategy="most_frequent")
    tuning_result = make_tuning_result(best_cv_score=0.70)

    selected, variant, selected_score = select_model_variant(
        baseline_estimator=baseline,
        baseline_cv_score=0.70,
        tuning_result=tuning_result,
    )

    assert selected is baseline
    assert variant == "baseline"
    assert selected_score == pytest.approx(0.70)


def test_save_tuning_results_creates_artifacts(
    tmp_path: Path,
) -> None:
    tuning_result = make_tuning_result(best_cv_score=0.91)

    results_path, parameters_path = save_tuning_results(
        tuning_result=tuning_result,
        metrics_directory=tmp_path,
    )

    assert results_path == tmp_path / TUNING_RESULTS_FILENAME
    assert parameters_path == tmp_path / BEST_PARAMETERS_FILENAME
    assert results_path.exists()
    assert parameters_path.exists()

    saved_results = pd.read_csv(results_path)

    assert len(saved_results) == 2
    assert saved_results["mean_test_score"].tolist() == pytest.approx(
        [0.91, 0.86]
    )

    with parameters_path.open("r", encoding="utf-8") as file:
        saved_parameters = json.load(file)

    assert saved_parameters["model_name"] == "dummy"
    assert saved_parameters["best_cv_f1_macro"] == pytest.approx(0.91)
    assert (
        saved_parameters["best_parameters"]["model__strategy"]
        == "stratified"
    )


def test_save_tuning_results_creates_missing_directory(
    tmp_path: Path,
) -> None:
    metrics_directory = tmp_path / "nested" / "metrics"

    results_path, parameters_path = save_tuning_results(
        tuning_result=make_tuning_result(),
        metrics_directory=metrics_directory,
    )

    assert metrics_directory.exists()
    assert results_path.exists()
    assert parameters_path.exists()


def test_save_tuning_results_overwrites_existing_files(
    tmp_path: Path,
) -> None:
    first_result = make_tuning_result(best_cv_score=0.70)
    second_result = make_tuning_result(best_cv_score=0.80)

    save_tuning_results(
        tuning_result=first_result,
        metrics_directory=tmp_path,
    )
    _, parameters_path = save_tuning_results(
        tuning_result=second_result,
        metrics_directory=tmp_path,
    )

    with parameters_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["best_cv_f1_macro"] == pytest.approx(0.80)
