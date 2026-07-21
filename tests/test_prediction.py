"""Tests for reusable prediction utilities.

These tests cover the application-facing inference API located in:

    src/titanic_passenger_class_prediction/prediction.py

They verify raw-data validation, deterministic feature engineering, model
validation, probability outputs, confidence calculations, batch prediction,
and single-passenger prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from titanic_passenger_class_prediction.config import (
    ID_COLUMN,
    TARGET_COLUMN,
)
from titanic_passenger_class_prediction.modeling import (
    MODEL_FEATURES,
    build_random_forest_pipeline,
)
from titanic_passenger_class_prediction.prediction import (
    CONFIDENCE_COLUMN,
    LOW_CONFIDENCE_COLUMN,
    LOW_CONFIDENCE_THRESHOLD,
    PREDICTION_COLUMN,
    PROBABILITY_MARGIN_COLUMN,
    RAW_PREDICTION_COLUMNS,
    SECOND_HIGHEST_PROBABILITY_COLUMN,
    add_context_columns,
    build_probability_column_name,
    generate_predictions,
    get_model_classes,
    predict_one_passenger,
    predict_passengers,
    prepare_prediction_data,
    validate_fitted_model,
    validate_raw_prediction_data,
)


def make_raw_dataframe(
    *,
    include_target: bool = True,
) -> pd.DataFrame:
    """Create representative raw Titanic passenger records."""
    dataframe = pd.DataFrame(
        {
            "PassengerId": [
                1001,
                1002,
                1003,
                1004,
                1005,
                1006,
                1007,
                1008,
                1009,
            ],
            "Survived": [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            "Name": [
                "Allen, Mrs. Anna",
                "Brown, Mr. William",
                "Clark, Miss. Emily",
                "Davis, Mr. Henry",
                "Evans, Dr. Sarah",
                "Foster, Master. James",
                "Green, Mrs. Mary",
                "Hill, Mr. Thomas",
                "Irwin, Miss. Alice",
            ],
            "Sex": [
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
            ],
            "Age": [
                35.0,
                41.0,
                22.0,
                28.0,
                52.0,
                8.0,
                30.0,
                None,
                19.0,
            ],
            "SibSp": [
                1,
                0,
                0,
                1,
                0,
                1,
                2,
                0,
                0,
            ],
            "Parch": [
                0,
                0,
                1,
                0,
                0,
                2,
                1,
                0,
                0,
            ],
            "Ticket": [
                "PC 17599",
                "113803",
                "SC/PARIS 2167",
                "A/5 21171",
                "PC 17757",
                "C.A. 33112",
                "347082",
                "STON/O2 3101282",
                "349909",
            ],
            "Fare": [
                90.0,
                75.0,
                30.0,
                8.0,
                120.0,
                26.0,
                18.0,
                7.5,
                10.5,
            ],
            "Cabin": [
                "C85",
                "B28",
                None,
                None,
                "B51 B53 B55",
                "F2",
                None,
                None,
                None,
            ],
            "Embarked": [
                "C",
                "S",
                "S",
                "S",
                "C",
                "S",
                "S",
                "Q",
                "S",
            ],
        }
    )

    if include_target:
        dataframe[TARGET_COLUMN] = [
            1,
            1,
            2,
            3,
            1,
            2,
            3,
            3,
            3,
        ]

    return dataframe


@pytest.fixture
def fitted_model() -> Pipeline:
    """Create a lightweight fitted production-style pipeline."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe(
            include_target=True
        )
    )

    model = build_random_forest_pipeline()

    model.fit(
        prepared_dataframe.loc[
            :,
            MODEL_FEATURES,
        ],
        prepared_dataframe[TARGET_COLUMN],
    )

    return model


def test_raw_prediction_columns_match_expected_schema() -> None:
    """The reusable inference schema should contain original inputs."""
    assert set(RAW_PREDICTION_COLUMNS) == {
        "PassengerId",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
    }


def test_validate_raw_prediction_data_accepts_valid_data() -> None:
    """Complete raw passenger data should pass validation."""
    validate_raw_prediction_data(
        make_raw_dataframe()
    )


def test_validate_raw_prediction_data_supports_unlabelled_data() -> None:
    """Inference must not require Pclass."""
    validate_raw_prediction_data(
        make_raw_dataframe(
            include_target=False
        )
    )


def test_validate_raw_prediction_data_rejects_non_dataframe() -> None:
    """Prediction input must be a dataframe."""
    with pytest.raises(
        TypeError,
        match="DataFrame",
    ):
        validate_raw_prediction_data(
            [{"PassengerId": 1}]
        )


def test_validate_raw_prediction_data_rejects_empty_data() -> None:
    """An empty dataframe should fail validation."""
    with pytest.raises(
        ValueError,
        match="at least one row",
    ):
        validate_raw_prediction_data(
            pd.DataFrame()
        )


def test_validate_raw_prediction_data_rejects_missing_column() -> None:
    """Missing raw inputs should be listed clearly."""
    dataframe = make_raw_dataframe().drop(
        columns=["Ticket"]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_raw_prediction_data(
            dataframe
        )


def test_validate_raw_prediction_data_rejects_missing_id() -> None:
    """Passenger identifiers cannot be missing."""
    dataframe = make_raw_dataframe()
    dataframe.loc[0, ID_COLUMN] = None

    with pytest.raises(
        ValueError,
        match="contains missing values",
    ):
        validate_raw_prediction_data(
            dataframe
        )


def test_validate_raw_prediction_data_rejects_duplicate_ids() -> None:
    """Passenger identifiers must remain unique."""
    dataframe = make_raw_dataframe()
    dataframe.loc[1, ID_COLUMN] = dataframe.loc[0, ID_COLUMN]

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_raw_prediction_data(
            dataframe
        )


def test_prepare_prediction_data_adds_all_model_features() -> None:
    """Raw rows should gain every engineered model input."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    assert len(prepared_dataframe) == 9
    assert set(MODEL_FEATURES).issubset(
        prepared_dataframe.columns
    )

    for column in [
        "FamilySize",
        "FareLog",
        "Title",
        "TicketPrefix",
        "HasCabin",
        "CabinDeck",
    ]:
        assert column in prepared_dataframe.columns


def test_prepare_prediction_data_preserves_target() -> None:
    """Pclass should survive preparation when available."""
    raw_dataframe = make_raw_dataframe(
        include_target=True
    )

    prepared_dataframe = prepare_prediction_data(
        raw_dataframe
    )

    pd.testing.assert_series_equal(
        prepared_dataframe[TARGET_COLUMN],
        raw_dataframe[TARGET_COLUMN],
    )


def test_prepare_prediction_data_supports_unlabelled_input() -> None:
    """Unlabelled rows should still be prediction-ready."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe(
            include_target=False
        )
    )

    assert TARGET_COLUMN not in prepared_dataframe.columns
    assert set(MODEL_FEATURES).issubset(
        prepared_dataframe.columns
    )


def test_prepare_prediction_data_does_not_change_row_count() -> None:
    """Feature engineering must remain one row per passenger."""
    raw_dataframe = make_raw_dataframe()

    prepared_dataframe = prepare_prediction_data(
        raw_dataframe
    )

    assert len(prepared_dataframe) == len(raw_dataframe)
    assert prepared_dataframe[ID_COLUMN].is_unique


def test_get_model_classes_returns_fitted_classes(
    fitted_model: Pipeline,
) -> None:
    """Class labels should be available from the fitted pipeline."""
    assert get_model_classes(fitted_model) == [
        1,
        2,
        3,
    ]


def test_get_model_classes_rejects_unfitted_model() -> None:
    """An unfitted classifier should not expose usable classes."""
    with pytest.raises(
        AttributeError,
        match="class labels",
    ):
        get_model_classes(
            DummyClassifier()
        )


def test_validate_fitted_model_accepts_classifier(
    fitted_model: Pipeline,
) -> None:
    """A fitted probability classifier should pass validation."""
    validate_fitted_model(
        fitted_model
    )


class PredictOnlyEstimator:
    """Minimal estimator intentionally lacking predict_proba."""

    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        return np.ones(
            len(features)
        )


def test_validate_fitted_model_requires_predict_proba() -> None:
    """The prediction service requires probabilities."""
    with pytest.raises(
        TypeError,
        match="predict_proba",
    ):
        validate_fitted_model(
            PredictOnlyEstimator()
        )


def test_probability_column_name_is_stable() -> None:
    """Class labels should map to predictable names."""
    assert (
        build_probability_column_name(1)
        == "ProbabilityClass1"
    )

    assert (
        build_probability_column_name(np.int64(3))
        == "ProbabilityClass3"
    )


def test_generate_predictions_creates_expected_columns(
    fitted_model: Pipeline,
) -> None:
    """Prepared rows should produce complete prediction outputs."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
    )

    expected_columns = {
        ID_COLUMN,
        PREDICTION_COLUMN,
        "ProbabilityClass1",
        "ProbabilityClass2",
        "ProbabilityClass3",
        CONFIDENCE_COLUMN,
        SECOND_HIGHEST_PROBABILITY_COLUMN,
        PROBABILITY_MARGIN_COLUMN,
        LOW_CONFIDENCE_COLUMN,
    }

    assert expected_columns.issubset(
        predictions.columns
    )
    assert len(predictions) == len(prepared_dataframe)


def test_generate_predictions_probabilities_sum_to_one(
    fitted_model: Pipeline,
) -> None:
    """Class probabilities should sum to one."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
    )

    probability_columns = [
        "ProbabilityClass1",
        "ProbabilityClass2",
        "ProbabilityClass3",
    ]

    assert np.allclose(
        predictions[probability_columns].sum(axis=1),
        1.0,
    )


def test_generate_predictions_confidence_is_max_probability(
    fitted_model: Pipeline,
) -> None:
    """Confidence should equal the largest class probability."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
    )

    probability_columns = [
        "ProbabilityClass1",
        "ProbabilityClass2",
        "ProbabilityClass3",
    ]

    expected = predictions[
        probability_columns
    ].max(axis=1)

    assert np.allclose(
        predictions[CONFIDENCE_COLUMN],
        expected,
    )


def test_generate_predictions_margin_is_correct(
    fitted_model: Pipeline,
) -> None:
    """Probability margin should compare the two leading classes."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
    )

    expected = (
        predictions[CONFIDENCE_COLUMN]
        - predictions[
            SECOND_HIGHEST_PROBABILITY_COLUMN
        ]
    )

    assert np.allclose(
        predictions[PROBABILITY_MARGIN_COLUMN],
        expected,
    )


def test_low_confidence_flag_matches_threshold(
    fitted_model: Pipeline,
) -> None:
    """Low-confidence flags should use the configured threshold."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
    )

    expected = (
        predictions[CONFIDENCE_COLUMN]
        < LOW_CONFIDENCE_THRESHOLD
    )

    pd.testing.assert_series_equal(
        predictions[LOW_CONFIDENCE_COLUMN],
        expected,
        check_names=False,
    )


def test_generate_predictions_rejects_missing_feature(
    fitted_model: Pipeline,
) -> None:
    """Prepared data must contain every model feature."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    ).drop(
        columns=["FareLog"]
    )

    with pytest.raises(
        ValueError,
        match="missing model features",
    ):
        generate_predictions(
            model=fitted_model,
            prepared_dataframe=prepared_dataframe,
        )


def test_add_context_columns_preserves_passengers(
    fitted_model: Pipeline,
) -> None:
    """Passenger context should merge one-to-one."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
    )

    enriched = add_context_columns(
        predictions=predictions,
        prepared_dataframe=prepared_dataframe,
    )

    assert len(enriched) == len(predictions)
    assert enriched[ID_COLUMN].tolist() == (
        prepared_dataframe[ID_COLUMN].tolist()
    )

    for column in [
        "Name",
        "Sex",
        "Age",
        "Fare",
        "Embarked",
    ]:
        assert column in enriched.columns


def test_predict_passengers_accepts_raw_dataframe(
    fitted_model: Pipeline,
) -> None:
    """The public batch API should perform feature engineering."""
    raw_dataframe = make_raw_dataframe(
        include_target=False
    )

    predictions = predict_passengers(
        raw_dataframe=raw_dataframe,
        model=fitted_model,
    )

    assert len(predictions) == len(raw_dataframe)
    assert predictions[ID_COLUMN].is_unique
    assert PREDICTION_COLUMN in predictions.columns
    assert CONFIDENCE_COLUMN in predictions.columns
    assert "Name" in predictions.columns


def test_predict_one_passenger_returns_mapping(
    fitted_model: Pipeline,
) -> None:
    """One raw passenger should produce one serializable result."""
    passenger = (
        make_raw_dataframe(
            include_target=False
        )
        .iloc[0]
        .to_dict()
    )

    result = predict_one_passenger(
        passenger=passenger,
        model=fitted_model,
    )

    assert isinstance(result, dict)
    assert result[ID_COLUMN] == passenger[ID_COLUMN]
    assert result[PREDICTION_COLUMN] in {
        1,
        2,
        3,
    }
    assert 0.0 <= result[CONFIDENCE_COLUMN] <= 1.0

    for column in [
        "ProbabilityClass1",
        "ProbabilityClass2",
        "ProbabilityClass3",
    ]:
        assert column in result


def test_predict_one_passenger_rejects_non_dictionary(
    fitted_model: Pipeline,
) -> None:
    """The single-record API should reject invalid input types."""
    with pytest.raises(
        TypeError,
        match="dictionary",
    ):
        predict_one_passenger(
            passenger=["not", "a", "dictionary"],
            model=fitted_model,
        )


def test_predict_one_passenger_rejects_empty_dictionary(
    fitted_model: Pipeline,
) -> None:
    """An empty passenger mapping cannot be predicted."""
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        predict_one_passenger(
            passenger={},
            model=fitted_model,
        )
