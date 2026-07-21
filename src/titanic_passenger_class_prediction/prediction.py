"""Reusable inference utilities for Titanic passenger-class prediction.

This module contains the application-facing prediction logic shared by:

- ``scripts/predict.py`` for batch CSV inference;
- the Streamlit prediction page;
- notebooks or other Python clients.

The module accepts raw Titanic-compatible passenger records, applies the same
deterministic feature engineering used during training, and then calls the
saved fitted model pipeline.

Keeping this logic inside the package prevents user interfaces and command-line
scripts from duplicating feature-engineering rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from titanic_passenger_class_prediction.config import ID_COLUMN
from titanic_passenger_class_prediction.features import add_passenger_features
from titanic_passenger_class_prediction.modeling import MODEL_FEATURES
from titanic_passenger_class_prediction.preprocessing import (
    standardize_column_types,
)


PREDICTION_COLUMN = "PredictedPclass"
CONFIDENCE_COLUMN = "PredictionConfidence"
SECOND_HIGHEST_PROBABILITY_COLUMN = "SecondHighestProbability"
PROBABILITY_MARGIN_COLUMN = "ProbabilityMargin"
LOW_CONFIDENCE_COLUMN = "LowConfidencePrediction"

LOW_CONFIDENCE_THRESHOLD = 0.60


RAW_PREDICTION_COLUMNS = (
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
)


def validate_raw_prediction_data(
    dataframe: pd.DataFrame,
) -> None:
    """Validate raw Titanic-compatible data before feature engineering.

    Parameters
    ----------
    dataframe
        Raw passenger records.

    Raises
    ------
    TypeError
        If the input is not a pandas dataframe.
    ValueError
        If the dataframe is empty, required columns are missing, or passenger
        identifiers are missing or duplicated.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Prediction input must be provided as a pandas DataFrame."
        )

    if dataframe.empty:
        raise ValueError(
            "Prediction dataframe must contain at least one row."
        )

    missing_columns = sorted(
        set(RAW_PREDICTION_COLUMNS)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Raw prediction data is missing required columns: "
            f"{missing_columns}"
        )

    if dataframe[ID_COLUMN].isna().any():
        raise ValueError(
            f"Identifier column '{ID_COLUMN}' contains missing values."
        )

    if dataframe[ID_COLUMN].duplicated().any():
        duplicated_ids = (
            dataframe.loc[
                dataframe[ID_COLUMN].duplicated(keep=False),
                ID_COLUMN,
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Identifier column '{ID_COLUMN}' contains duplicate values: "
            f"{duplicated_ids[:10]}"
        )


def prepare_prediction_data(
    raw_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare labelled or unlabelled passenger data for inference.

    The function applies the same deterministic transformations used during
    model training:

    1. validate the original passenger columns;
    2. standardize available column types;
    3. construct engineered passenger features;
    4. verify that all model inputs are present.

    Target columns such as ``Pclass`` are preserved when included, but are not
    required.

    Parameters
    ----------
    raw_dataframe
        Raw Titanic-compatible passenger records.

    Returns
    -------
    pandas.DataFrame
        Feature-engineered passenger records suitable for the saved model.
    """
    validate_raw_prediction_data(
        raw_dataframe
    )

    prepared_dataframe = standardize_column_types(
        raw_dataframe.copy()
    )

    prepared_dataframe = add_passenger_features(
        prepared_dataframe
    )

    missing_model_features = sorted(
        set(MODEL_FEATURES)
        - set(prepared_dataframe.columns)
    )

    if missing_model_features:
        raise ValueError(
            "Prepared prediction data is missing model features: "
            f"{missing_model_features}"
        )

    if len(prepared_dataframe) != len(raw_dataframe):
        raise ValueError(
            "Prediction preparation unexpectedly changed the number "
            "of passenger rows."
        )

    if not prepared_dataframe[ID_COLUMN].is_unique:
        raise ValueError(
            f"Prepared identifier column '{ID_COLUMN}' must remain unique."
        )

    return prepared_dataframe


def get_model_classes(
    model: BaseEstimator,
) -> list[Any]:
    """Return fitted class labels from an estimator or sklearn pipeline."""
    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is None and hasattr(
        model,
        "named_steps",
    ):
        final_estimator = model.named_steps.get(
            "model"
        )

        classes = getattr(
            final_estimator,
            "classes_",
            None,
        )

    if classes is None:
        raise AttributeError(
            "The fitted model does not expose class labels through "
            "'classes_'."
        )

    return [
        value.item()
        if hasattr(value, "item")
        else value
        for value in classes
    ]


def validate_fitted_model(
    model: BaseEstimator,
) -> None:
    """Confirm that a model supports class and probability prediction."""
    if not hasattr(
        model,
        "predict",
    ):
        raise TypeError(
            "Loaded model artifact does not implement predict()."
        )

    if not hasattr(
        model,
        "predict_proba",
    ):
        raise TypeError(
            "Loaded model artifact does not implement predict_proba()."
        )

    get_model_classes(model)


def build_probability_column_name(
    class_label: Any,
) -> str:
    """Return a stable class-probability output column name."""
    label = (
        class_label.item()
        if hasattr(class_label, "item")
        else class_label
    )

    return f"ProbabilityClass{label}"


def generate_predictions(
    model: BaseEstimator,
    prepared_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predicted classes, probabilities, and confidence values.

    Parameters
    ----------
    model
        Fitted classifier or pipeline.
    prepared_dataframe
        Feature-engineered passenger records containing ``MODEL_FEATURES``.

    Returns
    -------
    pandas.DataFrame
        Prediction outputs keyed by passenger identifier.
    """
    validate_fitted_model(
        model
    )

    if prepared_dataframe.empty:
        raise ValueError(
            "Cannot generate predictions for an empty dataframe."
        )

    missing_features = sorted(
        set(MODEL_FEATURES)
        - set(prepared_dataframe.columns)
    )

    if missing_features:
        raise ValueError(
            "Prediction dataframe is missing model features: "
            f"{missing_features}"
        )

    features = prepared_dataframe.loc[
        :,
        MODEL_FEATURES,
    ].copy()

    predictions = np.asarray(
        model.predict(features)
    )

    probabilities = np.asarray(
        model.predict_proba(features)
    )

    classes = get_model_classes(
        model
    )

    if predictions.shape[0] != len(
        prepared_dataframe
    ):
        raise ValueError(
            "Model returned an unexpected number of predictions."
        )

    expected_shape = (
        len(prepared_dataframe),
        len(classes),
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "predict_proba() must return a two-dimensional array."
        )

    if probabilities.shape != expected_shape:
        raise ValueError(
            "Probability output dimensions do not match the input rows "
            "and model classes."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "Model produced non-finite probability values."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Predicted class probabilities do not sum to one."
        )

    output = pd.DataFrame(
        {
            ID_COLUMN: (
                prepared_dataframe[ID_COLUMN]
                .reset_index(drop=True)
            ),
            PREDICTION_COLUMN: predictions,
        }
    )

    for class_index, class_label in enumerate(
        classes
    ):
        output[
            build_probability_column_name(
                class_label
            )
        ] = probabilities[:, class_index]

    output[CONFIDENCE_COLUMN] = probabilities.max(
        axis=1
    )

    sorted_probabilities = np.sort(
        probabilities,
        axis=1,
    )

    if probabilities.shape[1] >= 2:
        output[
            SECOND_HIGHEST_PROBABILITY_COLUMN
        ] = sorted_probabilities[:, -2]
    else:
        output[
            SECOND_HIGHEST_PROBABILITY_COLUMN
        ] = 0.0

    output[PROBABILITY_MARGIN_COLUMN] = (
        output[CONFIDENCE_COLUMN]
        - output[
            SECOND_HIGHEST_PROBABILITY_COLUMN
        ]
    )

    output[LOW_CONFIDENCE_COLUMN] = (
        output[CONFIDENCE_COLUMN]
        < LOW_CONFIDENCE_THRESHOLD
    )

    return output


def add_context_columns(
    predictions: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Attach selected passenger context to prediction outputs."""
    output = predictions.copy()

    context_columns = [
        column
        for column in [
            "Name",
            "Sex",
            "Age",
            "Fare",
            "Embarked",
        ]
        if column in prepared_dataframe.columns
    ]

    context = prepared_dataframe.loc[
        :,
        [
            ID_COLUMN,
            *context_columns,
        ],
    ].copy()

    output = output.merge(
        context,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    preferred_columns = [
        ID_COLUMN,
        *context_columns,
        PREDICTION_COLUMN,
    ]

    remaining_columns = [
        column
        for column in output.columns
        if column not in preferred_columns
    ]

    return output.loc[
        :,
        preferred_columns + remaining_columns,
    ]


def predict_passengers(
    raw_dataframe: pd.DataFrame,
    model: BaseEstimator,
) -> pd.DataFrame:
    """Predict passenger classes directly from raw Titanic records.

    This is the main reusable inference function for batch or interactive
    clients.
    """
    prepared_dataframe = prepare_prediction_data(
        raw_dataframe
    )

    predictions = generate_predictions(
        model=model,
        prepared_dataframe=prepared_dataframe,
    )

    return add_context_columns(
        predictions=predictions,
        prepared_dataframe=prepared_dataframe,
    )


def predict_one_passenger(
    passenger: dict[str, Any],
    model: BaseEstimator,
) -> dict[str, Any]:
    """Predict one raw passenger record.

    Parameters
    ----------
    passenger
        One raw passenger represented as a dictionary.
    model
        Fitted classifier or pipeline.

    Returns
    -------
    dict[str, Any]
        JSON-compatible prediction result.
    """
    if not isinstance(
        passenger,
        dict,
    ):
        raise TypeError(
            "Passenger input must be provided as a dictionary."
        )

    if not passenger:
        raise ValueError(
            "Passenger input dictionary cannot be empty."
        )

    predictions = predict_passengers(
        raw_dataframe=pd.DataFrame(
            [passenger]
        ),
        model=model,
    )

    if len(predictions) != 1:
        raise ValueError(
            "Single-passenger prediction must return exactly one row."
        )

    raw_result = predictions.iloc[
        0
    ].to_dict()

    result: dict[str, Any] = {}

    for key, value in raw_result.items():
        if hasattr(
            value,
            "item",
        ):
            value = value.item()

        if pd.isna(value):
            value = None

        result[key] = value

    return result
