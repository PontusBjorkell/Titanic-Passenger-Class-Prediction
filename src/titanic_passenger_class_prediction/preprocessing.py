"""Data-preparation pipeline for Titanic passenger data."""

from __future__ import annotations

import pandas as pd

from titanic_passenger_class_prediction.features import (
    add_passenger_features,
)
from titanic_passenger_class_prediction.validation import (
    data_quality_report,
    validate_required_columns,
)


def standardize_column_types(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply consistent data types without imputing missing values.

    Integer-like columns use the standard NumPy ``int64`` dtype when
    all values are present. If missing values occur, pandas' nullable
    ``Int64`` dtype is used instead.

    Continuous numeric columns use ``float64`` and string-like columns
    use ``object``. The original dataframe is not modified.
    """
    standardized = dataframe.copy()

    integer_columns = [
        "PassengerId",
        "Survived",
        "Pclass",
        "SibSp",
        "Parch",
    ]

    string_columns = [
        "Name",
        "Sex",
        "Ticket",
        "Cabin",
        "Embarked",
    ]

    for column in integer_columns:
        if column not in standardized.columns:
            continue

        numeric_values = pd.to_numeric(
            standardized[column],
            errors="raise",
        )

        # Ordinary int64 preserves the project's original behavior.
        # Nullable Int64 is used only when missing values are present.
        if numeric_values.isna().any():
            standardized[column] = numeric_values.astype(
                "Int64"
            )
        else:
            standardized[column] = numeric_values.astype(
                "int64"
            )

    for column in [
        "Age",
        "Fare",
    ]:
        if column in standardized.columns:
            standardized[column] = pd.to_numeric(
                standardized[column],
                errors="coerce",
            ).astype("float64")

    for column in string_columns:
        if column in standardized.columns:
            standardized[column] = standardized[
                column
            ].astype("object")

    return standardized


def prepare_passenger_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate, standardize, and enrich raw Titanic passenger data.

    This function performs deterministic preparation only. It does not
    impute missing values, encode categories, scale columns, or fit any
    learned transformations.

    Parameters
    ----------
    dataframe
        Raw Titanic training dataframe.

    Returns
    -------
    pandas.DataFrame
        Validated and feature-enriched passenger dataset.
    """
    validate_required_columns(dataframe)

    prepared = standardize_column_types(dataframe)
    prepared = add_passenger_features(prepared)

    return prepared


def build_preparation_summary(
    raw_dataframe: pd.DataFrame,
    processed_dataframe: pd.DataFrame,
) -> dict[str, object]:
    """Build metadata describing a data-preparation run."""
    return {
        "raw_quality": data_quality_report(raw_dataframe),
        "raw_shape": {
            "rows": int(raw_dataframe.shape[0]),
            "columns": int(raw_dataframe.shape[1]),
        },
        "processed_shape": {
            "rows": int(processed_dataframe.shape[0]),
            "columns": int(processed_dataframe.shape[1]),
        },
        "added_columns": sorted(
            set(processed_dataframe.columns)
            - set(raw_dataframe.columns)
        ),
    }