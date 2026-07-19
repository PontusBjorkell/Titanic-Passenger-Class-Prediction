"""Validation utilities for the Titanic passenger datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd


TRAIN_REQUIRED_COLUMNS = {
    "PassengerId",
    "Survived",
    "Pclass",
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

EXPECTED_PCLASS_VALUES = {1, 2, 3}
EXPECTED_SURVIVED_VALUES = {0, 1}
EXPECTED_SEX_VALUES = {"male", "female"}
EXPECTED_EMBARKED_VALUES = {"C", "Q", "S"}


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str] | None = None,
) -> None:
    """
    Confirm that all required columns are present.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    required = required_columns or TRAIN_REQUIRED_COLUMNS
    missing_columns = required.difference(dataframe.columns)

    if missing_columns:
        formatted = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {formatted}")


def find_unexpected_values(
    series: pd.Series,
    expected_values: set[Any],
) -> list[Any]:
    """
    Return non-null values that are not part of an expected set.
    """
    observed_values = set(series.dropna().unique())
    unexpected_values = observed_values.difference(expected_values)

    return sorted(unexpected_values, key=str)


def data_quality_report(dataframe: pd.DataFrame) -> dict[str, Any]:
    """
    Produce a structured data-quality report.

    The function does not modify the supplied dataframe.

    Returns
    -------
    dict[str, Any]
        Dataset dimensions, missing values, duplicates, data types,
        invalid ranges, and unexpected categorical values.
    """
    validate_required_columns(dataframe)

    missing_counts = dataframe.isna().sum()
    missing_percentages = dataframe.isna().mean().mul(100)

    report: dict[str, Any] = {
        "shape": {
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
        },
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "duplicate_passenger_ids": int(
            dataframe["PassengerId"].duplicated().sum()
        ),
        "missing_values": {
            column: {
                "count": int(missing_counts[column]),
                "percentage": round(
                    float(missing_percentages[column]),
                    2,
                ),
            }
            for column in dataframe.columns
            if missing_counts[column] > 0
        },
        "data_types": {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "numeric_columns": dataframe.select_dtypes(
            include="number"
        ).columns.tolist(),
        "categorical_columns": dataframe.select_dtypes(
            exclude="number"
        ).columns.tolist(),
        "invalid_values": {
            "negative_age_rows": int(dataframe["Age"].lt(0).sum()),
            "age_above_100_rows": int(dataframe["Age"].gt(100).sum()),
            "negative_fare_rows": int(dataframe["Fare"].lt(0).sum()),
            "negative_sibsp_rows": int(dataframe["SibSp"].lt(0).sum()),
            "negative_parch_rows": int(dataframe["Parch"].lt(0).sum()),
        },
        "unexpected_categories": {
            "Pclass": find_unexpected_values(
                dataframe["Pclass"],
                EXPECTED_PCLASS_VALUES,
            ),
            "Survived": find_unexpected_values(
                dataframe["Survived"],
                EXPECTED_SURVIVED_VALUES,
            ),
            "Sex": find_unexpected_values(
                dataframe["Sex"],
                EXPECTED_SEX_VALUES,
            ),
            "Embarked": find_unexpected_values(
                dataframe["Embarked"],
                EXPECTED_EMBARKED_VALUES,
            ),
        },
    }

    return report