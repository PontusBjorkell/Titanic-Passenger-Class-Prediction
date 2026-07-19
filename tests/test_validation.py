import pandas as pd
import pytest

from titanic_passenger_class_prediction.validation import (
    data_quality_report,
    validate_required_columns,
)


def make_valid_dataframe() -> pd.DataFrame:
    """Create a minimal valid Titanic-style dataframe."""
    return pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Survived": [0, 1],
            "Pclass": [3, 1],
            "Name": ["Passenger One", "Passenger Two"],
            "Sex": ["male", "female"],
            "Age": [22.0, 38.0],
            "SibSp": [1, 1],
            "Parch": [0, 0],
            "Ticket": ["A/5 21171", "PC 17599"],
            "Fare": [7.25, 71.2833],
            "Cabin": [None, "C85"],
            "Embarked": ["S", "C"],
        }
    )


def test_validate_required_columns_accepts_valid_data() -> None:
    dataframe = make_valid_dataframe()

    validate_required_columns(dataframe)


def test_validate_required_columns_rejects_missing_column() -> None:
    dataframe = make_valid_dataframe().drop(columns="Fare")

    with pytest.raises(ValueError, match="Fare"):
        validate_required_columns(dataframe)


def test_data_quality_report_returns_expected_shape() -> None:
    dataframe = make_valid_dataframe()

    report = data_quality_report(dataframe)

    assert report["shape"] == {"rows": 2, "columns": 12}
    assert report["duplicate_rows"] == 0
    assert report["invalid_values"]["negative_age_rows"] == 0
    assert report["unexpected_categories"]["Pclass"] == []