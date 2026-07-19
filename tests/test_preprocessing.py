import pandas as pd

from titanic_passenger_class_prediction.preprocessing import (
    build_preparation_summary,
    prepare_passenger_data,
    standardize_column_types,
)


def make_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Survived": [0, 1],
            "Pclass": [3, 1],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
            ],
            "Sex": ["male", "female"],
            "Age": [22.0, 38.0],
            "SibSp": [1, 0],
            "Parch": [0, 0],
            "Ticket": ["A/5 21171", "113803"],
            "Fare": [7.25, 53.10],
            "Cabin": [None, "C123"],
            "Embarked": ["S", "C"],
        }
    )


def test_standardize_column_types() -> None:
    dataframe = make_sample_dataframe()

    standardized = standardize_column_types(dataframe)

    assert str(standardized["PassengerId"].dtype) == "int64"
    assert str(standardized["Age"].dtype) == "float64"
    assert str(standardized["Name"].dtype) == "object"


def test_prepare_passenger_data_adds_features() -> None:
    raw = make_sample_dataframe()

    processed = prepare_passenger_data(raw)

    assert len(processed) == len(raw)
    assert "Title" in processed.columns
    assert "FamilySize" in processed.columns
    assert "CabinDeck" in processed.columns
    assert processed["Title"].tolist() == ["Mr", "Mrs"]


def test_prepare_passenger_data_does_not_modify_source() -> None:
    raw = make_sample_dataframe()

    prepare_passenger_data(raw)

    assert "Title" not in raw.columns
    assert "FamilySize" not in raw.columns


def test_build_preparation_summary() -> None:
    raw = make_sample_dataframe()
    processed = prepare_passenger_data(raw)

    summary = build_preparation_summary(raw, processed)

    assert summary["raw_shape"] == {
        "rows": 2,
        "columns": 12,
    }
    assert summary["processed_shape"]["rows"] == 2
    assert "Title" in summary["added_columns"]