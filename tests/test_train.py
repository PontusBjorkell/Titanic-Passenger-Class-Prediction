import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier

from scripts.train import (
    split_training_data,
    train_selected_model,
    validate_training_dataframe,
)
from titanic_passenger_class_prediction.modeling import (
    MODEL_FEATURES,
)


def make_training_dataframe(
    rows_per_class: int = 10,
) -> pd.DataFrame:
    """
    Create a compact prepared dataset with all modeling columns.
    """
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
    dataframe["FamilySizeGroup"] = [
        "Alone"
    ] * total_rows
    dataframe["CabinDeck"] = [
        "Unknown"
    ] * total_rows
    dataframe["TicketPrefix"] = [
        "NONE"
    ] * total_rows

    dataframe["Pclass"] = (
        ([1] * rows_per_class)
        + ([2] * rows_per_class)
        + ([3] * rows_per_class)
    )

    return dataframe


def test_validate_training_dataframe_accepts_valid_data() -> None:
    dataframe = make_training_dataframe()

    validate_training_dataframe(dataframe)


def test_validate_training_dataframe_rejects_empty_data() -> None:
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        validate_training_dataframe(
            pd.DataFrame()
        )


def test_validate_training_dataframe_rejects_missing_columns() -> None:
    dataframe = make_training_dataframe()
    dataframe = dataframe.drop(
        columns=[
            "FareLog",
        ]
    )

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


def test_split_training_data_preserves_all_rows() -> None:
    dataframe = make_training_dataframe()

    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = split_training_data(dataframe)

    assert (
        len(features_train)
        + len(features_test)
        == len(dataframe)
    )

    assert (
        len(target_train)
        + len(target_test)
        == len(dataframe)
    )

    assert list(features_train.columns) == MODEL_FEATURES
    assert list(features_test.columns) == MODEL_FEATURES


def test_split_training_data_is_stratified() -> None:
    dataframe = make_training_dataframe(
        rows_per_class=20,
    )

    (
        _,
        _,
        target_train,
        target_test,
    ) = split_training_data(dataframe)

    assert set(target_train.unique()) == {
        1,
        2,
        3,
    }

    assert set(target_test.unique()) == {
        1,
        2,
        3,
    }


def test_train_selected_model_fits_clone() -> None:
    dataframe = make_training_dataframe()

    features = dataframe.loc[
        :,
        MODEL_FEATURES,
    ]
    target = dataframe["Pclass"]

    original = DummyClassifier(
        strategy="most_frequent",
    )

    models: dict[str, BaseEstimator] = {
        "dummy": original,
    }

    fitted = train_selected_model(
        model_name="dummy",
        models=models,
        features=features,
        target=target,
    )

    predictions = fitted.predict(features)

    assert fitted is not original
    assert len(predictions) == len(features)


def test_train_selected_model_rejects_unknown_name() -> None:
    dataframe = make_training_dataframe()

    with pytest.raises(
        ValueError,
        match="Unknown model name",
    ):
        train_selected_model(
            model_name="missing",
            models={
                "dummy": DummyClassifier(),
            },
            features=dataframe[MODEL_FEATURES],
            target=dataframe["Pclass"],
        )