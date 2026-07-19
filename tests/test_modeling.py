import pandas as pd
from sklearn.pipeline import Pipeline

from titanic_passenger_class_prediction.data import load_processed_data

from titanic_passenger_class_prediction.modeling import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_candidate_models,
    build_preprocessor,
)


def make_model_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [22.0, 38.0, None, 35.0],
            "SibSp": [1, 1, 0, 1],
            "Parch": [0, 0, 0, 0],
            "Fare": [7.25, 71.28, 7.93, 53.10],
            "FamilySize": [2, 2, 1, 2],
            "IsAlone": [0, 0, 1, 0],
            "HasCabin": [0, 1, 0, 1],
            "CabinCount": [0, 1, 0, 1],
            "FareLog": [2.11, 4.28, 2.19, 3.99],
            "Sex": ["male", "female", "female", "female"],
            "Embarked": ["S", "C", "S", "S"],
            "Title": ["Mr", "Mrs", "Miss", "Mrs"],
            "FamilySizeGroup": [
                "Small",
                "Small",
                "Alone",
                "Small",
            ],
            "CabinDeck": [
                "Unknown",
                "C",
                "Unknown",
                "C",
            ],
            "TicketPrefix": [
                "A_5",
                "PC",
                "STON_O2",
                "NONE",
            ],
        }
    )


def test_feature_lists_do_not_overlap() -> None:
    assert not set(NUMERIC_FEATURES).intersection(
        CATEGORICAL_FEATURES
    )
    assert MODEL_FEATURES == (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )


def test_preprocessor_transforms_dataframe() -> None:
    dataframe = make_model_dataframe()

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(dataframe)

    assert transformed.shape[0] == len(dataframe)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
    assert not pd.isna(transformed).any()


def test_candidate_models_are_pipelines() -> None:
    models = build_candidate_models()

    assert set(models) == {
        "dummy",
        "logistic_regression",
        "random_forest",
    }

    for model in models.values():
        assert isinstance(model, Pipeline)


def test_candidate_models_can_fit_and_predict() -> None:
    dataframe = make_model_dataframe()
    target = pd.Series([3, 1, 3, 1])

    models = build_candidate_models()

    for model in models.values():
        model.fit(dataframe, target)
        predictions = model.predict(dataframe)

        assert len(predictions) == len(dataframe)



def test_preprocessor_transforms_processed_dataset() -> None:
    """
    Confirm that the modeling preprocessor accepts the actual
    processed dataset produced by the preparation pipeline.
    """
    dataframe = load_processed_data()
    features = dataframe[MODEL_FEATURES]

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(dataframe)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
    assert not pd.isna(transformed).any()