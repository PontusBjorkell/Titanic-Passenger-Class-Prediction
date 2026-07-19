"""Model definitions and preprocessing pipelines."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from titanic_passenger_class_prediction.config import RANDOM_STATE


NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Title",
    "FamilySizeGroup",
    "CabinDeck",
    "TicketPrefix",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Create the shared numeric and categorical preprocessing pipeline.

    Numeric values are median-imputed and standardized. Categorical
    values are mode-imputed and one-hot encoded.
    """
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_logistic_regression_pipeline() -> Pipeline:
    """Create an interpretable multinomial Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2_000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Create a Random Forest classification pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_candidate_models() -> dict[str, Pipeline]:
    """Return the initial candidate model registry."""
    return {
        "logistic_regression": (
            build_logistic_regression_pipeline()
        ),
        "random_forest": (
            build_random_forest_pipeline()
        ),
    }


def get_model_parameters() -> dict[str, dict[str, list[Any]]]:
    """
    Return compact hyperparameter search spaces.

    These search spaces are intentionally small because the Titanic
    dataset is limited in size.
    """
    return {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
        },
        "random_forest": {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 6, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.5],
        },
    }

def build_dummy_pipeline() -> Pipeline:
    """Create a majority-class baseline pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                DummyClassifier(
                    strategy="most_frequent",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

def build_candidate_models() -> dict[str, Pipeline]:
    """Return the initial candidate model registry."""
    return {
        "dummy": build_dummy_pipeline(),
        "logistic_regression": (
            build_logistic_regression_pipeline()
        ),
        "random_forest": (
            build_random_forest_pipeline()
        ),
    }