"""Model definitions and preprocessing pipelines."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from titanic_passenger_class_prediction.config import (
    N_JOBS,
    RANDOM_STATE,
)


# ---------------------------------------------------------------------------
# Modeling features
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Create the shared numeric and categorical preprocessing pipeline.

    Numeric features are median-imputed and standardized.

    Categorical features are mode-imputed and one-hot encoded. Unknown
    categories are ignored so the fitted pipeline can safely process
    categories that were not present during training.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted preprocessing transformer.
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


# ---------------------------------------------------------------------------
# Model pipelines
# ---------------------------------------------------------------------------

def build_dummy_pipeline() -> Pipeline:
    """
    Create a majority-class baseline pipeline.

    The dummy model establishes the minimum performance that trained
    models should exceed.
    """
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


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Create an interpretable Logistic Regression pipeline.

    Logistic Regression provides a strong linear baseline and is useful
    for understanding whether the engineered features separate passenger
    classes without requiring a highly complex model.
    """
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
    """
    Create a Random Forest classification pipeline.

    Random Forest can model nonlinear relationships and interactions
    between passenger characteristics without requiring manual feature
    interaction terms.
    """
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
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def build_candidate_models() -> dict[str, Pipeline]:
    """
    Return the initial candidate model registry.

    Returns
    -------
    dict[str, sklearn.pipeline.Pipeline]
        Mapping between stable model names and unfitted pipelines.
    """
    return {
        "dummy": build_dummy_pipeline(),
        "logistic_regression": (
            build_logistic_regression_pipeline()
        ),
        "random_forest": (
            build_random_forest_pipeline()
        ),
    }


# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------

def get_model_parameters() -> dict[str, dict[str, list[Any]]]:
    """
    Return compact hyperparameter search spaces.

    Parameter names use the ``model__`` prefix because the estimator is
    stored under the ``model`` step inside each scikit-learn pipeline.

    The search spaces are intentionally limited because the Titanic
    dataset is relatively small. A compact grid reduces unnecessary
    computation and lowers the risk of tuning excessively to noisy
    cross-validation results.

    Returns
    -------
    dict[str, dict[str, list[Any]]]
        Hyperparameter grids keyed by model name.
    """
    return {
        "logistic_regression": {
            "model__C": [
                0.01,
                0.1,
                1.0,
                10.0,
            ],
        },
        "random_forest": {
            "model__n_estimators": [
                300,
                500,
            ],
            "model__max_depth": [
                None,
                6,
                10,
            ],
            "model__min_samples_leaf": [
                1,
                2,
                4,
            ],
            "model__max_features": [
                "sqrt",
                0.5,
            ],
        },
    }