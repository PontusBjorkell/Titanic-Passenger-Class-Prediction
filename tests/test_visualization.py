"""Tests for model-evaluation visualization utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from titanic_passenger_class_prediction.visualization import (
    calculate_permutation_importance,
    ensure_figure_directory,
    extract_model_feature_importance,
    get_transformed_feature_names,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_importance,
)


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return a small deterministic multiclass dataset."""
    X = pd.DataFrame(
        {
            "Age": [
                22,
                38,
                26,
                35,
                35,
                54,
                2,
                27,
                14,
                4,
                58,
                20,
                39,
                30,
                45,
            ],
            "Fare": [
                7.25,
                71.28,
                7.93,
                53.10,
                8.05,
                51.86,
                21.08,
                11.13,
                30.07,
                16.70,
                26.55,
                8.05,
                83.16,
                13.00,
                90.00,
            ],
            "Sex": [
                "male",
                "female",
                "female",
                "female",
                "male",
                "male",
                "male",
                "male",
                "female",
                "female",
                "female",
                "male",
                "female",
                "male",
                "female",
            ],
            "Embarked": [
                "S",
                "C",
                "S",
                "S",
                "S",
                "S",
                "S",
                "S",
                "C",
                "S",
                "S",
                "S",
                "C",
                "S",
                "C",
            ],
        }
    )

    y = pd.Series(
        [3, 1, 3, 1, 3, 1, 3, 2, 2, 3, 1, 3, 1, 2, 1],
        name="Pclass",
    )

    return X, y


@pytest.fixture
def fitted_random_forest_pipeline(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> Pipeline:
    """Return a fitted preprocessing and random-forest pipeline."""
    X, y = classification_data

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["Age", "Fare"],
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                ["Sex", "Embarked"],
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=20,
                    max_depth=4,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def fitted_logistic_pipeline(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> Pipeline:
    """Return a fitted preprocessing and logistic-regression pipeline."""
    X, y = classification_data

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["Age", "Fare"],
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                ["Sex", "Embarked"],
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    return pipeline


def test_ensure_figure_directory_creates_directory(tmp_path: Path) -> None:
    """Figure directory creation should be recursive."""
    output_directory = tmp_path / "reports" / "figures"

    returned_path = ensure_figure_directory(output_directory)

    assert returned_path == output_directory
    assert output_directory.exists()
    assert output_directory.is_dir()


def test_plot_confusion_matrix_saves_image(tmp_path: Path) -> None:
    """The confusion-matrix function should create a non-empty image."""
    output_path = tmp_path / "confusion_matrix.png"

    returned_path = plot_confusion_matrix(
        y_true=[1, 1, 2, 2, 3, 3],
        y_pred=[1, 2, 2, 2, 3, 1],
        output_path=output_path,
        labels=[1, 2, 3],
        display_labels=["First", "Second", "Third"],
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_confusion_matrix_rejects_mismatched_lengths(
    tmp_path: Path,
) -> None:
    """Ground truth and predictions must have matching lengths."""
    with pytest.raises(ValueError, match="same number of rows"):
        plot_confusion_matrix(
            y_true=[1, 2, 3],
            y_pred=[1, 2],
            output_path=tmp_path / "matrix.png",
        )


def test_plot_confusion_matrix_rejects_empty_target(
    tmp_path: Path,
) -> None:
    """An empty target should not produce a confusion matrix."""
    with pytest.raises(ValueError, match="cannot be empty"):
        plot_confusion_matrix(
            y_true=[],
            y_pred=[],
            output_path=tmp_path / "matrix.png",
        )


def test_plot_confusion_matrix_rejects_invalid_normalization(
    tmp_path: Path,
) -> None:
    """Only sklearn-supported normalization modes should be accepted."""
    with pytest.raises(ValueError, match="normalize must be one of"):
        plot_confusion_matrix(
            y_true=[1, 2, 3],
            y_pred=[1, 2, 3],
            output_path=tmp_path / "matrix.png",
            normalize="invalid",
        )


def test_plot_rejects_invalid_file_extension(tmp_path: Path) -> None:
    """Figures should use a supported image or document extension."""
    with pytest.raises(ValueError, match="extension"):
        plot_confusion_matrix(
            y_true=[1, 2, 3],
            y_pred=[1, 2, 3],
            output_path=tmp_path / "matrix.txt",
        )


def test_get_transformed_feature_names_returns_names(
    fitted_random_forest_pipeline: Pipeline,
) -> None:
    """A fitted pipeline should expose transformed feature names."""
    feature_names = get_transformed_feature_names(
        fitted_random_forest_pipeline
    )

    assert isinstance(feature_names, np.ndarray)
    assert len(feature_names) > 0
    assert any("Age" in name for name in feature_names)
    assert any("Sex" in name for name in feature_names)


def test_get_transformed_feature_names_requires_pipeline() -> None:
    """Non-pipeline estimators should be rejected."""
    with pytest.raises(TypeError, match="Pipeline"):
        get_transformed_feature_names(RandomForestClassifier())


def test_get_transformed_feature_names_requires_preprocessor() -> None:
    """The pipeline must use the expected preprocessing step name."""
    pipeline = Pipeline(
        steps=[
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    with pytest.raises(ValueError, match="preprocessor"):
        get_transformed_feature_names(pipeline)


def test_extract_random_forest_feature_importance(
    fitted_random_forest_pipeline: Pipeline,
) -> None:
    """Tree-based model importances should be extracted and sorted."""
    result = extract_model_feature_importance(
        fitted_random_forest_pipeline
    )

    assert list(result.columns) == ["feature", "importance"]
    assert not result.empty
    assert result["importance"].is_monotonic_decreasing
    assert (result["importance"] >= 0).all()
    assert result["importance"].sum() == pytest.approx(1.0)


def test_extract_logistic_feature_importance(
    fitted_logistic_pipeline: Pipeline,
) -> None:
    """Linear-model coefficient magnitudes should be supported."""
    result = extract_model_feature_importance(
        fitted_logistic_pipeline
    )

    assert list(result.columns) == ["feature", "importance"]
    assert not result.empty
    assert result["importance"].is_monotonic_decreasing
    assert (result["importance"] >= 0).all()


def test_extract_feature_importance_requires_model_step(
    fitted_random_forest_pipeline: Pipeline,
) -> None:
    """The expected model step must exist."""
    invalid_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                fitted_random_forest_pipeline.named_steps["preprocessor"],
            ),
            (
                "classifier",
                fitted_random_forest_pipeline.named_steps["model"],
            ),
        ]
    )

    with pytest.raises(ValueError, match="model"):
        extract_model_feature_importance(invalid_pipeline)


def test_plot_feature_importance_saves_image(
    fitted_random_forest_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    """Model-derived importance should be saved as an image."""
    output_path = tmp_path / "feature_importance.png"

    returned_path = plot_feature_importance(
        pipeline=fitted_random_forest_pipeline,
        output_path=output_path,
        top_n=5,
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_feature_importance_rejects_nonpositive_top_n(
    fitted_random_forest_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    """The number of displayed features must be positive."""
    with pytest.raises(ValueError, match="greater than zero"):
        plot_feature_importance(
            pipeline=fitted_random_forest_pipeline,
            output_path=tmp_path / "importance.png",
            top_n=0,
        )


def test_calculate_permutation_importance_returns_original_columns(
    fitted_random_forest_pipeline: Pipeline,
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Permutation importance should use the original input columns."""
    X, y = classification_data

    result = calculate_permutation_importance(
        pipeline=fitted_random_forest_pipeline,
        X=X,
        y=y,
        n_repeats=3,
        random_state=42,
    )

    assert list(result.columns) == [
        "feature",
        "importance_mean",
        "importance_std",
    ]
    assert set(result["feature"]) == set(X.columns)
    assert result["importance_mean"].is_monotonic_decreasing
    assert len(result) == X.shape[1]


def test_calculate_permutation_importance_rejects_empty_features(
    fitted_random_forest_pipeline: Pipeline,
) -> None:
    """An empty feature matrix should be rejected."""
    empty_X = pd.DataFrame(columns=["Age", "Fare", "Sex", "Embarked"])

    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_permutation_importance(
            pipeline=fitted_random_forest_pipeline,
            X=empty_X,
            y=[],
            n_repeats=2,
        )


def test_calculate_permutation_importance_rejects_mismatched_lengths(
    fitted_random_forest_pipeline: Pipeline,
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Features and targets must contain matching row counts."""
    X, y = classification_data

    with pytest.raises(ValueError, match="same number of rows"):
        calculate_permutation_importance(
            pipeline=fitted_random_forest_pipeline,
            X=X,
            y=y.iloc[:-1],
            n_repeats=2,
        )


def test_calculate_permutation_importance_rejects_invalid_repeats(
    fitted_random_forest_pipeline: Pipeline,
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Permutation repeat count must be positive."""
    X, y = classification_data

    with pytest.raises(ValueError, match="greater than zero"):
        calculate_permutation_importance(
            pipeline=fitted_random_forest_pipeline,
            X=X,
            y=y,
            n_repeats=0,
        )


def test_plot_permutation_importance_saves_image(
    fitted_random_forest_pipeline: Pipeline,
    classification_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Permutation importance should be saved as an image."""
    X, y = classification_data
    output_path = tmp_path / "permutation_importance.png"

    returned_path = plot_permutation_importance(
        pipeline=fitted_random_forest_pipeline,
        X=X,
        y=y,
        output_path=output_path,
        top_n=4,
        n_repeats=3,
        random_state=42,
    )

    assert returned_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_functions_close_created_figures(
    fitted_random_forest_pipeline: Pipeline,
    classification_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Plot helpers should not leave Matplotlib figures open."""
    X, y = classification_data
    open_figures_before = set(plt.get_fignums())

    plot_confusion_matrix(
        y_true=y,
        y_pred=fitted_random_forest_pipeline.predict(X),
        output_path=tmp_path / "matrix.png",
    )

    plot_feature_importance(
        pipeline=fitted_random_forest_pipeline,
        output_path=tmp_path / "model_importance.png",
        top_n=4,
    )

    plot_permutation_importance(
        pipeline=fitted_random_forest_pipeline,
        X=X,
        y=y,
        output_path=tmp_path / "permutation.png",
        top_n=4,
        n_repeats=2,
    )

    open_figures_after = set(plt.get_fignums())

    assert open_figures_after == open_figures_before