"""Tests for automated model-reporting utilities."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from titanic_passenger_class_prediction.reporting import (
    ReportArtifacts,
    classification_report_dataframe,
    generate_model_report,
    generate_visual_reports,
    save_classification_report,
    write_training_summary,
)


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return a compact deterministic three-class dataset."""
    features = pd.DataFrame(
        {
            "Age": [22, 38, 26, 35, 54, 2, 27, 14, 58, 20, 39, 30],
            "Fare": [7.25, 71.28, 7.93, 53.10, 51.86, 21.08,
                     11.13, 30.07, 26.55, 8.05, 83.16, 13.00],
            "Sex": [
                "male", "female", "female", "female", "male", "male",
                "male", "female", "female", "male", "female", "male",
            ],
        }
    )
    target = pd.Series(
        [3, 1, 3, 1, 1, 3, 2, 2, 1, 3, 1, 2],
        name="Pclass",
    )
    return features, target


@pytest.fixture
def fitted_pipeline(
    classification_data: tuple[pd.DataFrame, pd.Series],
) -> Pipeline:
    """Return a fitted preprocessing and random-forest pipeline."""
    features, target = classification_data
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
                ["Sex"],
            ),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=15,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(features, target)
    return pipeline


@pytest.fixture
def report_mapping() -> dict[str, object]:
    """Return a representative sklearn classification-report mapping."""
    return {
        "1": {
            "precision": 0.80,
            "recall": 1.00,
            "f1-score": 0.89,
            "support": 4.0,
        },
        "2": {
            "precision": 1.00,
            "recall": 0.67,
            "f1-score": 0.80,
            "support": 3.0,
        },
        "3": {
            "precision": 1.00,
            "recall": 0.80,
            "f1-score": 0.89,
            "support": 5.0,
        },
        "accuracy": 0.83,
        "macro avg": {
            "precision": 0.93,
            "recall": 0.82,
            "f1-score": 0.86,
            "support": 12.0,
        },
        "weighted avg": {
            "precision": 0.92,
            "recall": 0.83,
            "f1-score": 0.86,
            "support": 12.0,
        },
    }


def make_comparison() -> pd.DataFrame:
    """Return a representative sorted model-comparison table."""
    return pd.DataFrame(
        {
            "model": ["random_forest", "logistic_regression", "dummy"],
            "cv_accuracy_mean": [0.91, 0.87, 0.49],
            "cv_balanced_accuracy_mean": [0.90, 0.86, 0.33],
            "cv_f1_macro_mean": [0.90, 0.85, 0.22],
            "cv_f1_macro_std": [0.02, 0.03, 0.01],
        }
    )


def make_test_metrics(
    report_mapping: dict[str, object],
) -> dict[str, object]:
    """Return holdout metrics consumed by the reporting workflow."""
    return {
        "accuracy": 0.83,
        "balanced_accuracy": 0.82,
        "f1_macro": 0.86,
        "labels": [1, 2, 3],
        "classification_report": report_mapping,
        "test_rows": 12,
    }


def test_classification_report_dataframe_returns_tidy_table(
    report_mapping: dict[str, object],
) -> None:
    """Classification metrics should become one row per report label."""
    result = classification_report_dataframe(report_mapping)

    assert list(result.columns[:5]) == [
        "label",
        "precision",
        "recall",
        "f1-score",
        "support",
    ]
    assert result["label"].tolist() == [
        "1", "2", "3", "accuracy", "macro avg", "weighted avg"
    ]
    accuracy_row = result.loc[result["label"] == "accuracy"].iloc[0]
    assert accuracy_row["f1-score"] == pytest.approx(0.83)


def test_classification_report_dataframe_rejects_empty_mapping() -> None:
    """An empty classification report should be rejected."""
    with pytest.raises(ValueError, match="cannot be empty"):
        classification_report_dataframe({})


def test_save_classification_report_creates_csv(
    report_mapping: dict[str, object],
    tmp_path: Path,
) -> None:
    """The classification report should be persisted as readable CSV."""
    output_path = tmp_path / "nested" / "classification_report.csv"

    returned_path = save_classification_report(
        classification_report=report_mapping,
        output_path=output_path,
    )
    loaded = pd.read_csv(output_path)

    assert returned_path == output_path
    assert output_path.exists()
    assert "macro avg" in loaded["label"].tolist()


def test_write_training_summary_creates_markdown(
    report_mapping: dict[str, object],
    tmp_path: Path,
) -> None:
    """The Markdown report should contain selection and metric details."""
    output_path = tmp_path / "training_summary.md"

    returned_path = write_training_summary(
        model_name="random_forest",
        comparison=make_comparison(),
        test_metrics=make_test_metrics(report_mapping),
        training_rows=700,
        test_rows=191,
        output_path=output_path,
    )
    contents = output_path.read_text(encoding="utf-8")

    assert returned_path == output_path
    assert "# Model Training Summary" in contents
    assert "**random_forest**" in contents
    assert "| Macro F1 | 86.00% |" in contents
    assert "Selected-model Macro F1: 0.9000" in contents
    assert "logistic_regression" in contents
    assert "Training rows: 700" in contents


def test_write_training_summary_rejects_empty_comparison(
    report_mapping: dict[str, object],
    tmp_path: Path,
) -> None:
    """A training summary requires model-comparison results."""
    with pytest.raises(ValueError, match="comparison cannot be empty"):
        write_training_summary(
            model_name="random_forest",
            comparison=pd.DataFrame(),
            test_metrics=make_test_metrics(report_mapping),
            training_rows=10,
            test_rows=3,
            output_path=tmp_path / "summary.md",
        )


def test_write_training_summary_requires_metrics(tmp_path: Path) -> None:
    """Core holdout metrics must be present."""
    with pytest.raises(ValueError, match="missing required values"):
        write_training_summary(
            model_name="random_forest",
            comparison=make_comparison(),
            test_metrics={"accuracy": 0.8},
            training_rows=10,
            test_rows=3,
            output_path=tmp_path / "summary.md",
        )


def test_generate_visual_reports_creates_figures(
    fitted_pipeline: Pipeline,
    classification_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """All supported evaluation figures should be generated."""
    features, target = classification_data

    paths = generate_visual_reports(
        pipeline=fitted_pipeline,
        features=features,
        target=target,
        labels=[1, 2, 3],
        confusion_matrix_path=tmp_path / "confusion.png",
        feature_importance_path=tmp_path / "feature.png",
        permutation_importance_path=tmp_path / "permutation.png",
        top_n=3,
        permutation_repeats=2,
        n_jobs=1,
    )

    assert all(path is not None and path.exists() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths if path is not None)


def test_generate_visual_reports_allows_unsupported_model_importance(
    classification_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Permutation reporting should continue for models without importance."""
    features, target = classification_data
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", ["Age", "Fare"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["Sex"],
            ),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    pipeline.fit(features, target)

    (
    confusion_path,
    feature_path,
    aggregated_path,
    permutation_path,
) = generate_visual_reports(
    pipeline=pipeline,
    features=features,
    target=target,
    confusion_matrix_path=tmp_path / "confusion.png",
    feature_importance_path=tmp_path / "feature.png",
    permutation_importance_path=tmp_path / "permutation.png",
    top_n=3,
    permutation_repeats=2,
    n_jobs=1,
)

    assert confusion_path.exists()
    assert feature_path is None
    assert aggregated_path is None
    assert not (tmp_path / "feature.png").exists()
    assert not (
        tmp_path / "aggregated_feature_importance.png"
    ).exists()
    assert permutation_path.exists()


def test_generate_visual_reports_rejects_empty_features(
    fitted_pipeline: Pipeline,
    tmp_path: Path,
) -> None:
    """Reporting cannot operate on an empty holdout feature matrix."""
    with pytest.raises(ValueError, match="features cannot be empty"):
        generate_visual_reports(
            pipeline=fitted_pipeline,
            features=pd.DataFrame(),
            target=[],
            confusion_matrix_path=tmp_path / "confusion.png",
            feature_importance_path=tmp_path / "feature.png",
            permutation_importance_path=tmp_path / "permutation.png",
        )


def test_generate_model_report_creates_complete_artifact_set(
    fitted_pipeline: Pipeline,
    classification_data: tuple[pd.DataFrame, pd.Series],
    report_mapping: dict[str, object],
    tmp_path: Path,
) -> None:
    """The orchestration function should return all generated paths."""
    features, target = classification_data

    artifacts = generate_model_report(
        model_name="random_forest",
        pipeline=fitted_pipeline,
        comparison=make_comparison(),
        test_metrics=make_test_metrics(report_mapping),
        features_test=features,
        target_test=target,
        training_rows=700,
        labels=[1, 2, 3],
        classification_report_path=tmp_path / "classification.csv",
        training_summary_path=tmp_path / "summary.md",
        confusion_matrix_path=tmp_path / "confusion.png",
        feature_importance_path=tmp_path / "feature.png",
        permutation_importance_path=tmp_path / "permutation.png",
        top_n=3,
        permutation_repeats=2,
        n_jobs=1,
    )

    assert isinstance(artifacts, ReportArtifacts)
    assert artifacts.classification_report_path.exists()
    assert artifacts.training_summary_path.exists()
    assert artifacts.confusion_matrix_path.exists()
    assert artifacts.feature_importance_path is not None
    assert artifacts.feature_importance_path.exists()
    assert artifacts.permutation_importance_path.exists()


def test_generate_model_report_requires_classification_report(
    fitted_pipeline: Pipeline,
    classification_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """The orchestration layer requires evaluation's detailed report."""
    features, target = classification_data

    with pytest.raises(ValueError, match="classification_report mapping"):
        generate_model_report(
            model_name="random_forest",
            pipeline=fitted_pipeline,
            comparison=make_comparison(),
            test_metrics={
                "accuracy": 0.8,
                "balanced_accuracy": 0.8,
                "f1_macro": 0.8,
            },
            features_test=features,
            target_test=target,
            training_rows=10,
            classification_report_path=tmp_path / "classification.csv",
            training_summary_path=tmp_path / "summary.md",
            confusion_matrix_path=tmp_path / "confusion.png",
            feature_importance_path=tmp_path / "feature.png",
            permutation_importance_path=tmp_path / "permutation.png",
        )
