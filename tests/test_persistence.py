import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from titanic_passenger_class_prediction.persistence import (
    build_model_metadata,
    ensure_artifact_directories,
    load_model,
    save_json_report,
    save_model,
    save_model_comparison,
    save_model_metadata,
    save_test_metrics,
)


def test_ensure_artifact_directories_creates_directories(
    tmp_path: Path,
) -> None:
    models_directory = tmp_path / "models"
    metrics_directory = tmp_path / "metrics"

    ensure_artifact_directories(
        models_directory=models_directory,
        metrics_directory=metrics_directory,
    )

    assert models_directory.exists()
    assert models_directory.is_dir()

    assert metrics_directory.exists()
    assert metrics_directory.is_dir()


def test_save_and_load_model(
    tmp_path: Path,
) -> None:
    features = pd.DataFrame(
        {
            "value": [
                1,
                2,
                3,
                4,
            ],
        }
    )

    target = pd.Series(
        [
            1,
            1,
            2,
            2,
        ]
    )

    estimator = DummyClassifier(
        strategy="most_frequent",
    )
    estimator.fit(
        features,
        target,
    )

    output_path = tmp_path / "model.joblib"

    saved_path = save_model(
        estimator=estimator,
        output_path=output_path,
    )

    loaded_model = load_model(
        input_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()

    predictions = loaded_model.predict(features)

    assert len(predictions) == len(features)


def test_load_model_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(
        FileNotFoundError,
        match="Model artifact not found",
    ):
        load_model(
            input_path=missing_path,
        )


def test_save_model_comparison(
    tmp_path: Path,
) -> None:
    comparison = pd.DataFrame(
        {
            "model": [
                "random_forest",
                "logistic_regression",
            ],
            "cv_f1_macro_mean": [
                0.72,
                0.68,
            ],
        }
    )

    output_path = tmp_path / "comparison.csv"

    saved_path = save_model_comparison(
        comparison=comparison,
        output_path=output_path,
    )

    loaded = pd.read_csv(output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert loaded.equals(comparison)


def test_save_model_comparison_rejects_empty_table(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="empty model-comparison table",
    ):
        save_model_comparison(
            comparison=pd.DataFrame(),
            output_path=tmp_path / "comparison.csv",
        )


def test_save_json_report(
    tmp_path: Path,
) -> None:
    report = {
        "accuracy": 0.75,
        "model": "random_forest",
    }

    output_path = tmp_path / "report.json"

    saved_path = save_json_report(
        report=report,
        output_path=output_path,
    )

    with output_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    assert saved_path == output_path
    assert loaded == report


def test_save_json_report_supports_numpy_scalars(
    tmp_path: Path,
) -> None:
    report = {
        "accuracy": np.float64(0.75),
        "rows": np.int64(100),
    }

    output_path = tmp_path / "report.json"

    save_json_report(
        report=report,
        output_path=output_path,
    )

    with output_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    assert loaded["accuracy"] == 0.75
    assert loaded["rows"] == 100


def test_save_json_report_rejects_empty_mapping(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="empty JSON report",
    ):
        save_json_report(
            report={},
            output_path=tmp_path / "report.json",
        )


def test_save_test_metrics(
    tmp_path: Path,
) -> None:
    metrics = {
        "accuracy": 0.78,
        "balanced_accuracy": 0.76,
        "f1_macro": 0.75,
    }

    output_path = tmp_path / "test_metrics.json"

    saved_path = save_test_metrics(
        metrics=metrics,
        output_path=output_path,
    )

    with output_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    assert saved_path == output_path
    assert loaded == metrics


def test_build_model_metadata() -> None:
    metadata = build_model_metadata(
        model_name="random_forest",
        target_column="Pclass",
        feature_columns=[
            "Age",
            "Fare",
            "Sex",
        ],
        training_rows=700,
        test_rows=191,
        selected_metric="cv_f1_macro_mean",
        selected_metric_value=0.71,
        additional_metadata={
            "random_state": 42,
        },
    )

    assert metadata["model_name"] == "random_forest"
    assert metadata["target_column"] == "Pclass"

    assert metadata["feature_columns"] == [
        "Age",
        "Fare",
        "Sex",
    ]

    assert metadata["training_rows"] == 700
    assert metadata["test_rows"] == 191

    assert metadata["selected_metric"] == (
        "cv_f1_macro_mean"
    )

    assert metadata["selected_metric_value"] == 0.71
    assert metadata["random_state"] == 42
    assert "created_at_utc" in metadata


@pytest.mark.parametrize(
    (
        "model_name",
        "target_column",
        "feature_columns",
        "training_rows",
        "test_rows",
        "selected_metric",
        "expected_message",
    ),
    [
        (
            "",
            "Pclass",
            ["Age"],
            10,
            5,
            "cv_f1_macro_mean",
            "Model name",
        ),
        (
            "random_forest",
            "",
            ["Age"],
            10,
            5,
            "cv_f1_macro_mean",
            "Target column",
        ),
        (
            "random_forest",
            "Pclass",
            [],
            10,
            5,
            "cv_f1_macro_mean",
            "feature column",
        ),
        (
            "random_forest",
            "Pclass",
            ["Age"],
            0,
            5,
            "cv_f1_macro_mean",
            "Training row count",
        ),
        (
            "random_forest",
            "Pclass",
            ["Age"],
            10,
            0,
            "cv_f1_macro_mean",
            "Test row count",
        ),
        (
            "random_forest",
            "Pclass",
            ["Age"],
            10,
            5,
            "",
            "Selected metric",
        ),
    ],
)
def test_build_model_metadata_rejects_invalid_values(
    model_name: str,
    target_column: str,
    feature_columns: list[str],
    training_rows: int,
    test_rows: int,
    selected_metric: str,
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        build_model_metadata(
            model_name=model_name,
            target_column=target_column,
            feature_columns=feature_columns,
            training_rows=training_rows,
            test_rows=test_rows,
            selected_metric=selected_metric,
            selected_metric_value=0.70,
        )


def test_save_model_metadata(
    tmp_path: Path,
) -> None:
    metadata = {
        "model_name": "logistic_regression",
        "target_column": "Pclass",
        "training_rows": 700,
    }

    output_path = tmp_path / "metadata.json"

    saved_path = save_model_metadata(
        metadata=metadata,
        output_path=output_path,
    )

    with output_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    assert saved_path == output_path
    assert loaded == metadata