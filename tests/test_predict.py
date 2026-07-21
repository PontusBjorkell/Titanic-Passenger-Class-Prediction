"""Tests for the batch prediction command-line workflow.

These tests cover:

    scripts/predict.py

Reusable inference behavior is tested separately in ``test_prediction.py``.
This file focuses on CSV loading, external evaluation, metadata, artifacts,
and the complete batch workflow.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from titanic_passenger_class_prediction.config import (
    ID_COLUMN,
    TARGET_COLUMN,
)
from titanic_passenger_class_prediction.modeling import (
    MODEL_FEATURES,
    build_random_forest_pipeline,
)
from titanic_passenger_class_prediction.persistence import (
    save_model,
)
from titanic_passenger_class_prediction.prediction import (
    CONFIDENCE_COLUMN,
    PREDICTION_COLUMN,
    prepare_prediction_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "predict.py"


def load_predict_script() -> ModuleType:
    """Load scripts/predict.py directly from its filesystem path."""
    if not PREDICT_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"Prediction script not found: {PREDICT_SCRIPT_PATH}"
        )

    module_name = "predict_script_for_tests"

    specification = importlib.util.spec_from_file_location(
        module_name,
        PREDICT_SCRIPT_PATH,
    )

    if specification is None or specification.loader is None:
        raise ImportError(
            "Could not create an import specification for "
            f"{PREDICT_SCRIPT_PATH}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[module_name] = module

    try:
        specification.loader.exec_module(
            module
        )
    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    return module


predict_script = load_predict_script()


CORRECT_COLUMN = predict_script.CORRECT_COLUMN
EXTERNAL_CLASSIFICATION_REPORT_FILENAME = (
    predict_script.EXTERNAL_CLASSIFICATION_REPORT_FILENAME
)
EXTERNAL_METRICS_FILENAME = (
    predict_script.EXTERNAL_METRICS_FILENAME
)
PREDICTION_METADATA_FILENAME = (
    predict_script.PREDICTION_METADATA_FILENAME
)
PREDICTIONS_FILENAME = (
    predict_script.PREDICTIONS_FILENAME
)

PredictionArtifacts = (
    predict_script.PredictionArtifacts
)

build_prediction_metadata = (
    predict_script.build_prediction_metadata
)
evaluate_external_predictions = (
    predict_script.evaluate_external_predictions
)
load_input_data = (
    predict_script.load_input_data
)
run_prediction_workflow = (
    predict_script.run_prediction_workflow
)
save_json = (
    predict_script.save_json
)
save_prediction_artifacts = (
    predict_script.save_prediction_artifacts
)


def make_raw_dataframe(
    *,
    include_target: bool = True,
) -> pd.DataFrame:
    """Create representative raw Titanic passenger records."""
    dataframe = pd.DataFrame(
        {
            "PassengerId": [
                1001,
                1002,
                1003,
                1004,
                1005,
                1006,
                1007,
                1008,
                1009,
            ],
            "Survived": [
                1,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            "Name": [
                "Allen, Mrs. Anna",
                "Brown, Mr. William",
                "Clark, Miss. Emily",
                "Davis, Mr. Henry",
                "Evans, Dr. Sarah",
                "Foster, Master. James",
                "Green, Mrs. Mary",
                "Hill, Mr. Thomas",
                "Irwin, Miss. Alice",
            ],
            "Sex": [
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
            ],
            "Age": [
                35.0,
                41.0,
                22.0,
                28.0,
                52.0,
                8.0,
                30.0,
                None,
                19.0,
            ],
            "SibSp": [
                1,
                0,
                0,
                1,
                0,
                1,
                2,
                0,
                0,
            ],
            "Parch": [
                0,
                0,
                1,
                0,
                0,
                2,
                1,
                0,
                0,
            ],
            "Ticket": [
                "PC 17599",
                "113803",
                "SC/PARIS 2167",
                "A/5 21171",
                "PC 17757",
                "C.A. 33112",
                "347082",
                "STON/O2 3101282",
                "349909",
            ],
            "Fare": [
                90.0,
                75.0,
                30.0,
                8.0,
                120.0,
                26.0,
                18.0,
                7.5,
                10.5,
            ],
            "Cabin": [
                "C85",
                "B28",
                None,
                None,
                "B51 B53 B55",
                "F2",
                None,
                None,
                None,
            ],
            "Embarked": [
                "C",
                "S",
                "S",
                "S",
                "C",
                "S",
                "S",
                "Q",
                "S",
            ],
        }
    )

    if include_target:
        dataframe[TARGET_COLUMN] = [
            1,
            1,
            2,
            3,
            1,
            2,
            3,
            3,
            3,
        ]

    return dataframe


@pytest.fixture
def fitted_model() -> Pipeline:
    """Create a lightweight fitted production-style model."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe(
            include_target=True
        )
    )

    model = build_random_forest_pipeline()

    model.fit(
        prepared_dataframe.loc[
            :,
            MODEL_FEATURES,
        ],
        prepared_dataframe[TARGET_COLUMN],
    )

    return model


@pytest.fixture
def labelled_csv_path(
    tmp_path: Path,
) -> Path:
    """Save labelled external input as a temporary CSV."""
    input_path = tmp_path / "external_test.csv"

    make_raw_dataframe(
        include_target=True
    ).to_csv(
        input_path,
        index=False,
    )

    return input_path


def test_predict_script_loaded() -> None:
    """The script should expose the expected workflow."""
    assert predict_script is not None
    assert hasattr(
        predict_script,
        "run_prediction_workflow",
    )


def test_load_input_data_reads_csv(
    labelled_csv_path: Path,
) -> None:
    """A valid CSV should be loaded."""
    dataframe = load_input_data(
        labelled_csv_path
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )
    assert len(dataframe) == 9
    assert ID_COLUMN in dataframe.columns


def test_load_input_data_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A nonexistent input should fail."""
    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        load_input_data(
            tmp_path / "missing.csv"
        )


def test_load_input_data_rejects_empty_csv(
    tmp_path: Path,
) -> None:
    """A header-only CSV should fail."""
    input_path = tmp_path / "empty.csv"

    make_raw_dataframe().iloc[
        0:0
    ].to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        load_input_data(
            input_path
        )


def test_evaluate_external_predictions_returns_metrics() -> None:
    """External evaluation should calculate multiclass metrics."""
    true_target = pd.Series(
        [
            1,
            1,
            2,
            2,
            3,
            3,
        ]
    )

    predicted_target = [
        1,
        1,
        2,
        3,
        3,
        3,
    ]

    metrics, report = evaluate_external_predictions(
        true_target=true_target,
        predicted_target=predicted_target,
        labels=[
            1,
            2,
            3,
        ],
    )

    assert metrics["rows"] == 6
    assert metrics["accuracy"] == pytest.approx(
        5 / 6
    )
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert 0 <= metrics["f1_macro"] <= 1

    assert np.asarray(
        metrics["confusion_matrix"]
    ).shape == (
        3,
        3,
    )

    assert isinstance(
        report,
        pd.DataFrame,
    )
    assert not report.empty
    assert "label" in report.columns


def test_evaluate_external_predictions_rejects_missing_target() -> None:
    """Ground truth cannot contain missing labels."""
    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        evaluate_external_predictions(
            true_target=pd.Series(
                [
                    1,
                    2,
                    None,
                ]
            ),
            predicted_target=[
                1,
                2,
                3,
            ],
            labels=[
                1,
                2,
                3,
            ],
        )


def test_evaluate_external_predictions_rejects_unknown_class() -> None:
    """External labels must belong to known model classes."""
    with pytest.raises(
        ValueError,
        match="not recognized",
    ):
        evaluate_external_predictions(
            true_target=pd.Series(
                [
                    1,
                    2,
                    4,
                ]
            ),
            predicted_target=[
                1,
                2,
                3,
            ],
            labels=[
                1,
                2,
                3,
            ],
        )


def test_build_prediction_metadata_contains_distribution(
    fitted_model: Pipeline,
    tmp_path: Path,
) -> None:
    """Metadata should describe prediction balance and confidence."""
    raw_dataframe = make_raw_dataframe()

    prepared_dataframe = prepare_prediction_data(
        raw_dataframe
    )

    predictions = (
        predict_script.predict_passengers(
            raw_dataframe=raw_dataframe,
            model=fitted_model,
        )
    )

    metadata = build_prediction_metadata(
        raw_dataframe=raw_dataframe,
        prepared_dataframe=prepared_dataframe,
        predictions=predictions,
        input_path=tmp_path / "input.csv",
        model_path=tmp_path / "model.joblib",
        model_classes=[
            1,
            2,
            3,
        ],
        external_metrics=None,
    )

    assert metadata["raw_rows"] == 9
    assert metadata["prediction_rows"] == 9
    assert metadata["model_classes"] == [
        1,
        2,
        3,
    ]

    assert sum(
        metadata[
            "predicted_class_counts"
        ].values()
    ) == 9

    assert (
        0
        <= metadata[
            "mean_prediction_confidence"
        ]
        <= 1
    )


def test_save_json_creates_json(
    tmp_path: Path,
) -> None:
    """JSON payloads should be persisted correctly."""
    output_path = tmp_path / "metadata.json"

    save_json(
        {
            "rows": np.int64(10),
            "path": tmp_path,
        },
        output_path,
    )

    payload = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["rows"] == 10
    assert payload["path"] == str(tmp_path)


def test_save_json_rejects_empty_payload(
    tmp_path: Path,
) -> None:
    """Empty reports should not be written."""
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        save_json(
            {},
            tmp_path / "empty.json",
        )


def test_save_prediction_artifacts_creates_files(
    tmp_path: Path,
) -> None:
    """Prediction and evaluation artifacts should be created."""
    predictions = pd.DataFrame(
        {
            ID_COLUMN: [
                1,
                2,
            ],
            PREDICTION_COLUMN: [
                1,
                3,
            ],
            CONFIDENCE_COLUMN: [
                0.9,
                0.8,
            ],
        }
    )

    metadata = {
        "prediction_rows": 2,
    }

    metrics = {
        "accuracy": 0.5,
    }

    report = pd.DataFrame(
        {
            "label": [
                "1",
                "3",
            ],
            "precision": [
                1.0,
                0.5,
            ],
        }
    )

    artifacts = save_prediction_artifacts(
        predictions=predictions,
        metadata=metadata,
        output_directory=tmp_path,
        external_metrics=metrics,
        classification_report_dataframe=report,
    )

    assert isinstance(
        artifacts,
        PredictionArtifacts,
    )

    assert (
        tmp_path / PREDICTIONS_FILENAME
    ).exists()

    assert (
        tmp_path / PREDICTION_METADATA_FILENAME
    ).exists()

    assert (
        tmp_path / EXTERNAL_METRICS_FILENAME
    ).exists()

    assert (
        tmp_path
        / EXTERNAL_CLASSIFICATION_REPORT_FILENAME
    ).exists()


def test_run_prediction_workflow_with_labelled_data(
    fitted_model: Pipeline,
    labelled_csv_path: Path,
    tmp_path: Path,
) -> None:
    """The labelled workflow should evaluate and export results."""
    model_path = tmp_path / "model.joblib"
    output_directory = tmp_path / "predictions"

    save_model(
        estimator=fitted_model,
        output_path=model_path,
    )

    result = run_prediction_workflow(
        input_path=labelled_csv_path,
        model_path=model_path,
        output_directory=output_directory,
        evaluate_when_target_available=True,
    )

    assert len(result.predictions) == 9
    assert result.external_metrics is not None
    assert result.classification_report is not None

    assert TARGET_COLUMN in result.predictions.columns
    assert CORRECT_COLUMN in result.predictions.columns

    assert result.artifacts.predictions_path.exists()
    assert result.artifacts.metadata_path.exists()

    assert result.artifacts.metrics_path is not None
    assert result.artifacts.metrics_path.exists()

    assert (
        result.artifacts.classification_report_path
        is not None
    )
    assert (
        result.artifacts.classification_report_path
        .exists()
    )


def test_run_prediction_workflow_with_unlabelled_data(
    fitted_model: Pipeline,
    tmp_path: Path,
) -> None:
    """Target-free input should still create predictions."""
    input_path = tmp_path / "unlabelled.csv"
    model_path = tmp_path / "model.joblib"
    output_directory = tmp_path / "outputs"

    make_raw_dataframe(
        include_target=False
    ).to_csv(
        input_path,
        index=False,
    )

    save_model(
        estimator=fitted_model,
        output_path=model_path,
    )

    result = run_prediction_workflow(
        input_path=input_path,
        model_path=model_path,
        output_directory=output_directory,
    )

    assert result.external_metrics is None
    assert result.classification_report is None
    assert TARGET_COLUMN not in result.predictions.columns
    assert CORRECT_COLUMN not in result.predictions.columns
    assert result.artifacts.metrics_path is None
    assert (
        result.artifacts.classification_report_path
        is None
    )

    assert result.artifacts.predictions_path.exists()
    assert result.artifacts.metadata_path.exists()


def test_run_prediction_workflow_can_skip_evaluation(
    fitted_model: Pipeline,
    labelled_csv_path: Path,
    tmp_path: Path,
) -> None:
    """Evaluation may be disabled for labelled inputs."""
    model_path = tmp_path / "model.joblib"

    save_model(
        estimator=fitted_model,
        output_path=model_path,
    )

    result = run_prediction_workflow(
        input_path=labelled_csv_path,
        model_path=model_path,
        output_directory=tmp_path / "outputs",
        evaluate_when_target_available=False,
    )

    assert result.external_metrics is None
    assert result.classification_report is None
    assert TARGET_COLUMN not in result.predictions.columns


def test_saved_metadata_matches_returned_metadata(
    fitted_model: Pipeline,
    labelled_csv_path: Path,
    tmp_path: Path,
) -> None:
    """Persisted metadata should match the workflow result."""
    model_path = tmp_path / "model.joblib"

    save_model(
        estimator=fitted_model,
        output_path=model_path,
    )

    result = run_prediction_workflow(
        input_path=labelled_csv_path,
        model_path=model_path,
        output_directory=tmp_path / "outputs",
    )

    saved_metadata = json.loads(
        result.artifacts.metadata_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        saved_metadata["prediction_rows"]
        == result.metadata["prediction_rows"]
    )

    assert (
        saved_metadata["predicted_class_counts"]
        == result.metadata["predicted_class_counts"]
    )


def test_prediction_output_has_unique_ids(
    fitted_model: Pipeline,
    labelled_csv_path: Path,
    tmp_path: Path,
) -> None:
    """The workflow should remain one row per passenger."""
    model_path = tmp_path / "model.joblib"

    save_model(
        estimator=fitted_model,
        output_path=model_path,
    )

    result = run_prediction_workflow(
        input_path=labelled_csv_path,
        model_path=model_path,
        output_directory=tmp_path / "outputs",
    )

    assert result.predictions[ID_COLUMN].is_unique
    assert len(result.predictions) == len(
        make_raw_dataframe()
    )
