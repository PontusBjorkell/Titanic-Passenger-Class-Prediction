"""Tests for scripts/predict.py.

The prediction script is loaded directly from its filesystem path because
the repository's pytest configuration may expose only the ``src`` directory
on Python's import path.

All tests use temporary files and lightweight fitted scikit-learn pipelines.
They do not overwrite the project's real model, datasets, or prediction
artifacts.
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
from sklearn.dummy import DummyClassifier
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "predict.py"


def load_predict_module() -> ModuleType:
    """Load scripts/predict.py directly from its filesystem path.

    Registering the module in ``sys.modules`` before execution is necessary
    because predict.py defines dataclasses. Python's dataclass machinery
    expects the module to be registered while class definitions are created.
    """
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

    module = importlib.util.module_from_spec(specification)

    sys.modules[module_name] = module

    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


predict_module = load_predict_module()


CONFIDENCE_COLUMN = predict_module.CONFIDENCE_COLUMN
CORRECT_COLUMN = predict_module.CORRECT_COLUMN

EXTERNAL_CLASSIFICATION_REPORT_FILENAME = (
    predict_module.EXTERNAL_CLASSIFICATION_REPORT_FILENAME
)
EXTERNAL_METRICS_FILENAME = (
    predict_module.EXTERNAL_METRICS_FILENAME
)

LOW_CONFIDENCE_THRESHOLD = (
    predict_module.LOW_CONFIDENCE_THRESHOLD
)

PREDICTION_COLUMN = predict_module.PREDICTION_COLUMN
PREDICTION_METADATA_FILENAME = (
    predict_module.PREDICTION_METADATA_FILENAME
)
PREDICTIONS_FILENAME = predict_module.PREDICTIONS_FILENAME

PredictionArtifacts = predict_module.PredictionArtifacts

add_context_columns = predict_module.add_context_columns
build_prediction_metadata = (
    predict_module.build_prediction_metadata
)
build_probability_column_name = (
    predict_module.build_probability_column_name
)
evaluate_external_predictions = (
    predict_module.evaluate_external_predictions
)
generate_predictions = predict_module.generate_predictions
get_model_classes = predict_module.get_model_classes
load_input_data = predict_module.load_input_data
prepare_prediction_data = (
    predict_module.prepare_prediction_data
)
run_prediction_workflow = (
    predict_module.run_prediction_workflow
)
save_json = predict_module.save_json
save_prediction_artifacts = (
    predict_module.save_prediction_artifacts
)
validate_fitted_model = (
    predict_module.validate_fitted_model
)
validate_raw_prediction_data = (
    predict_module.validate_raw_prediction_data
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


def make_training_dataframe() -> pd.DataFrame:
    """Create prepared data suitable for fitting a three-class model."""
    raw_dataframe = make_raw_dataframe(
        include_target=True
    )

    return prepare_prediction_data(
        raw_dataframe
    )


@pytest.fixture
def fitted_model() -> Pipeline:
    """Create a fitted production-style passenger-class pipeline."""
    prepared_dataframe = make_training_dataframe()

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
def raw_csv_path(
    tmp_path: Path,
) -> Path:
    """Save representative labelled external input as CSV."""
    input_path = tmp_path / "external_test.csv"

    make_raw_dataframe(
        include_target=True
    ).to_csv(
        input_path,
        index=False,
    )

    return input_path


def test_predict_script_was_loaded() -> None:
    """The direct script loader should expose the expected workflow."""
    assert predict_module is not None
    assert hasattr(
        predict_module,
        "run_prediction_workflow",
    )


def test_load_input_data_reads_csv(
    raw_csv_path: Path,
) -> None:
    """A valid CSV should be loaded as a dataframe."""
    dataframe = load_input_data(
        raw_csv_path
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
    """A nonexistent CSV should raise FileNotFoundError."""
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
    """A CSV containing headers but no rows should fail."""
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


def test_validate_raw_prediction_data_accepts_valid_data() -> None:
    """A complete raw Titanic dataframe should pass validation."""
    validate_raw_prediction_data(
        make_raw_dataframe()
    )


def test_validate_raw_prediction_data_does_not_require_target() -> None:
    """Inference should support genuinely unlabeled passenger data."""
    validate_raw_prediction_data(
        make_raw_dataframe(
            include_target=False
        )
    )


def test_validate_raw_prediction_data_rejects_empty_data() -> None:
    """An empty dataframe should fail validation."""
    with pytest.raises(
        ValueError,
        match="at least one row",
    ):
        validate_raw_prediction_data(
            pd.DataFrame()
        )


def test_validate_raw_prediction_data_rejects_missing_column() -> None:
    """Missing raw feature inputs should be identified."""
    dataframe = make_raw_dataframe().drop(
        columns=["Ticket"]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_raw_prediction_data(
            dataframe
        )


def test_validate_raw_prediction_data_rejects_missing_id() -> None:
    """Passenger identifiers cannot be missing."""
    dataframe = make_raw_dataframe()

    dataframe.loc[
        0,
        ID_COLUMN,
    ] = None

    with pytest.raises(
        ValueError,
        match="contains missing values",
    ):
        validate_raw_prediction_data(
            dataframe
        )


def test_validate_raw_prediction_data_rejects_duplicate_ids() -> None:
    """Passenger identifiers must be unique."""
    dataframe = make_raw_dataframe()

    dataframe.loc[
        1,
        ID_COLUMN,
    ] = dataframe.loc[
        0,
        ID_COLUMN,
    ]

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_raw_prediction_data(
            dataframe
        )


def test_prepare_prediction_data_adds_model_features() -> None:
    """Raw data should gain every engineered model feature."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    assert len(prepared_dataframe) == 9

    assert set(
        MODEL_FEATURES
    ).issubset(
        prepared_dataframe.columns
    )

    assert "FamilySize" in prepared_dataframe.columns
    assert "FareLog" in prepared_dataframe.columns
    assert "Title" in prepared_dataframe.columns
    assert "TicketPrefix" in prepared_dataframe.columns


def test_prepare_prediction_data_preserves_target_when_present() -> None:
    """True passenger-class labels should survive preparation."""
    raw_dataframe = make_raw_dataframe(
        include_target=True
    )

    prepared_dataframe = prepare_prediction_data(
        raw_dataframe
    )

    assert TARGET_COLUMN in prepared_dataframe.columns

    pd.testing.assert_series_equal(
        prepared_dataframe[TARGET_COLUMN],
        raw_dataframe[TARGET_COLUMN],
    )


def test_prepare_prediction_data_supports_unlabeled_input() -> None:
    """Target-free data should still support feature preparation."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe(
            include_target=False
        )
    )

    assert TARGET_COLUMN not in prepared_dataframe.columns

    assert set(
        MODEL_FEATURES
    ).issubset(
        prepared_dataframe.columns
    )


def test_get_model_classes_returns_fitted_classes(
    fitted_model: Pipeline,
) -> None:
    """Class labels should be extracted from the fitted pipeline."""
    classes = get_model_classes(
        fitted_model
    )

    assert classes == [
        1,
        2,
        3,
    ]


def test_get_model_classes_rejects_unfitted_estimator() -> None:
    """An estimator without fitted classes should fail."""
    with pytest.raises(
        AttributeError,
        match="class labels",
    ):
        get_model_classes(
            DummyClassifier()
        )


def test_validate_fitted_model_accepts_probability_classifier(
    fitted_model: Pipeline,
) -> None:
    """A fitted classifier with probability support should pass."""
    validate_fitted_model(
        fitted_model
    )


class PredictOnlyEstimator:
    """Minimal estimator that intentionally lacks predict_proba."""

    def predict(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """Return a constant prediction."""
        return np.ones(
            len(features)
        )


def test_validate_fitted_model_requires_predict_proba() -> None:
    """The workflow requires probability outputs."""
    with pytest.raises(
        TypeError,
        match="predict_proba",
    ):
        validate_fitted_model(
            PredictOnlyEstimator()
        )


def test_probability_column_name_is_stable() -> None:
    """Class labels should map to predictable output names."""
    assert (
        build_probability_column_name(1)
        == "ProbabilityClass1"
    )

    assert (
        build_probability_column_name(
            np.int64(3)
        )
        == "ProbabilityClass3"
    )


def test_generate_predictions_creates_expected_columns(
    fitted_model: Pipeline,
) -> None:
    """Inference should include predictions and class probabilities."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
    )

    expected_columns = {
        ID_COLUMN,
        PREDICTION_COLUMN,
        "ProbabilityClass1",
        "ProbabilityClass2",
        "ProbabilityClass3",
        CONFIDENCE_COLUMN,
        "SecondHighestProbability",
        "ProbabilityMargin",
        "LowConfidencePrediction",
    }

    assert expected_columns.issubset(
        predictions.columns
    )

    assert len(predictions) == len(
        prepared_dataframe
    )


def test_generate_predictions_probabilities_sum_to_one(
    fitted_model: Pipeline,
) -> None:
    """Class probabilities should sum to one for every passenger."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
    )

    probability_columns = [
        "ProbabilityClass1",
        "ProbabilityClass2",
        "ProbabilityClass3",
    ]

    probability_sums = predictions[
        probability_columns
    ].sum(
        axis=1
    )

    assert np.allclose(
        probability_sums,
        1.0,
    )


def test_generate_predictions_confidence_matches_max_probability(
    fitted_model: Pipeline,
) -> None:
    """Prediction confidence should equal the highest probability."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
    )

    probability_columns = [
        "ProbabilityClass1",
        "ProbabilityClass2",
        "ProbabilityClass3",
    ]

    expected_confidence = predictions[
        probability_columns
    ].max(
        axis=1
    )

    assert np.allclose(
        predictions[CONFIDENCE_COLUMN],
        expected_confidence,
    )


def test_low_confidence_flag_matches_threshold(
    fitted_model: Pipeline,
) -> None:
    """Low-confidence flags should reflect the configured threshold."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
    )

    expected_flags = (
        predictions[CONFIDENCE_COLUMN]
        < LOW_CONFIDENCE_THRESHOLD
    )

    pd.testing.assert_series_equal(
        predictions["LowConfidencePrediction"],
        expected_flags,
        check_names=False,
    )


def test_generate_predictions_rejects_missing_model_feature(
    fitted_model: Pipeline,
) -> None:
    """Prepared inference data must contain every model input."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    ).drop(
        columns=["FareLog"]
    )

    with pytest.raises(
        ValueError,
        match="missing model features",
    ):
        generate_predictions(
            model=fitted_model,
            prepared_dataframe=prepared_dataframe,
        )


def test_add_context_columns_preserves_prediction_rows(
    fitted_model: Pipeline,
) -> None:
    """Passenger context should merge one-to-one by identifier."""
    prepared_dataframe = prepare_prediction_data(
        make_raw_dataframe()
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
    )

    enriched_predictions = add_context_columns(
        predictions=predictions,
        prepared_dataframe=prepared_dataframe,
    )

    assert len(
        enriched_predictions
    ) == len(
        predictions
    )

    assert "Name" in enriched_predictions.columns
    assert "Sex" in enriched_predictions.columns
    assert "Fare" in enriched_predictions.columns

    assert enriched_predictions[
        ID_COLUMN
    ].tolist() == prepared_dataframe[
        ID_COLUMN
    ].tolist()


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

    assert metrics[
        "accuracy"
    ] == pytest.approx(
        5 / 6
    )

    assert (
        0
        <= metrics["balanced_accuracy"]
        <= 1
    )

    assert (
        0
        <= metrics["f1_macro"]
        <= 1
    )

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
    """External labels must not contain missing values."""
    target = pd.Series(
        [
            1,
            2,
            None,
        ]
    )

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        evaluate_external_predictions(
            true_target=target,
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
    """Ground truth cannot include a class unknown to the model."""
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
    """Metadata should describe class balance and confidence."""
    raw_dataframe = make_raw_dataframe()

    prepared_dataframe = prepare_prediction_data(
        raw_dataframe
    )

    predictions = generate_predictions(
        model=fitted_model,
        prepared_dataframe=prepared_dataframe,
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

    assert (
        metadata[
            "low_confidence_threshold"
        ]
        == LOW_CONFIDENCE_THRESHOLD
    )


def test_save_json_creates_formatted_json(
    tmp_path: Path,
) -> None:
    """JSON payloads should be written correctly."""
    output_path = tmp_path / "metadata.json"

    save_json(
        {
            "rows": np.int64(10),
            "path": tmp_path,
        },
        output_path,
    )

    loaded_payload = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert loaded_payload["rows"] == 10
    assert loaded_payload["path"] == str(
        tmp_path
    )


def test_save_json_rejects_empty_payload(
    tmp_path: Path,
) -> None:
    """Empty JSON reports should not be saved."""
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        save_json(
            {},
            tmp_path / "empty.json",
        )


def test_save_prediction_artifacts_creates_all_files(
    tmp_path: Path,
) -> None:
    """Prediction and evaluation artifacts should be saved."""
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
        tmp_path
        / PREDICTION_METADATA_FILENAME
    ).exists()

    assert (
        tmp_path
        / EXTERNAL_METRICS_FILENAME
    ).exists()

    assert (
        tmp_path
        / EXTERNAL_CLASSIFICATION_REPORT_FILENAME
    ).exists()


def test_run_prediction_workflow_with_labeled_data(
    fitted_model: Pipeline,
    raw_csv_path: Path,
    tmp_path: Path,
) -> None:
    """The labelled workflow should evaluate and export results."""
    model_path = tmp_path / "model.joblib"

    output_directory = (
        tmp_path / "predictions"
    )

    save_model(
        estimator=fitted_model,
        output_path=model_path,
    )

    result = run_prediction_workflow(
        input_path=raw_csv_path,
        model_path=model_path,
        output_directory=output_directory,
        evaluate_when_target_available=True,
    )

    assert len(
        result.predictions
    ) == 9

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


def test_run_prediction_workflow_with_unlabeled_data(
    fitted_model: Pipeline,
    tmp_path: Path,
) -> None:
    """The workflow should support target-free input data."""
    input_path = tmp_path / "unlabeled.csv"
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
    raw_csv_path: Path,
    tmp_path: Path,
) -> None:
    """Evaluation may be disabled for a labelled input file."""
    model_path = tmp_path / "model.joblib"

    save_model(
        estimator=fitted_model,
        output_path=model_path,
    )

    result = run_prediction_workflow(
        input_path=raw_csv_path,
        model_path=model_path,
        output_directory=tmp_path / "outputs",
        evaluate_when_target_available=False,
    )

    assert result.external_metrics is None
    assert result.classification_report is None
    assert TARGET_COLUMN not in result.predictions.columns


def test_run_prediction_workflow_metadata_matches_saved_json(
    fitted_model: Pipeline,
    raw_csv_path: Path,
    tmp_path: Path,
) -> None:
    """Returned metadata should match its persisted JSON file."""
    model_path = tmp_path / "model.joblib"

    save_model(
        estimator=fitted_model,
        output_path=model_path,
    )

    result = run_prediction_workflow(
        input_path=raw_csv_path,
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
        == result.metadata[
            "prediction_rows"
        ]
    )

    assert (
        saved_metadata[
            "predicted_class_counts"
        ]
        == result.metadata[
            "predicted_class_counts"
        ]
    )


def test_prediction_output_has_unique_passenger_ids(
    fitted_model: Pipeline,
    raw_csv_path: Path,
    tmp_path: Path,
) -> None:
    """Prediction output should remain one row per passenger."""
    model_path = tmp_path / "model.joblib"

    save_model(
        estimator=fitted_model,
        output_path=model_path,
    )

    result = run_prediction_workflow(
        input_path=raw_csv_path,
        model_path=model_path,
        output_directory=tmp_path / "outputs",
    )

    assert result.predictions[
        ID_COLUMN
    ].is_unique

    assert len(
        result.predictions
    ) == len(
        make_raw_dataframe()
    )