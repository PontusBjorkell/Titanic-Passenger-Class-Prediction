"""Generate passenger-class predictions for an external Titanic dataset.

This command-line script is intentionally thin. Reusable inference logic lives
in ``titanic_passenger_class_prediction.prediction`` so that batch prediction,
Streamlit, notebooks, and tests all use the same feature-engineering and model
prediction code.

Examples
--------
Run inference on the default raw test file:

    python scripts/predict.py

Use another input file:

    python scripts/predict.py --input data/raw/test.csv

Use another saved model:

    python scripts/predict.py \
        --model artifacts/models/best_model.joblib

Write artifacts to another directory:

    python scripts/predict.py \
        --output-dir artifacts/predictions

Disable evaluation even when Pclass is present:

    python scripts/predict.py --skip-evaluation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from titanic_passenger_class_prediction.config import (
    ARTIFACTS_DIR,
    BEST_MODEL_PATH,
    ID_COLUMN,
    TARGET_COLUMN,
    TEST_DATA_PATH,
)
from titanic_passenger_class_prediction.modeling import MODEL_FEATURES
from titanic_passenger_class_prediction.persistence import load_model
from titanic_passenger_class_prediction.prediction import (
    CONFIDENCE_COLUMN,
    LOW_CONFIDENCE_COLUMN,
    LOW_CONFIDENCE_THRESHOLD,
    PREDICTION_COLUMN,
    get_model_classes,
    predict_passengers,
    prepare_prediction_data,
    validate_fitted_model,
)


DEFAULT_PREDICTIONS_DIRECTORY = ARTIFACTS_DIR / "predictions"

PREDICTIONS_FILENAME = "test_predictions.csv"
PREDICTION_METADATA_FILENAME = "test_prediction_summary.json"
EXTERNAL_METRICS_FILENAME = "external_test_metrics.json"
EXTERNAL_CLASSIFICATION_REPORT_FILENAME = (
    "external_classification_report.csv"
)

CORRECT_COLUMN = "PredictionCorrect"


@dataclass(frozen=True)
class PredictionArtifacts:
    """Paths created by one prediction run."""

    predictions_path: Path
    metadata_path: Path
    metrics_path: Path | None = None
    classification_report_path: Path | None = None


@dataclass(frozen=True)
class PredictionRunResult:
    """In-memory results and generated artifact paths."""

    predictions: pd.DataFrame
    metadata: dict[str, Any]
    external_metrics: dict[str, Any] | None
    classification_report: pd.DataFrame | None
    artifacts: PredictionArtifacts


def load_input_data(input_path: Path) -> pd.DataFrame:
    """Load a raw Titanic-compatible CSV file."""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prediction input file not found: {input_path}"
        )

    if not input_path.is_file():
        raise ValueError(
            f"Prediction input path is not a file: {input_path}"
        )

    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError(
            f"Prediction input dataset is empty: {input_path}"
        )

    return dataframe


def evaluate_external_predictions(
    true_target: pd.Series,
    predicted_target: Sequence[Any],
    labels: Sequence[Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate predictions against external ground-truth labels."""
    target = pd.Series(true_target).reset_index(drop=True)
    predictions = pd.Series(predicted_target).reset_index(drop=True)

    if len(target) != len(predictions):
        raise ValueError(
            "True and predicted target lengths do not match."
        )

    if target.isna().any():
        raise ValueError(
            f"External target column '{TARGET_COLUMN}' contains "
            "missing values."
        )

    normalized_labels = [
        value.item() if hasattr(value, "item") else value
        for value in labels
    ]

    unexpected_labels = sorted(
        set(target.unique()) - set(normalized_labels)
    )

    if unexpected_labels:
        raise ValueError(
            "External target contains classes not recognized by the "
            f"model: {unexpected_labels}"
        )

    report_dictionary = classification_report(
        target,
        predictions,
        labels=normalized_labels,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = (
        pd.DataFrame(report_dictionary)
        .transpose()
        .rename_axis("label")
        .reset_index()
    )

    confusion = confusion_matrix(
        target,
        predictions,
        labels=normalized_labels,
    )

    metrics = {
        "evaluation_type": "external_labeled_test_set",
        "target_column": TARGET_COLUMN,
        "rows": int(len(target)),
        "labels": normalized_labels,
        "accuracy": float(
            accuracy_score(target, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, predictions)
        ),
        "f1_macro": float(
            f1_score(
                target,
                predictions,
                labels=normalized_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion.tolist(),
    }

    return metrics, report_dataframe


def build_prediction_metadata(
    *,
    raw_dataframe: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
    predictions: pd.DataFrame,
    input_path: Path,
    model_path: Path,
    model_classes: Sequence[Any],
    external_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a JSON-compatible summary of an inference run."""
    class_distribution = (
        predictions[PREDICTION_COLUMN]
        .value_counts()
        .sort_index()
    )

    class_distribution_percent = (
        predictions[PREDICTION_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "model_path": str(model_path),
        "raw_rows": int(raw_dataframe.shape[0]),
        "raw_columns": int(raw_dataframe.shape[1]),
        "prepared_rows": int(prepared_dataframe.shape[0]),
        "prepared_columns": int(prepared_dataframe.shape[1]),
        "prediction_rows": int(predictions.shape[0]),
        "target_available": bool(
            TARGET_COLUMN in prepared_dataframe.columns
        ),
        "external_evaluation_performed": (
            external_metrics is not None
        ),
        "model_classes": [
            value.item() if hasattr(value, "item") else value
            for value in model_classes
        ],
        "model_features": list(MODEL_FEATURES),
        "predicted_class_counts": {
            str(label): int(count)
            for label, count in class_distribution.items()
        },
        "predicted_class_percentages": {
            str(label): float(percentage)
            for label, percentage
            in class_distribution_percent.items()
        },
        "mean_prediction_confidence": float(
            predictions[CONFIDENCE_COLUMN].mean()
        ),
        "median_prediction_confidence": float(
            predictions[CONFIDENCE_COLUMN].median()
        ),
        "minimum_prediction_confidence": float(
            predictions[CONFIDENCE_COLUMN].min()
        ),
        "maximum_prediction_confidence": float(
            predictions[CONFIDENCE_COLUMN].max()
        ),
        "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "low_confidence_count": int(
            predictions[LOW_CONFIDENCE_COLUMN].sum()
        ),
        "low_confidence_percentage": float(
            predictions[LOW_CONFIDENCE_COLUMN].mean() * 100
        ),
        "external_metrics": external_metrics,
    }


def _json_default(value: Any) -> Any:
    """Convert common non-native values for JSON serialization."""
    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


def save_json(
    payload: dict[str, Any],
    output_path: Path,
) -> Path:
    """Save a non-empty mapping as formatted JSON."""
    if not payload:
        raise ValueError(
            "Cannot save an empty JSON payload."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )

    return output_path


def save_prediction_artifacts(
    *,
    predictions: pd.DataFrame,
    metadata: dict[str, Any],
    output_directory: Path,
    external_metrics: dict[str, Any] | None = None,
    classification_report_dataframe: pd.DataFrame | None = None,
) -> PredictionArtifacts:
    """Persist predictions, metadata, and optional evaluation outputs."""
    if predictions.empty:
        raise ValueError(
            "Cannot save an empty predictions dataframe."
        )

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = (
        output_directory / PREDICTIONS_FILENAME
    )
    metadata_path = (
        output_directory / PREDICTION_METADATA_FILENAME
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    save_json(
        metadata,
        metadata_path,
    )

    metrics_path: Path | None = None
    report_path: Path | None = None

    if external_metrics is not None:
        metrics_path = (
            output_directory / EXTERNAL_METRICS_FILENAME
        )

        save_json(
            external_metrics,
            metrics_path,
        )

    if classification_report_dataframe is not None:
        if classification_report_dataframe.empty:
            raise ValueError(
                "Classification report dataframe cannot be empty."
            )

        report_path = (
            output_directory
            / EXTERNAL_CLASSIFICATION_REPORT_FILENAME
        )

        classification_report_dataframe.to_csv(
            report_path,
            index=False,
        )

    return PredictionArtifacts(
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        metrics_path=metrics_path,
        classification_report_path=report_path,
    )


def run_prediction_workflow(
    *,
    input_path: Path = TEST_DATA_PATH,
    model_path: Path = BEST_MODEL_PATH,
    output_directory: Path = DEFAULT_PREDICTIONS_DIRECTORY,
    evaluate_when_target_available: bool = True,
) -> PredictionRunResult:
    """Execute the complete external inference workflow."""
    input_path = Path(input_path)
    model_path = Path(model_path)
    output_directory = Path(output_directory)

    raw_dataframe = load_input_data(
        input_path
    )

    prepared_dataframe = prepare_prediction_data(
        raw_dataframe
    )

    model = load_model(
        model_path
    )

    validate_fitted_model(
        model
    )

    model_classes = get_model_classes(
        model
    )

    predictions = predict_passengers(
        raw_dataframe=raw_dataframe,
        model=model,
    )

    external_metrics: dict[str, Any] | None = None
    report_dataframe: pd.DataFrame | None = None

    target_available = (
        TARGET_COLUMN in prepared_dataframe.columns
    )

    if (
        evaluate_when_target_available
        and target_available
    ):
        external_metrics, report_dataframe = (
            evaluate_external_predictions(
                true_target=prepared_dataframe[
                    TARGET_COLUMN
                ],
                predicted_target=predictions[
                    PREDICTION_COLUMN
                ],
                labels=model_classes,
            )
        )

        predictions[TARGET_COLUMN] = (
            prepared_dataframe[TARGET_COLUMN]
            .reset_index(drop=True)
        )

        predictions[CORRECT_COLUMN] = (
            predictions[PREDICTION_COLUMN]
            == predictions[TARGET_COLUMN]
        )

        preferred_columns = [
            ID_COLUMN,
            "Name",
            "Sex",
            "Age",
            "Fare",
            "Embarked",
            TARGET_COLUMN,
            PREDICTION_COLUMN,
            CORRECT_COLUMN,
        ]

        existing_preferred = [
            column
            for column in preferred_columns
            if column in predictions.columns
        ]

        other_columns = [
            column
            for column in predictions.columns
            if column not in existing_preferred
        ]

        predictions = predictions.loc[
            :,
            existing_preferred + other_columns,
        ]

    metadata = build_prediction_metadata(
        raw_dataframe=raw_dataframe,
        prepared_dataframe=prepared_dataframe,
        predictions=predictions,
        input_path=input_path,
        model_path=model_path,
        model_classes=model_classes,
        external_metrics=external_metrics,
    )

    artifacts = save_prediction_artifacts(
        predictions=predictions,
        metadata=metadata,
        output_directory=output_directory,
        external_metrics=external_metrics,
        classification_report_dataframe=report_dataframe,
    )

    return PredictionRunResult(
        predictions=predictions,
        metadata=metadata,
        external_metrics=external_metrics,
        classification_report=report_dataframe,
        artifacts=artifacts,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate passenger-class predictions using the "
            "saved Titanic model pipeline."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=TEST_DATA_PATH,
        help=(
            "Raw Titanic CSV used for prediction. "
            f"Default: {TEST_DATA_PATH}"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=BEST_MODEL_PATH,
        help=(
            "Saved fitted model pipeline. "
            f"Default: {BEST_MODEL_PATH}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIRECTORY,
        help=(
            "Directory for prediction artifacts. "
            f"Default: {DEFAULT_PREDICTIONS_DIRECTORY}"
        ),
    )

    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help=(
            "Do not calculate external metrics even when the "
            f"input contains '{TARGET_COLUMN}'."
        ),
    )

    return parser


def print_prediction_summary(
    result: PredictionRunResult,
) -> None:
    """Print concise details about a completed run."""
    print()
    print("=" * 78)
    print("EXTERNAL PREDICTION WORKFLOW COMPLETE")
    print("=" * 78)

    print(
        f"Predicted rows: {len(result.predictions):,}"
    )

    print(
        "Mean prediction confidence: "
        f"{result.metadata['mean_prediction_confidence']:.2%}"
    )

    print(
        "Low-confidence predictions: "
        f"{result.metadata['low_confidence_count']:,} "
        f"({result.metadata['low_confidence_percentage']:.2f}%)"
    )

    print()
    print("Predicted class counts:")

    for class_label, count in (
        result.metadata[
            "predicted_class_counts"
        ].items()
    ):
        print(
            f"  Class {class_label}: {count:,}"
        )

    if result.external_metrics is not None:
        print()
        print("External labeled-test performance:")
        print(
            "  Accuracy: "
            f"{result.external_metrics['accuracy']:.4f}"
        )
        print(
            "  Balanced accuracy: "
            f"{result.external_metrics['balanced_accuracy']:.4f}"
        )
        print(
            "  Macro F1: "
            f"{result.external_metrics['f1_macro']:.4f}"
        )
    else:
        print()
        print(
            "External evaluation was not performed because it "
            "was disabled or the target was unavailable."
        )

    print()
    print("Artifacts:")
    print(
        f"  Predictions: {result.artifacts.predictions_path}"
    )
    print(
        f"  Metadata: {result.artifacts.metadata_path}"
    )

    if result.artifacts.metrics_path is not None:
        print(
            "  External metrics: "
            f"{result.artifacts.metrics_path}"
        )

    if (
        result.artifacts.classification_report_path
        is not None
    ):
        print(
            "  External classification report: "
            f"{result.artifacts.classification_report_path}"
        )


def main() -> None:
    """Run external inference from the command line."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    result = run_prediction_workflow(
        input_path=arguments.input,
        model_path=arguments.model,
        output_directory=arguments.output_dir,
        evaluate_when_target_available=(
            not arguments.skip_evaluation
        ),
    )

    print_prediction_summary(
        result
    )


if __name__ == "__main__":
    main()
