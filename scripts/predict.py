"""Generate passenger-class predictions for an external Titanic dataset.

This script loads the fitted model pipeline produced by ``scripts/train.py``,
prepares a raw Titanic CSV using the project's deterministic feature
engineering, generates class predictions and probabilities, and saves the
results as reusable artifacts.

When the input dataset contains the true target column ``Pclass``, the script
also performs external evaluation and saves:

- accuracy;
- balanced accuracy;
- Macro F1;
- classification report;
- confusion matrix values.

This external evaluation is separate from the internal holdout evaluation
created during model training.

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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
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
from titanic_passenger_class_prediction.modeling import (
    MODEL_FEATURES,
)
from titanic_passenger_class_prediction.persistence import (
    load_model,
)
from titanic_passenger_class_prediction.features import (
    add_passenger_features,
)
from titanic_passenger_class_prediction.preprocessing import (
    standardize_column_types,
)


DEFAULT_PREDICTIONS_DIRECTORY = ARTIFACTS_DIR / "predictions"

PREDICTIONS_FILENAME = "test_predictions.csv"
PREDICTION_METADATA_FILENAME = "test_prediction_summary.json"
EXTERNAL_METRICS_FILENAME = "external_test_metrics.json"
EXTERNAL_CLASSIFICATION_REPORT_FILENAME = (
    "external_classification_report.csv"
)

PREDICTION_COLUMN = "PredictedPclass"
CONFIDENCE_COLUMN = "PredictionConfidence"
CORRECT_COLUMN = "PredictionCorrect"

LOW_CONFIDENCE_THRESHOLD = 0.60


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
    """Load a raw external Titanic CSV.

    Parameters
    ----------
    input_path
        Path to a raw Titanic-compatible CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded raw input data.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    ValueError
        If the input file is not a regular file or contains no rows.
    """
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


def validate_raw_prediction_data(
    dataframe: pd.DataFrame,
) -> None:
    """Validate raw external data before feature engineering.

    The deterministic preparation pipeline requires the original Titanic
    passenger columns used to construct engineered features.
    """
    if dataframe.empty:
        raise ValueError(
            "Prediction dataframe must contain at least one row."
        )

    required_raw_columns = {
        "PassengerId",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
    }

    missing_columns = sorted(
        required_raw_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Raw prediction data is missing required columns: "
            f"{missing_columns}"
        )

    if dataframe[ID_COLUMN].isna().any():
        raise ValueError(
            f"Identifier column '{ID_COLUMN}' contains missing values."
        )

    if dataframe[ID_COLUMN].duplicated().any():
        duplicated_ids = (
            dataframe.loc[
                dataframe[ID_COLUMN].duplicated(keep=False),
                ID_COLUMN,
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Identifier column '{ID_COLUMN}' contains duplicate values: "
            f"{duplicated_ids[:10]}"
        )


def prepare_prediction_data(
    raw_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare labelled or unlabelled passenger data for inference.

    Prediction data should not be required to contain the target column
    ``Pclass`` or the historical outcome column ``Survived``.

    The project's general ``prepare_passenger_data`` function is designed
    for labelled training data and therefore validates against the complete
    training schema. This inference-specific function instead:

    1. validates only the raw columns required to construct model features;
    2. standardizes available column types;
    3. applies the same deterministic feature engineering used in training;
    4. confirms that every model feature is available;
    5. preserves Pclass and Survived when they are present.

    Parameters
    ----------
    raw_dataframe
        Raw Titanic-compatible passenger data. The dataframe may be labelled
        or unlabelled.

    Returns
    -------
    pandas.DataFrame
        Type-standardized and feature-engineered passenger data suitable for
        the fitted model pipeline.

    Raises
    ------
    ValueError
        If required inference columns are missing, passenger identifiers are
        invalid, or the engineered output lacks model features.
    """
    validate_raw_prediction_data(
        raw_dataframe
    )

    # The standardization function converts only columns that are actually
    # present. It therefore safely supports both labelled and unlabelled
    # inference datasets.
    prepared_dataframe = standardize_column_types(
        raw_dataframe
    )

    # Feature engineering depends only on passenger attributes such as name,
    # age, fare, ticket, family counts, and cabin. It does not require Pclass
    # or Survived.
    prepared_dataframe = add_passenger_features(
        prepared_dataframe
    )

    missing_model_features = sorted(
        set(MODEL_FEATURES)
        - set(prepared_dataframe.columns)
    )

    if missing_model_features:
        raise ValueError(
            "Prepared prediction data is missing model features: "
            f"{missing_model_features}"
        )

    if len(prepared_dataframe) != len(raw_dataframe):
        raise ValueError(
            "Prediction preparation unexpectedly changed the number "
            "of passenger rows."
        )

    if not prepared_dataframe[
        ID_COLUMN
    ].is_unique:
        raise ValueError(
            f"Prepared identifier column '{ID_COLUMN}' "
            "must remain unique."
        )

    return prepared_dataframe


def get_model_classes(
    model: BaseEstimator,
) -> list[Any]:
    """Extract class labels from a fitted estimator or pipeline."""
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        final_estimator = model.named_steps.get("model")
        classes = getattr(final_estimator, "classes_", None)

    if classes is None:
        raise AttributeError(
            "The fitted model does not expose class labels through "
            "'classes_'."
        )

    return [
        value.item() if hasattr(value, "item") else value
        for value in classes
    ]


def validate_fitted_model(
    model: BaseEstimator,
) -> None:
    """Confirm that the loaded artifact supports prediction."""
    if not hasattr(model, "predict"):
        raise TypeError(
            "Loaded model artifact does not implement predict()."
        )

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Loaded model artifact does not implement predict_proba(). "
            "Probability outputs are required by this workflow."
        )

    get_model_classes(model)


def build_probability_column_name(
    class_label: Any,
) -> str:
    """Create a stable output name for one class-probability column."""
    label = (
        class_label.item()
        if hasattr(class_label, "item")
        else class_label
    )

    return f"ProbabilityClass{label}"


def generate_predictions(
    model: BaseEstimator,
    prepared_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predicted classes, probabilities, and confidence values."""
    validate_fitted_model(model)

    if prepared_dataframe.empty:
        raise ValueError(
            "Cannot generate predictions for an empty dataframe."
        )

    missing_features = sorted(
        set(MODEL_FEATURES) - set(prepared_dataframe.columns)
    )

    if missing_features:
        raise ValueError(
            "Prediction dataframe is missing model features: "
            f"{missing_features}"
        )

    features = prepared_dataframe.loc[:, MODEL_FEATURES].copy()

    predictions = np.asarray(model.predict(features))
    probabilities = np.asarray(model.predict_proba(features))
    classes = get_model_classes(model)

    if predictions.shape[0] != len(prepared_dataframe):
        raise ValueError(
            "Model returned an unexpected number of predictions."
        )

    if probabilities.ndim != 2:
        raise ValueError(
            "predict_proba() must return a two-dimensional array."
        )

    if probabilities.shape != (
        len(prepared_dataframe),
        len(classes),
    ):
        raise ValueError(
            "Probability output dimensions do not match the input rows "
            "and model classes."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model produced non-finite probability values."
        )

    if not np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    ):
        raise ValueError(
            "Predicted class probabilities do not sum to one."
        )

    identifiers = prepared_dataframe[ID_COLUMN].reset_index(
        drop=True
    )

    output = pd.DataFrame(
        {
            ID_COLUMN: identifiers,
            PREDICTION_COLUMN: predictions,
        }
    )

    for class_index, class_label in enumerate(classes):
        output[
            build_probability_column_name(class_label)
        ] = probabilities[:, class_index]

    output[CONFIDENCE_COLUMN] = probabilities.max(axis=1)

    output["SecondHighestProbability"] = np.sort(
        probabilities,
        axis=1,
    )[:, -2]

    output["ProbabilityMargin"] = (
        output[CONFIDENCE_COLUMN]
        - output["SecondHighestProbability"]
    )

    output["LowConfidencePrediction"] = (
        output[CONFIDENCE_COLUMN]
        < LOW_CONFIDENCE_THRESHOLD
    )

    return output


def add_context_columns(
    predictions: pd.DataFrame,
    prepared_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Attach useful passenger context without duplicating model features."""
    output = predictions.copy()

    context_columns = [
        column
        for column in [
            "Name",
            "Sex",
            "Age",
            "Fare",
            "Embarked",
        ]
        if column in prepared_dataframe.columns
    ]

    context = prepared_dataframe[
        [ID_COLUMN, *context_columns]
    ].copy()

    output = output.merge(
        context,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    ordered_columns = [
        ID_COLUMN,
        *context_columns,
        PREDICTION_COLUMN,
    ]

    remaining_columns = [
        column
        for column in output.columns
        if column not in ordered_columns
    ]

    return output.loc[
        :,
        ordered_columns + remaining_columns,
    ]


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
            balanced_accuracy_score(
                target,
                predictions,
            )
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
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
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
        "low_confidence_threshold": (
            LOW_CONFIDENCE_THRESHOLD
        ),
        "low_confidence_count": int(
            predictions["LowConfidencePrediction"].sum()
        ),
        "low_confidence_percentage": float(
            predictions[
                "LowConfidencePrediction"
            ].mean()
            * 100
        ),
        "external_metrics": external_metrics,
    }


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

    raw_dataframe = load_input_data(input_path)
    prepared_dataframe = prepare_prediction_data(
        raw_dataframe
    )

    model = load_model(model_path)
    validate_fitted_model(model)

    model_classes = get_model_classes(model)

    predictions = generate_predictions(
        model=model,
        prepared_dataframe=prepared_dataframe,
    )

    predictions = add_context_columns(
        predictions=predictions,
        prepared_dataframe=prepared_dataframe,
    )

    external_metrics: dict[str, Any] | None = None
    report_dataframe: pd.DataFrame | None = None

    target_available = (
        TARGET_COLUMN in prepared_dataframe.columns
    )

    if evaluate_when_target_available and target_available:
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
        f"Predicted rows: "
        f"{len(result.predictions):,}"
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
        print(f"  Class {class_label}: {count:,}")

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
        f"  Predictions: "
        f"{result.artifacts.predictions_path}"
    )
    print(
        f"  Metadata: "
        f"{result.artifacts.metadata_path}"
    )

    if result.artifacts.metrics_path is not None:
        print(
            f"  External metrics: "
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

    print_prediction_summary(result)


if __name__ == "__main__":
    main()