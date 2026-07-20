"""Persistence utilities for trained models and evaluation artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

from titanic_passenger_class_prediction.config import (
    BEST_MODEL_FILENAME,
    METRICS_DIR,
    MODEL_COMPARISON_FILENAME,
    MODEL_METADATA_FILENAME,
    MODELS_DIR,
    TEST_METRICS_FILENAME,
)


def ensure_artifact_directories(
    models_directory: Path = MODELS_DIR,
    metrics_directory: Path = METRICS_DIR,
) -> None:
    """
    Create model and metrics artifact directories when necessary.

    Parameters
    ----------
    models_directory
        Directory used for serialized model pipelines.
    metrics_directory
        Directory used for evaluation tables and JSON reports.
    """
    models_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_model(
    estimator: BaseEstimator,
    output_path: Path | None = None,
) -> Path:
    """
    Serialize a fitted scikit-learn estimator or pipeline.

    The complete pipeline should be saved rather than only the final
    estimator. This preserves the fitted preprocessing steps required
    to make predictions on new passenger data.

    Parameters
    ----------
    estimator
        Fitted scikit-learn estimator or pipeline.
    output_path
        Optional destination path. When omitted, the configured default
        model artifact path is used.

    Returns
    -------
    pathlib.Path
        Path of the saved model artifact.
    """
    destination = output_path or (
        MODELS_DIR / BEST_MODEL_FILENAME
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        estimator,
        destination,
    )

    return destination


def load_model(
    input_path: Path | None = None,
) -> BaseEstimator:
    """
    Load a serialized scikit-learn estimator or pipeline.

    Parameters
    ----------
    input_path
        Optional model artifact path. When omitted, the configured
        default path is used.

    Returns
    -------
    sklearn.base.BaseEstimator
        Deserialized model or pipeline.

    Raises
    ------
    FileNotFoundError
        If the requested model artifact does not exist.
    """
    source = input_path or (
        MODELS_DIR / BEST_MODEL_FILENAME
    )

    if not source.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {source}"
        )

    return joblib.load(source)


def save_model_comparison(
    comparison: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """
    Save the cross-validation model-comparison table as CSV.

    Parameters
    ----------
    comparison
        Model-comparison table.
    output_path
        Optional destination path.

    Returns
    -------
    pathlib.Path
        Path of the saved CSV file.

    Raises
    ------
    ValueError
        If the comparison table is empty.
    """
    if comparison.empty:
        raise ValueError(
            "Cannot save an empty model-comparison table."
        )

    destination = output_path or (
        METRICS_DIR / MODEL_COMPARISON_FILENAME
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        destination,
        index=False,
    )

    return destination


def save_json_report(
    report: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """
    Save a mapping as a formatted JSON report.

    Parameters
    ----------
    report
        Mapping containing JSON-compatible report values.
    output_path
        Destination JSON path.

    Returns
    -------
    pathlib.Path
        Path of the saved JSON report.

    Raises
    ------
    ValueError
        If the report mapping is empty.
    """
    if not report:
        raise ValueError(
            "Cannot save an empty JSON report."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            sort_keys=True,
            default=_json_default,
        )

    return output_path


def save_test_metrics(
    metrics: Mapping[str, Any],
    output_path: Path | None = None,
) -> Path:
    """
    Save final holdout-test metrics as JSON.

    Parameters
    ----------
    metrics
        Mapping containing final test-set metrics.
    output_path
        Optional destination path.

    Returns
    -------
    pathlib.Path
        Path of the saved metrics report.
    """
    destination = output_path or (
        METRICS_DIR / TEST_METRICS_FILENAME
    )

    return save_json_report(
        report=metrics,
        output_path=destination,
    )


def build_model_metadata(
    model_name: str,
    target_column: str,
    feature_columns: list[str],
    training_rows: int,
    test_rows: int,
    selected_metric: str,
    selected_metric_value: float,
    additional_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build structured metadata for a trained model artifact.

    Parameters
    ----------
    model_name
        Stable name of the selected model.
    target_column
        Name of the prediction target.
    feature_columns
        Input columns expected by the model pipeline.
    training_rows
        Number of rows used to fit the final model.
    test_rows
        Number of rows reserved for final evaluation.
    selected_metric
        Metric used to rank candidate models.
    selected_metric_value
        Cross-validation value achieved by the selected model.
    additional_metadata
        Optional extra metadata fields.

    Returns
    -------
    dict[str, Any]
        JSON-compatible metadata dictionary.

    Raises
    ------
    ValueError
        If required string values are empty, feature columns are empty,
        or row counts are invalid.
    """
    if not model_name.strip():
        raise ValueError(
            "Model name must not be empty."
        )

    if not target_column.strip():
        raise ValueError(
            "Target column must not be empty."
        )

    if not feature_columns:
        raise ValueError(
            "At least one feature column is required."
        )

    if training_rows <= 0:
        raise ValueError(
            "Training row count must be positive."
        )

    if test_rows <= 0:
        raise ValueError(
            "Test row count must be positive."
        )

    if not selected_metric.strip():
        raise ValueError(
            "Selected metric must not be empty."
        )

    metadata: dict[str, Any] = {
        "model_name": model_name,
        "target_column": target_column,
        "feature_columns": list(feature_columns),
        "training_rows": int(training_rows),
        "test_rows": int(test_rows),
        "selected_metric": selected_metric,
        "selected_metric_value": float(
            selected_metric_value
        ),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    if additional_metadata:
        metadata.update(
            dict(additional_metadata)
        )

    return metadata


def save_model_metadata(
    metadata: Mapping[str, Any],
    output_path: Path | None = None,
) -> Path:
    """
    Save model metadata as JSON.

    Parameters
    ----------
    metadata
        Structured model metadata.
    output_path
        Optional destination path.

    Returns
    -------
    pathlib.Path
        Path of the saved metadata report.
    """
    destination = output_path or (
        METRICS_DIR / MODEL_METADATA_FILENAME
    )

    return save_json_report(
        report=metadata,
        output_path=destination,
    )


def _json_default(value: Any) -> Any:
    """
    Convert common non-native scalar values into JSON-compatible values.

    This supports NumPy and pandas scalar objects that expose an
    ``item`` method.

    Parameters
    ----------
    value
        Value that the standard JSON encoder cannot serialize.

    Returns
    -------
    Any
        JSON-compatible representation.

    Raises
    ------
    TypeError
        If the value cannot be converted safely.
    """
    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        "Object of type "
        f"{type(value).__name__} "
        "is not JSON serializable."
    )