"""Train, evaluate, select, and persist passenger-class models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split

from titanic_passenger_class_prediction.config import (
    BEST_MODEL_FILENAME,
    CV_FOLDS,
    DEFAULT_PROCESSED_FILENAME,
    METRICS_DIR,
    MODEL_COMPARISON_FILENAME,
    MODEL_METADATA_FILENAME,
    MODELS_DIR,
    N_JOBS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_METRICS_FILENAME,
    TEST_SIZE,
)
from titanic_passenger_class_prediction.data import (
    load_processed_data,
)
from titanic_passenger_class_prediction.evaluation import (
    compare_models_cv,
    evaluate_holdout,
    select_best_model_name,
)
from titanic_passenger_class_prediction.modeling import (
    MODEL_FEATURES,
    build_candidate_models,
)
from titanic_passenger_class_prediction.persistence import (
    build_model_metadata,
    ensure_artifact_directories,
    save_model,
    save_model_comparison,
    save_model_metadata,
    save_test_metrics,
)


SELECTION_METRIC = "cv_f1_macro_mean"


def validate_training_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate that the prepared dataset can support model training.

    Parameters
    ----------
    dataframe
        Prepared passenger dataset.

    Raises
    ------
    ValueError
        If the dataset is empty, required columns are missing, the
        target contains missing values, or insufficient target classes
        are present.
    """
    if dataframe.empty:
        raise ValueError(
            "The processed training dataset is empty."
        )

    required_columns = {
        TARGET_COLUMN,
        *MODEL_FEATURES,
    }

    missing_columns = sorted(
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Processed training data is missing required columns: "
            f"{missing_columns}"
        )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains missing values."
        )

    if dataframe[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "Model training requires at least two target classes."
        )


def split_training_data(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Create a reproducible stratified train-test split.

    Parameters
    ----------
    dataframe
        Validated prepared passenger dataset.

    Returns
    -------
    tuple
        ``X_train``, ``X_test``, ``y_train``, and ``y_test``.
    """
    features = dataframe.loc[:, MODEL_FEATURES].copy()
    target = dataframe.loc[:, TARGET_COLUMN].copy()

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def train_selected_model(
    model_name: str,
    models: dict[str, BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
) -> BaseEstimator:
    """
    Clone and fit the selected model.

    Parameters
    ----------
    model_name
        Selected registry key.
    models
        Candidate-model registry.
    features
        Training features.
    target
        Training target.

    Returns
    -------
    sklearn.base.BaseEstimator
        Fitted estimator or pipeline.
    """
    if model_name not in models:
        raise ValueError(
            f"Unknown model name: {model_name}"
        )

    estimator = clone(models[model_name])

    estimator.fit(
        features,
        target,
    )

    return estimator


def run_training(
    processed_filename: str = DEFAULT_PROCESSED_FILENAME,
    models_directory: Path = MODELS_DIR,
    metrics_directory: Path = METRICS_DIR,
) -> dict[str, Any]:
    """
    Execute the complete model-training workflow.

    Parameters
    ----------
    processed_filename
        Prepared Parquet dataset filename.
    models_directory
        Directory for serialized model artifacts.
    metrics_directory
        Directory for evaluation reports.

    Returns
    -------
    dict[str, Any]
        Summary of the completed training run.
    """
    print("Loading processed passenger data...")

    dataframe = load_processed_data(
        filename=processed_filename,
    )

    validate_training_dataframe(dataframe)

    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = split_training_data(dataframe)

    print(
        "Training rows: "
        f"{len(features_train):,}"
    )
    print(
        "Test rows: "
        f"{len(features_test):,}"
    )

    models = build_candidate_models()

    print(
        "\nComparing candidate models with "
        f"{CV_FOLDS}-fold cross-validation..."
    )

    comparison = compare_models_cv(
        models=models,
        features=features_train,
        target=target_train,
    )

    display_columns = [
        "model",
        "cv_accuracy_mean",
        "cv_balanced_accuracy_mean",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
    ]

    print()
    print(
        comparison.loc[:, display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    best_model_name = select_best_model_name(
        comparison
    )

    print(
        "\nSelected model: "
        f"{best_model_name}"
    )

    fitted_model = train_selected_model(
        model_name=best_model_name,
        models=models,
        features=features_train,
        target=target_train,
    )

    test_metrics = evaluate_holdout(
        estimator=fitted_model,
        features=features_test,
        target=target_test,
        labels=sorted(
            dataframe[TARGET_COLUMN]
            .dropna()
            .unique()
            .tolist()
        ),
    )

    print("\nHoldout-test performance:")
    print(
        "  Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        "  Balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )
    print(
        "  Macro F1: "
        f"{test_metrics['f1_macro']:.4f}"
    )

    ensure_artifact_directories(
        models_directory=models_directory,
        metrics_directory=metrics_directory,
    )

    model_path = models_directory / BEST_MODEL_FILENAME
    comparison_path = (
        metrics_directory / MODEL_COMPARISON_FILENAME
    )
    test_metrics_path = (
        metrics_directory / TEST_METRICS_FILENAME
    )
    metadata_path = (
        metrics_directory / MODEL_METADATA_FILENAME
    )

    best_row = comparison.iloc[0]

    metadata = build_model_metadata(
        model_name=best_model_name,
        target_column=TARGET_COLUMN,
        feature_columns=MODEL_FEATURES,
        training_rows=len(features_train),
        test_rows=len(features_test),
        selected_metric=SELECTION_METRIC,
        selected_metric_value=float(
            best_row[SELECTION_METRIC]
        ),
        additional_metadata={
            "processed_filename": processed_filename,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "n_jobs": N_JOBS,
            "candidate_models": list(models),
            "target_classes": sorted(
                int(value)
                for value in target_train.unique()
            ),
            "holdout_accuracy": (
                test_metrics["accuracy"]
            ),
            "holdout_balanced_accuracy": (
                test_metrics["balanced_accuracy"]
            ),
            "holdout_f1_macro": (
                test_metrics["f1_macro"]
            ),
        },
    )

    save_model(
        estimator=fitted_model,
        output_path=model_path,
    )

    save_model_comparison(
        comparison=comparison,
        output_path=comparison_path,
    )

    save_test_metrics(
        metrics=test_metrics,
        output_path=test_metrics_path,
    )

    save_model_metadata(
        metadata=metadata,
        output_path=metadata_path,
    )

    print("\nArtifacts saved:")
    print(f"  Model: {model_path}")
    print(f"  Comparison: {comparison_path}")
    print(f"  Test metrics: {test_metrics_path}")
    print(f"  Metadata: {metadata_path}")

    return {
        "best_model_name": best_model_name,
        "model_path": model_path,
        "comparison_path": comparison_path,
        "test_metrics_path": test_metrics_path,
        "metadata_path": metadata_path,
        "comparison": comparison,
        "test_metrics": test_metrics,
    }


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate passenger-class "
            "classification models."
        )
    )

    parser.add_argument(
        "--processed-filename",
        default=DEFAULT_PROCESSED_FILENAME,
        help=(
            "Filename inside data/processed. "
            f"Default: {DEFAULT_PROCESSED_FILENAME}"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the command-line training workflow.
    """
    arguments = parse_arguments()

    run_training(
        processed_filename=(
            arguments.processed_filename
        ),
    )


if __name__ == "__main__":
    main()