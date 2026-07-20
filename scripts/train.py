"""Train, tune, evaluate, report, and persist passenger-class models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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
from titanic_passenger_class_prediction.data import load_processed_data
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
from titanic_passenger_class_prediction.reporting import (
    ReportArtifacts,
    generate_model_report,
)
from titanic_passenger_class_prediction.tuning import (
    DEFAULT_SEARCH_ITERATIONS,
    TuningResult,
    tune_model,
)


SELECTION_METRIC = "cv_f1_macro_mean"
TUNING_RESULTS_FILENAME = "randomized_search_results.csv"
BEST_PARAMETERS_FILENAME = "best_hyperparameters.json"


def validate_training_dataframe(dataframe: pd.DataFrame) -> None:
    """Validate that the prepared dataset can support model training."""
    if dataframe.empty:
        raise ValueError("The processed training dataset is empty.")

    required_columns = {TARGET_COLUMN, *MODEL_FEATURES}
    missing_columns = sorted(required_columns - set(dataframe.columns))

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified train-test split."""
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
    """Clone and fit one estimator from the candidate-model registry."""
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}")

    estimator = clone(models[model_name])
    estimator.fit(features, target)

    return estimator


def select_model_variant(
    baseline_estimator: BaseEstimator,
    baseline_cv_score: float,
    tuning_result: TuningResult,
) -> tuple[BaseEstimator, str, float]:
    """
    Select the baseline or tuned variant using training-only CV Macro F1.

    The baseline wins ties. The holdout set is not used for model selection.
    """
    if tuning_result.best_cv_score > baseline_cv_score:
        return (
            tuning_result.best_estimator,
            "tuned",
            float(tuning_result.best_cv_score),
        )

    return baseline_estimator, "baseline", float(baseline_cv_score)


def save_tuning_results(
    tuning_result: TuningResult,
    metrics_directory: Path,
) -> tuple[Path, Path]:
    """Save the complete search table and the best parameter configuration."""
    metrics_directory.mkdir(parents=True, exist_ok=True)

    results_path = metrics_directory / TUNING_RESULTS_FILENAME
    parameters_path = metrics_directory / BEST_PARAMETERS_FILENAME

    tuning_result.search_results.to_csv(results_path, index=False)

    payload = {
        "model_name": tuning_result.model_name,
        "best_cv_f1_macro": float(tuning_result.best_cv_score),
        "best_parameters": tuning_result.best_parameters,
    }

    with parameters_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=_json_default)

    return results_path, parameters_path


def _json_default(value: Any) -> Any:
    """Convert common NumPy/pandas scalars and paths for JSON output."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )


def _print_model_comparison(comparison: pd.DataFrame) -> None:
    """Print the most useful cross-validation columns."""
    preferred_columns = [
        "model",
        "cv_accuracy_mean",
        "cv_balanced_accuracy_mean",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
    ]
    display_columns = [
        column for column in preferred_columns if column in comparison.columns
    ]

    print()
    print(
        comparison.loc[:, display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


def _print_holdout_metrics(test_metrics: dict[str, Any]) -> None:
    """Print final holdout metrics."""
    print("\nHoldout-test performance:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(
        "  Balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )
    print(f"  Macro F1: {test_metrics['f1_macro']:.4f}")


def _print_report_artifacts(report_artifacts: ReportArtifacts) -> None:
    """Print all report paths supported by the current reporting module."""
    print("\nReports generated:")
    print(
        "  Classification report: "
        f"{report_artifacts.classification_report_path}"
    )
    print(
        "  Training summary: "
        f"{report_artifacts.training_summary_path}"
    )
    print(
        "  Confusion matrix: "
        f"{report_artifacts.confusion_matrix_path}"
    )

    if report_artifacts.feature_importance_path is not None:
        print(
            "  Feature importance: "
            f"{report_artifacts.feature_importance_path}"
        )

    aggregated_path = getattr(
        report_artifacts,
        "aggregated_feature_importance_path",
        None,
    )
    if aggregated_path is not None:
        print(f"  Aggregated feature importance: {aggregated_path}")

    print(
        "  Permutation importance: "
        f"{report_artifacts.permutation_importance_path}"
    )


def run_training(
    processed_filename: str = DEFAULT_PROCESSED_FILENAME,
    models_directory: Path = MODELS_DIR,
    metrics_directory: Path = METRICS_DIR,
    tune_selected_model: bool = True,
    tuning_iterations: int = DEFAULT_SEARCH_ITERATIONS,
) -> dict[str, Any]:
    """
    Execute baseline comparison, optional tuning, final evaluation, and saving.

    Model selection uses cross-validation on the training partition only.
    The holdout partition is evaluated once after the final variant is chosen.
    """
    if tuning_iterations <= 0:
        raise ValueError("tuning_iterations must be greater than zero.")

    print("Loading processed passenger data...")
    dataframe = load_processed_data(filename=processed_filename)
    validate_training_dataframe(dataframe)

    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = split_training_data(dataframe)

    print(f"Training rows: {len(features_train):,}")
    print(f"Test rows: {len(features_test):,}")

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
    _print_model_comparison(comparison)

    best_model_name = select_best_model_name(comparison)
    best_row = comparison.iloc[0]
    baseline_cv_score = float(best_row[SELECTION_METRIC])

    print(f"\nSelected baseline model: {best_model_name}")

    baseline_model = train_selected_model(
        model_name=best_model_name,
        models=models,
        features=features_train,
        target=target_train,
    )

    final_model = baseline_model
    selected_variant = "baseline"
    selected_cv_score = baseline_cv_score

    tuning_result: TuningResult | None = None
    tuning_results_path: Path | None = None
    best_parameters_path: Path | None = None

    if tune_selected_model:
        print(
            "\nTuning selected model with "
            f"{tuning_iterations} randomized configurations..."
        )

        try:
            tuning_result = tune_model(
                model_name=best_model_name,
                estimator=clone(models[best_model_name]),
                features=features_train,
                target=target_train,
                n_iter=tuning_iterations,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            )
        except ValueError as error:
            # Some candidate models may intentionally have no tuning space.
            print(f"Tuning skipped: {error}")
        else:
            (
                final_model,
                selected_variant,
                selected_cv_score,
            ) = select_model_variant(
                baseline_estimator=baseline_model,
                baseline_cv_score=baseline_cv_score,
                tuning_result=tuning_result,
            )

            print(f"  Baseline CV Macro F1: {baseline_cv_score:.4f}")
            print(
                "  Tuned CV Macro F1: "
                f"{tuning_result.best_cv_score:.4f}"
            )
            print(f"  Selected variant: {selected_variant}")

    if not isinstance(final_model, Pipeline):
        raise TypeError(
            "The selected estimator must be a scikit-learn Pipeline "
            "so preprocessing and prediction remain coupled."
        )

    labels = sorted(
        dataframe[TARGET_COLUMN].dropna().unique().tolist()
    )

    test_metrics = evaluate_holdout(
        estimator=final_model,
        features=features_test,
        target=target_test,
        labels=labels,
    )
    _print_holdout_metrics(test_metrics)

    ensure_artifact_directories(
        models_directory=models_directory,
        metrics_directory=metrics_directory,
    )

    model_path = models_directory / BEST_MODEL_FILENAME
    comparison_path = metrics_directory / MODEL_COMPARISON_FILENAME
    test_metrics_path = metrics_directory / TEST_METRICS_FILENAME
    metadata_path = metrics_directory / MODEL_METADATA_FILENAME

    if tuning_result is not None:
        (
            tuning_results_path,
            best_parameters_path,
        ) = save_tuning_results(
            tuning_result=tuning_result,
            metrics_directory=metrics_directory,
        )

    metadata = build_model_metadata(
        model_name=best_model_name,
        target_column=TARGET_COLUMN,
        feature_columns=MODEL_FEATURES,
        training_rows=len(features_train),
        test_rows=len(features_test),
        selected_metric=SELECTION_METRIC,
        selected_metric_value=selected_cv_score,
        additional_metadata={
            "processed_filename": processed_filename,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "n_jobs": N_JOBS,
            "candidate_models": list(models),
            "target_classes": [int(value) for value in labels],
            "model_variant": selected_variant,
            "baseline_cv_f1_macro": baseline_cv_score,
            "tuning_enabled": tune_selected_model,
            "tuning_iterations": (
                tuning_iterations if tune_selected_model else 0
            ),
            "tuned_cv_f1_macro": (
                float(tuning_result.best_cv_score)
                if tuning_result is not None
                else None
            ),
            "best_hyperparameters": (
                tuning_result.best_parameters
                if tuning_result is not None
                else {}
            ),
            "holdout_accuracy": test_metrics["accuracy"],
            "holdout_balanced_accuracy": (
                test_metrics["balanced_accuracy"]
            ),
            "holdout_f1_macro": test_metrics["f1_macro"],
        },
    )

    save_model(estimator=final_model, output_path=model_path)
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

    report_model_name = f"{best_model_name} ({selected_variant})"
    report_artifacts = generate_model_report(
        model_name=report_model_name,
        pipeline=final_model,
        comparison=comparison,
        test_metrics=test_metrics,
        features_test=features_test,
        target_test=target_test,
        training_rows=len(features_train),
        labels=labels,
        display_labels=[
            f"Class {int(label)}"
            for label in labels
        ],
    )

    print("\nArtifacts saved:")
    print(f"  Model: {model_path}")
    print(f"  Comparison: {comparison_path}")
    print(f"  Test metrics: {test_metrics_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Selected model variant: {selected_variant}")
    print(f"  Selected CV Macro F1: {selected_cv_score:.4f}")

    if tuning_results_path is not None:
        print(f"  Randomized-search results: {tuning_results_path}")
    if best_parameters_path is not None:
        print(f"  Best hyperparameters: {best_parameters_path}")

    _print_report_artifacts(report_artifacts)

    return {
        "best_model_name": best_model_name,
        "selected_variant": selected_variant,
        "selected_cv_score": selected_cv_score,
        "baseline_cv_score": baseline_cv_score,
        "final_model": final_model,
        "tuning_result": tuning_result,
        "model_path": model_path,
        "comparison_path": comparison_path,
        "test_metrics_path": test_metrics_path,
        "metadata_path": metadata_path,
        "tuning_results_path": tuning_results_path,
        "best_parameters_path": best_parameters_path,
        "report_artifacts": report_artifacts,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "metadata": metadata,
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train, tune, evaluate, report, and persist "
            "passenger-class classification models."
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
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help=(
            "Skip randomized hyperparameter tuning and use the "
            "strongest baseline model."
        ),
    )
    parser.add_argument(
        "--tuning-iterations",
        type=int,
        default=DEFAULT_SEARCH_ITERATIONS,
        help=(
            "Number of randomized parameter configurations. "
            f"Default: {DEFAULT_SEARCH_ITERATIONS}"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the command-line training workflow."""
    arguments = parse_arguments()

    run_training(
        processed_filename=arguments.processed_filename,
        tune_selected_model=not arguments.skip_tuning,
        tuning_iterations=arguments.tuning_iterations,
    )


if __name__ == "__main__":
    main()
