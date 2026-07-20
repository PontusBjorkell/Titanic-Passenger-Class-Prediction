"""Generate reusable model-training reports and evaluation figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.pipeline import Pipeline

from titanic_passenger_class_prediction.config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    N_JOBS,
    PERMUTATION_IMPORTANCE_PATH,
    PERMUTATION_IMPORTANCE_REPEATS,
    RANDOM_STATE,
    TOP_FEATURES_TO_DISPLAY,
    TRAINING_SUMMARY_PATH,
)
from titanic_passenger_class_prediction.visualization import (
    aggregate_model_feature_importance,
    plot_aggregated_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_importance,
)


@dataclass(frozen=True)
class ReportArtifacts:
    """Paths created by a completed model-reporting workflow."""

    classification_report_path: Path
    training_summary_path: Path
    confusion_matrix_path: Path
    feature_importance_path: Path | None
    aggregated_feature_importance_path: Path | None
    permutation_importance_path: Path


def classification_report_dataframe(
    classification_report: Mapping[str, Any],
) -> pd.DataFrame:
    """Convert a scikit-learn classification report into a tidy table.

    Parameters
    ----------
    classification_report
        Mapping returned by ``classification_report(..., output_dict=True)``.

    Returns
    -------
    pandas.DataFrame
        Report table with one row per class or aggregate metric.
    """
    if not classification_report:
        raise ValueError("classification_report cannot be empty.")

    rows: list[dict[str, Any]] = []

    for label, values in classification_report.items():
        if isinstance(values, Mapping):
            row = {"label": str(label), **dict(values)}
        else:
            row = {
                "label": str(label),
                "precision": None,
                "recall": None,
                "f1-score": float(values),
                "support": None,
            }

        rows.append(row)

    dataframe = pd.DataFrame(rows)
    expected_columns = [
        "label",
        "precision",
        "recall",
        "f1-score",
        "support",
    ]

    for column in expected_columns:
        if column not in dataframe.columns:
            dataframe[column] = None

    remaining_columns = [
        column
        for column in dataframe.columns
        if column not in expected_columns
    ]

    return dataframe.loc[:, expected_columns + remaining_columns]


def save_classification_report(
    classification_report: Mapping[str, Any],
    output_path: Path = CLASSIFICATION_REPORT_PATH,
) -> Path:
    """Save a classification report as CSV."""
    report_table = classification_report_dataframe(classification_report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_table.to_csv(output_path, index=False)

    return output_path


def write_training_summary(
    *,
    model_name: str,
    comparison: pd.DataFrame,
    test_metrics: Mapping[str, Any],
    training_rows: int,
    test_rows: int,
    top_features: Sequence[str] | None = None,
    output_path: Path = TRAINING_SUMMARY_PATH,
) -> Path:
    """Write a concise Markdown summary of model selection and evaluation."""
    if not model_name.strip():
        raise ValueError("model_name cannot be empty.")

    if comparison.empty:
        raise ValueError("comparison cannot be empty.")

    required_metrics = {
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
    }
    missing_metrics = sorted(required_metrics - set(test_metrics))

    if missing_metrics:
        raise ValueError(
            "test_metrics is missing required values: "
            f"{missing_metrics}"
        )

    if training_rows <= 0:
        raise ValueError("training_rows must be greater than zero.")

    if test_rows <= 0:
        raise ValueError("test_rows must be greater than zero.")

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

    if "model" in comparison.columns:
        selected_rows = comparison.loc[comparison["model"] == model_name]
    else:
        selected_rows = pd.DataFrame()

    selected_cv_f1: float | None = None
    if (
        not selected_rows.empty
        and "cv_f1_macro_mean" in selected_rows.columns
    ):
        selected_cv_f1 = float(
            selected_rows.iloc[0]["cv_f1_macro_mean"]
        )

    lines = [
        "# Model Training Summary",
        "",
        "## Selected model",
        "",
        f"**{model_name}**",
        "",
        "## Dataset",
        "",
        f"- Training rows: {training_rows:,}",
        f"- Holdout rows: {test_rows:,}",
        "",
        "## Cross-validation",
        "",
    ]

    if selected_cv_f1 is not None:
        lines.append(f"- Selected-model Macro F1: {selected_cv_f1:.4f}")
    else:
        lines.append("- Selected-model Macro F1: unavailable")

    lines.extend(
        [
            "",
            "## Holdout evaluation",
            "",
            "| Metric | Score |",
            "|---|---:|",
            f"| Accuracy | {float(test_metrics['accuracy']):.2%} |",
            (
                "| Balanced accuracy | "
                f"{float(test_metrics['balanced_accuracy']):.2%} |"
            ),
            f"| Macro F1 | {float(test_metrics['f1_macro']):.2%} |",
            "",
        ]
    )

    if top_features:
        lines.extend(["## Most influential original features", ""])
        lines.extend(
            f"{index}. {feature}"
            for index, feature in enumerate(top_features, start=1)
        )
        lines.append("")

    lines.extend(["## Candidate-model comparison", ""])

    if display_columns:
        lines.extend(_dataframe_to_markdown(comparison.loc[:, display_columns]))
    else:
        lines.append("No displayable comparison columns were available.")

    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    return output_path


def generate_visual_reports(
    *,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: Sequence[Any],
    labels: Sequence[Any] | None = None,
    display_labels: Sequence[str] | None = None,
    confusion_matrix_path: Path = CONFUSION_MATRIX_PATH,
    feature_importance_path: Path = FEATURE_IMPORTANCE_PATH,
    aggregated_feature_importance_path: Path | None = None,
    permutation_importance_path: Path = PERMUTATION_IMPORTANCE_PATH,
    top_n: int = TOP_FEATURES_TO_DISPLAY,
    permutation_repeats: int = PERMUTATION_IMPORTANCE_REPEATS,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = N_JOBS,
) -> tuple[Path, Path | None, Path | None, Path]:
    """Generate confusion-matrix and feature-importance figures.

    Model-derived feature importance is optional because not every estimator
    exposes coefficients or impurity-based importance values. Permutation
    importance remains available for any fitted predictive pipeline.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a scikit-learn Pipeline.")

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")

    if features.empty:
        raise ValueError("features cannot be empty.")

    if aggregated_feature_importance_path is None:
        aggregated_feature_importance_path = (
            feature_importance_path.with_name(
                "aggregated_feature_importance.png"
            )
        )

    predictions = pipeline.predict(features)

    saved_confusion_matrix_path = plot_confusion_matrix(
        y_true=target,
        y_pred=predictions,
        output_path=confusion_matrix_path,
        labels=labels,
        display_labels=display_labels,
    )

    saved_feature_importance_path: Path | None

    try:
        model_display_name = pipeline.named_steps["model"].__class__.__name__
        saved_feature_importance_path = plot_feature_importance(
            pipeline=pipeline,
            output_path=feature_importance_path,
            top_n=top_n,
            title=f"{model_display_name} Feature Importance",
        )
        saved_aggregated_feature_importance_path = (
            plot_aggregated_feature_importance(
                pipeline=pipeline,
                original_features=features.columns.tolist(),
                output_path=aggregated_feature_importance_path,
                top_n=top_n,
            )
        )
    except TypeError as exc:
        if "feature_importances_ or coef_" not in str(exc):
            raise
        saved_feature_importance_path = None
        saved_aggregated_feature_importance_path = None

    saved_permutation_importance_path = plot_permutation_importance(
        pipeline=pipeline,
        X=features,
        y=target,
        output_path=permutation_importance_path,
        top_n=top_n,
        n_repeats=permutation_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return (
        saved_confusion_matrix_path,
        saved_feature_importance_path,
        saved_aggregated_feature_importance_path,
        saved_permutation_importance_path,
    )


def generate_model_report(
    *,
    model_name: str,
    pipeline: Pipeline,
    comparison: pd.DataFrame,
    test_metrics: Mapping[str, Any],
    features_test: pd.DataFrame,
    target_test: Sequence[Any],
    training_rows: int,
    labels: Sequence[Any] | None = None,
    display_labels: Sequence[str] | None = None,
    classification_report_path: Path = CLASSIFICATION_REPORT_PATH,
    training_summary_path: Path = TRAINING_SUMMARY_PATH,
    confusion_matrix_path: Path = CONFUSION_MATRIX_PATH,
    feature_importance_path: Path = FEATURE_IMPORTANCE_PATH,
    aggregated_feature_importance_path: Path | None = None,
    permutation_importance_path: Path = PERMUTATION_IMPORTANCE_PATH,
    top_n: int = TOP_FEATURES_TO_DISPLAY,
    permutation_repeats: int = PERMUTATION_IMPORTANCE_REPEATS,
    random_state: int = RANDOM_STATE,
    n_jobs: int | None = N_JOBS,
) -> ReportArtifacts:
    """Generate all tabular, Markdown, and visual training reports."""
    classification_report = test_metrics.get("classification_report")

    if not isinstance(classification_report, Mapping):
        raise ValueError(
            "test_metrics must contain a classification_report mapping."
        )

    saved_classification_report_path = save_classification_report(
        classification_report=classification_report,
        output_path=classification_report_path,
    )

    (
        saved_confusion_matrix_path,
        saved_feature_importance_path,
        saved_aggregated_feature_importance_path,
        saved_permutation_importance_path,
    ) = generate_visual_reports(
        pipeline=pipeline,
        features=features_test,
        target=target_test,
        labels=labels,
        display_labels=display_labels,
        confusion_matrix_path=confusion_matrix_path,
        feature_importance_path=feature_importance_path,
        aggregated_feature_importance_path=(
            aggregated_feature_importance_path
        ),
        permutation_importance_path=permutation_importance_path,
        top_n=top_n,
        permutation_repeats=permutation_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    top_features: list[str] = []
    if saved_aggregated_feature_importance_path is not None:
        aggregated_table = aggregate_model_feature_importance(
            pipeline=pipeline,
            original_features=features_test.columns.tolist(),
        )
        top_features = (
            aggregated_table.head(5)["feature"].astype(str).tolist()
        )

    saved_training_summary_path = write_training_summary(
        model_name=model_name,
        comparison=comparison,
        test_metrics=test_metrics,
        training_rows=training_rows,
        test_rows=len(features_test),
        top_features=top_features,
        output_path=training_summary_path,
    )

    return ReportArtifacts(
        classification_report_path=saved_classification_report_path,
        training_summary_path=saved_training_summary_path,
        confusion_matrix_path=saved_confusion_matrix_path,
        feature_importance_path=saved_feature_importance_path,
        aggregated_feature_importance_path=(
            saved_aggregated_feature_importance_path
        ),
        permutation_importance_path=saved_permutation_importance_path,
    )


def _dataframe_to_markdown(dataframe: pd.DataFrame) -> list[str]:
    """Render a compact DataFrame as Markdown without optional dependencies."""
    headers = [str(column) for column in dataframe.columns]
    rows = [
        [_format_markdown_value(value) for value in row]
        for row in dataframe.itertuples(index=False, name=None)
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)

    return lines


def _format_markdown_value(value: Any) -> str:
    """Format scalar values for stable Markdown output."""
    if pd.isna(value):
        return ""

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value).replace("|", "\\|")
