
"""Model-performance dashboard for the Titanic Streamlit application."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from titanic_passenger_class_prediction.config import (  # noqa: E402
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    TEST_METRICS_PATH,
)

from utils import (  # noqa: E402
    apply_global_styles,
    dataframe_to_csv_bytes,
    format_metric_value,
    load_json,
    load_table,
    normalize_classification_report,
    render_missing_artifact,
    render_page_header,
    safe_metric,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Model Performance · Titanic",
    page_icon="📈",
    layout="wide",
)

apply_global_styles()


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

BEST_HYPERPARAMETERS_PATH = (
    METRICS_DIR / "best_hyperparameters.json"
)
RANDOMIZED_SEARCH_RESULTS_PATH = (
    METRICS_DIR / "randomized_search_results.csv"
)

INTERNAL_CLASSIFICATION_REPORT_CANDIDATES = [
    PROJECT_ROOT / "reports" / "classification_report.csv",
    METRICS_DIR / "classification_report.csv",
]

EXTERNAL_PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "predictions"
    / "external_nonoverlap"
)

EXTERNAL_METRICS_PATH = (
    EXTERNAL_PREDICTIONS_DIR
    / "external_test_metrics.json"
)
EXTERNAL_REPORT_PATH = (
    EXTERNAL_PREDICTIONS_DIR
    / "external_classification_report.csv"
)
EXTERNAL_PREDICTIONS_PATH = (
    EXTERNAL_PREDICTIONS_DIR
    / "test_predictions.csv"
)
EXTERNAL_SUMMARY_PATH = (
    EXTERNAL_PREDICTIONS_DIR
    / "test_prediction_summary.json"
)

FIGURE_PATHS = {
    "Internal confusion matrix": FIGURES_DIR / "confusion_matrix.png",
    "Aggregated feature importance": (
        FIGURES_DIR / "aggregated_feature_importance.png"
    ),
    "Encoded feature importance": (
        FIGURES_DIR / "feature_importance.png"
    ),
    "Permutation importance": (
        FIGURES_DIR / "permutation_importance.png"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_first_existing(
    candidates: list[Path],
) -> Path | None:
    """Return the first existing file from a candidate list."""
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def scalar_metrics(
    report: dict[str, Any] | None,
) -> dict[str, float]:
    """Extract finite scalar numeric values from a JSON report."""
    if not isinstance(report, dict):
        return {}

    output: dict[str, float] = {}

    for key, value in report.items():
        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)):
            numeric = float(value)

            if np.isfinite(numeric):
                output[key] = numeric

    return output


def metrics_table(
    metrics: dict[str, float],
) -> pd.DataFrame:
    """Convert a scalar metric mapping into a readable table."""
    return pd.DataFrame(
        {
            "Metric": [
                key.replace("_", " ").title()
                for key in metrics
            ],
            "Score": list(metrics.values()),
        }
    )


def confusion_matrix_from_report(
    report: dict[str, Any] | None,
) -> np.ndarray | None:
    """Extract a square confusion matrix from a JSON report."""
    if not isinstance(report, dict):
        return None

    matrix = np.asarray(
        report.get("confusion_matrix", [])
    )

    if matrix.ndim != 2:
        return None

    if matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.size == 0:
        return None

    return matrix


def render_confusion_matrix(
    matrix: np.ndarray,
    title: str,
    labels: list[str] | None = None,
) -> None:
    """Render a confusion matrix as a readable dataframe and heatmap."""
    if labels is None:
        labels = [
            f"Class {index + 1}"
            for index in range(matrix.shape[0])
        ]

    matrix_frame = pd.DataFrame(
        matrix,
        index=[
            f"Actual {label}"
            for label in labels
        ],
        columns=[
            f"Predicted {label}"
            for label in labels
        ],
    )

    st.markdown(f"#### {title}")

    st.dataframe(
        matrix_frame.style.background_gradient(
            axis=None,
        ).format("{:.0f}"),
        use_container_width=True,
    )


def render_metric_cards(
    report: dict[str, Any],
    prefix: str = "",
) -> None:
    """Render the main model metrics in compact cards."""
    columns = st.columns(4)

    with columns[0]:
        st.metric(
            f"{prefix}Accuracy".strip(),
            format_metric_value(
                safe_metric(report, "accuracy"),
                percentage=True,
            ),
        )

    with columns[1]:
        st.metric(
            f"{prefix}Balanced accuracy".strip(),
            format_metric_value(
                safe_metric(
                    report,
                    "balanced_accuracy",
                ),
                percentage=True,
            ),
        )

    with columns[2]:
        st.metric(
            f"{prefix}Macro F1".strip(),
            format_metric_value(
                safe_metric(report, "f1_macro"),
                percentage=True,
            ),
        )

    with columns[3]:
        weighted_f1 = (
            safe_metric(report, "f1_weighted")
            or safe_metric(report, "weighted_f1")
        )

        st.metric(
            f"{prefix}Weighted F1".strip(),
            format_metric_value(
                weighted_f1,
                percentage=True,
            ),
        )


def render_image_artifact(
    title: str,
    path: Path,
    caption: str,
) -> None:
    """Render a saved figure or a helpful missing-artifact message."""
    st.markdown(f"### {title}")

    if path.exists():
        st.image(
            str(path),
            use_container_width=True,
            caption=caption,
        )
    else:
        st.info(
            f"The figure is not available at `{path}`."
        )


def probability_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return class-probability columns from prediction artifacts."""
    return [
        column
        for column in dataframe.columns
        if (
            column.lower().startswith("probabilityclass")
            or "probability_class" in column.lower()
        )
    ]


def confidence_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Find the saved prediction-confidence column."""
    candidates = [
        "PredictionConfidence",
        "prediction_confidence",
        "Confidence",
        "confidence",
        "MaxProbability",
        "max_probability",
    ]

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    return None


def format_model_comparison(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Select and order useful model-comparison columns."""
    if dataframe.empty:
        return dataframe

    preferred = [
        "model",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_balanced_accuracy_mean",
        "cv_balanced_accuracy_std",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
        "fit_time_mean",
    ]

    columns = [
        column
        for column in preferred
        if column in dataframe.columns
    ]

    output = (
        dataframe.loc[:, columns].copy()
        if columns
        else dataframe.copy()
    )

    if "cv_f1_macro_mean" in output.columns:
        output = output.sort_values(
            "cv_f1_macro_mean",
            ascending=False,
        )

    return output


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------

render_page_header(
    title="Model Performance",
    subtitle=(
        "Review model selection, internal hold-out evaluation, external "
        "validation, feature importance, and prediction confidence using "
        "the saved artifacts produced by the training pipeline."
    ),
    icon="📈",
)

model_comparison = load_table(
    MODEL_COMPARISON_PATH
)
model_metadata = load_json(
    MODEL_METADATA_PATH,
    default={},
)
internal_metrics = load_json(
    TEST_METRICS_PATH,
    default={},
)
best_hyperparameters = load_json(
    BEST_HYPERPARAMETERS_PATH,
    default={},
)
randomized_search_results = load_table(
    RANDOMIZED_SEARCH_RESULTS_PATH
)

internal_report_path = find_first_existing(
    INTERNAL_CLASSIFICATION_REPORT_CANDIDATES
)
internal_classification_report = (
    normalize_classification_report(
        load_table(internal_report_path)
    )
    if internal_report_path is not None
    else pd.DataFrame()
)

external_metrics = load_json(
    EXTERNAL_METRICS_PATH,
    default={},
)
external_classification_report = normalize_classification_report(
    load_table(EXTERNAL_REPORT_PATH)
)
external_predictions = load_table(
    EXTERNAL_PREDICTIONS_PATH
)
external_summary = load_json(
    EXTERNAL_SUMMARY_PATH,
    default={},
)


# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------

selected_model_raw = str(
    model_metadata.get(
        "model_name",
        "Unavailable",
    )
)

selected_model = (
    selected_model_raw
    if len(selected_model_raw) <= 28
    else selected_model_raw[:25] + "..."
)

summary_columns = st.columns(5)

with summary_columns[0]:
    st.metric(
        "Selected model",
        selected_model,
    )

with summary_columns[1]:
    st.metric(
        "Internal accuracy",
        format_metric_value(
            safe_metric(
                internal_metrics,
                "accuracy",
            ),
            percentage=True,
        ),
    )

with summary_columns[2]:
    st.metric(
        "Internal Macro F1",
        format_metric_value(
            safe_metric(
                internal_metrics,
                "f1_macro",
            ),
            percentage=True,
        ),
    )

with summary_columns[3]:
    st.metric(
        "External accuracy",
        format_metric_value(
            safe_metric(
                external_metrics,
                "accuracy",
            ),
            percentage=True,
        ),
    )

with summary_columns[4]:
    st.metric(
        "External Macro F1",
        format_metric_value(
            safe_metric(
                external_metrics,
                "f1_macro",
            ),
            percentage=True,
        ),
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

(
    overview_tab,
    comparison_tab,
    internal_tab,
    external_tab,
    importance_tab,
    confidence_tab,
) = st.tabs(
    [
        "Overview",
        "Model comparison",
        "Internal evaluation",
        "External validation",
        "Feature importance",
        "Prediction confidence",
    ]
)


# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------

with overview_tab:
    st.subheader("Selected model")

    overview_left, overview_right = st.columns(
        [1.15, 0.85]
    )

    with overview_left:
        metadata_rows = []

        for key in [
            "model_name",
            "selected_metric",
            "selected_metric_value",
            "training_rows",
            "test_rows",
            "target_column",
            "created_at_utc",
        ]:
            if key in model_metadata:
                metadata_rows.append(
                    {
                        "Field": (
                            key.replace("_", " ").title()
                        ),
                        "Value": model_metadata[key],
                    }
                )

        if metadata_rows:
            st.dataframe(
                pd.DataFrame(metadata_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            render_missing_artifact(
                title="Model metadata",
                path=MODEL_METADATA_PATH,
                command="python scripts/train.py",
            )

    with overview_right:
        st.markdown("#### Best hyperparameters")

        if isinstance(
            best_hyperparameters,
            dict,
        ) and best_hyperparameters:
            hyperparameter_table = pd.DataFrame(
                {
                    "Hyperparameter": list(
                        best_hyperparameters.keys()
                    ),
                    "Value": [
                        str(value)
                        for value
                        in best_hyperparameters.values()
                    ],
                }
            )

            st.dataframe(
                hyperparameter_table,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No saved hyperparameter artifact is available."
            )

    st.markdown("#### Evaluation design")

    st.markdown(
        """
        The project distinguishes between two evaluation settings:

        - **Internal hold-out evaluation** measures performance on unseen rows
          drawn from the original training dataset.
        - **Independent external validation** applies the saved model to a
          separately prepared, non-overlapping Titanic dataset.

        Comparing these settings helps reveal whether the model generalizes
        beyond a single random split of the original source.
        """
    )

    internal_scalar = scalar_metrics(
        internal_metrics
    )
    external_scalar = scalar_metrics(
        external_metrics
    )

    shared_keys = [
        key
        for key in internal_scalar
        if key in external_scalar
    ]

    if shared_keys:
        comparison_frame = pd.DataFrame(
            {
                "Metric": [
                    key.replace("_", " ").title()
                    for key in shared_keys
                ],
                "Internal": [
                    internal_scalar[key]
                    for key in shared_keys
                ],
                "External": [
                    external_scalar[key]
                    for key in shared_keys
                ],
            }
        )

        comparison_frame["Change"] = (
            comparison_frame["External"]
            - comparison_frame["Internal"]
        )

        st.dataframe(
            comparison_frame.style.format(
                {
                    "Internal": "{:.4f}",
                    "External": "{:.4f}",
                    "Change": "{:+.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        chart_frame = comparison_frame.set_index(
            "Metric"
        )[["Internal", "External"]]

        st.bar_chart(chart_frame)


# ---------------------------------------------------------------------------
# Model comparison tab
# ---------------------------------------------------------------------------

with comparison_tab:
    st.subheader("Candidate-model comparison")

    comparison_display = format_model_comparison(
        model_comparison
    )

    if comparison_display.empty:
        render_missing_artifact(
            title="Model-comparison table",
            path=MODEL_COMPARISON_PATH,
            command="python scripts/train.py",
        )
    else:
        numeric_formatters = {
            column: "{:.4f}"
            for column
            in comparison_display.columns
            if column != "model"
            and pd.api.types.is_numeric_dtype(
                comparison_display[column]
            )
        }

        styled_comparison = comparison_display.style

        if numeric_formatters:
            styled_comparison = styled_comparison.format(
                numeric_formatters
            )

        st.dataframe(
            styled_comparison,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Download model-comparison table",
            data=dataframe_to_csv_bytes(
                model_comparison
            ),
            file_name=MODEL_COMPARISON_PATH.name,
            mime="text/csv",
            use_container_width=True,
        )

        chart_metric_options = [
            column
            for column
            in comparison_display.columns
            if column != "model"
            and pd.api.types.is_numeric_dtype(
                comparison_display[column]
            )
        ]

        if (
            "model" in comparison_display.columns
            and chart_metric_options
        ):
            selected_metric = st.selectbox(
                "Comparison metric",
                options=chart_metric_options,
                index=(
                    chart_metric_options.index(
                        "cv_f1_macro_mean"
                    )
                    if "cv_f1_macro_mean"
                    in chart_metric_options
                    else 0
                ),
            )

            chart_data = (
                comparison_display
                .set_index("model")[
                    [selected_metric]
                ]
            )

            st.bar_chart(chart_data)

    st.subheader("Randomized-search results")

    if randomized_search_results.empty:
        st.info(
            "No randomized-search result table is available."
        )
    else:
        if len(randomized_search_results) <= 5:
            result_limit = len(randomized_search_results)
            st.caption(
                f"All {result_limit} randomized-search result row(s) are displayed."
            )
        else:
            result_limit = st.slider(
                "Rows to display",
                min_value=5,
                max_value=min(
                    100,
                    len(randomized_search_results),
                ),
                value=min(
                    20,
                    len(randomized_search_results),
                ),
            )

        ranking_column = next(
            (
                column
                for column
                in randomized_search_results.columns
                if column.startswith("rank_test_")
            ),
            None,
        )

        search_display = (
            randomized_search_results.sort_values(
                ranking_column
            )
            if ranking_column
            else randomized_search_results
        )

        st.dataframe(
            search_display.head(result_limit),
            use_container_width=True,
            hide_index=True,
            height=500,
        )


# ---------------------------------------------------------------------------
# Internal evaluation tab
# ---------------------------------------------------------------------------

with internal_tab:
    st.subheader("Internal hold-out evaluation")

    if not isinstance(
        internal_metrics,
        dict,
    ) or not internal_metrics:
        render_missing_artifact(
            title="Internal evaluation metrics",
            path=TEST_METRICS_PATH,
            command="python scripts/train.py",
        )
    else:
        render_metric_cards(
            internal_metrics
        )

        scalar_table = metrics_table(
            scalar_metrics(internal_metrics)
        )

        if not scalar_table.empty:
            st.dataframe(
                scalar_table.style.format(
                    {"Score": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        internal_matrix = confusion_matrix_from_report(
            internal_metrics
        )

        if internal_matrix is not None:
            render_confusion_matrix(
                internal_matrix,
                title="Internal confusion matrix",
            )
        elif FIGURE_PATHS[
            "Internal confusion matrix"
        ].exists():
            st.image(
                str(
                    FIGURE_PATHS[
                        "Internal confusion matrix"
                    ]
                ),
                use_container_width=True,
                caption=(
                    "Internal hold-out confusion matrix"
                ),
            )
        else:
            st.info(
                "No internal confusion matrix was found."
            )

    st.markdown("#### Internal classification report")

    if internal_classification_report.empty:
        st.info(
            "No internal classification-report CSV was found. "
            "The scalar metrics and confusion matrix remain available."
        )
    else:
        st.dataframe(
            internal_classification_report,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# External validation tab
# ---------------------------------------------------------------------------

with external_tab:
    st.subheader("Independent external validation")

    st.markdown(
        """
        External validation applies the trained model to a separately
        prepared Titanic dataset after overlapping passenger records have
        been removed. This is a stricter test than evaluating only on the
        internal hold-out split.
        """
    )

    if not isinstance(
        external_metrics,
        dict,
    ) or not external_metrics:
        render_missing_artifact(
            title="External validation metrics",
            path=EXTERNAL_METRICS_PATH,
            command=(
                "python scripts/predict.py "
                "--input data/external/"
                "titanic_external_nonoverlap.csv "
                "--output-dir artifacts/predictions/"
                "external_nonoverlap"
            ),
        )
    else:
        render_metric_cards(
            external_metrics
        )

        external_scalar_table = metrics_table(
            scalar_metrics(external_metrics)
        )

        if not external_scalar_table.empty:
            st.dataframe(
                external_scalar_table.style.format(
                    {"Score": "{:.4f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        external_matrix = confusion_matrix_from_report(
            external_metrics
        )

        if external_matrix is not None:
            render_confusion_matrix(
                external_matrix,
                title="External confusion matrix",
            )

    st.markdown(
        "#### External classification report"
    )

    if external_classification_report.empty:
        st.info(
            "No external classification-report artifact is available."
        )
    else:
        st.dataframe(
            external_classification_report,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Download external classification report",
            data=dataframe_to_csv_bytes(
                external_classification_report
            ),
            file_name=EXTERNAL_REPORT_PATH.name,
            mime="text/csv",
            use_container_width=True,
        )

    if isinstance(
        external_summary,
        dict,
    ) and external_summary:
        st.markdown(
            "#### External prediction summary"
        )

        summary_table = pd.DataFrame(
            {
                "Statistic": [
                    key.replace("_", " ").title()
                    for key in external_summary
                ],
                "Value": list(
                    external_summary.values()
                ),
            }
        )

        st.dataframe(
            summary_table,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Feature-importance tab
# ---------------------------------------------------------------------------

with importance_tab:
    st.subheader("Model interpretation")

    st.markdown(
        """
        The figures below describe which variables the fitted model relies on.
        Model-derived importance and permutation importance answer related but
        different questions and should not be interpreted as causal effects.
        """
    )

    render_image_artifact(
        title="Aggregated feature importance",
        path=FIGURE_PATHS[
            "Aggregated feature importance"
        ],
        caption=(
            "Importance aggregated back to the original input columns."
        ),
    )

    render_image_artifact(
        title="Encoded feature importance",
        path=FIGURE_PATHS[
            "Encoded feature importance"
        ],
        caption=(
            "Importance of the transformed features used by the fitted model."
        ),
    )

    render_image_artifact(
        title="Permutation importance",
        path=FIGURE_PATHS[
            "Permutation importance"
        ],
        caption=(
            "Performance decrease after independently permuting each "
            "original input feature."
        ),
    )


# ---------------------------------------------------------------------------
# Prediction-confidence tab
# ---------------------------------------------------------------------------

with confidence_tab:
    st.subheader("External prediction confidence")

    if external_predictions.empty:
        render_missing_artifact(
            title="External prediction file",
            path=EXTERNAL_PREDICTIONS_PATH,
            command=(
                "python scripts/predict.py "
                "--input data/external/"
                "titanic_external_nonoverlap.csv "
                "--output-dir artifacts/predictions/"
                "external_nonoverlap"
            ),
        )
    else:
        confidence_name = confidence_column(
            external_predictions
        )
        probability_names = probability_columns(
            external_predictions
        )

        if (
            confidence_name is None
            and probability_names
        ):
            probability_frame = (
                external_predictions[
                    probability_names
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
            )

            external_predictions = (
                external_predictions.copy()
            )
            external_predictions[
                "PredictionConfidence"
            ] = probability_frame.max(axis=1)
            confidence_name = "PredictionConfidence"

        if confidence_name is None:
            st.info(
                "The prediction artifact does not contain a confidence "
                "column or class-probability columns."
            )
            st.dataframe(
                external_predictions,
                use_container_width=True,
                hide_index=True,
            )
        else:
            confidence_values = pd.to_numeric(
                external_predictions[
                    confidence_name
                ],
                errors="coerce",
            ).dropna()

            confidence_metric_columns = st.columns(
                4
            )

            with confidence_metric_columns[0]:
                st.metric(
                    "Mean confidence",
                    (
                        f"{confidence_values.mean():.1%}"
                        if not confidence_values.empty
                        else "Unavailable"
                    ),
                )

            with confidence_metric_columns[1]:
                st.metric(
                    "Median confidence",
                    (
                        f"{confidence_values.median():.1%}"
                        if not confidence_values.empty
                        else "Unavailable"
                    ),
                )

            with confidence_metric_columns[2]:
                low_confidence_column = next(
                    (
                        column
                        for column
                        in [
                            "LowConfidencePrediction",
                            "low_confidence_prediction",
                        ]
                        if column
                        in external_predictions.columns
                    ),
                    None,
                )

                if low_confidence_column:
                    low_confidence_count = int(
                        external_predictions[
                            low_confidence_column
                        ]
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    )
                else:
                    low_confidence_count = int(
                        (
                            pd.to_numeric(
                                external_predictions[
                                    confidence_name
                                ],
                                errors="coerce",
                            )
                            < 0.60
                        ).sum()
                    )

                st.metric(
                    "Low-confidence predictions",
                    f"{low_confidence_count:,}",
                )

            with confidence_metric_columns[3]:
                st.metric(
                    "Prediction rows",
                    f"{len(external_predictions):,}",
                )

            if not confidence_values.empty:
                histogram_data = pd.DataFrame(
                    {
                        "Confidence band": pd.cut(
                            confidence_values,
                            bins=[
                                0.0,
                                0.5,
                                0.6,
                                0.7,
                                0.8,
                                0.9,
                                1.0,
                            ],
                            include_lowest=True,
                        ).astype(str)
                    }
                )

                confidence_distribution = (
                    histogram_data[
                        "Confidence band"
                    ]
                    .value_counts()
                    .sort_index()
                    .rename_axis(
                        "Confidence band"
                    )
                    .to_frame("Predictions")
                )

                st.bar_chart(
                    confidence_distribution
                )

            st.markdown(
                "#### Lowest-confidence predictions"
            )

            sorted_predictions = (
                external_predictions.assign(
                    _confidence=pd.to_numeric(
                        external_predictions[
                            confidence_name
                        ],
                        errors="coerce",
                    )
                )
                .sort_values(
                    "_confidence",
                    ascending=True,
                )
                .drop(
                    columns="_confidence"
                )
            )

            display_columns = [
                column
                for column
                in [
                    "PassengerId",
                    "Name",
                    "Sex",
                    "Age",
                    "Fare",
                    "Embarked",
                    "PredictedPclass",
                    confidence_name,
                    "ProbabilityMargin",
                    "PredictionCorrect",
                ]
                if column
                in sorted_predictions.columns
            ]

            if probability_names:
                display_columns.extend(
                    [
                        column
                        for column
                        in probability_names
                        if column
                        not in display_columns
                    ]
                )

            if not display_columns:
                display_columns = list(
                    sorted_predictions.columns
                )

            if len(sorted_predictions) <= 5:
                row_limit = len(sorted_predictions)
                st.caption(
                    f"All {row_limit} available prediction row(s) are displayed."
                )
            else:
                row_limit = st.slider(
                    "Number of low-confidence rows",
                    min_value=5,
                    max_value=min(
                        50,
                        len(sorted_predictions),
                    ),
                    value=min(
                        15,
                        len(sorted_predictions),
                    ),
                )

            st.dataframe(
                sorted_predictions.loc[
                    :,
                    display_columns,
                ].head(row_limit),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                label="Download external predictions",
                data=dataframe_to_csv_bytes(
                    external_predictions
                ),
                file_name=EXTERNAL_PREDICTIONS_PATH.name,
                mime="text/csv",
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()

st.caption(
    "This page reads model artifacts generated by the repository's training, "
    "reporting, and external-prediction workflows. It does not retrain the "
    "model inside Streamlit."
)
