
"""Main Streamlit entry point for the Titanic passenger-class project.

Run from the repository root with:

    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Ensure the src-layout package is importable when Streamlit is launched
# directly from the repository root.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from titanic_passenger_class_prediction.config import (  # noqa: E402
    DATABASE_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    PROCESSED_DATA_PATH,
    REPORTS_DIR,
    TEST_METRICS_PATH,
)

from utils import (  # noqa: E402
    apply_global_styles,
    artifact_status_table,
    format_metric_value,
    load_json,
    load_table,
    render_architecture,
    render_page_header,
    safe_metric,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Titanic Passenger Class Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🚢 Titanic ML Project")

    st.markdown(
        """
        A production-inspired portfolio project combining:

        - exploratory analysis,
        - SQL analytics,
        - machine learning,
        - external validation,
        - automated testing,
        - and interactive prediction.
        """
    )

    st.divider()

    st.subheader("Navigation")
    st.caption(
        "Use the page menu above to explore the data, SQL analyses, "
        "model performance, and prediction demo."
    )

    st.divider()

    st.subheader("Repository status")

    status = artifact_status_table(
        {
            "Processed data": PROCESSED_DATA_PATH,
            "SQLite database": DATABASE_PATH,
            "Model comparison": MODEL_COMPARISON_PATH,
            "Test metrics": TEST_METRICS_PATH,
            "Model metadata": MODEL_METADATA_PATH,
            "SQL reports": REPORTS_DIR / "sql",
        }
    )

    st.dataframe(
        status,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

render_page_header(
    title="Titanic Passenger Class Prediction",
    subtitle=(
        "End-to-end machine-learning pipeline with SQL analytics, "
        "external validation, and interactive inference"
    ),
    icon="🚢",
)

st.markdown(
    """
    This application presents the outputs of a modular machine-learning
    repository. Data preparation, feature engineering, model training,
    evaluation, SQL reporting, and prediction are implemented as reusable
    Python components. The dashboard focuses on communicating those outputs
    clearly and interactively.
    """
)


# ---------------------------------------------------------------------------
# Main metrics
# ---------------------------------------------------------------------------

metrics = load_json(TEST_METRICS_PATH, default={})
metadata = load_json(MODEL_METADATA_PATH, default={})
comparison = load_table(MODEL_COMPARISON_PATH)

metric_columns = st.columns(4)

with metric_columns[0]:
    st.metric(
        "Selected model",
        str(metadata.get("model_name", "Unavailable")),
    )

with metric_columns[1]:
    st.metric(
        "Hold-out accuracy",
        format_metric_value(
            safe_metric(metrics, "accuracy"),
            percentage=True,
        ),
    )

with metric_columns[2]:
    st.metric(
        "Balanced accuracy",
        format_metric_value(
            safe_metric(metrics, "balanced_accuracy"),
            percentage=True,
        ),
    )

with metric_columns[3]:
    st.metric(
        "Macro F1",
        format_metric_value(
            safe_metric(metrics, "f1_macro"),
            percentage=True,
        ),
    )


# ---------------------------------------------------------------------------
# Project highlights
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Project highlights")

highlight_columns = st.columns(3)

with highlight_columns[0]:
    st.markdown(
        """
        ### 🧱 Modular architecture

        The repository separates data preparation, feature engineering,
        modelling, evaluation, persistence, reporting, SQL, tests, and the
        application layer.
        """
    )

with highlight_columns[1]:
    st.markdown(
        """
        ### 🗄️ SQL analytics

        Twenty named SQL analyses are executed against a reproducible SQLite
        database and exported as reusable reporting tables.
        """
    )

with highlight_columns[2]:
    st.markdown(
        """
        ### 🌍 External validation

        The trained model is evaluated on a separately prepared,
        non-overlapping Titanic dataset in addition to the internal hold-out
        set.
        """
    )


# ---------------------------------------------------------------------------
# Model comparison preview
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Model-selection overview")

if comparison.empty:
    st.info(
        "The model-comparison artifact is not available yet. "
        "Run the training workflow to generate it."
    )
else:
    display_columns = [
        column
        for column in [
            "model",
            "cv_accuracy_mean",
            "cv_balanced_accuracy_mean",
            "cv_f1_macro_mean",
            "cv_f1_macro_std",
        ]
        if column in comparison.columns
    ]

    preview = (
        comparison.loc[:, display_columns]
        if display_columns
        else comparison
    )

    if "cv_f1_macro_mean" in preview.columns:
        preview = preview.sort_values(
            "cv_f1_macro_mean",
            ascending=False,
        )

    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True,
    )

    if "cv_f1_macro_mean" in preview.columns and "model" in preview.columns:
        chart_data = (
            preview.set_index("model")[["cv_f1_macro_mean"]]
        )
        st.bar_chart(chart_data)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Repository architecture")

render_architecture()

st.caption(
    "The Streamlit application reads saved datasets, database outputs, "
    "metrics, figures, and the fitted model rather than duplicating the "
    "training workflow."
)


# ---------------------------------------------------------------------------
# Page guide
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Application pages")

page_columns = st.columns(4)

with page_columns[0]:
    st.markdown(
        """
        ### 1 · Data Explorer

        Inspect passenger records, distributions, missingness, and class
        structure using interactive filters.
        """
    )

with page_columns[1]:
    st.markdown(
        """
        ### 2 · SQL Analytics

        Browse the twenty exported SQL analyses, review their descriptions,
        visualize results, and download tables.
        """
    )

with page_columns[2]:
    st.markdown(
        """
        ### 3 · Model Performance

        Examine model comparison, hold-out metrics, confusion matrices,
        classification reports, feature importance, and external validation.
        """
    )

with page_columns[3]:
    st.markdown(
        """
        ### 4 · Predict Passenger Class

        Enter passenger information, run the saved pipeline, and inspect class
        probabilities and confidence.
        """
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Built with Python, pandas, scikit-learn, SQLite, pytest, and Streamlit."
)
