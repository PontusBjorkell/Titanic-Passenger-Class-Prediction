
"""Interactive SQL analytics browser for the Titanic Streamlit application."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
    REPORTS_DIR,
)

from utils import (  # noqa: E402
    apply_global_styles,
    dataframe_to_csv_bytes,
    discover_sql_reports,
    load_json,
    load_table,
    prettify_name,
    render_page_header,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SQL Analytics · Titanic",
    page_icon="🗄️",
    layout="wide",
)

apply_global_styles()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SQL_REPORTS_DIR = REPORTS_DIR / "sql"
MANIFEST_PATH = SQL_REPORTS_DIR / "analysis_manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_numeric_prefix(name: str) -> str:
    """Remove a leading numeric query prefix from a report name."""
    parts = name.split("_", maxsplit=1)

    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]

    return name


def report_title(path: Path) -> str:
    """Create a readable title from an exported SQL report filename."""
    return prettify_name(
        strip_numeric_prefix(path.stem)
    )


def load_manifest_metadata(
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    """Index manifest metadata by query name."""
    manifest = load_json(
        manifest_path,
        default={},
    )

    if not isinstance(manifest, dict):
        return {}

    queries = manifest.get("queries", [])

    if not isinstance(queries, list):
        return {}

    metadata: dict[str, dict[str, Any]] = {}

    for item in queries:
        if isinstance(item, dict) and item.get("name"):
            metadata[str(item["name"])] = item

    return metadata


def is_numeric_series(series: pd.Series) -> bool:
    """Return whether a series is suitable for numeric plotting."""
    if pd.api.types.is_numeric_dtype(series):
        return True

    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    return converted.notna().mean() >= 0.80


def numeric_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return plot-ready numeric columns."""
    return [
        column
        for column in dataframe.columns
        if is_numeric_series(dataframe[column])
    ]


def categorical_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """Return candidate label columns for charts."""
    return [
        column
        for column in dataframe.columns
        if column not in numeric_columns(dataframe)
    ]


def coerce_numeric_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Convert mostly numeric object columns to numeric values."""
    output = dataframe.copy()

    for column in output.columns:
        if is_numeric_series(output[column]):
            output[column] = pd.to_numeric(
                output[column],
                errors="coerce",
            )

    return output


def choose_default_label_column(
    dataframe: pd.DataFrame,
) -> str | None:
    """Select a sensible default label column."""
    categorical = categorical_columns(dataframe)

    if categorical:
        return categorical[0]

    if len(dataframe.columns) > 1:
        return str(dataframe.columns[0])

    return None


def choose_default_value_columns(
    dataframe: pd.DataFrame,
    label_column: str | None,
) -> list[str]:
    """Select sensible default numeric columns."""
    candidates = [
        column
        for column in numeric_columns(dataframe)
        if column != label_column
    ]

    return candidates[:3]


def render_auto_chart(
    dataframe: pd.DataFrame,
) -> None:
    """Render a configurable chart for a SQL result."""
    if dataframe.empty:
        st.info("The selected analysis returned no rows.")
        return

    chart_frame = coerce_numeric_columns(dataframe)
    available_numeric = numeric_columns(chart_frame)

    if not available_numeric:
        st.info(
            "This result is primarily categorical, so the table is the "
            "most informative presentation."
        )
        return

    st.markdown("#### Visualization")

    label_default = choose_default_label_column(chart_frame)

    chart_control_columns = st.columns([1, 1, 1])

    with chart_control_columns[0]:
        label_options = ["Row number"] + [
            str(column)
            for column in chart_frame.columns
            if column not in available_numeric
        ]

        if label_default and label_default in label_options:
            default_label_index = label_options.index(label_default)
        else:
            default_label_index = 0

        selected_label = st.selectbox(
            "Category or index",
            options=label_options,
            index=default_label_index,
            key="sql_chart_label",
        )

    with chart_control_columns[1]:
        default_values = choose_default_value_columns(
            chart_frame,
            None if selected_label == "Row number" else selected_label,
        )

        selected_values = st.multiselect(
            "Numeric values",
            options=available_numeric,
            default=default_values,
            key="sql_chart_values",
        )

    with chart_control_columns[2]:
        chart_type = st.selectbox(
            "Chart type",
            options=[
                "Bar chart",
                "Line chart",
                "Area chart",
            ],
            key="sql_chart_type",
        )

    if not selected_values:
        st.info("Select at least one numeric column to create a chart.")
        return

    plot_data = chart_frame.loc[:, selected_values].copy()

    if selected_label != "Row number":
        labels = (
            chart_frame[selected_label]
            .fillna("Missing")
            .astype(str)
        )

        # Avoid unreadable charts when every row has a unique verbose label.
        labels = labels.str.slice(0, 60)
        plot_data.index = labels

    # Limit highly granular result sets in the chart while preserving the
    # complete table below.
    if len(plot_data) > 50:
        st.caption(
            "The chart displays the first 50 rows for readability. "
            "The full result remains available in the table and download."
        )
        plot_data = plot_data.head(50)

    if chart_type == "Bar chart":
        st.bar_chart(plot_data)
    elif chart_type == "Line chart":
        st.line_chart(plot_data)
    else:
        st.area_chart(plot_data)


def analysis_group(name: str) -> str:
    """Assign a broad topic group from the report name."""
    normalized = name.lower()

    if any(
        token in normalized
        for token in [
            "family",
            "sibsp",
            "parch",
        ]
    ):
        return "Family"

    if any(
        token in normalized
        for token in [
            "fare",
            "ticket",
        ]
    ):
        return "Fare and ticket"

    if any(
        token in normalized
        for token in [
            "cabin",
            "deck",
        ]
    ):
        return "Cabin"

    if any(
        token in normalized
        for token in [
            "embark",
            "port",
        ]
    ):
        return "Embarkation"

    if any(
        token in normalized
        for token in [
            "class",
            "survival",
            "outcome",
        ]
    ):
        return "Class and outcomes"

    if any(
        token in normalized
        for token in [
            "age",
            "sex",
            "title",
        ]
    ):
        return "Demographics"

    if any(
        token in normalized
        for token in [
            "missing",
            "duplicate",
            "feature",
            "overview",
        ]
    ):
        return "Data quality and overview"

    return "Other"


# ---------------------------------------------------------------------------
# Load reports
# ---------------------------------------------------------------------------

render_page_header(
    title="SQL Analytics",
    subtitle=(
        "Browse the named SQL analyses generated from the project's SQLite "
        "database. Each result can be inspected, visualized, searched, and "
        "downloaded."
    ),
    icon="🗄️",
)

report_paths = discover_sql_reports(
    SQL_REPORTS_DIR
)

manifest_metadata = load_manifest_metadata(
    MANIFEST_PATH
)

if not report_paths:
    st.warning(
        f"No exported SQL reports were found in `{SQL_REPORTS_DIR}`."
    )
    st.markdown(
        "Generate the database and reports from the repository root:"
    )
    st.code(
        "python scripts/build_database.py\n"
        "python scripts/run_analysis.py",
        language="bash",
    )
    st.stop()


# ---------------------------------------------------------------------------
# Analysis catalogue
# ---------------------------------------------------------------------------

catalogue_records: list[dict[str, Any]] = []

for path in report_paths:
    metadata = manifest_metadata.get(
        path.stem,
        {},
    )

    catalogue_records.append(
        {
            "name": path.stem,
            "title": report_title(path),
            "description": metadata.get(
                "description",
                "",
            ),
            "group": analysis_group(path.stem),
            "path": path,
            "rows": metadata.get("rows"),
            "columns": metadata.get("columns"),
        }
    )

catalogue = pd.DataFrame(catalogue_records)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Analysis browser")

    search_term = st.text_input(
        "Search analyses",
        placeholder="e.g. family, fare, cabin",
    ).strip()

    group_options = sorted(
        catalogue["group"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_groups = st.multiselect(
        "Topic",
        options=group_options,
        default=group_options,
    )

    filtered_catalogue = catalogue.copy()

    if selected_groups:
        filtered_catalogue = filtered_catalogue[
            filtered_catalogue["group"].isin(selected_groups)
        ]

    if search_term:
        search_blob = (
            filtered_catalogue["title"].fillna("")
            + " "
            + filtered_catalogue["description"].fillna("")
            + " "
            + filtered_catalogue["name"].fillna("")
        )

        filtered_catalogue = filtered_catalogue[
            search_blob.str.contains(
                search_term,
                case=False,
                na=False,
                regex=False,
            )
        ]

    if filtered_catalogue.empty:
        st.info(
            "No analyses match the current search and topic filters."
        )
        st.stop()

    analysis_options = filtered_catalogue["name"].tolist()

    selected_analysis_name = st.selectbox(
        "Choose analysis",
        options=analysis_options,
        format_func=lambda name: (
            filtered_catalogue.loc[
                filtered_catalogue["name"] == name,
                "title",
            ].iloc[0]
        ),
    )

    st.divider()

    st.caption(
        f"{len(filtered_catalogue):,} of "
        f"{len(catalogue):,} analyses available"
    )


# ---------------------------------------------------------------------------
# Selected analysis
# ---------------------------------------------------------------------------

selected_record = filtered_catalogue.loc[
    filtered_catalogue["name"] == selected_analysis_name
].iloc[0]

selected_path = Path(selected_record["path"])
result = load_table(selected_path)

st.subheader(str(selected_record["title"]))

description = str(
    selected_record.get("description", "")
).strip()

if description and description.lower() != "nan":
    st.markdown(description)
else:
    st.caption(
        "This result was generated from a named query in "
        "`sql/analysis_queries.sql`."
    )

metadata_columns = st.columns(4)

with metadata_columns[0]:
    st.metric(
        "Rows",
        f"{len(result):,}",
    )

with metadata_columns[1]:
    st.metric(
        "Columns",
        f"{result.shape[1]:,}",
    )

with metadata_columns[2]:
    st.metric(
        "Topic",
        str(selected_record["group"]),
    )

with metadata_columns[3]:
    st.metric(
        "Query ID",
        selected_analysis_name.split("_", maxsplit=1)[0],
    )


# ---------------------------------------------------------------------------
# Table and visualization
# ---------------------------------------------------------------------------

st.divider()

table_tab, chart_tab, catalogue_tab = st.tabs(
    [
        "Result table",
        "Visualization",
        "Analysis catalogue",
    ]
)

with table_tab:
    st.markdown("#### Query result")

    if result.empty:
        st.info(
            "The selected query returned no rows."
        )
    else:
        text_search = st.text_input(
            "Search within this result",
            placeholder="Filter rows by any displayed value",
        ).strip()

        displayed_result = result.copy()

        if text_search:
            row_text = (
                displayed_result
                .astype(str)
                .agg(" ".join, axis=1)
            )

            displayed_result = displayed_result[
                row_text.str.contains(
                    text_search,
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        st.caption(
            f"Displaying {len(displayed_result):,} of "
            f"{len(result):,} row(s)."
        )

        st.dataframe(
            displayed_result,
            use_container_width=True,
            hide_index=True,
            height=540,
        )

        st.download_button(
            label="Download selected SQL result",
            data=dataframe_to_csv_bytes(result),
            file_name=selected_path.name,
            mime="text/csv",
            use_container_width=True,
        )

with chart_tab:
    render_auto_chart(result)

with catalogue_tab:
    st.markdown("#### Available SQL analyses")

    catalogue_display = catalogue.loc[
        :,
        [
            "title",
            "group",
            "description",
            "rows",
            "columns",
        ],
    ].rename(
        columns={
            "title": "Analysis",
            "group": "Topic",
            "description": "Description",
            "rows": "Rows",
            "columns": "Columns",
        }
    )

    st.dataframe(
        catalogue_display,
        use_container_width=True,
        hide_index=True,
        height=560,
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()

st.caption(
    "SQL results are generated by `scripts/run_analysis.py` from named query "
    "blocks in `sql/analysis_queries.sql` and exported to `reports/sql/`."
)
