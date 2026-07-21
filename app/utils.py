
"""Reusable helpers for the Titanic Streamlit application."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def apply_global_styles() -> None:
    """Apply a restrained, consistent visual style to the application."""
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1450px;
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            h1, h2, h3 {
                letter-spacing: -0.02em;
            }

            div[data-testid="stMetric"] {
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-radius: 0.8rem;
                padding: 1rem;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(128, 128, 128, 0.20);
                border-radius: 0.5rem;
                overflow: hidden;
            }

            .project-subtitle {
                font-size: 1.15rem;
                opacity: 0.80;
                margin-top: -0.5rem;
                margin-bottom: 1.5rem;
            }

            .architecture-box {
                border: 1px solid rgba(128, 128, 128, 0.25);
                border-radius: 0.8rem;
                padding: 1rem 1.2rem;
                text-align: center;
                min-height: 7rem;
            }

            .architecture-arrow {
                text-align: center;
                font-size: 1.6rem;
                opacity: 0.65;
                padding-top: 1.8rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(
    title: str,
    subtitle: str | None = None,
    icon: str | None = None,
) -> None:
    """Render a consistent page title and optional subtitle."""
    heading = f"{icon} {title}" if icon else title
    st.title(heading)

    if subtitle:
        st.markdown(
            f'<div class="project-subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Safe artifact loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_json(
    path: str | Path,
    default: Any = None,
) -> Any:
    """Load JSON safely, returning ``default`` when unavailable."""
    path = Path(path)

    if not path.exists() or not path.is_file():
        return default

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return default


@st.cache_data(show_spinner=False)
def load_table(
    path: str | Path,
) -> pd.DataFrame:
    """Load CSV or Parquet into a DataFrame.

    Missing or unreadable files return an empty DataFrame so the application
    can show a helpful message rather than crash.
    """
    path = Path(path)

    if not path.exists() or not path.is_file():
        return pd.DataFrame()

    try:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)

        raise ValueError(
            f"Unsupported table format: {path.suffix}"
        )
    except (OSError, ValueError, ImportError):
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_joblib_model(path: str | Path) -> Any:
    """Load and cache a serialized joblib model."""
    import joblib

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}"
        )

    return joblib.load(path)


def read_markdown(
    path: str | Path,
    default: str = "",
) -> str:
    """Read a Markdown file safely."""
    path = Path(path)

    if not path.exists() or not path.is_file():
        return default

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return default


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def safe_metric(
    metrics: Mapping[str, Any] | None,
    key: str,
) -> float | None:
    """Return a finite numeric metric when available."""
    if not metrics or key not in metrics:
        return None

    value = metrics[key]

    if isinstance(value, bool):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(numeric):
        return None

    return numeric


def format_metric_value(
    value: float | None,
    *,
    percentage: bool = False,
    decimals: int = 2,
) -> str:
    """Format an optional metric value for display."""
    if value is None:
        return "Unavailable"

    if percentage:
        return f"{value:.{decimals}%}"

    return f"{value:.{decimals}f}"


def prettify_name(value: str) -> str:
    """Convert snake_case names into readable labels."""
    return (
        value.replace("_", " ")
        .replace("-", " ")
        .strip()
        .title()
    )


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Convert a DataFrame into UTF-8 CSV bytes for downloads."""
    return dataframe.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Artifact and repository helpers
# ---------------------------------------------------------------------------

def artifact_status_table(
    artifacts: Mapping[str, str | Path],
) -> pd.DataFrame:
    """Build a compact availability table for files and directories."""
    records = []

    for label, raw_path in artifacts.items():
        path = Path(raw_path)
        records.append(
            {
                "Artifact": label,
                "Status": "Available" if path.exists() else "Missing",
            }
        )

    return pd.DataFrame(records)


def render_architecture() -> None:
    """Render the high-level project workflow."""
    columns = st.columns([1, 0.25, 1, 0.25, 1, 0.25, 1])

    boxes = [
        (
            "Raw data",
            "CSV datasets and external validation source",
        ),
        (
            "Preparation",
            "Validation, cleaning, feature engineering, Parquet, SQLite",
        ),
        (
            "Analytics and ML",
            "SQL views, named analyses, training, tuning, evaluation",
        ),
        (
            "Reporting",
            "Artifacts, figures, notebook, and Streamlit application",
        ),
    ]

    for box_index, (title, description) in enumerate(boxes):
        column_index = box_index * 2

        with columns[column_index]:
            st.markdown(
                f"""
                <div class="architecture-box">
                    <strong>{title}</strong><br><br>
                    <span>{description}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if box_index < len(boxes) - 1:
            with columns[column_index + 1]:
                st.markdown(
                    '<div class="architecture-arrow">→</div>',
                    unsafe_allow_html=True,
                )


def discover_sql_reports(
    reports_directory: str | Path,
) -> list[Path]:
    """Return exported SQL CSV files in stable filename order."""
    directory = Path(reports_directory)

    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.glob("*.csv")
        if path.is_file()
    )


def normalize_classification_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize common classification-report CSV layouts."""
    if dataframe.empty:
        return dataframe

    output = dataframe.copy()

    first_column = str(output.columns[0])

    if first_column.startswith("Unnamed"):
        output = output.rename(
            columns={output.columns[0]: "label"}
        )

    if "label" not in output.columns and output.index.name:
        output = output.reset_index()

    return output


def render_missing_artifact(
    title: str,
    path: str | Path,
    command: str | None = None,
) -> None:
    """Display a consistent missing-artifact message."""
    st.warning(f"{title} is not available at `{Path(path)}`.")

    if command:
        st.code(command, language="bash")
