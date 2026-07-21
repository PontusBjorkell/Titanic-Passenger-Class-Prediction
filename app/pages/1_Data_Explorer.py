
"""Interactive dataset explorer for the Titanic Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

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
    PROCESSED_DATA_PATH,
    TARGET_COLUMN,
)

from utils import (  # noqa: E402
    apply_global_styles,
    dataframe_to_csv_bytes,
    load_table,
    render_missing_artifact,
    render_page_header,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Data Explorer · Titanic",
    page_icon="🔎",
    layout="wide",
)

apply_global_styles()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def available_values(
    dataframe: pd.DataFrame,
    column: str,
) -> list[str]:
    """Return sorted non-null string values from a column."""
    if column not in dataframe.columns:
        return []

    return sorted(
        dataframe[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def numeric_range(
    dataframe: pd.DataFrame,
    column: str,
) -> tuple[float, float] | None:
    """Return finite minimum and maximum values for a numeric column."""
    if column not in dataframe.columns:
        return None

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(values.min()), float(values.max())


def apply_filters(
    dataframe: pd.DataFrame,
    *,
    passenger_classes: list[str],
    sexes: list[str],
    embarkation_ports: list[str],
    age_range: tuple[float, float] | None,
    fare_range: tuple[float, float] | None,
    cabin_status: str,
    family_status: str,
) -> pd.DataFrame:
    """Apply all interactive sidebar filters."""
    filtered = dataframe.copy()

    if passenger_classes and TARGET_COLUMN in filtered.columns:
        filtered = filtered[
            filtered[TARGET_COLUMN].astype(str).isin(passenger_classes)
        ]

    if sexes and "Sex" in filtered.columns:
        filtered = filtered[
            filtered["Sex"].astype(str).isin(sexes)
        ]

    if embarkation_ports and "Embarked" in filtered.columns:
        filtered = filtered[
            filtered["Embarked"].astype(str).isin(embarkation_ports)
        ]

    if age_range is not None and "Age" in filtered.columns:
        ages = pd.to_numeric(
            filtered["Age"],
            errors="coerce",
        )
        filtered = filtered[
            ages.between(age_range[0], age_range[1])
        ]

    if fare_range is not None and "Fare" in filtered.columns:
        fares = pd.to_numeric(
            filtered["Fare"],
            errors="coerce",
        )
        filtered = filtered[
            fares.between(fare_range[0], fare_range[1])
        ]

    if cabin_status != "All":
        if "HasCabin" in filtered.columns:
            has_cabin = (
                pd.to_numeric(
                    filtered["HasCabin"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )
        elif "Cabin" in filtered.columns:
            has_cabin = filtered["Cabin"].notna().astype(int)
        else:
            has_cabin = pd.Series(
                0,
                index=filtered.index,
            )

        target_value = 1 if cabin_status == "Cabin recorded" else 0
        filtered = filtered[has_cabin == target_value]

    if family_status != "All":
        if "FamilySize" in filtered.columns:
            family_size = pd.to_numeric(
                filtered["FamilySize"],
                errors="coerce",
            ).fillna(1)
        else:
            sibsp = pd.to_numeric(
                filtered.get("SibSp", 0),
                errors="coerce",
            ).fillna(0)
            parch = pd.to_numeric(
                filtered.get("Parch", 0),
                errors="coerce",
            ).fillna(0)
            family_size = sibsp + parch + 1

        if family_status == "Travelling alone":
            filtered = filtered[family_size == 1]
        else:
            filtered = filtered[family_size > 1]

    return filtered


def missingness_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact missing-data summary."""
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "Column",
                "Missing values",
                "Missing percentage",
            ]
        )

    missing = dataframe.isna().sum()
    summary = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing values": missing.values,
            "Missing percentage": (
                missing.values / len(dataframe)
            ),
        }
    )

    return (
        summary[summary["Missing values"] > 0]
        .sort_values(
            "Missing percentage",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

render_page_header(
    title="Data Explorer",
    subtitle=(
        "Interactively inspect the processed passenger dataset and examine "
        "how demographic, ticket, fare, family, and cabin variables relate "
        "to passenger class."
    ),
    icon="🔎",
)

dataframe = load_table(PROCESSED_DATA_PATH)

if dataframe.empty:
    render_missing_artifact(
        title="Processed passenger dataset",
        path=PROCESSED_DATA_PATH,
        command="python scripts/prepare_data.py",
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")

    class_options = available_values(
        dataframe,
        TARGET_COLUMN,
    )
    selected_classes = st.multiselect(
        "Passenger class",
        options=class_options,
        default=class_options,
    )

    sex_options = available_values(
        dataframe,
        "Sex",
    )
    selected_sexes = st.multiselect(
        "Sex",
        options=sex_options,
        default=sex_options,
    )

    embarked_options = available_values(
        dataframe,
        "Embarked",
    )
    selected_embarked = st.multiselect(
        "Embarkation port",
        options=embarked_options,
        default=embarked_options,
    )

    age_bounds = numeric_range(
        dataframe,
        "Age",
    )

    if age_bounds is not None:
        selected_age_range = st.slider(
            "Age range",
            min_value=float(age_bounds[0]),
            max_value=float(age_bounds[1]),
            value=(
                float(age_bounds[0]),
                float(age_bounds[1]),
            ),
        )
    else:
        selected_age_range = None

    fare_bounds = numeric_range(
        dataframe,
        "Fare",
    )

    if fare_bounds is not None:
        selected_fare_range = st.slider(
            "Fare range",
            min_value=float(fare_bounds[0]),
            max_value=float(fare_bounds[1]),
            value=(
                float(fare_bounds[0]),
                float(fare_bounds[1]),
            ),
        )
    else:
        selected_fare_range = None

    selected_cabin_status = st.selectbox(
        "Cabin information",
        options=[
            "All",
            "Cabin recorded",
            "Cabin missing",
        ],
    )

    selected_family_status = st.selectbox(
        "Travel group",
        options=[
            "All",
            "Travelling alone",
            "Travelling with family",
        ],
    )

    st.divider()

    if st.button(
        "Reset filters",
        use_container_width=True,
    ):
        st.rerun()


filtered = apply_filters(
    dataframe,
    passenger_classes=selected_classes,
    sexes=selected_sexes,
    embarkation_ports=selected_embarked,
    age_range=selected_age_range,
    fare_range=selected_fare_range,
    cabin_status=selected_cabin_status,
    family_status=selected_family_status,
)


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

metric_columns = st.columns(5)

with metric_columns[0]:
    st.metric(
        "Displayed passengers",
        f"{len(filtered):,}",
        delta=f"{len(filtered) - len(dataframe):,} vs. full data",
    )

with metric_columns[1]:
    st.metric(
        "Dataset rows",
        f"{len(dataframe):,}",
    )

with metric_columns[2]:
    median_age = (
        pd.to_numeric(
            filtered["Age"],
            errors="coerce",
        ).median()
        if "Age" in filtered.columns
        else float("nan")
    )
    st.metric(
        "Median age",
        (
            f"{median_age:.1f}"
            if pd.notna(median_age)
            else "Unavailable"
        ),
    )

with metric_columns[3]:
    median_fare = (
        pd.to_numeric(
            filtered["Fare"],
            errors="coerce",
        ).median()
        if "Fare" in filtered.columns
        else float("nan")
    )
    st.metric(
        "Median fare",
        (
            f"{median_fare:.2f}"
            if pd.notna(median_fare)
            else "Unavailable"
        ),
    )

with metric_columns[4]:
    if TARGET_COLUMN in filtered.columns and not filtered.empty:
        dominant_class = (
            filtered[TARGET_COLUMN]
            .value_counts()
            .idxmax()
        )
        dominant_share = (
            filtered[TARGET_COLUMN]
            .value_counts(normalize=True)
            .max()
        )
        dominant_label = (
            f"Class {dominant_class} "
            f"({dominant_share:.1%})"
        )
    else:
        dominant_label = "Unavailable"

    st.metric(
        "Largest class",
        dominant_label,
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

overview_tab, distributions_tab, quality_tab, records_tab = st.tabs(
    [
        "Overview",
        "Distributions",
        "Data quality",
        "Passenger records",
    ]
)


# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------

with overview_tab:
    st.subheader("Passenger-class composition")

    if filtered.empty:
        st.info(
            "No rows match the selected filters."
        )
    else:
        left, right = st.columns(2)

        with left:
            if TARGET_COLUMN in filtered.columns:
                class_counts = (
                    filtered[TARGET_COLUMN]
                    .value_counts()
                    .sort_index()
                    .rename_axis("Passenger class")
                    .to_frame("Passengers")
                )

                st.bar_chart(class_counts)

        with right:
            if (
                TARGET_COLUMN in filtered.columns
                and "Sex" in filtered.columns
            ):
                class_sex = pd.crosstab(
                    filtered["Sex"],
                    filtered[TARGET_COLUMN],
                    normalize="columns",
                )

                st.markdown(
                    "#### Sex composition within each class"
                )
                st.dataframe(
                    class_sex.style.format("{:.1%}"),
                    use_container_width=True,
                )

        st.markdown("#### Class profile")

        profile_columns = [
            column
            for column in [
                TARGET_COLUMN,
                "Age",
                "Fare",
                "FamilySize",
                "HasCabin",
            ]
            if column in filtered.columns
        ]

        if (
            TARGET_COLUMN in profile_columns
            and len(profile_columns) > 1
        ):
            aggregations = {
                column: "mean"
                for column in profile_columns
                if column != TARGET_COLUMN
            }

            profile = (
                filtered[profile_columns]
                .groupby(TARGET_COLUMN)
                .agg(aggregations)
                .reset_index()
            )

            profile = profile.rename(
                columns={
                    "Age": "Mean age",
                    "Fare": "Mean fare",
                    "FamilySize": "Mean family size",
                    "HasCabin": "Cabin-recording rate",
                }
            )

            formatters = {
                column: "{:.2f}"
                for column in profile.columns
                if column != TARGET_COLUMN
            }

            if "Cabin-recording rate" in profile.columns:
                formatters[
                    "Cabin-recording rate"
                ] = "{:.1%}"

            st.dataframe(
                profile.style.format(formatters),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Distributions tab
# ---------------------------------------------------------------------------

with distributions_tab:
    if filtered.empty:
        st.info(
            "No rows match the selected filters."
        )
    else:
        numeric_candidates = [
            column
            for column in [
                "Age",
                "Fare",
                "FamilySize",
                "SibSp",
                "Parch",
                "TicketGroupSize",
            ]
            if column in filtered.columns
        ]

        if numeric_candidates:
            selected_numeric = st.selectbox(
                "Numeric variable",
                options=numeric_candidates,
            )

            plot_frame = filtered[
                [selected_numeric]
            ].copy()

            plot_frame[selected_numeric] = pd.to_numeric(
                plot_frame[selected_numeric],
                errors="coerce",
            )

            plot_frame = plot_frame.dropna()

            if not plot_frame.empty:
                st.bar_chart(
                    plot_frame[selected_numeric]
                    .round(0)
                    .value_counts()
                    .sort_index()
                )

                st.dataframe(
                    plot_frame[selected_numeric]
                    .describe()
                    .rename("Value")
                    .to_frame(),
                    use_container_width=True,
                )

        categorical_candidates = [
            column
            for column in [
                "Sex",
                "Embarked",
                "Title",
                "FamilySizeGroup",
                "CabinDeck",
                "TicketPrefix",
            ]
            if column in filtered.columns
        ]

        if categorical_candidates:
            st.markdown("#### Categorical distribution")

            selected_categorical = st.selectbox(
                "Categorical variable",
                options=categorical_candidates,
            )

            category_counts = (
                filtered[selected_categorical]
                .fillna("Missing")
                .astype(str)
                .value_counts()
                .head(20)
                .rename_axis(selected_categorical)
                .to_frame("Passengers")
            )

            st.bar_chart(category_counts)
            st.dataframe(
                category_counts,
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Data quality tab
# ---------------------------------------------------------------------------

with quality_tab:
    st.subheader("Missing-data profile")

    missing_summary = missingness_table(filtered)

    if missing_summary.empty:
        st.success(
            "The filtered dataset contains no missing values."
        )
    else:
        st.dataframe(
            missing_summary.style.format(
                {
                    "Missing percentage": "{:.1%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            missing_summary.set_index("Column")[
                ["Missing percentage"]
            ]
        )

    st.subheader("Duplicate and identifier checks")

    check_columns = st.columns(3)

    with check_columns[0]:
        st.metric(
            "Duplicate rows",
            f"{int(filtered.duplicated().sum()):,}",
        )

    with check_columns[1]:
        passenger_id_duplicates = (
            int(
                filtered["PassengerId"]
                .duplicated()
                .sum()
            )
            if "PassengerId" in filtered.columns
            else 0
        )
        st.metric(
            "Duplicate passenger IDs",
            f"{passenger_id_duplicates:,}",
        )

    with check_columns[2]:
        st.metric(
            "Columns",
            f"{filtered.shape[1]:,}",
        )

    st.subheader("Column types")

    dtypes = pd.DataFrame(
        {
            "Column": filtered.columns,
            "Data type": [
                str(dtype)
                for dtype in filtered.dtypes
            ],
            "Unique values": [
                int(filtered[column].nunique(dropna=True))
                for column in filtered.columns
            ],
        }
    )

    st.dataframe(
        dtypes,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Passenger records tab
# ---------------------------------------------------------------------------

with records_tab:
    st.subheader("Passenger records")

    search_text = st.text_input(
        "Search by passenger name, ticket, cabin, or passenger ID",
        placeholder="Enter part of a name, ticket, cabin, or ID",
    ).strip()

    records = filtered.copy()

    if search_text:
        searchable_columns = [
            column
            for column in [
                "PassengerId",
                "Name",
                "Ticket",
                "Cabin",
            ]
            if column in records.columns
        ]

        if searchable_columns:
            search_mask = pd.Series(
                False,
                index=records.index,
            )

            for column in searchable_columns:
                search_mask = search_mask | (
                    records[column]
                    .astype(str)
                    .str.contains(
                        search_text,
                        case=False,
                        na=False,
                        regex=False,
                    )
                )

            records = records[search_mask]

    preferred_columns = [
        column
        for column in [
            "PassengerId",
            "Name",
            TARGET_COLUMN,
            "Sex",
            "Age",
            "Fare",
            "Embarked",
            "FamilySize",
            "Title",
            "CabinDeck",
            "Ticket",
        ]
        if column in records.columns
    ]

    remaining_columns = [
        column
        for column in records.columns
        if column not in preferred_columns
    ]

    displayed_records = records.loc[
        :,
        preferred_columns + remaining_columns,
    ]

    st.caption(
        f"Displaying {len(displayed_records):,} passenger record(s)."
    )

    st.dataframe(
        displayed_records,
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    st.download_button(
        label="Download filtered passenger records",
        data=dataframe_to_csv_bytes(displayed_records),
        file_name="filtered_titanic_passengers.csv",
        mime="text/csv",
        use_container_width=True,
        disabled=displayed_records.empty,
    )
