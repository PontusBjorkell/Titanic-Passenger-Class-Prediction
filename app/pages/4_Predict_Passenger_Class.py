
"""Interactive passenger-class prediction page for the Titanic Streamlit app."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import (  # noqa: E402
    apply_global_styles,
    load_json,
    render_page_header,
)

from titanic_passenger_class_prediction.prediction import (  # noqa: E402
    CONFIDENCE_COLUMN,
    PREDICTION_COLUMN,
    build_probability_column_name,
    get_model_classes,
    predict_one_passenger,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Predict Passenger Class · Titanic",
    page_icon="🎫",
    layout="wide",
)

apply_global_styles()


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

MODEL_METADATA_PATH = METRICS_DIR / "model_metadata.json"

MODEL_CANDIDATES = [
    MODELS_DIR / "best_model.joblib",
    MODELS_DIR / "trained_model.joblib",
    MODELS_DIR / "model.joblib",
    MODELS_DIR / "passenger_class_model.joblib",
    MODELS_DIR / "titanic_model.joblib",
]


def discover_model_path() -> Path | None:
    """Locate the saved fitted model without hard-coding one filename."""
    metadata = load_json(
        MODEL_METADATA_PATH,
        default={},
    )

    if isinstance(metadata, dict):
        for key in [
            "model_path",
            "artifact_path",
            "saved_model_path",
        ]:
            value = metadata.get(key)

            if value:
                candidate = Path(str(value))

                if not candidate.is_absolute():
                    candidate = PROJECT_ROOT / candidate

                if candidate.exists():
                    return candidate

    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    discovered = sorted(
        list(MODELS_DIR.glob("*.joblib"))
        + list(MODELS_DIR.glob("*.pkl"))
        + list(MODELS_DIR.glob("*.pickle"))
    )

    return discovered[0] if discovered else None


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> Any:
    """Load the serialized prediction pipeline once per Streamlit session."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

CLASS_LABELS = {
    1: "First class",
    2: "Second class",
    3: "Third class",
    "1": "First class",
    "2": "Second class",
    "3": "Third class",
}


def class_name(value: Any) -> str:
    """Return a human-readable passenger-class label."""
    return CLASS_LABELS.get(
        value,
        f"Class {value}",
    )


def confidence_description(
    confidence: float,
) -> tuple[str, str]:
    """Describe confidence without overstating model certainty."""
    if confidence >= 0.80:
        return (
            "High model confidence",
            "The model strongly favors one class for this input.",
        )

    if confidence >= 0.60:
        return (
            "Moderate model confidence",
            "The model favors one class, but alternatives remain plausible.",
        )

    return (
        "Low model confidence",
        "The class probabilities are relatively close. Interpret cautiously.",
    )



def build_passenger_record(
    *,
    passenger_id: int,
    name: str,
    sex: str,
    age: float,
    sibsp: int,
    parch: int,
    ticket: str,
    fare: float,
    cabin: str,
    embarked: str,
) -> dict[str, Any]:
    """Build one raw Titanic-compatible passenger record."""
    return {
        "PassengerId": passenger_id,
        "Name": name,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": ticket,
        "Fare": fare,
        "Cabin": cabin or None,
        "Embarked": embarked,
    }


PRESETS: dict[str, dict[str, Any]] = {
    "Custom passenger": {},
    "Affluent solo traveller": {
        "name": "Mrs. Eleanor Whitmore",
        "sex": "female",
        "age": 38.0,
        "sibsp": 0,
        "parch": 0,
        "ticket": "PC 17599",
        "fare": 82.17,
        "cabin": "B28",
        "embarked": "C",
    },
    "Young family traveller": {
        "name": "Mr. Thomas Carter",
        "sex": "male",
        "age": 29.0,
        "sibsp": 1,
        "parch": 2,
        "ticket": "347082",
        "fare": 31.28,
        "cabin": "",
        "embarked": "S",
    },
    "Low-fare solo traveller": {
        "name": "Mr. Samuel Reed",
        "sex": "male",
        "age": 24.0,
        "sibsp": 0,
        "parch": 0,
        "ticket": "A/5 21171",
        "fare": 7.25,
        "cabin": "",
        "embarked": "S",
    },
}


# ---------------------------------------------------------------------------
# Page header and model status
# ---------------------------------------------------------------------------

render_page_header(
    title="Predict Passenger Class",
    subtitle=(
        "Enter a passenger profile and use the saved production pipeline to "
        "estimate whether the passenger most closely resembles first-, "
        "second-, or third-class travellers in the training data."
    ),
    icon="🎫",
)

model_path = discover_model_path()

if model_path is None:
    st.error(
        "No fitted model artifact was found in "
        f"`{MODELS_DIR}`."
    )
    st.markdown(
        "Train and save the model from the repository root:"
    )
    st.code(
        "python scripts/train.py",
        language="bash",
    )
    st.stop()

try:
    model = load_model(
        str(model_path)
    )
except Exception as error:
    st.error(
        "The saved model could not be loaded."
    )
    st.exception(error)
    st.stop()

metadata = load_json(
    MODEL_METADATA_PATH,
    default={},
)

status_columns = st.columns(3)

with status_columns[0]:
    st.metric(
        "Model status",
        "Loaded",
    )

with status_columns[1]:
    st.metric(
        "Artifact",
        model_path.name,
    )

with status_columns[2]:
    model_name = (
        metadata.get("model_name")
        if isinstance(metadata, dict)
        else None
    )
    st.metric(
        "Estimator",
        str(model_name or type(model).__name__),
    )


# ---------------------------------------------------------------------------
# Preset selection
# ---------------------------------------------------------------------------

st.subheader("Passenger profile")

preset_name = st.selectbox(
    "Start from a profile",
    options=list(PRESETS),
    help=(
        "Presets are illustrative examples only. Every field remains "
        "editable before prediction."
    ),
)

preset = PRESETS[preset_name]


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

with st.form(
    "passenger_prediction_form",
    clear_on_submit=False,
):
    identity_column, travel_column = st.columns(2)

    with identity_column:
        st.markdown("#### Passenger details")

        passenger_id = st.number_input(
            "Passenger ID",
            min_value=1,
            max_value=999999,
            value=9999,
            step=1,
        )

        name = st.text_input(
            "Passenger name",
            value=str(
                preset.get(
                    "name",
                    "Mr. Example Passenger",
                )
            ),
        )

        sex = st.selectbox(
            "Sex",
            options=["male", "female"],
            index=(
                1
                if preset.get("sex") == "female"
                else 0
            ),
        )

        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=float(
                preset.get("age", 30.0)
            ),
            step=1.0,
        )

        sibsp = st.number_input(
            "Siblings or spouses aboard",
            min_value=0,
            max_value=10,
            value=int(
                preset.get("sibsp", 0)
            ),
            step=1,
        )

        parch = st.number_input(
            "Parents or children aboard",
            min_value=0,
            max_value=10,
            value=int(
                preset.get("parch", 0)
            ),
            step=1,
        )

    with travel_column:
        st.markdown("#### Ticket and journey")

        ticket = st.text_input(
            "Ticket",
            value=str(
                preset.get(
                    "ticket",
                    "A/5 21171",
                )
            ),
        )

        fare = st.number_input(
            "Fare",
            min_value=0.0,
            max_value=1000.0,
            value=float(
                preset.get("fare", 15.0)
            ),
            step=0.50,
        )

        cabin = st.text_input(
            "Cabin",
            value=str(
                preset.get("cabin", "")
            ),
            placeholder=(
                "Leave blank when cabin is unknown"
            ),
        )

        embarked_options = ["S", "C", "Q"]
        embarked_default = str(
            preset.get("embarked", "S")
        )

        embarked = st.selectbox(
            "Embarkation port",
            options=embarked_options,
            index=(
                embarked_options.index(
                    embarked_default
                )
                if embarked_default
                in embarked_options
                else 0
            ),
            format_func=lambda value: {
                "S": "Southampton (S)",
                "C": "Cherbourg (C)",
                "Q": "Queenstown (Q)",
            }[value],
        )

        family_size = (
            int(sibsp)
            + int(parch)
            + 1
        )

        st.metric(
            "Calculated family size",
            family_size,
        )

    submitted = st.form_submit_button(
        "Predict passenger class",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if submitted:
    validation_errors = []

    if not name.strip():
        validation_errors.append(
            "Passenger name cannot be empty."
        )

    if not ticket.strip():
        validation_errors.append(
            "Ticket cannot be empty."
        )

    if validation_errors:
        for message in validation_errors:
            st.error(message)
        st.stop()

    passenger_record = build_passenger_record(
        passenger_id=int(passenger_id),
        name=name.strip(),
        sex=sex,
        age=float(age),
        sibsp=int(sibsp),
        parch=int(parch),
        ticket=ticket.strip(),
        fare=float(fare),
        cabin=cabin.strip(),
        embarked=embarked,
    )

    passenger_frame = pd.DataFrame(
        [passenger_record]
    )

    try:
        prediction_result = predict_one_passenger(
            passenger=passenger_record,
            model=model,
        )

        predicted_class = prediction_result[
            PREDICTION_COLUMN
        ]

        classes = get_model_classes(
            model
        )

        probability_rows = []

        for class_value in classes:
            probability_column = (
                build_probability_column_name(
                    class_value
                )
            )

            probability_rows.append(
                {
                    "Passenger class": class_name(
                        class_value
                    ),
                    "Class value": class_value,
                    "Probability": float(
                        prediction_result[
                            probability_column
                        ]
                    ),
                }
            )

        table = (
            pd.DataFrame(
                probability_rows
            )
            .sort_values(
                "Probability",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        confidence = float(
            prediction_result[
                CONFIDENCE_COLUMN
            ]
        )

    except Exception as error:
        st.error(
            "Prediction failed while preparing the passenger features "
            "or calling the saved model."
        )
        st.markdown(
            "The exact raw one-row input sent to the reusable prediction "
            "service is shown below."
        )
        st.dataframe(
            passenger_frame,
            use_container_width=True,
            hide_index=True,
        )
        st.exception(error)
        st.stop()

    st.divider()
    st.subheader("Prediction result")

    result_left, result_right = st.columns(
        [0.85, 1.15]
    )

    with result_left:
        st.success(
            f"### {class_name(predicted_class)}"
        )

        st.metric(
            "Predicted class value",
            predicted_class,
        )

        st.markdown(
            """
            This is a **model estimate**, not a historical fact about a real
            passenger. It reflects patterns learned from the training data.
            """
        )

    with result_right:
        confidence_title, confidence_text = (
            confidence_description(
                confidence
            )
        )

        st.metric(
            confidence_title,
            f"{confidence:.1%}",
        )

        st.caption(
            confidence_text
        )

        probability_chart = (
            table.set_index(
                "Passenger class"
            )[["Probability"]]
        )

        st.bar_chart(
            probability_chart
        )

        st.dataframe(
            table.style.format(
                {
                    "Probability": "{:.1%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Submitted model input")

    display_frame = passenger_frame.copy()
    display_frame["CalculatedFamilySize"] = (
        int(sibsp)
        + int(parch)
        + 1
    )
    display_frame["CabinRecorded"] = (
        "Yes"
        if cabin.strip()
        else "No"
    )

    st.dataframe(
        display_frame,
        use_container_width=True,
        hide_index=True,
    )

    prediction_export = display_frame.copy()
    prediction_export[
        "PredictedPclass"
    ] = predicted_class
    prediction_export[
        "PredictedClassLabel"
    ] = class_name(predicted_class)

    for class_value in classes:
        probability_column = (
            build_probability_column_name(
                class_value
            )
        )

        prediction_export[
            probability_column
        ] = float(
            prediction_result[
                probability_column
            ]
        )

    prediction_export[
        CONFIDENCE_COLUMN
    ] = confidence

    st.download_button(
        label="Download this prediction",
        data=prediction_export.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="passenger_class_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Methodology notes
# ---------------------------------------------------------------------------

st.divider()

with st.expander(
    "How to interpret this page"
):
    st.markdown(
        """
        - The model predicts **passenger class**, not survival.
        - The prediction is based on associations in the historical dataset.
        - A high probability means the model strongly favors one class
          relative to the others; it does not imply certainty.
        - Inputs outside the range of the training data may produce less
          reliable predictions.
        - Passenger class is a historical socioeconomic category. Model
          outputs should not be interpreted as judgments about personal worth.
        """
    )

with st.expander(
    "Raw input schema"
):
    st.code(
        json.dumps(
            {
                "PassengerId": "integer",
                "Name": "string",
                "Sex": "male or female",
                "Age": "numeric",
                "SibSp": "integer",
                "Parch": "integer",
                "Ticket": "string",
                "Fare": "numeric",
                "Cabin": "string or missing",
                "Embarked": "S, C, or Q",
            },
            indent=2,
        ),
        language="json",
    )

st.caption(
    "The application loads the saved fitted pipeline from `artifacts/models/`. "
    "Reusable feature engineering and inference are performed through "
    "`titanic_passenger_class_prediction.prediction`; Streamlit does not "
    "retrain the model."
)
