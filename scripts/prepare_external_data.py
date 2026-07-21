"""Prepare and audit an external Titanic passenger dataset.

The external source uses a different schema from the Kaggle-style data used
by this project. This script converts the external data into the project's
canonical passenger format and checks for overlap with the labelled training
dataset.

The external source contains passenger and crew categories. Only passengers
from first, second, and third class are retained because the model predicts
the three passenger classes:

    1st -> 1
    2nd -> 2
    3rd -> 3

Generated outputs
-----------------
data/external/titanic_external_clean.csv
    All harmonized first-, second-, and third-class passenger rows.

data/external/titanic_external_nonoverlap.csv
    Harmonized rows that do not match the training data by normalized name
    or normalized ticket.

reports/external/external_overlap_audit.csv
    Passenger-level overlap flags and matching evidence.

reports/external/external_overlap_summary.json
    Dataset dimensions, filtering counts, overlap counts, and output paths.

Examples
--------
Run with the project defaults:

    python scripts/prepare_external_data.py

Specify another external input file:

    python scripts/prepare_external_data.py \
        --external-input data/raw/titanic_external.csv

Use another labelled training file:

    python scripts/prepare_external_data.py \
        --training-input data/raw/train.csv

Disable removal of overlapping records from the strict dataset:

    python scripts/prepare_external_data.py \
        --overlap-rule both
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from titanic_passenger_class_prediction.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)


DEFAULT_EXTERNAL_INPUT_PATH = (
    RAW_DATA_DIR / "titanic_external.csv"
)

DEFAULT_TRAINING_INPUT_PATH = (
    RAW_DATA_DIR / "train.csv"
)

DEFAULT_EXTERNAL_OUTPUT_DIRECTORY = (
    DATA_DIR / "external"
)

DEFAULT_EXTERNAL_REPORT_DIRECTORY = (
    REPORTS_DIR / "external"
)

CLEAN_EXTERNAL_FILENAME = (
    "titanic_external_clean.csv"
)

NONOVERLAP_EXTERNAL_FILENAME = (
    "titanic_external_nonoverlap.csv"
)

OVERLAP_AUDIT_FILENAME = (
    "external_overlap_audit.csv"
)

OVERLAP_SUMMARY_FILENAME = (
    "external_overlap_summary.json"
)


EXTERNAL_REQUIRED_COLUMNS = {
    "name",
    "gender",
    "age",
    "class",
    "embarked",
    "ticketno",
    "fare",
    "sibsp",
    "parch",
    "survived",
}

TRAINING_MATCH_COLUMNS = {
    "Name",
    "Ticket",
}

PASSENGER_CLASS_MAPPING = {
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
}

SURVIVAL_MAPPING = {
    "no": 0,
    "yes": 1,
}

EMBARKATION_MAPPING = {
    "belfast": "B",
    "cherbourg": "C",
    "queenstown": "Q",
    "southampton": "S",
}

OverlapRule = Literal[
    "either",
    "both",
    "name",
    "ticket",
]


@dataclass(frozen=True)
class ExternalPreparationArtifacts:
    """Paths created by one external-data preparation run."""

    clean_external_path: Path
    nonoverlap_external_path: Path
    overlap_audit_path: Path
    overlap_summary_path: Path


@dataclass(frozen=True)
class ExternalPreparationResult:
    """In-memory output and artifact paths from one run."""

    clean_external: pd.DataFrame
    nonoverlap_external: pd.DataFrame
    overlap_audit: pd.DataFrame
    summary: dict[str, Any]
    artifacts: ExternalPreparationArtifacts


def load_csv(
    path: Path,
    *,
    dataset_name: str,
) -> pd.DataFrame:
    """Load a non-empty CSV file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{dataset_name} path is not a file: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"{dataset_name} dataset is empty: {path}"
        )

    return dataframe


def validate_external_schema(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the source columns required for harmonization."""
    missing_columns = sorted(
        EXTERNAL_REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "External Titanic data is missing required columns: "
            f"{missing_columns}"
        )


def validate_training_schema(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the training columns needed for overlap checks."""
    missing_columns = sorted(
        TRAINING_MATCH_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training data is missing columns required for "
            f"overlap detection: {missing_columns}"
        )


def normalize_whitespace(
    value: Any,
) -> str:
    """Collapse repeated whitespace and return stripped text."""
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def remove_accents(
    value: str,
) -> str:
    """Remove Unicode accents while preserving basic characters."""
    normalized = unicodedata.normalize(
        "NFKD",
        value,
    )

    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


def normalize_name(
    value: Any,
) -> str:
    """Create a stable passenger-name key for overlap detection.

    The normalization:

    - removes accents;
    - lowercases text;
    - removes punctuation;
    - collapses whitespace.

    Titles and given names are retained because they help distinguish
    passengers who share surnames.
    """
    text = normalize_whitespace(value)

    if not text:
        return ""

    text = remove_accents(text).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return normalize_whitespace(text)


def normalize_ticket(
    value: Any,
) -> str:
    """Create a punctuation-insensitive ticket key."""
    text = normalize_whitespace(value)

    if not text:
        return ""

    text = remove_accents(text).upper()

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text,
    )


def normalize_external_class(
    series: pd.Series,
) -> pd.Series:
    """Normalize external class strings."""
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
    )


def normalize_survival(
    series: pd.Series,
) -> pd.Series:
    """Map yes/no survival values to nullable integers."""
    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    unexpected_values = sorted(
        set(
            normalized.dropna().unique()
        )
        - set(SURVIVAL_MAPPING)
    )

    if unexpected_values:
        raise ValueError(
            "Unexpected external survival values: "
            f"{unexpected_values}"
        )

    return (
        normalized
        .map(SURVIVAL_MAPPING)
        .astype("Int64")
    )


def normalize_embarked_value(
    value: Any,
) -> str | None:
    """Convert embarkation labels to compact codes when possible.

    Missing or blank values are returned as ``None``. When the resulting
    series is converted to pandas' nullable string dtype, pandas represents
    these values internally as ``pd.NA``.
    """
    if pd.isna(value):
        return None

    text = normalize_whitespace(value)

    if not text:
        return None

    lowered = text.lower()

    if lowered in EMBARKATION_MAPPING:
        return EMBARKATION_MAPPING[lowered]

    first_character = text[0].upper()

    if first_character in {
        "B",
        "C",
        "Q",
        "S",
    }:
        return first_character

    return text.upper()


def coerce_nullable_integer(
    series: pd.Series,
    *,
    column_name: str,
) -> pd.Series:
    """Convert a series to nullable integer values."""
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    non_missing_original = series.notna()
    conversion_failures = (
        non_missing_original
        & numeric.isna()
    )

    if conversion_failures.any():
        examples = (
            series.loc[conversion_failures]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Column '{column_name}' contains values that "
            f"cannot be converted to integers: {examples}"
        )

    non_integer_values = (
        numeric.dropna()
        % 1
        != 0
    )

    if non_integer_values.any():
        examples = (
            numeric.dropna()
            .loc[non_integer_values]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Column '{column_name}' contains non-integer "
            f"values: {examples}"
        )

    return numeric.astype("Int64")


def harmonize_external_data(
    external_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert the external source to the project's canonical schema."""
    validate_external_schema(
        external_dataframe
    )

    source = external_dataframe.copy()

    source["_normalized_source_class"] = (
        normalize_external_class(
            source["class"]
        )
    )

    source_row_count = int(
        len(source)
    )

    retained_mask = source[
        "_normalized_source_class"
    ].isin(
        PASSENGER_CLASS_MAPPING
    )

    passenger_source = (
        source.loc[retained_mask]
        .copy()
        .reset_index(drop=True)
    )

    excluded_source = (
        source.loc[~retained_mask]
        .copy()
    )

    passenger_source["Pclass"] = (
        passenger_source[
            "_normalized_source_class"
        ]
        .map(PASSENGER_CLASS_MAPPING)
        .astype("Int64")
    )

    passenger_source["Survived"] = (
        normalize_survival(
            passenger_source["survived"]
        )
    )

    clean = pd.DataFrame(
        {
            "PassengerId": pd.Series(
                np.arange(
                    1,
                    len(passenger_source) + 1,
                ),
                dtype="Int64",
            ),
            "Survived": passenger_source[
                "Survived"
            ],
            "Pclass": passenger_source[
                "Pclass"
            ],
            "Name": (
                passenger_source["name"]
                .map(normalize_whitespace)
                .astype("string")
            ),
            "Sex": (
                passenger_source["gender"]
                .astype("string")
                .str.strip()
                .str.lower()
            ),
            "Age": pd.to_numeric(
                passenger_source["age"],
                errors="coerce",
            ),
            "SibSp": coerce_nullable_integer(
                passenger_source["sibsp"],
                column_name="sibsp",
            ),
            "Parch": coerce_nullable_integer(
                passenger_source["parch"],
                column_name="parch",
            ),
            "Ticket": (
                passenger_source["ticketno"]
                .map(normalize_whitespace)
                .replace("", pd.NA)
                .astype("string")
            ),
            "Fare": pd.to_numeric(
                passenger_source["fare"],
                errors="coerce",
            ),
            "Cabin": pd.Series(
                pd.NA,
                index=passenger_source.index,
                dtype="string",
            ),
            "Embarked": (
                passenger_source["embarked"]
                .map(normalize_embarked_value)
                .astype("string")
            ),
            "Country": (
                passenger_source["country"]
                .map(normalize_whitespace)
                .replace("", pd.NA)
                .astype("string")
            ),
            "ExternalSourceRow": pd.Series(
                passenger_source.index + 1,
                dtype="Int64",
            ),
        }
    )

    if clean["PassengerId"].duplicated().any():
        raise ValueError(
            "Synthetic external passenger identifiers are not unique."
        )

    if clean["Pclass"].isna().any():
        raise ValueError(
            "External passenger-class conversion produced missing values."
        )

    if clean["Name"].isna().any():
        raise ValueError(
            "External data contains missing passenger names."
        )

    if clean["Sex"].isna().any():
        raise ValueError(
            "External data contains missing sex values."
        )

    unexpected_sexes = sorted(
        set(
            clean["Sex"].dropna().unique()
        )
        - {
            "female",
            "male",
        }
    )

    if unexpected_sexes:
        raise ValueError(
            "Unexpected values in external gender column: "
            f"{unexpected_sexes}"
        )

    excluded_class_counts = (
        excluded_source[
            "_normalized_source_class"
        ]
        .fillna("missing")
        .value_counts()
        .sort_index()
        .to_dict()
    )

    filter_summary = {
        "source_rows": source_row_count,
        "retained_passenger_rows": int(
            len(clean)
        ),
        "excluded_nonpassenger_rows": int(
            len(excluded_source)
        ),
        "retained_first_class_rows": int(
            (clean["Pclass"] == 1).sum()
        ),
        "retained_second_class_rows": int(
            (clean["Pclass"] == 2).sum()
        ),
        "retained_third_class_rows": int(
            (clean["Pclass"] == 3).sum()
        ),
        "excluded_class_counts": {
            str(key): int(value)
            for key, value
            in excluded_class_counts.items()
        },
    }

    return clean, filter_summary


def prepare_training_match_keys(
    training_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create normalized name and ticket keys for training rows."""
    validate_training_schema(
        training_dataframe
    )

    training_keys = pd.DataFrame(
        {
            "TrainingRowNumber": pd.Series(
                np.arange(
                    1,
                    len(training_dataframe) + 1,
                ),
                dtype="Int64",
            ),
            "TrainingName": (
                training_dataframe["Name"]
                .astype("string")
            ),
            "TrainingTicket": (
                training_dataframe["Ticket"]
                .astype("string")
            ),
            "NormalizedName": (
                training_dataframe["Name"]
                .map(normalize_name)
            ),
            "NormalizedTicket": (
                training_dataframe["Ticket"]
                .map(normalize_ticket)
            ),
        }
    )

    if "PassengerId" in training_dataframe.columns:
        training_keys[
            "TrainingPassengerId"
        ] = training_dataframe[
            "PassengerId"
        ].reset_index(
            drop=True
        )

    return training_keys


def build_match_lookup(
    training_keys: pd.DataFrame,
    *,
    key_column: str,
    value_column: str,
) -> dict[str, list[Any]]:
    """Map one normalized key to matching training values."""
    valid = training_keys.loc[
        training_keys[key_column].ne("")
        & training_keys[key_column].notna()
    ]

    lookup: dict[str, list[Any]] = {}

    for key, group in valid.groupby(
        key_column,
        dropna=False,
    ):
        values = (
            group[value_column]
            .dropna()
            .tolist()
        )

        lookup[str(key)] = values

    return lookup


def audit_overlap(
    clean_external: pd.DataFrame,
    training_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Flag external records matching the training data."""
    training_keys = prepare_training_match_keys(
        training_dataframe
    )

    training_name_set = set(
        training_keys.loc[
            training_keys["NormalizedName"].ne(""),
            "NormalizedName",
        ]
    )

    training_ticket_set = set(
        training_keys.loc[
            training_keys["NormalizedTicket"].ne(""),
            "NormalizedTicket",
        ]
    )

    name_id_column = (
        "TrainingPassengerId"
        if "TrainingPassengerId"
        in training_keys.columns
        else "TrainingRowNumber"
    )

    name_id_lookup = build_match_lookup(
        training_keys,
        key_column="NormalizedName",
        value_column=name_id_column,
    )

    ticket_id_lookup = build_match_lookup(
        training_keys,
        key_column="NormalizedTicket",
        value_column=name_id_column,
    )

    audit = clean_external.copy()

    audit["NormalizedName"] = (
        audit["Name"].map(
            normalize_name
        )
    )

    audit["NormalizedTicket"] = (
        audit["Ticket"].map(
            normalize_ticket
        )
    )

    audit["NameMatch"] = (
        audit["NormalizedName"].ne("")
        & audit["NormalizedName"].isin(
            training_name_set
        )
    )

    audit["TicketMatch"] = (
        audit["NormalizedTicket"].ne("")
        & audit["NormalizedTicket"].isin(
            training_ticket_set
        )
    )

    audit["NameAndTicketMatch"] = (
        audit["NameMatch"]
        & audit["TicketMatch"]
    )

    audit["NameOrTicketMatch"] = (
        audit["NameMatch"]
        | audit["TicketMatch"]
    )

    audit["MatchedTrainingIdsByName"] = (
        audit["NormalizedName"].map(
            lambda key: json.dumps(
                name_id_lookup.get(
                    str(key),
                    [],
                )
            )
        )
    )

    audit["MatchedTrainingIdsByTicket"] = (
        audit["NormalizedTicket"].map(
            lambda key: json.dumps(
                ticket_id_lookup.get(
                    str(key),
                    [],
                )
            )
        )
    )

    audit["OverlapEvidence"] = np.select(
        [
            audit["NameAndTicketMatch"],
            audit["NameMatch"],
            audit["TicketMatch"],
        ],
        [
            "name_and_ticket",
            "name_only",
            "ticket_only",
        ],
        default="none",
    )

    return audit


def get_overlap_mask(
    audit_dataframe: pd.DataFrame,
    *,
    overlap_rule: OverlapRule,
) -> pd.Series:
    """Select which records count as overlapping."""
    if overlap_rule == "either":
        return audit_dataframe[
            "NameOrTicketMatch"
        ].astype(bool)

    if overlap_rule == "both":
        return audit_dataframe[
            "NameAndTicketMatch"
        ].astype(bool)

    if overlap_rule == "name":
        return audit_dataframe[
            "NameMatch"
        ].astype(bool)

    if overlap_rule == "ticket":
        return audit_dataframe[
            "TicketMatch"
        ].astype(bool)

    raise ValueError(
        f"Unsupported overlap rule: {overlap_rule}"
    )


def build_nonoverlap_external(
    overlap_audit: pd.DataFrame,
    *,
    overlap_rule: OverlapRule = "either",
) -> pd.DataFrame:
    """Remove records judged to overlap with training data."""
    overlap_mask = get_overlap_mask(
        overlap_audit,
        overlap_rule=overlap_rule,
    )

    audit_only_columns = {
        "NormalizedName",
        "NormalizedTicket",
        "NameMatch",
        "TicketMatch",
        "NameAndTicketMatch",
        "NameOrTicketMatch",
        "MatchedTrainingIdsByName",
        "MatchedTrainingIdsByTicket",
        "OverlapEvidence",
    }

    output_columns = [
        column
        for column in overlap_audit.columns
        if column not in audit_only_columns
    ]

    nonoverlap = (
        overlap_audit.loc[
            ~overlap_mask,
            output_columns,
        ]
        .copy()
        .reset_index(drop=True)
    )

    return nonoverlap


def build_summary(
    *,
    external_input_path: Path,
    training_input_path: Path,
    clean_external: pd.DataFrame,
    nonoverlap_external: pd.DataFrame,
    overlap_audit: pd.DataFrame,
    filter_summary: dict[str, Any],
    overlap_rule: OverlapRule,
    artifacts: ExternalPreparationArtifacts,
) -> dict[str, Any]:
    """Build a JSON-compatible audit summary."""
    overlap_mask = get_overlap_mask(
        overlap_audit,
        overlap_rule=overlap_rule,
    )

    summary = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "external_input_path": str(
            external_input_path
        ),
        "training_input_path": str(
            training_input_path
        ),
        "overlap_rule_for_strict_dataset": (
            overlap_rule
        ),
        **filter_summary,
        "clean_external_rows": int(
            len(clean_external)
        ),
        "clean_external_columns": int(
            clean_external.shape[1]
        ),
        "name_match_rows": int(
            overlap_audit["NameMatch"].sum()
        ),
        "ticket_match_rows": int(
            overlap_audit["TicketMatch"].sum()
        ),
        "name_and_ticket_match_rows": int(
            overlap_audit[
                "NameAndTicketMatch"
            ].sum()
        ),
        "name_or_ticket_match_rows": int(
            overlap_audit[
                "NameOrTicketMatch"
            ].sum()
        ),
        "overlap_rows_under_selected_rule": int(
            overlap_mask.sum()
        ),
        "nonoverlap_rows": int(
            len(nonoverlap_external)
        ),
        "nonoverlap_percentage_of_clean": float(
            100.0
            * len(nonoverlap_external)
            / len(clean_external)
        ),
        "clean_class_counts": {
            str(class_label): int(count)
            for class_label, count
            in clean_external[
                "Pclass"
            ]
            .value_counts()
            .sort_index()
            .items()
        },
        "nonoverlap_class_counts": {
            str(class_label): int(count)
            for class_label, count
            in nonoverlap_external[
                "Pclass"
            ]
            .value_counts()
            .sort_index()
            .items()
        },
        "missing_value_counts": {
            str(column): int(count)
            for column, count
            in clean_external.isna()
            .sum()
            .items()
        },
        "artifacts": {
            key: str(value)
            for key, value
            in asdict(artifacts).items()
        },
    }

    return summary


def save_json(
    payload: dict[str, Any],
    output_path: Path,
) -> Path:
    """Save a dictionary as formatted JSON."""
    if not payload:
        raise ValueError(
            "Cannot save an empty JSON payload."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=_json_default,
        )

    return output_path


def _json_default(
    value: Any,
) -> Any:
    """Convert common non-native objects for JSON."""
    if hasattr(value, "item"):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


def save_external_artifacts(
    *,
    clean_external: pd.DataFrame,
    nonoverlap_external: pd.DataFrame,
    overlap_audit: pd.DataFrame,
    summary: dict[str, Any],
    external_output_directory: Path,
    report_directory: Path,
) -> ExternalPreparationArtifacts:
    """Save harmonized datasets and overlap reports."""
    external_output_directory = Path(
        external_output_directory
    )

    report_directory = Path(
        report_directory
    )

    external_output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_external_path = (
        external_output_directory
        / CLEAN_EXTERNAL_FILENAME
    )

    nonoverlap_external_path = (
        external_output_directory
        / NONOVERLAP_EXTERNAL_FILENAME
    )

    overlap_audit_path = (
        report_directory
        / OVERLAP_AUDIT_FILENAME
    )

    overlap_summary_path = (
        report_directory
        / OVERLAP_SUMMARY_FILENAME
    )

    clean_external.to_csv(
        clean_external_path,
        index=False,
    )

    nonoverlap_external.to_csv(
        nonoverlap_external_path,
        index=False,
    )

    overlap_audit.to_csv(
        overlap_audit_path,
        index=False,
    )

    save_json(
        summary,
        overlap_summary_path,
    )

    return ExternalPreparationArtifacts(
        clean_external_path=clean_external_path,
        nonoverlap_external_path=(
            nonoverlap_external_path
        ),
        overlap_audit_path=overlap_audit_path,
        overlap_summary_path=overlap_summary_path,
    )


def run_external_preparation(
    *,
    external_input_path: Path = (
        DEFAULT_EXTERNAL_INPUT_PATH
    ),
    training_input_path: Path = (
        DEFAULT_TRAINING_INPUT_PATH
    ),
    external_output_directory: Path = (
        DEFAULT_EXTERNAL_OUTPUT_DIRECTORY
    ),
    report_directory: Path = (
        DEFAULT_EXTERNAL_REPORT_DIRECTORY
    ),
    overlap_rule: OverlapRule = "either",
) -> ExternalPreparationResult:
    """Run schema harmonization and overlap auditing."""
    external_input_path = Path(
        external_input_path
    )

    training_input_path = Path(
        training_input_path
    )

    external_dataframe = load_csv(
        external_input_path,
        dataset_name="External Titanic",
    )

    training_dataframe = load_csv(
        training_input_path,
        dataset_name="Training",
    )

    clean_external, filter_summary = (
        harmonize_external_data(
            external_dataframe
        )
    )

    overlap_audit = audit_overlap(
        clean_external,
        training_dataframe,
    )

    nonoverlap_external = (
        build_nonoverlap_external(
            overlap_audit,
            overlap_rule=overlap_rule,
        )
    )

    placeholder_artifacts = (
        ExternalPreparationArtifacts(
            clean_external_path=(
                Path(external_output_directory)
                / CLEAN_EXTERNAL_FILENAME
            ),
            nonoverlap_external_path=(
                Path(external_output_directory)
                / NONOVERLAP_EXTERNAL_FILENAME
            ),
            overlap_audit_path=(
                Path(report_directory)
                / OVERLAP_AUDIT_FILENAME
            ),
            overlap_summary_path=(
                Path(report_directory)
                / OVERLAP_SUMMARY_FILENAME
            ),
        )
    )

    summary = build_summary(
        external_input_path=external_input_path,
        training_input_path=training_input_path,
        clean_external=clean_external,
        nonoverlap_external=nonoverlap_external,
        overlap_audit=overlap_audit,
        filter_summary=filter_summary,
        overlap_rule=overlap_rule,
        artifacts=placeholder_artifacts,
    )

    artifacts = save_external_artifacts(
        clean_external=clean_external,
        nonoverlap_external=nonoverlap_external,
        overlap_audit=overlap_audit,
        summary=summary,
        external_output_directory=(
            external_output_directory
        ),
        report_directory=report_directory,
    )

    return ExternalPreparationResult(
        clean_external=clean_external,
        nonoverlap_external=nonoverlap_external,
        overlap_audit=overlap_audit,
        summary=summary,
        artifacts=artifacts,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Harmonize an external Titanic dataset and "
            "audit overlap with the training data."
        )
    )

    parser.add_argument(
        "--external-input",
        type=Path,
        default=DEFAULT_EXTERNAL_INPUT_PATH,
        help=(
            "External Titanic CSV. "
            f"Default: {DEFAULT_EXTERNAL_INPUT_PATH}"
        ),
    )

    parser.add_argument(
        "--training-input",
        type=Path,
        default=DEFAULT_TRAINING_INPUT_PATH,
        help=(
            "Labelled training CSV used for overlap checks. "
            f"Default: {DEFAULT_TRAINING_INPUT_PATH}"
        ),
    )

    parser.add_argument(
        "--external-output-dir",
        type=Path,
        default=DEFAULT_EXTERNAL_OUTPUT_DIRECTORY,
        help=(
            "Directory for harmonized external datasets. "
            f"Default: {DEFAULT_EXTERNAL_OUTPUT_DIRECTORY}"
        ),
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_EXTERNAL_REPORT_DIRECTORY,
        help=(
            "Directory for overlap audit reports. "
            f"Default: {DEFAULT_EXTERNAL_REPORT_DIRECTORY}"
        ),
    )

    parser.add_argument(
        "--overlap-rule",
        choices=[
            "either",
            "both",
            "name",
            "ticket",
        ],
        default="either",
        help=(
            "Rule used to exclude overlaps from the strict "
            "external dataset. 'either' is the most conservative."
        ),
    )

    return parser


def print_summary(
    result: ExternalPreparationResult,
) -> None:
    """Print a concise preparation summary."""
    summary = result.summary

    print()
    print("=" * 78)
    print("EXTERNAL TITANIC DATA PREPARATION COMPLETE")
    print("=" * 78)

    print(
        f"Source rows: "
        f"{summary['source_rows']:,}"
    )

    print(
        "Retained passenger rows: "
        f"{summary['retained_passenger_rows']:,}"
    )

    print(
        "Excluded non-passenger rows: "
        f"{summary['excluded_nonpassenger_rows']:,}"
    )

    print()
    print("Overlap audit:")

    print(
        f"  Name matches: "
        f"{summary['name_match_rows']:,}"
    )

    print(
        f"  Ticket matches: "
        f"{summary['ticket_match_rows']:,}"
    )

    print(
        "  Name or ticket matches: "
        f"{summary['name_or_ticket_match_rows']:,}"
    )

    print(
        "  Name and ticket matches: "
        f"{summary['name_and_ticket_match_rows']:,}"
    )

    print(
        "  Strict non-overlap rows: "
        f"{summary['nonoverlap_rows']:,}"
    )

    print()
    print("Artifacts:")

    print(
        f"  Clean external data: "
        f"{result.artifacts.clean_external_path}"
    )

    print(
        f"  Non-overlap data: "
        f"{result.artifacts.nonoverlap_external_path}"
    )

    print(
        f"  Overlap audit: "
        f"{result.artifacts.overlap_audit_path}"
    )

    print(
        f"  Summary: "
        f"{result.artifacts.overlap_summary_path}"
    )


def main() -> None:
    """Run external data preparation from the command line."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    result = run_external_preparation(
        external_input_path=(
            arguments.external_input
        ),
        training_input_path=(
            arguments.training_input
        ),
        external_output_directory=(
            arguments.external_output_dir
        ),
        report_directory=(
            arguments.report_dir
        ),
        overlap_rule=arguments.overlap_rule,
    )

    print_summary(result)


if __name__ == "__main__":
    main()