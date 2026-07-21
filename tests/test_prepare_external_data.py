"""Tests for scripts/prepare_external_data.py.

The external-data preparation script is loaded directly from its file path.
All tests use temporary CSV files and directories.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "prepare_external_data.py"
)


def load_external_module() -> ModuleType:
    """Load the external preparation script directly."""
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"External preparation script not found: "
            f"{SCRIPT_PATH}"
        )

    module_name = (
        "prepare_external_data_for_tests"
    )

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            SCRIPT_PATH,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            f"Could not import {SCRIPT_PATH}"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[module_name] = module

    try:
        specification.loader.exec_module(
            module
        )
    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    return module


external_module = load_external_module()


CLEAN_EXTERNAL_FILENAME = (
    external_module.CLEAN_EXTERNAL_FILENAME
)

NONOVERLAP_EXTERNAL_FILENAME = (
    external_module.NONOVERLAP_EXTERNAL_FILENAME
)

OVERLAP_AUDIT_FILENAME = (
    external_module.OVERLAP_AUDIT_FILENAME
)

OVERLAP_SUMMARY_FILENAME = (
    external_module.OVERLAP_SUMMARY_FILENAME
)

audit_overlap = external_module.audit_overlap

build_nonoverlap_external = (
    external_module.build_nonoverlap_external
)

get_overlap_mask = (
    external_module.get_overlap_mask
)

harmonize_external_data = (
    external_module.harmonize_external_data
)

load_csv = external_module.load_csv

normalize_name = external_module.normalize_name

normalize_ticket = (
    external_module.normalize_ticket
)

run_external_preparation = (
    external_module.run_external_preparation
)

validate_external_schema = (
    external_module.validate_external_schema
)

validate_training_schema = (
    external_module.validate_training_schema
)


def make_external_dataframe() -> pd.DataFrame:
    """Create passenger and crew source records."""
    return pd.DataFrame(
        {
            "name": [
                "Abbott, Mr. Rossmore Edward",
                "Andersson, Miss. Ellis Anna Maria",
                "Brown, Mr. Thomas",
                "Crewman, Mr. Engine",
                "Waiter, Mr. Restaurant",
                "Unknown, Mrs. New Passenger",
            ],
            "gender": [
                "male",
                "female",
                "male",
                "male",
                "male",
                "female",
            ],
            "age": [
                16.0,
                2.0,
                35.0,
                40.0,
                30.0,
                28.0,
            ],
            "class": [
                "3rd",
                "3rd",
                "1st",
                "engineering crew",
                "restaurant staff",
                "2nd",
            ],
            "embarked": [
                "Southampton",
                "Southampton",
                "Cherbourg",
                "Belfast",
                "Southampton",
                "Queenstown",
            ],
            "country": [
                "United States",
                "Sweden",
                "United Kingdom",
                "United Kingdom",
                "France",
                "Finland",
            ],
            "ticketno": [
                "C.A. 2673",
                "347082",
                "PC 17599",
                None,
                None,
                "NEW 123",
            ],
            "fare": [
                20.25,
                31.275,
                100.0,
                None,
                None,
                25.0,
            ],
            "sibsp": [
                1.0,
                4.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "parch": [
                1.0,
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "survived": [
                "no",
                "no",
                "yes",
                "no",
                "yes",
                "yes",
            ],
        }
    )


def make_training_dataframe() -> pd.DataFrame:
    """Create labelled rows with controlled overlap."""
    return pd.DataFrame(
        {
            "PassengerId": [
                1,
                2,
                3,
            ],
            "Name": [
                "Abbott, Mr. Rossmore Edward",
                "Different Passenger",
                "Ticket Match Passenger",
            ],
            "Ticket": [
                "UNRELATED",
                "347082",
                "PC-17599",
            ],
            "Pclass": [
                3,
                3,
                1,
            ],
        }
    )


def test_script_loaded() -> None:
    """The external script should load successfully."""
    assert external_module is not None
    assert hasattr(
        external_module,
        "run_external_preparation",
    )


def test_load_csv_reads_nonempty_file(
    tmp_path: Path,
) -> None:
    """A valid CSV should load."""
    path = tmp_path / "external.csv"

    make_external_dataframe().to_csv(
        path,
        index=False,
    )

    dataframe = load_csv(
        path,
        dataset_name="External",
    )

    assert len(dataframe) == 6


def test_load_csv_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """A missing source should fail."""
    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        load_csv(
            tmp_path / "missing.csv",
            dataset_name="External",
        )


def test_validate_external_schema_accepts_valid_data() -> None:
    """The representative external schema should pass."""
    validate_external_schema(
        make_external_dataframe()
    )


def test_validate_external_schema_rejects_missing_column() -> None:
    """Missing external columns should be reported."""
    dataframe = (
        make_external_dataframe()
        .drop(columns=["class"])
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_external_schema(
            dataframe
        )


def test_validate_training_schema_accepts_match_columns() -> None:
    """Name and Ticket are sufficient for overlap checks."""
    validate_training_schema(
        make_training_dataframe()
    )


def test_validate_training_schema_rejects_missing_ticket() -> None:
    """Training overlap detection requires Ticket."""
    dataframe = (
        make_training_dataframe()
        .drop(columns=["Ticket"])
    )

    with pytest.raises(
        ValueError,
        match="overlap detection",
    ):
        validate_training_schema(
            dataframe
        )


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        (
            "  Abbott, Mr. Rossmore Edward ",
            "abbott mr rossmore edward",
        ),
        (
            "Björkell, Mr. Pontus",
            "bjorkell mr pontus",
        ),
        (
            None,
            "",
        ),
    ],
)
def test_normalize_name(
    raw_name: object,
    expected: str,
) -> None:
    """Name normalization should remove punctuation and accents."""
    assert normalize_name(
        raw_name
    ) == expected


@pytest.mark.parametrize(
    ("raw_ticket", "expected"),
    [
        (
            "PC 17599",
            "PC17599",
        ),
        (
            "C.A. 2673",
            "CA2673",
        ),
        (
            "STON/O2. 3101282",
            "STONO23101282",
        ),
        (
            None,
            "",
        ),
    ],
)
def test_normalize_ticket(
    raw_ticket: object,
    expected: str,
) -> None:
    """Ticket normalization should ignore punctuation."""
    assert normalize_ticket(
        raw_ticket
    ) == expected


def test_harmonize_external_data_filters_crew() -> None:
    """Only first-, second-, and third-class passengers should remain."""
    clean, summary = harmonize_external_data(
        make_external_dataframe()
    )

    assert len(clean) == 4

    assert set(
        clean["Pclass"]
    ) == {
        1,
        2,
        3,
    }

    assert summary[
        "source_rows"
    ] == 6

    assert summary[
        "retained_passenger_rows"
    ] == 4

    assert summary[
        "excluded_nonpassenger_rows"
    ] == 2


def test_harmonize_external_data_maps_schema() -> None:
    """The harmonized output should follow project naming."""
    clean, _ = harmonize_external_data(
        make_external_dataframe()
    )

    expected_columns = {
        "PassengerId",
        "Survived",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
        "Country",
        "ExternalSourceRow",
    }

    assert expected_columns == set(
        clean.columns
    )

    assert clean[
        "PassengerId"
    ].is_unique

    assert clean[
        "Cabin"
    ].isna().all()

    assert set(
        clean["Survived"]
    ) == {
        0,
        1,
    }


def test_harmonize_external_data_maps_embarkation() -> None:
    """Full port names should become compact codes."""
    clean, _ = harmonize_external_data(
        make_external_dataframe()
    )

    assert set(
        clean["Embarked"].dropna()
    ) == {
        "C",
        "Q",
        "S",
    }


def test_audit_overlap_detects_name_and_ticket_matches() -> None:
    """The audit should distinguish matching evidence."""
    clean, _ = harmonize_external_data(
        make_external_dataframe()
    )

    audit = audit_overlap(
        clean,
        make_training_dataframe(),
    )

    abbott = audit.loc[
        audit["Name"].str.contains(
            "Abbott"
        )
    ].iloc[0]

    andersson = audit.loc[
        audit["Name"].str.contains(
            "Andersson"
        )
    ].iloc[0]

    brown = audit.loc[
        audit["Name"].str.contains(
            "Brown"
        )
    ].iloc[0]

    new_passenger = audit.loc[
        audit["Name"].str.contains(
            "New Passenger"
        )
    ].iloc[0]

    assert bool(
        abbott["NameMatch"]
    )

    assert not bool(
        abbott["TicketMatch"]
    )

    assert not bool(
        andersson["NameMatch"]
    )

    assert bool(
        andersson["TicketMatch"]
    )

    assert not bool(
        brown["NameMatch"]
    )

    assert bool(
        brown["TicketMatch"]
    )

    assert not bool(
        new_passenger["NameOrTicketMatch"]
    )


def test_get_overlap_mask_supports_all_rules() -> None:
    """Each configured rule should return a boolean mask."""
    clean, _ = harmonize_external_data(
        make_external_dataframe()
    )

    audit = audit_overlap(
        clean,
        make_training_dataframe(),
    )

    for rule in [
        "either",
        "both",
        "name",
        "ticket",
    ]:
        mask = get_overlap_mask(
            audit,
            overlap_rule=rule,
        )

        assert len(mask) == len(
            audit
        )

        assert mask.dtype == bool


def test_build_nonoverlap_external_uses_conservative_rule() -> None:
    """The either rule should remove name or ticket matches."""
    clean, _ = harmonize_external_data(
        make_external_dataframe()
    )

    audit = audit_overlap(
        clean,
        make_training_dataframe(),
    )

    nonoverlap = build_nonoverlap_external(
        audit,
        overlap_rule="either",
    )

    assert len(nonoverlap) == 1

    assert (
        nonoverlap.iloc[0]["Name"]
        == "Unknown, Mrs. New Passenger"
    )

    assert "NormalizedName" not in (
        nonoverlap.columns
    )

    assert "NameMatch" not in (
        nonoverlap.columns
    )


def test_run_external_preparation_end_to_end(
    tmp_path: Path,
) -> None:
    """The complete workflow should save all artifacts."""
    external_path = (
        tmp_path / "external.csv"
    )

    training_path = (
        tmp_path / "train.csv"
    )

    external_output_directory = (
        tmp_path / "data_external"
    )

    report_directory = (
        tmp_path / "reports_external"
    )

    make_external_dataframe().to_csv(
        external_path,
        index=False,
    )

    make_training_dataframe().to_csv(
        training_path,
        index=False,
    )

    result = run_external_preparation(
        external_input_path=external_path,
        training_input_path=training_path,
        external_output_directory=(
            external_output_directory
        ),
        report_directory=report_directory,
        overlap_rule="either",
    )

    assert len(
        result.clean_external
    ) == 4

    assert len(
        result.nonoverlap_external
    ) == 1

    assert (
        external_output_directory
        / CLEAN_EXTERNAL_FILENAME
    ).exists()

    assert (
        external_output_directory
        / NONOVERLAP_EXTERNAL_FILENAME
    ).exists()

    assert (
        report_directory
        / OVERLAP_AUDIT_FILENAME
    ).exists()

    assert (
        report_directory
        / OVERLAP_SUMMARY_FILENAME
    ).exists()


def test_saved_summary_contains_overlap_counts(
    tmp_path: Path,
) -> None:
    """The JSON report should describe the selected overlap rule."""
    external_path = (
        tmp_path / "external.csv"
    )

    training_path = (
        tmp_path / "train.csv"
    )

    output_directory = (
        tmp_path / "output"
    )

    report_directory = (
        tmp_path / "reports"
    )

    make_external_dataframe().to_csv(
        external_path,
        index=False,
    )

    make_training_dataframe().to_csv(
        training_path,
        index=False,
    )

    result = run_external_preparation(
        external_input_path=external_path,
        training_input_path=training_path,
        external_output_directory=(
            output_directory
        ),
        report_directory=report_directory,
        overlap_rule="either",
    )

    summary = json.loads(
        result.artifacts.overlap_summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert summary[
        "source_rows"
    ] == 6

    assert summary[
        "retained_passenger_rows"
    ] == 4

    assert summary[
        "nonoverlap_rows"
    ] == 1

    assert summary[
        "overlap_rule_for_strict_dataset"
    ] == "either"


def test_nonoverlap_output_preserves_target_labels(
    tmp_path: Path,
) -> None:
    """The strict external dataset should remain evaluable."""
    external_path = (
        tmp_path / "external.csv"
    )

    training_path = (
        tmp_path / "train.csv"
    )

    make_external_dataframe().to_csv(
        external_path,
        index=False,
    )

    make_training_dataframe().to_csv(
        training_path,
        index=False,
    )

    result = run_external_preparation(
        external_input_path=external_path,
        training_input_path=training_path,
        external_output_directory=(
            tmp_path / "output"
        ),
        report_directory=(
            tmp_path / "reports"
        ),
    )

    assert "Pclass" in (
        result.nonoverlap_external.columns
    )

    assert not result.nonoverlap_external[
        "Pclass"
    ].isna().any()