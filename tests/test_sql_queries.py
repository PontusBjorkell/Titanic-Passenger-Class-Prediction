"""Integration tests for the production analytical SQL queries.

The complete sql/analysis_queries.sql file is executed against a temporary
Titanic-style SQLite database.

These tests ensure that:

- every query has a unique name;
- the expected production analyses exist;
- all queries are read-only SELECT statements;
- every query executes successfully;
- every result has named columns;
- the entire analysis workflow can export all production results.

No real project data or database is modified.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ANALYSIS_PATH = PROJECT_ROOT / "scripts" / "run_analysis.py"
ANALYSIS_SQL_PATH = PROJECT_ROOT / "sql" / "analysis_queries.sql"


EXPECTED_QUERY_NAMES = [
    "01_dataset_overview",
    "02_passenger_class_distribution",
    "03_class_profile",
    "04_survival_by_class_and_sex",
    "05_fare_statistics_by_class",
    "06_age_profile_by_class",
    "07_embarkation_class_mix",
    "08_embarkation_outcomes",
    "09_family_profile",
    "10_family_survival_profile",
    "11_cabin_profile",
    "12_cabin_access_and_outcomes",
    "13_title_profile",
    "14_ticket_prefix_profile",
    "15_high_fare_passengers",
    "16_largest_family_groups",
    "17_feature_summary",
    "18_missing_data_by_class",
    "19_class_prediction_feature_profile",
    "20_potential_duplicate_tickets",
]


def load_run_analysis_module() -> ModuleType:
    """Import scripts/run_analysis.py from its path."""
    if not RUN_ANALYSIS_PATH.exists():
        raise FileNotFoundError(
            f"Analysis script not found: "
            f"{RUN_ANALYSIS_PATH}"
        )

    module_name = "run_analysis_for_sql_tests"

    specification = importlib.util.spec_from_file_location(
        module_name,
        RUN_ANALYSIS_PATH,
    )

    if specification is None or specification.loader is None:
        raise ImportError(
            f"Could not import {RUN_ANALYSIS_PATH}"
        )

    module = importlib.util.module_from_spec(specification)

    # Register the module before execution so dataclasses can resolve
    # the module namespace correctly.
    sys.modules[module_name] = module

    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


@pytest.fixture(scope="module")
def run_analysis_module() -> ModuleType:
    """Provide the imported analysis module."""
    return load_run_analysis_module()


def build_representative_passenger_frame() -> pd.DataFrame:
    """Construct data covering the values used by all SQL analyses."""
    records = [
        {
            "PassengerId": 1,
            "Survived": 1,
            "Pclass": 1,
            "Name": "Allen, Mr. William",
            "Sex": "male",
            "Age": 35.0,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "PC 17599",
            "Fare": 120.00,
            "Cabin": "C85",
            "Embarked": "C",
            "FamilySize": 1,
            "IsAlone": 1,
            "FamilySizeGroup": "Alone",
            "Title": "Mr",
            "TicketPrefix": "PC",
            "CabinCount": 1,
            "CabinDeck": "C",
            "HasCabin": 1,
            "FareLog": np.log1p(120.00),
        },
        {
            "PassengerId": 2,
            "Survived": 1,
            "Pclass": 1,
            "Name": "Brown, Mrs. Alice",
            "Sex": "female",
            "Age": 38.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "PC 17599",
            "Fare": 120.00,
            "Cabin": "C123",
            "Embarked": "C",
            "FamilySize": 2,
            "IsAlone": 0,
            "FamilySizeGroup": "Small",
            "Title": "Mrs",
            "TicketPrefix": "PC",
            "CabinCount": 1,
            "CabinDeck": "C",
            "HasCabin": 1,
            "FareLog": np.log1p(120.00),
        },
        {
            "PassengerId": 3,
            "Survived": 0,
            "Pclass": 1,
            "Name": "Clark, Mr. Edward",
            "Sex": "male",
            "Age": 62.0,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "113803",
            "Fare": 80.00,
            "Cabin": "B28",
            "Embarked": "S",
            "FamilySize": 1,
            "IsAlone": 1,
            "FamilySizeGroup": "Alone",
            "Title": "Mr",
            "TicketPrefix": "NONE",
            "CabinCount": 1,
            "CabinDeck": "B",
            "HasCabin": 1,
            "FareLog": np.log1p(80.00),
        },
        {
            "PassengerId": 4,
            "Survived": 1,
            "Pclass": 2,
            "Name": "Davis, Miss. Emily",
            "Sex": "female",
            "Age": 24.0,
            "SibSp": 0,
            "Parch": 1,
            "Ticket": "SC PARIS 2167",
            "Fare": 32.50,
            "Cabin": None,
            "Embarked": "S",
            "FamilySize": 2,
            "IsAlone": 0,
            "FamilySizeGroup": "Small",
            "Title": "Miss",
            "TicketPrefix": "SC_PARIS",
            "CabinCount": 0,
            "CabinDeck": "Unknown",
            "HasCabin": 0,
            "FareLog": np.log1p(32.50),
        },
        {
            "PassengerId": 5,
            "Survived": 0,
            "Pclass": 2,
            "Name": "Evans, Mr. Henry",
            "Sex": "male",
            "Age": 42.0,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "248738",
            "Fare": 13.00,
            "Cabin": None,
            "Embarked": "S",
            "FamilySize": 1,
            "IsAlone": 1,
            "FamilySizeGroup": "Alone",
            "Title": "Mr",
            "TicketPrefix": "NONE",
            "CabinCount": 0,
            "CabinDeck": "Unknown",
            "HasCabin": 0,
            "FareLog": np.log1p(13.00),
        },
        {
            "PassengerId": 6,
            "Survived": 1,
            "Pclass": 2,
            "Name": "Foster, Master. James",
            "Sex": "male",
            "Age": 8.0,
            "SibSp": 1,
            "Parch": 2,
            "Ticket": "C.A. 33112",
            "Fare": 26.00,
            "Cabin": "F2",
            "Embarked": "S",
            "FamilySize": 4,
            "IsAlone": 0,
            "FamilySizeGroup": "Medium",
            "Title": "Master",
            "TicketPrefix": "CA",
            "CabinCount": 1,
            "CabinDeck": "F",
            "HasCabin": 1,
            "FareLog": np.log1p(26.00),
        },
        {
            "PassengerId": 7,
            "Survived": 0,
            "Pclass": 3,
            "Name": "Green, Mr. Thomas",
            "Sex": "male",
            "Age": 21.0,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": None,
            "Embarked": "S",
            "FamilySize": 1,
            "IsAlone": 1,
            "FamilySizeGroup": "Alone",
            "Title": "Mr",
            "TicketPrefix": "A5",
            "CabinCount": 0,
            "CabinDeck": "Unknown",
            "HasCabin": 0,
            "FareLog": np.log1p(7.25),
        },
        {
            "PassengerId": 8,
            "Survived": 1,
            "Pclass": 3,
            "Name": "Hill, Miss. Sarah",
            "Sex": "female",
            "Age": 18.0,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "STON/O2 3101282",
            "Fare": 7.92,
            "Cabin": None,
            "Embarked": "Q",
            "FamilySize": 1,
            "IsAlone": 1,
            "FamilySizeGroup": "Alone",
            "Title": "Miss",
            "TicketPrefix": "STONO2",
            "CabinCount": 0,
            "CabinDeck": "Unknown",
            "HasCabin": 0,
            "FareLog": np.log1p(7.92),
        },
        {
            "PassengerId": 9,
            "Survived": 0,
            "Pclass": 3,
            "Name": "Irwin, Mrs. Mary",
            "Sex": "female",
            "Age": 29.0,
            "SibSp": 3,
            "Parch": 2,
            "Ticket": "347082",
            "Fare": 31.28,
            "Cabin": None,
            "Embarked": "S",
            "FamilySize": 6,
            "IsAlone": 0,
            "FamilySizeGroup": "Large",
            "Title": "Mrs",
            "TicketPrefix": "NONE",
            "CabinCount": 0,
            "CabinDeck": "Unknown",
            "HasCabin": 0,
            "FareLog": np.log1p(31.28),
        },
        {
            "PassengerId": 10,
            "Survived": 0,
            "Pclass": 3,
            "Name": "Jones, Mr. Arthur",
            "Sex": "male",
            "Age": None,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "347082",
            "Fare": 31.28,
            "Cabin": None,
            "Embarked": None,
            "FamilySize": 2,
            "IsAlone": 0,
            "FamilySizeGroup": "Small",
            "Title": "Mr",
            "TicketPrefix": "NONE",
            "CabinCount": 0,
            "CabinDeck": "Unknown",
            "HasCabin": 0,
            "FareLog": np.log1p(31.28),
        },
        {
            "PassengerId": 11,
            "Survived": 1,
            "Pclass": 1,
            "Name": "King, Dr. Anna",
            "Sex": "female",
            "Age": 48.0,
            "SibSp": 0,
            "Parch": 0,
            "Ticket": "PC 17757",
            "Fare": 512.33,
            "Cabin": "B51 B53 B55",
            "Embarked": "C",
            "FamilySize": 1,
            "IsAlone": 1,
            "FamilySizeGroup": "Alone",
            "Title": "Rare",
            "TicketPrefix": "PC",
            "CabinCount": 3,
            "CabinDeck": "B",
            "HasCabin": 1,
            "FareLog": np.log1p(512.33),
        },
        {
            "PassengerId": 12,
            "Survived": 0,
            "Pclass": 3,
            "Name": "Lewis, Master. Peter",
            "Sex": "male",
            "Age": 4.0,
            "SibSp": 4,
            "Parch": 2,
            "Ticket": "CA 2144",
            "Fare": 46.90,
            "Cabin": None,
            "Embarked": "S",
            "FamilySize": 7,
            "IsAlone": 0,
            "FamilySizeGroup": "Large",
            "Title": "Master",
            "TicketPrefix": "CA",
            "CabinCount": 0,
            "CabinDeck": "Unknown",
            "HasCabin": 0,
            "FareLog": np.log1p(46.90),
        },
    ]

    return pd.DataFrame.from_records(records)


def create_production_views(
    connection: sqlite3.Connection,
) -> None:
    """Create the views referenced by production analysis queries."""
    connection.executescript(
        """
        DROP VIEW IF EXISTS vw_class_profile;
        DROP VIEW IF EXISTS vw_embarkation_class_mix;
        DROP VIEW IF EXISTS vw_family_profile;
        DROP VIEW IF EXISTS vw_cabin_profile;

        CREATE VIEW vw_class_profile AS
        SELECT
            Pclass AS passenger_class,
            COUNT(*) AS passenger_count,
            ROUND(AVG(Age), 2) AS average_age,
            ROUND(AVG(Fare), 2) AS average_fare,
            ROUND(AVG(FamilySize), 2) AS average_family_size,
            ROUND(100.0 * AVG(IsAlone), 2)
                AS percent_travelling_alone,
            ROUND(100.0 * AVG(HasCabin), 2)
                AS percent_with_cabin
        FROM passengers
        GROUP BY Pclass;

        CREATE VIEW vw_embarkation_class_mix AS
        WITH port_class_counts AS (
            SELECT
                COALESCE(Embarked, 'Unknown')
                    AS embarkation_port,
                Pclass AS passenger_class,
                COUNT(*) AS passenger_count
            FROM passengers
            GROUP BY
                COALESCE(Embarked, 'Unknown'),
                Pclass
        ),
        port_totals AS (
            SELECT
                embarkation_port,
                SUM(passenger_count) AS port_total
            FROM port_class_counts
            GROUP BY embarkation_port
        )
        SELECT
            counts.embarkation_port,
            counts.passenger_class,
            counts.passenger_count,
            ROUND(
                100.0 * counts.passenger_count
                / totals.port_total,
                2
            ) AS percentage_within_port
        FROM port_class_counts AS counts
        JOIN port_totals AS totals
            ON counts.embarkation_port
                = totals.embarkation_port;

        CREATE VIEW vw_family_profile AS
        SELECT
            FamilySizeGroup AS family_size_group,
            Pclass AS passenger_class,
            COUNT(*) AS passenger_count,
            ROUND(AVG(Fare), 2) AS average_fare
        FROM passengers
        GROUP BY FamilySizeGroup, Pclass;

        CREATE VIEW vw_cabin_profile AS
        SELECT
            CabinDeck AS cabin_deck,
            Pclass AS passenger_class,
            COUNT(*) AS passenger_count
        FROM passengers
        GROUP BY CabinDeck, Pclass;
        """
    )


@pytest.fixture(scope="module")
def production_style_database(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Create one representative database for SQL integration tests."""
    temporary_directory = tmp_path_factory.mktemp(
        "production_sql_database"
    )

    database_path = (
        temporary_directory / "titanic_test.db"
    )

    passengers = build_representative_passenger_frame()

    with sqlite3.connect(database_path) as connection:
        passengers.to_sql(
            "passengers",
            connection,
            if_exists="replace",
            index=False,
        )

        create_production_views(connection)
        connection.commit()

    return database_path


def test_production_sql_file_exists() -> None:
    """The production SQL analysis file must be present."""
    assert ANALYSIS_SQL_PATH.exists()
    assert ANALYSIS_SQL_PATH.is_file()


def test_production_sql_contains_expected_queries(
    run_analysis_module: ModuleType,
) -> None:
    """The SQL file should expose the intended twenty analyses."""
    queries = run_analysis_module.load_named_queries(
        ANALYSIS_SQL_PATH
    )

    actual_names = [query.name for query in queries]

    assert actual_names == EXPECTED_QUERY_NAMES


def test_production_query_names_are_unique(
    run_analysis_module: ModuleType,
) -> None:
    """Every query must map to a unique CSV filename."""
    queries = run_analysis_module.load_named_queries(
        ANALYSIS_SQL_PATH
    )

    names = [query.name for query in queries]

    assert len(names) == len(set(names))


def test_every_production_query_has_description(
    run_analysis_module: ModuleType,
) -> None:
    """Descriptions support the manifest and Streamlit interface."""
    queries = run_analysis_module.load_named_queries(
        ANALYSIS_SQL_PATH
    )

    missing_descriptions = [
        query.name
        for query in queries
        if not query.description
    ]

    assert missing_descriptions == []


@pytest.mark.parametrize(
    "forbidden_token",
    [
        "INSERT ",
        "UPDATE ",
        "DELETE ",
        "DROP ",
        "ALTER ",
        "REPLACE ",
        "CREATE ",
        "ATTACH ",
        "DETACH ",
        "VACUUM ",
    ],
)
def test_production_queries_are_read_only(
    run_analysis_module: ModuleType,
    forbidden_token: str,
) -> None:
    """Analytical queries must not modify the database."""
    queries = run_analysis_module.load_named_queries(
        ANALYSIS_SQL_PATH
    )

    offending_queries = []

    for query in queries:
        normalized_sql = (
            " ".join(query.sql.upper().split()) + " "
        )

        if forbidden_token in normalized_sql:
            offending_queries.append(query.name)

    assert offending_queries == []


def test_every_production_query_executes(
    run_analysis_module: ModuleType,
    production_style_database: Path,
) -> None:
    """All production queries should run against valid input data."""
    queries = run_analysis_module.load_named_queries(
        ANALYSIS_SQL_PATH
    )

    failures: list[str] = []

    for query in queries:
        try:
            result = (
                run_analysis_module.query_database(
                    query=query.sql,
                    database_path=production_style_database,
                )
            )
        except Exception as error:
            failures.append(
                f"{query.name}: "
                f"{type(error).__name__}: {error}"
            )
            continue

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) > 0
        assert all(
            str(column).strip()
            for column in result.columns
        )

    assert failures == [], (
        "One or more production SQL queries failed:\n"
        + "\n".join(failures)
    )


def test_each_query_exports_to_its_own_csv(
    run_analysis_module: ModuleType,
    production_style_database: Path,
    tmp_path: Path,
) -> None:
    """Every production analysis should create a separate CSV."""
    output_directory = tmp_path / "query_exports"

    results = run_analysis_module.run_analyses(
        database_path=production_style_database,
        sql_path=ANALYSIS_SQL_PATH,
        output_directory=output_directory,
    )

    assert len(results) == len(EXPECTED_QUERY_NAMES)

    for query_name in EXPECTED_QUERY_NAMES:
        output_path = (
            output_directory / f"{query_name}.csv"
        )

        assert output_path.exists(), (
            f"Expected output was not created: "
            f"{output_path}"
        )

        exported = pd.read_csv(output_path)

        assert len(exported.columns) > 0


def test_production_manifest_describes_all_queries(
    run_analysis_module: ModuleType,
    production_style_database: Path,
    tmp_path: Path,
) -> None:
    """The manifest should contain all exported analyses."""
    output_directory = tmp_path / "manifest_test"

    run_analysis_module.run_analyses(
        database_path=production_style_database,
        sql_path=ANALYSIS_SQL_PATH,
        output_directory=output_directory,
    )

    manifest_path = (
        output_directory / "analysis_manifest.json"
    )

    assert manifest_path.exists()

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["query_count"] == len(
        EXPECTED_QUERY_NAMES
    )

    manifest_names = [
        query_metadata["name"]
        for query_metadata in manifest["queries"]
    ]

    assert manifest_names == EXPECTED_QUERY_NAMES

    for query_metadata in manifest["queries"]:
        assert query_metadata["description"]
        assert query_metadata["rows"] >= 0
        assert query_metadata["columns"] > 0
        assert query_metadata["column_names"]


def test_dataset_overview_result_is_correct(
    run_analysis_module: ModuleType,
    production_style_database: Path,
) -> None:
    """The overview query should report known fixture totals."""
    query = next(
        query
        for query in run_analysis_module.load_named_queries(
            ANALYSIS_SQL_PATH
        )
        if query.name == "01_dataset_overview"
    )

    result = run_analysis_module.query_database(
        query=query.sql,
        database_path=production_style_database,
    )

    assert int(result.loc[0, "passenger_count"]) == 12
    assert int(result.loc[0, "unique_passengers"]) == 12
    assert int(result.loc[0, "passenger_classes"]) == 3
    assert int(result.loc[0, "missing_age_count"]) == 1
    assert int(
        result.loc[0, "missing_embarkation_count"]
    ) == 1


def test_class_distribution_totals_match_dataset(
    run_analysis_module: ModuleType,
    production_style_database: Path,
) -> None:
    """Class-distribution counts should sum to all passengers."""
    query = next(
        query
        for query in run_analysis_module.load_named_queries(
            ANALYSIS_SQL_PATH
        )
        if query.name
        == "02_passenger_class_distribution"
    )

    result = run_analysis_module.query_database(
        query=query.sql,
        database_path=production_style_database,
    )

    assert set(result["passenger_class"]) == {
        1,
        2,
        3,
    }

    assert int(result["passenger_count"].sum()) == 12

    assert result[
        "dataset_share_percent"
    ].sum() == pytest.approx(
        100.0,
        abs=0.02,
    )


def test_high_fare_query_is_sorted_descending(
    run_analysis_module: ModuleType,
    production_style_database: Path,
) -> None:
    """The high-fare output should place the largest fare first."""
    query = next(
        query
        for query in run_analysis_module.load_named_queries(
            ANALYSIS_SQL_PATH
        )
        if query.name == "15_high_fare_passengers"
    )

    result = run_analysis_module.query_database(
        query=query.sql,
        database_path=production_style_database,
    )

    fares = result["fare"].tolist()

    assert fares == sorted(
        fares,
        reverse=True,
    )

    assert result.loc[0, "passenger_id"] == 11
    assert result.loc[0, "fare"] == pytest.approx(
        512.33
    )


def test_duplicate_ticket_query_finds_shared_tickets(
    run_analysis_module: ModuleType,
    production_style_database: Path,
) -> None:
    """The duplicate-ticket query should return known shared tickets."""
    query = next(
        query
        for query in run_analysis_module.load_named_queries(
            ANALYSIS_SQL_PATH
        )
        if query.name
        == "20_potential_duplicate_tickets"
    )

    result = run_analysis_module.query_database(
        query=query.sql,
        database_path=production_style_database,
    )

    tickets = set(result["ticket"].astype(str))

    assert "PC 17599" in tickets
    assert "347082" in tickets

    assert (
        result["passenger_count"] > 1
    ).all()