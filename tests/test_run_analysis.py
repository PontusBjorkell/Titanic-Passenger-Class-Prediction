"""Tests for scripts/run_analysis.py.

The tests import the analysis script as a Python module and exercise its
parsing, validation, execution, export, and manifest-generation behavior.

All database and output operations use pytest temporary directories.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ANALYSIS_PATH = PROJECT_ROOT / "scripts" / "run_analysis.py"


def load_run_analysis_module() -> ModuleType:
    """Import scripts/run_analysis.py from its filesystem path."""
    if not RUN_ANALYSIS_PATH.exists():
        raise FileNotFoundError(
            f"Expected analysis script was not found: "
            f"{RUN_ANALYSIS_PATH}"
        )

    module_name = "run_analysis_for_tests"

    specification = importlib.util.spec_from_file_location(
        module_name,
        RUN_ANALYSIS_PATH,
    )

    if specification is None or specification.loader is None:
        raise ImportError(
            f"Could not import module from {RUN_ANALYSIS_PATH}"
        )

    module = importlib.util.module_from_spec(specification)

    # Dataclasses and other runtime features expect the module to be
    # registered while its source code is being executed.
    sys.modules[module_name] = module

    try:
        specification.loader.exec_module(module)
    except Exception:
        # Avoid retaining a partially imported module after a failure.
        sys.modules.pop(module_name, None)
        raise

    return module


@pytest.fixture(scope="module")
def run_analysis_module() -> ModuleType:
    """Provide the imported run_analysis module."""
    return load_run_analysis_module()


@pytest.fixture
def miniature_database(tmp_path: Path) -> Path:
    """Create a minimal passengers database for orchestration tests."""
    database_path = tmp_path / "miniature_titanic.db"

    frame = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "Pclass": [1, 1, 2, 3, 3],
            "Sex": [
                "female",
                "male",
                "female",
                "male",
                "female",
            ],
            "Age": [29.0, 45.0, 31.0, 22.0, None],
            "Fare": [100.0, 75.0, 30.0, 8.0, 12.0],
            "Survived": [1, 0, 1, 0, 1],
        }
    )

    with sqlite3.connect(database_path) as connection:
        frame.to_sql(
            "passengers",
            connection,
            if_exists="replace",
            index=False,
        )

    return database_path


@pytest.fixture
def miniature_sql_file(tmp_path: Path) -> Path:
    """Create a small named-query SQL file."""
    sql_path = tmp_path / "analysis_queries.sql"

    sql_path.write_text(
        """
-- name: 01_dataset_overview
-- description: Count all passengers.
SELECT
    COUNT(*) AS passenger_count
FROM passengers;


-- name: 02_class_distribution
-- description: Count passengers by class.
SELECT
    Pclass AS passenger_class,
    COUNT(*) AS passenger_count
FROM passengers
GROUP BY Pclass
ORDER BY Pclass;


-- name: 03_average_fare
-- description: Calculate the average fare.
SELECT
    ROUND(AVG(Fare), 2) AS average_fare
FROM passengers;
""".strip(),
        encoding="utf-8",
    )

    return sql_path


def test_validate_query_name_accepts_safe_names(
    run_analysis_module: ModuleType,
) -> None:
    """Valid filename-safe query names should be accepted."""
    assert (
        run_analysis_module.validate_query_name(
            "01_dataset_overview"
        )
        == "01_dataset_overview"
    )

    assert (
        run_analysis_module.validate_query_name(
            "query-name-2"
        )
        == "query-name-2"
    )


@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        "   ",
        "query name",
        "query/name",
        "query.name",
        "../query",
        "query;drop",
    ],
)
def test_validate_query_name_rejects_unsafe_names(
    run_analysis_module: ModuleType,
    invalid_name: str,
) -> None:
    """Unsafe query names should be rejected."""
    with pytest.raises(ValueError):
        run_analysis_module.validate_query_name(
            invalid_name
        )


def test_parse_named_queries_returns_expected_queries(
    run_analysis_module: ModuleType,
) -> None:
    """The parser should extract names, descriptions, and SQL."""
    sql_text = """
-- name: first_query
-- description: Return one.
SELECT 1 AS value;

-- name: second_query
-- description: Return two.
SELECT 2 AS value;
"""

    queries = run_analysis_module.parse_named_queries(
        sql_text
    )

    assert len(queries) == 2

    assert queries[0].name == "first_query"
    assert queries[0].description == "Return one."
    assert "SELECT 1 AS value" in queries[0].sql

    assert queries[1].name == "second_query"
    assert queries[1].description == "Return two."
    assert "SELECT 2 AS value" in queries[1].sql


def test_parse_named_queries_preserves_query_order(
    run_analysis_module: ModuleType,
) -> None:
    """Queries should remain in their source-file order."""
    sql_text = """
-- name: query_c
SELECT 3 AS value;

-- name: query_a
SELECT 1 AS value;

-- name: query_b
SELECT 2 AS value;
"""

    queries = run_analysis_module.parse_named_queries(
        sql_text
    )

    assert [query.name for query in queries] == [
        "query_c",
        "query_a",
        "query_b",
    ]


def test_parse_named_queries_allows_missing_description(
    run_analysis_module: ModuleType,
) -> None:
    """Descriptions should be optional."""
    queries = run_analysis_module.parse_named_queries(
        """
-- name: simple_query
SELECT 1 AS value;
"""
    )

    assert len(queries) == 1
    assert queries[0].description is None


@pytest.mark.parametrize(
    "sql_text",
    [
        "",
        "   \n\n",
        "-- This file contains no named queries\nSELECT 1;",
    ],
)
def test_parse_named_queries_rejects_missing_queries(
    run_analysis_module: ModuleType,
    sql_text: str,
) -> None:
    """An empty or unnamed SQL file should be rejected."""
    with pytest.raises(ValueError):
        run_analysis_module.parse_named_queries(
            sql_text
        )


def test_parse_named_queries_rejects_query_without_sql(
    run_analysis_module: ModuleType,
) -> None:
    """A name marker without executable SQL should fail."""
    with pytest.raises(ValueError):
        run_analysis_module.parse_named_queries(
            """
-- name: empty_query
-- description: This query has no SQL.
"""
        )


def test_parse_named_queries_rejects_duplicate_names(
    run_analysis_module: ModuleType,
) -> None:
    """Duplicate query names would overwrite CSV files and must fail."""
    sql_text = """
-- name: duplicate_query
SELECT 1 AS value;

-- name: duplicate_query
SELECT 2 AS value;
"""

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        run_analysis_module.parse_named_queries(
            sql_text
        )


def test_load_named_queries_reads_file(
    run_analysis_module: ModuleType,
    miniature_sql_file: Path,
) -> None:
    """Named queries should be loadable from disk."""
    queries = run_analysis_module.load_named_queries(
        miniature_sql_file
    )

    assert len(queries) == 3
    assert queries[0].name == "01_dataset_overview"
    assert queries[-1].name == "03_average_fare"


def test_load_named_queries_rejects_missing_file(
    run_analysis_module: ModuleType,
    tmp_path: Path,
) -> None:
    """A nonexistent analysis file should raise FileNotFoundError."""
    missing_file = tmp_path / "missing.sql"

    with pytest.raises(FileNotFoundError):
        run_analysis_module.load_named_queries(
            missing_file
        )


def test_select_queries_returns_all_when_none_requested(
    run_analysis_module: ModuleType,
) -> None:
    """No requested-name filter should select all queries."""
    queries = [
        run_analysis_module.NamedQuery(
            name="query_1",
            sql="SELECT 1;",
        ),
        run_analysis_module.NamedQuery(
            name="query_2",
            sql="SELECT 2;",
        ),
    ]

    selected = run_analysis_module.select_queries(
        queries,
        requested_names=None,
    )

    assert selected == queries


def test_select_queries_returns_requested_subset_in_file_order(
    run_analysis_module: ModuleType,
) -> None:
    """Selection should preserve SQL-file order."""
    queries = [
        run_analysis_module.NamedQuery(
            name="query_1",
            sql="SELECT 1;",
        ),
        run_analysis_module.NamedQuery(
            name="query_2",
            sql="SELECT 2;",
        ),
        run_analysis_module.NamedQuery(
            name="query_3",
            sql="SELECT 3;",
        ),
    ]

    selected = run_analysis_module.select_queries(
        queries,
        requested_names=[
            "query_3",
            "query_1",
        ],
    )

    assert [query.name for query in selected] == [
        "query_1",
        "query_3",
    ]


def test_select_queries_rejects_unknown_name(
    run_analysis_module: ModuleType,
) -> None:
    """An unavailable query name should fail clearly."""
    queries = [
        run_analysis_module.NamedQuery(
            name="known_query",
            sql="SELECT 1;",
        )
    ]

    with pytest.raises(
        ValueError,
        match="Unknown query",
    ):
        run_analysis_module.select_queries(
            queries,
            requested_names=["unknown_query"],
        )


def test_select_queries_rejects_duplicate_request(
    run_analysis_module: ModuleType,
) -> None:
    """The same requested query should not be accepted twice."""
    queries = [
        run_analysis_module.NamedQuery(
            name="query_1",
            sql="SELECT 1;",
        )
    ]

    with pytest.raises(
        ValueError,
        match="more than once",
    ):
        run_analysis_module.select_queries(
            queries,
            requested_names=[
                "query_1",
                "query_1",
            ],
        )


def test_validate_database_accepts_passengers_table(
    run_analysis_module: ModuleType,
    miniature_database: Path,
) -> None:
    """A database containing passengers should pass validation."""
    run_analysis_module.validate_database(
        miniature_database
    )


def test_validate_database_rejects_missing_file(
    run_analysis_module: ModuleType,
    tmp_path: Path,
) -> None:
    """A missing database should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_analysis_module.validate_database(
            tmp_path / "missing.db"
        )


def test_validate_database_rejects_database_without_passengers(
    run_analysis_module: ModuleType,
    tmp_path: Path,
) -> None:
    """The required passengers table must exist."""
    database_path = tmp_path / "wrong_schema.db"

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE unrelated_table (value INTEGER)"
        )
        connection.commit()

    with pytest.raises(
        ValueError,
        match="passengers",
    ):
        run_analysis_module.validate_database(
            database_path
        )


def test_export_query_result_creates_csv(
    run_analysis_module: ModuleType,
    miniature_database: Path,
    tmp_path: Path,
) -> None:
    """Executing one query should produce a readable CSV."""
    query = run_analysis_module.NamedQuery(
        name="class_counts",
        description="Count passengers by class.",
        sql="""
            SELECT
                Pclass AS passenger_class,
                COUNT(*) AS passenger_count
            FROM passengers
            GROUP BY Pclass
            ORDER BY Pclass
        """,
    )

    output_directory = tmp_path / "exports"

    result = run_analysis_module.export_query_result(
        query=query,
        database_path=miniature_database,
        output_directory=output_directory,
    )

    output_path = output_directory / "class_counts.csv"

    assert output_path.exists()
    assert result.name == "class_counts"
    assert result.rows == 3
    assert result.columns == 2
    assert result.column_names == [
        "passenger_class",
        "passenger_count",
    ]

    exported = pd.read_csv(output_path)

    assert exported["passenger_class"].tolist() == [
        1,
        2,
        3,
    ]
    assert exported["passenger_count"].tolist() == [
        2,
        1,
        2,
    ]


def test_save_manifest_creates_valid_json(
    run_analysis_module: ModuleType,
    tmp_path: Path,
) -> None:
    """The analysis manifest should contain query metadata."""
    output_directory = tmp_path / "reports"

    results = [
        run_analysis_module.AnalysisResult(
            name="sample_query",
            description="A sample query.",
            output_path=str(
                output_directory / "sample_query.csv"
            ),
            rows=4,
            columns=2,
            column_names=["column_a", "column_b"],
        )
    ]

    manifest_path = run_analysis_module.save_manifest(
        results=results,
        output_directory=output_directory,
        database_path=tmp_path / "database.db",
        sql_path=tmp_path / "queries.sql",
    )

    assert manifest_path.exists()

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["query_count"] == 1
    assert manifest["queries"][0]["name"] == "sample_query"
    assert manifest["queries"][0]["rows"] == 4
    assert manifest["queries"][0]["columns"] == 2
    assert manifest["queries"][0]["column_names"] == [
        "column_a",
        "column_b",
    ]


def test_build_summary_dataframe(
    run_analysis_module: ModuleType,
) -> None:
    """Exported results should be convertible into a summary table."""
    results = [
        run_analysis_module.AnalysisResult(
            name="query_1",
            description=None,
            output_path="reports/query_1.csv",
            rows=10,
            columns=3,
            column_names=["a", "b", "c"],
        )
    ]

    summary = (
        run_analysis_module.build_summary_dataframe(
            results
        )
    )

    assert isinstance(summary, pd.DataFrame)
    assert summary.shape == (1, 4)
    assert summary.loc[0, "query"] == "query_1"
    assert summary.loc[0, "rows"] == 10
    assert summary.loc[0, "columns"] == 3


def test_run_analyses_end_to_end(
    run_analysis_module: ModuleType,
    miniature_database: Path,
    miniature_sql_file: Path,
    tmp_path: Path,
) -> None:
    """A complete temporary run should export every query and a manifest."""
    output_directory = tmp_path / "sql_reports"

    results = run_analysis_module.run_analyses(
        database_path=miniature_database,
        sql_path=miniature_sql_file,
        output_directory=output_directory,
    )

    assert len(results) == 3

    expected_files = {
        "01_dataset_overview.csv",
        "02_class_distribution.csv",
        "03_average_fare.csv",
        "analysis_manifest.json",
    }

    actual_files = {
        path.name
        for path in output_directory.iterdir()
        if path.is_file()
    }

    assert expected_files == actual_files

    overview = pd.read_csv(
        output_directory / "01_dataset_overview.csv"
    )

    assert int(overview.loc[0, "passenger_count"]) == 5

    class_distribution = pd.read_csv(
        output_directory / "02_class_distribution.csv"
    )

    assert class_distribution[
        "passenger_count"
    ].tolist() == [2, 1, 2]

    manifest = json.loads(
        (
            output_directory / "analysis_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["query_count"] == 3


def test_run_analyses_can_execute_selected_queries(
    run_analysis_module: ModuleType,
    miniature_database: Path,
    miniature_sql_file: Path,
    tmp_path: Path,
) -> None:
    """The workflow should support a requested subset."""
    output_directory = tmp_path / "selected_reports"

    results = run_analysis_module.run_analyses(
        database_path=miniature_database,
        sql_path=miniature_sql_file,
        output_directory=output_directory,
        requested_names=[
            "02_class_distribution",
        ],
    )

    assert len(results) == 1
    assert results[0].name == "02_class_distribution"

    assert (
        output_directory
        / "02_class_distribution.csv"
    ).exists()

    assert not (
        output_directory
        / "01_dataset_overview.csv"
    ).exists()


def test_run_analyses_reports_failing_query(
    run_analysis_module: ModuleType,
    miniature_database: Path,
    tmp_path: Path,
) -> None:
    """A failed query should identify the named analysis."""
    sql_path = tmp_path / "broken_queries.sql"

    sql_path.write_text(
        """
-- name: broken_query
-- description: References a missing column.
SELECT MissingColumn
FROM passengers;
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="broken_query",
    ):
        run_analysis_module.run_analyses(
            database_path=miniature_database,
            sql_path=sql_path,
            output_directory=tmp_path / "broken_output",
        )