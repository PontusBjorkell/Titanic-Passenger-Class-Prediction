"""Run named Titanic SQL analyses and export their results as CSV files.

The script reads named query blocks from ``sql/analysis_queries.sql``.
Each block begins with a marker of the form:

    -- name: 01_dataset_overview
    -- description: Optional human-readable description
    SELECT ...;

Every query is executed against the configured SQLite database and saved as
an independent CSV file inside ``reports/sql``. A manifest containing query
metadata and output dimensions is also produced.

Examples
--------
Run all analyses:

    python scripts/run_analysis.py

List available analyses without executing them:

    python scripts/run_analysis.py --list

Run selected analyses:

    python scripts/run_analysis.py \
        --query 01_dataset_overview \
        --query 03_class_profile

Use custom files or directories:

    python scripts/run_analysis.py \
        --sql-file sql/analysis_queries.sql \
        --output-dir reports/sql
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from titanic_passenger_class_prediction.config import (
    DATABASE_PATH,
    REPORTS_DIR,
    SQL_DIR,
)
from titanic_passenger_class_prediction.database import (
    list_database_objects,
    query_database,
)


DEFAULT_ANALYSIS_SQL_PATH = SQL_DIR / "analysis_queries.sql"
DEFAULT_OUTPUT_DIRECTORY = REPORTS_DIR / "sql"
DEFAULT_MANIFEST_FILENAME = "analysis_manifest.json"

QUERY_NAME_PATTERN = re.compile(
    r"^\s*--\s*name\s*:\s*(?P<name>[A-Za-z0-9_-]+)\s*$",
    flags=re.IGNORECASE,
)

QUERY_DESCRIPTION_PATTERN = re.compile(
    r"^\s*--\s*description\s*:\s*(?P<description>.+?)\s*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class NamedQuery:
    """Represent one parsed SQL analysis."""

    name: str
    sql: str
    description: str | None = None


@dataclass(frozen=True)
class AnalysisResult:
    """Describe one successfully exported SQL result."""

    name: str
    description: str | None
    output_path: str
    rows: int
    columns: int
    column_names: list[str]


def validate_query_name(name: str) -> str:
    """Validate and normalize a query name.

    Query names are also used as output filenames, so only letters, numbers,
    underscores, and hyphens are accepted.
    """
    normalized = name.strip()

    if not normalized:
        raise ValueError("Query name cannot be empty.")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", normalized):
        raise ValueError(
            "Query names may contain only letters, numbers, "
            "underscores, and hyphens. "
            f"Received: {name!r}"
        )

    return normalized


def parse_named_queries(sql_text: str) -> list[NamedQuery]:
    """Parse named query blocks from SQL text.

    Parameters
    ----------
    sql_text
        Complete contents of ``analysis_queries.sql``.

    Returns
    -------
    list[NamedQuery]
        Queries in their original file order.

    Raises
    ------
    ValueError
        If the file is empty, contains duplicate names, or contains a query
        marker without executable SQL.
    """
    if not sql_text.strip():
        raise ValueError("The SQL analysis file is empty.")

    parsed_queries: list[NamedQuery] = []

    current_name: str | None = None
    current_description: str | None = None
    current_sql_lines: list[str] = []

    def finalize_current_query() -> None:
        nonlocal current_name
        nonlocal current_description
        nonlocal current_sql_lines

        if current_name is None:
            return

        sql = "\n".join(current_sql_lines).strip()

        if not sql:
            raise ValueError(
                f"Query {current_name!r} does not contain SQL."
            )

        parsed_queries.append(
            NamedQuery(
                name=current_name,
                description=current_description,
                sql=sql,
            )
        )

        current_name = None
        current_description = None
        current_sql_lines = []

    for line_number, line in enumerate(
        sql_text.splitlines(),
        start=1,
    ):
        name_match = QUERY_NAME_PATTERN.match(line)

        if name_match:
            finalize_current_query()

            current_name = validate_query_name(
                name_match.group("name")
            )
            current_description = None
            current_sql_lines = []
            continue

        description_match = QUERY_DESCRIPTION_PATTERN.match(line)

        if description_match and current_name is not None:
            current_description = description_match.group(
                "description"
            ).strip()
            continue

        if current_name is not None:
            current_sql_lines.append(line)

    finalize_current_query()

    if not parsed_queries:
        raise ValueError(
            "No named SQL queries were found. "
            "Each query must begin with '-- name: query_name'."
        )

    query_names = [query.name for query in parsed_queries]
    duplicate_names = sorted(
        {
            name
            for name in query_names
            if query_names.count(name) > 1
        }
    )

    if duplicate_names:
        raise ValueError(
            "Duplicate SQL query names found: "
            + ", ".join(duplicate_names)
        )

    return parsed_queries


def load_named_queries(
    sql_path: Path = DEFAULT_ANALYSIS_SQL_PATH,
) -> list[NamedQuery]:
    """Load and parse named SQL analyses from disk."""
    sql_path = Path(sql_path)

    if not sql_path.exists():
        raise FileNotFoundError(
            f"SQL analysis file not found: {sql_path}"
        )

    if not sql_path.is_file():
        raise ValueError(
            f"SQL analysis path is not a file: {sql_path}"
        )

    sql_text = sql_path.read_text(encoding="utf-8")

    return parse_named_queries(sql_text)


def validate_database(
    database_path: Path = DATABASE_PATH,
) -> None:
    """Confirm that the SQLite database contains required objects."""
    database_path = Path(database_path)

    if not database_path.exists():
        raise FileNotFoundError(
            f"Titanic database not found: {database_path}. "
            "Run 'python scripts/build_database.py' first."
        )

    objects = list_database_objects(
        database_path=database_path,
    )

    object_names = set(objects["name"].astype(str))

    if "passengers" not in object_names:
        raise ValueError(
            "The database does not contain the required "
            "'passengers' table. Rebuild the database with "
            "'python scripts/build_database.py'."
        )


def select_queries(
    queries: Sequence[NamedQuery],
    requested_names: Sequence[str] | None = None,
) -> list[NamedQuery]:
    """Return all queries or a requested subset while preserving file order."""
    if not requested_names:
        return list(queries)

    normalized_names = [
        validate_query_name(name)
        for name in requested_names
    ]

    duplicate_requests = sorted(
        {
            name
            for name in normalized_names
            if normalized_names.count(name) > 1
        }
    )

    if duplicate_requests:
        raise ValueError(
            "The same query was requested more than once: "
            + ", ".join(duplicate_requests)
        )

    available_by_name = {
        query.name: query
        for query in queries
    }

    unknown_names = [
        name
        for name in normalized_names
        if name not in available_by_name
    ]

    if unknown_names:
        raise ValueError(
            "Unknown query name(s): "
            + ", ".join(unknown_names)
            + ". Available queries: "
            + ", ".join(available_by_name)
        )

    requested_name_set = set(normalized_names)

    return [
        query
        for query in queries
        if query.name in requested_name_set
    ]


def export_query_result(
    query: NamedQuery,
    database_path: Path,
    output_directory: Path,
) -> AnalysisResult:
    """Execute one query and save its result as CSV."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    dataframe = query_database(
        query=query.sql,
        database_path=database_path,
    )

    output_path = output_directory / f"{query.name}.csv"

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return AnalysisResult(
        name=query.name,
        description=query.description,
        output_path=str(output_path),
        rows=int(dataframe.shape[0]),
        columns=int(dataframe.shape[1]),
        column_names=[
            str(column)
            for column in dataframe.columns
        ],
    )


def save_manifest(
    results: Sequence[AnalysisResult],
    output_directory: Path,
    database_path: Path,
    sql_path: Path,
) -> Path:
    """Save machine-readable metadata for an analysis run."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        output_directory / DEFAULT_MANIFEST_FILENAME
    )

    manifest = {
        "database_path": str(database_path),
        "sql_path": str(sql_path),
        "output_directory": str(output_directory),
        "query_count": len(results),
        "queries": [
            asdict(result)
            for result in results
        ],
    }

    with manifest_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return manifest_path


def build_summary_dataframe(
    results: Sequence[AnalysisResult],
) -> pd.DataFrame:
    """Build a concise terminal summary of exported results."""
    return pd.DataFrame(
        [
            {
                "query": result.name,
                "rows": result.rows,
                "columns": result.columns,
                "output": result.output_path,
            }
            for result in results
        ]
    )


def run_analyses(
    *,
    database_path: Path = DATABASE_PATH,
    sql_path: Path = DEFAULT_ANALYSIS_SQL_PATH,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    requested_names: Sequence[str] | None = None,
) -> list[AnalysisResult]:
    """Run and export all requested SQL analyses."""
    database_path = Path(database_path)
    sql_path = Path(sql_path)
    output_directory = Path(output_directory)

    validate_database(database_path)

    available_queries = load_named_queries(sql_path)

    selected_queries = select_queries(
        queries=available_queries,
        requested_names=requested_names,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[AnalysisResult] = []

    for position, query in enumerate(
        selected_queries,
        start=1,
    ):
        print(
            f"[{position:02d}/{len(selected_queries):02d}] "
            f"Running {query.name}..."
        )

        try:
            result = export_query_result(
                query=query,
                database_path=database_path,
                output_directory=output_directory,
            )
        except Exception as error:
            raise RuntimeError(
                f"SQL analysis {query.name!r} failed."
            ) from error

        results.append(result)

        print(
            f"     Exported {result.rows:,} row(s) and "
            f"{result.columns:,} column(s)."
        )

    save_manifest(
        results=results,
        output_directory=output_directory,
        database_path=database_path,
        sql_path=sql_path,
    )

    return results


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Execute named Titanic SQL analyses and export "
            "the results as CSV files."
        )
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help=(
            "Path to the SQLite database. "
            f"Default: {DATABASE_PATH}"
        ),
    )

    parser.add_argument(
        "--sql-file",
        type=Path,
        default=DEFAULT_ANALYSIS_SQL_PATH,
        help=(
            "Path to the named SQL analysis file. "
            f"Default: {DEFAULT_ANALYSIS_SQL_PATH}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory for exported CSV files. "
            f"Default: {DEFAULT_OUTPUT_DIRECTORY}"
        ),
    )

    parser.add_argument(
        "--query",
        action="append",
        dest="query_names",
        help=(
            "Run one named query. Repeat this option to run "
            "multiple selected queries. By default, all queries run."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_queries",
        help=(
            "List available query names and descriptions "
            "without executing them."
        ),
    )

    return parser


def list_available_queries(
    sql_path: Path,
) -> None:
    """Print query names and descriptions."""
    queries = load_named_queries(sql_path)

    print(f"Available analyses in {sql_path}:")
    print()

    for query in queries:
        description = (
            query.description
            or "No description provided."
        )
        print(f"- {query.name}")
        print(f"  {description}")


def main() -> None:
    """Run the command-line SQL analytics workflow."""
    parser = build_argument_parser()
    arguments = parser.parse_args()

    if arguments.list_queries:
        list_available_queries(arguments.sql_file)
        return

    results = run_analyses(
        database_path=arguments.database,
        sql_path=arguments.sql_file,
        output_directory=arguments.output_dir,
        requested_names=arguments.query_names,
    )

    summary = build_summary_dataframe(results)

    print()
    print("=" * 88)
    print("SQL ANALYSIS COMPLETE")
    print("=" * 88)
    print()
    print(summary.to_string(index=False))
    print()
    print(
        f"Exported {len(results)} analysis file(s) to: "
        f"{arguments.output_dir}"
    )
    print(
        "Manifest saved to: "
        f"{arguments.output_dir / DEFAULT_MANIFEST_FILENAME}"
    )


if __name__ == "__main__":
    main()