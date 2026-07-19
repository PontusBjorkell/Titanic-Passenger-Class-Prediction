"""SQLite utilities for Titanic passenger analytics."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from titanic_passenger_class_prediction.config import DATABASE_PATH


PASSENGERS_TABLE = "passengers"


def create_connection(
    database_path: Path = DATABASE_PATH,
) -> sqlite3.Connection:
    """
    Create a SQLite connection.

    The database directory is created automatically when needed.
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)

    return sqlite3.connect(database_path)


def write_passengers_table(
    dataframe: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
    if_exists: str = "replace",
) -> None:
    """
    Write processed passenger data to SQLite.
    """
    with create_connection(database_path) as connection:
        dataframe.to_sql(
            PASSENGERS_TABLE,
            connection,
            if_exists=if_exists,
            index=False,
        )


def execute_sql_script(
    script_path: Path,
    database_path: Path = DATABASE_PATH,
) -> None:
    """
    Execute a SQL script against the Titanic database.
    """
    if not script_path.exists():
        raise FileNotFoundError(
            f"SQL script not found: {script_path}"
        )

    sql = script_path.read_text(encoding="utf-8")

    with create_connection(database_path) as connection:
        connection.executescript(sql)


def query_database(
    query: str,
    database_path: Path = DATABASE_PATH,
    parameters: tuple[object, ...] | None = None,
) -> pd.DataFrame:
    """
    Run a SQL query and return the result as a dataframe.
    """
    with create_connection(database_path) as connection:
        return pd.read_sql_query(
            query,
            connection,
            params=parameters,
        )


def list_database_objects(
    database_path: Path = DATABASE_PATH,
) -> pd.DataFrame:
    """
    List user-created tables and views in the database.
    """
    query = """
        SELECT
            name,
            type
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name;
    """

    return query_database(query, database_path)