"""Tests for the SQLite database utility module.

These tests use temporary SQLite databases created by pytest. They do not
read from or modify the project's real database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from titanic_passenger_class_prediction.database import (
    list_database_objects,
    query_database,
)


@pytest.fixture
def temporary_database(tmp_path: Path) -> Path:
    """Create a small SQLite database for database utility tests."""
    database_path = tmp_path / "test_titanic.db"

    passengers = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Pclass": [1, 2, 3, 1],
            "Name": [
                "Allen, Mr. William",
                "Baker, Mrs. Anne",
                "Clark, Miss. Emily",
                "Davis, Dr. Robert",
            ],
            "Sex": ["male", "female", "female", "male"],
            "Age": [35.0, 28.0, 19.0, 52.0],
            "Fare": [80.0, 30.0, 8.5, 120.0],
            "Survived": [1, 1, 0, 1],
        }
    )

    with sqlite3.connect(database_path) as connection:
        passengers.to_sql(
            "passengers",
            connection,
            if_exists="replace",
            index=False,
        )

        connection.execute(
            """
            CREATE VIEW vw_class_counts AS
            SELECT
                Pclass AS passenger_class,
                COUNT(*) AS passenger_count
            FROM passengers
            GROUP BY Pclass
            """
        )

        connection.commit()

    return database_path


def test_query_database_returns_dataframe(
    temporary_database: Path,
) -> None:
    """A valid SELECT query should return a pandas DataFrame."""
    result = query_database(
        query="""
            SELECT
                PassengerId,
                Pclass,
                Fare
            FROM passengers
            ORDER BY PassengerId
        """,
        database_path=temporary_database,
    )

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (4, 3)
    assert list(result.columns) == [
        "PassengerId",
        "Pclass",
        "Fare",
    ]


def test_query_database_returns_expected_values(
    temporary_database: Path,
) -> None:
    """Query results should contain the values stored in SQLite."""
    result = query_database(
        query="""
            SELECT
                Pclass,
                COUNT(*) AS passenger_count
            FROM passengers
            GROUP BY Pclass
            ORDER BY Pclass
        """,
        database_path=temporary_database,
    )

    expected = pd.DataFrame(
        {
            "Pclass": [1, 2, 3],
            "passenger_count": [2, 1, 1],
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected,
        check_dtype=False,
    )


def test_query_database_can_use_view(
    temporary_database: Path,
) -> None:
    """The query helper should work with database views."""
    result = query_database(
        query="""
            SELECT *
            FROM vw_class_counts
            ORDER BY passenger_class
        """,
        database_path=temporary_database,
    )

    assert result["passenger_class"].tolist() == [1, 2, 3]
    assert result["passenger_count"].tolist() == [2, 1, 1]


def test_query_database_can_create_new_sqlite_database(
    tmp_path: Path,
) -> None:
    """SQLite may create a database automatically when the path is new."""
    database_path = tmp_path / "new_database.db"

    result = query_database(
        query="SELECT 1 AS value",
        database_path=database_path,
    )

    assert database_path.exists()
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)
    assert int(result.loc[0, "value"]) == 1


def test_query_database_raises_for_invalid_sql(
    temporary_database: Path,
) -> None:
    """Malformed SQL should raise an exception."""
    with pytest.raises(Exception):
        query_database(
            query="SELEKT * FROM passengers",
            database_path=temporary_database,
        )


def test_query_database_raises_for_missing_table(
    temporary_database: Path,
) -> None:
    """Queries referencing nonexistent tables should fail."""
    with pytest.raises(Exception):
        query_database(
            query="SELECT * FROM nonexistent_table",
            database_path=temporary_database,
        )


def test_list_database_objects_returns_dataframe(
    temporary_database: Path,
) -> None:
    """Database object inspection should return a DataFrame."""
    objects = list_database_objects(
        database_path=temporary_database,
    )

    assert isinstance(objects, pd.DataFrame)
    assert not objects.empty
    assert "name" in objects.columns


def test_list_database_objects_finds_passengers_table(
    temporary_database: Path,
) -> None:
    """The passengers table should appear in the object listing."""
    objects = list_database_objects(
        database_path=temporary_database,
    )

    assert "passengers" in set(objects["name"].astype(str))


def test_list_database_objects_finds_analysis_view(
    temporary_database: Path,
) -> None:
    """Views should also appear in the object listing."""
    objects = list_database_objects(
        database_path=temporary_database,
    )

    assert "vw_class_counts" in set(
        objects["name"].astype(str)
    )


def test_list_database_objects_contains_object_types(
    temporary_database: Path,
) -> None:
    """The listing should identify tables and views where supported."""
    objects = list_database_objects(
        database_path=temporary_database,
    )

    if "type" not in objects.columns:
        pytest.skip(
            "list_database_objects does not expose object types."
        )

    object_types = set(objects["type"].astype(str).str.lower())

    assert "table" in object_types
    assert "view" in object_types


def test_database_connection_is_not_left_locked(
    temporary_database: Path,
) -> None:
    """Database helper calls should close their connections."""
    query_database(
        query="SELECT * FROM passengers",
        database_path=temporary_database,
    )

    with sqlite3.connect(temporary_database) as connection:
        connection.execute(
            """
            INSERT INTO passengers (
                PassengerId,
                Pclass,
                Name,
                Sex,
                Age,
                Fare,
                Survived
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                5,
                3,
                "Evans, Mr. Thomas",
                "male",
                41.0,
                9.0,
                0,
            ),
        )
        connection.commit()

    result = query_database(
        query="SELECT COUNT(*) AS passenger_count FROM passengers",
        database_path=temporary_database,
    )

    assert int(result.loc[0, "passenger_count"]) == 5