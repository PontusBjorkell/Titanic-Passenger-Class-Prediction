from pathlib import Path

import pandas as pd

from titanic_passenger_class_prediction.database import (
    list_database_objects,
    query_database,
    write_passengers_table,
)


def test_write_and_query_passengers_table(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.sqlite"

    dataframe = pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Pclass": [3, 1],
            "Fare": [7.25, 71.28],
        }
    )

    write_passengers_table(
        dataframe,
        database_path=database_path,
    )

    result = query_database(
        """
        SELECT
            Pclass,
            COUNT(*) AS passenger_count
        FROM passengers
        GROUP BY Pclass
        ORDER BY Pclass;
        """,
        database_path=database_path,
    )

    assert result["Pclass"].tolist() == [1, 3]
    assert result["passenger_count"].tolist() == [1, 1]


def test_list_database_objects(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "test.sqlite"

    dataframe = pd.DataFrame(
        {
            "PassengerId": [1],
            "Pclass": [3],
        }
    )

    write_passengers_table(
        dataframe,
        database_path=database_path,
    )

    objects = list_database_objects(
        database_path=database_path,
    )

    assert "passengers" in objects["name"].tolist()
    assert "table" in objects["type"].tolist()