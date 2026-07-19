"""Build the Titanic SQLite analytics database."""

from __future__ import annotations

from titanic_passenger_class_prediction.config import (
    DATABASE_PATH,
    SQL_DIR,
)
from titanic_passenger_class_prediction.data import (
    load_processed_data,
)
from titanic_passenger_class_prediction.database import (
    execute_sql_script,
    list_database_objects,
    write_passengers_table,
)


def main() -> None:
    """Create the database, passenger table, and analytical views."""
    dataframe = load_processed_data()

    write_passengers_table(dataframe)

    execute_sql_script(
        SQL_DIR / "create_views.sql",
    )

    objects = list_database_objects()

    print(f"Database built at: {DATABASE_PATH}")
    print()
    print(objects.to_string(index=False))


if __name__ == "__main__":
    main()