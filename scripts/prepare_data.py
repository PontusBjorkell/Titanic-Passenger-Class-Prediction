"""Prepare and save the Titanic passenger dataset."""

from __future__ import annotations

import json

from titanic_passenger_class_prediction.config import (
    ARTIFACTS_DIR,
    DEFAULT_PROCESSED_FILENAME,
)
from titanic_passenger_class_prediction.data import (
    load_raw_data,
    save_processed_data,
)
from titanic_passenger_class_prediction.preprocessing import (
    build_preparation_summary,
    prepare_passenger_data,
)


def main() -> None:
    """Run the complete deterministic data-preparation workflow."""
    raw_dataframe = load_raw_data()
    processed_dataframe = prepare_passenger_data(raw_dataframe)

    output_path = save_processed_data(
        processed_dataframe,
        DEFAULT_PROCESSED_FILENAME,
    )

    summary = build_preparation_summary(
        raw_dataframe,
        processed_dataframe,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = ARTIFACTS_DIR / "data_preparation_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(
        f"Prepared {len(processed_dataframe):,} passenger records."
    )
    print(f"Processed data saved to: {output_path}")
    print(f"Preparation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()