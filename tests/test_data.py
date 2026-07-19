from pathlib import Path

import pandas as pd
import pytest

from titanic_passenger_class_prediction.data import (
    load_processed_data,
    load_raw_data,
    save_processed_data,
)


def test_load_raw_data_returns_dataframe() -> None:
    dataframe = load_raw_data()

    assert isinstance(dataframe, pd.DataFrame)
    assert not dataframe.empty
    assert "PassengerId" in dataframe.columns
    assert "Pclass" in dataframe.columns


def test_load_raw_data_rejects_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_data("file_that_does_not_exist.csv")


def test_save_and_load_processed_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Pclass": [3, 1],
        }
    )

    monkeypatch.setattr(
        "titanic_passenger_class_prediction.data.PROCESSED_DATA_DIR",
        tmp_path,
    )

    saved_path = save_processed_data(sample, "sample.parquet")
    loaded = load_processed_data("sample.parquet")

    assert saved_path.exists()
    pd.testing.assert_frame_equal(loaded, sample)