"""Functions for loading and saving Titanic datasets."""

from pathlib import Path

import pandas as pd

from titanic_passenger_class_prediction.config import (
    DEFAULT_PROCESSED_FILENAME,
    DEFAULT_RAW_FILENAME,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)


def load_raw_data(
    filename: str = DEFAULT_RAW_FILENAME,
) -> pd.DataFrame:
    """
    Load a raw Titanic CSV dataset.

    Parameters
    ----------
    filename
        Name of the CSV file inside ``data/raw``.

    Returns
    -------
    pandas.DataFrame
        Loaded raw dataset.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    """
    filepath = RAW_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find raw dataset at '{filepath}'. "
            "Place the file inside data/raw/."
        )

    return pd.read_csv(filepath)


def load_processed_data(
    filename: str = DEFAULT_PROCESSED_FILENAME,
) -> pd.DataFrame:
    """
    Load a processed Titanic dataset from Parquet.
    """
    filepath = PROCESSED_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find processed dataset at '{filepath}'. "
            "Run the data-preparation pipeline first."
        )

    return pd.read_parquet(filepath)


def save_processed_data(
    dataframe: pd.DataFrame,
    filename: str = DEFAULT_PROCESSED_FILENAME,
) -> Path:
    """
    Save a processed Titanic dataset as Parquet.

    Returns
    -------
    pathlib.Path
        Path of the saved file.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    filepath = PROCESSED_DATA_DIR / filename
    dataframe.to_parquet(filepath, index=False)

    return filepath