"""Titanic passenger class analytics and machine learning package."""

from titanic_passenger_class_prediction.config import (
    RANDOM_STATE,
    TARGET_COLUMN,
)
from titanic_passenger_class_prediction.data import (
    load_processed_data,
    load_raw_data,
    save_processed_data,
)

__version__ = "0.1.0"

__all__ = [
    "RANDOM_STATE",
    "TARGET_COLUMN",
    "load_processed_data",
    "load_raw_data",
    "save_processed_data",
]