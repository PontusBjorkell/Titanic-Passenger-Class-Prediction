"""Project-wide configuration and filesystem paths."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SQL_DIR = PROJECT_ROOT / "sql"
DATABASE_PATH = DATA_DIR / "titanic.sqlite"

DEFAULT_RAW_FILENAME = "train.csv"
DEFAULT_PROCESSED_FILENAME = "passengers.parquet"

RANDOM_STATE = 42
TARGET_COLUMN = "Pclass"
ID_COLUMN = "PassengerId"