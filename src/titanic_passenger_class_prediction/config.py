"""Project-wide configuration and filesystem paths."""

from pathlib import Path


# ---------------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SQL_DIR = PROJECT_ROOT / "sql"


# ---------------------------------------------------------------------------
# Data and database paths
# ---------------------------------------------------------------------------

DATABASE_PATH = DATA_DIR / "titanic.sqlite"

DEFAULT_RAW_FILENAME = "train.csv"
DEFAULT_PROCESSED_FILENAME = "passengers.parquet"


# ---------------------------------------------------------------------------
# Model artifact filenames
# ---------------------------------------------------------------------------

BEST_MODEL_FILENAME = "best_model.joblib"
MODEL_COMPARISON_FILENAME = "model_comparison.csv"
TEST_METRICS_FILENAME = "test_metrics.json"
MODEL_METADATA_FILENAME = "model_metadata.json"


# ---------------------------------------------------------------------------
# Reproducibility and training settings
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_JOBS = -1


# ---------------------------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------------------------

TARGET_COLUMN = "Pclass"
ID_COLUMN = "PassengerId"