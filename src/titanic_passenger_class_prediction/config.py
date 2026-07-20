"""Central configuration for the Titanic passenger-class prediction project.

All reusable paths, filenames, model settings, and reporting settings are
defined here so that application modules and tests share one source of truth.
"""

from __future__ import annotations

from pathlib import Path


# =============================================================================
# Project root
# =============================================================================

# File location:
# project_root/src/titanic_passenger_class_prediction/config.py
#
# parents[0] = src/titanic_passenger_class_prediction
# parents[1] = src
# parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Project directories
# =============================================================================

CONFIG_DIR = PROJECT_ROOT / "config"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SQL_DIR = PROJECT_ROOT / "sql"


# =============================================================================
# Data filenames
# =============================================================================

# DEFAULT_RAW_FILENAME is retained because existing data.py imports it.
DEFAULT_RAW_FILENAME = "train.csv"

DEFAULT_TRAIN_FILENAME = "train.csv"
DEFAULT_TEST_FILENAME = "test.csv"
DEFAULT_PROCESSED_FILENAME = "passengers.parquet"
DEFAULT_DATABASE_FILENAME = "titanic.sqlite"


# =============================================================================
# Artifact filenames
# =============================================================================

DEFAULT_MODEL_FILENAME = "best_model.joblib"
DEFAULT_BEST_MODEL_FILENAME = DEFAULT_MODEL_FILENAME

DEFAULT_MODEL_COMPARISON_FILENAME = "model_comparison.csv"
DEFAULT_TEST_METRICS_FILENAME = "test_metrics.json"
DEFAULT_MODEL_METADATA_FILENAME = "model_metadata.json"


# =============================================================================
# Report filenames
# =============================================================================

DEFAULT_CLASSIFICATION_REPORT_FILENAME = "classification_report.csv"
DEFAULT_TRAINING_SUMMARY_FILENAME = "training_summary.md"

DEFAULT_CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
DEFAULT_FEATURE_IMPORTANCE_FILENAME = "feature_importance.png"
DEFAULT_PERMUTATION_IMPORTANCE_FILENAME = "permutation_importance.png"


# =============================================================================
# Data paths
# =============================================================================

# Original/default raw dataset path used by the training workflow.
RAW_DATA_PATH = RAW_DATA_DIR / DEFAULT_RAW_FILENAME

TRAIN_DATA_PATH = RAW_DATA_DIR / DEFAULT_TRAIN_FILENAME
TEST_DATA_PATH = RAW_DATA_DIR / DEFAULT_TEST_FILENAME

PROCESSED_DATA_PATH = (
    PROCESSED_DATA_DIR / DEFAULT_PROCESSED_FILENAME
)

DATABASE_PATH = DATA_DIR / DEFAULT_DATABASE_FILENAME


# =============================================================================
# Artifact paths
# =============================================================================

BEST_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_FILENAME

MODEL_COMPARISON_PATH = (
    METRICS_DIR / DEFAULT_MODEL_COMPARISON_FILENAME
)

TEST_METRICS_PATH = (
    METRICS_DIR / DEFAULT_TEST_METRICS_FILENAME
)

MODEL_METADATA_PATH = (
    METRICS_DIR / DEFAULT_MODEL_METADATA_FILENAME
)


# =============================================================================
# Report paths
# =============================================================================

CLASSIFICATION_REPORT_PATH = (
    REPORTS_DIR / DEFAULT_CLASSIFICATION_REPORT_FILENAME
)

TRAINING_SUMMARY_PATH = (
    REPORTS_DIR / DEFAULT_TRAINING_SUMMARY_FILENAME
)

CONFUSION_MATRIX_PATH = (
    FIGURES_DIR / DEFAULT_CONFUSION_MATRIX_FILENAME
)

FEATURE_IMPORTANCE_PATH = (
    FIGURES_DIR / DEFAULT_FEATURE_IMPORTANCE_FILENAME
)

PERMUTATION_IMPORTANCE_PATH = (
    FIGURES_DIR / DEFAULT_PERMUTATION_IMPORTANCE_FILENAME
)


# =============================================================================
# Dataset configuration
# =============================================================================

TARGET_COLUMN = "Pclass"
ID_COLUMN = "PassengerId"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_JOBS = -1


# =============================================================================
# Visualization configuration
# =============================================================================

FIGURE_DPI = 150
TOP_FEATURES_TO_DISPLAY = 20
PERMUTATION_IMPORTANCE_REPEATS = 20


# =============================================================================
# Compatibility aliases
#
# These names preserve imports used by existing modules and tests. They allow
# the project to adopt clearer canonical names without breaking older code.
# =============================================================================

# Data filename aliases
RAW_FILENAME = DEFAULT_RAW_FILENAME
TRAIN_FILENAME = DEFAULT_TRAIN_FILENAME
TEST_FILENAME = DEFAULT_TEST_FILENAME
PROCESSED_FILENAME = DEFAULT_PROCESSED_FILENAME
DATABASE_FILENAME = DEFAULT_DATABASE_FILENAME

# Additional default filename aliases
DEFAULT_DATA_FILENAME = DEFAULT_RAW_FILENAME
DEFAULT_RAW_DATA_FILENAME = DEFAULT_RAW_FILENAME
DEFAULT_TRAIN_DATA_FILENAME = DEFAULT_TRAIN_FILENAME
DEFAULT_TEST_DATA_FILENAME = DEFAULT_TEST_FILENAME
DEFAULT_PROCESSED_DATA_FILENAME = DEFAULT_PROCESSED_FILENAME

# Data path aliases
RAW_PATH = RAW_DATA_PATH
RAW_TRAIN_PATH = TRAIN_DATA_PATH
RAW_TEST_PATH = TEST_DATA_PATH

TRAIN_PATH = TRAIN_DATA_PATH
TEST_PATH = TEST_DATA_PATH

DEFAULT_RAW_PATH = RAW_DATA_PATH
DEFAULT_TRAIN_PATH = TRAIN_DATA_PATH
DEFAULT_TEST_PATH = TEST_DATA_PATH
DEFAULT_PROCESSED_PATH = PROCESSED_DATA_PATH

PROCESSED_PATH = PROCESSED_DATA_PATH
PROCESSED_DATASET_PATH = PROCESSED_DATA_PATH

DB_PATH = DATABASE_PATH
SQLITE_PATH = DATABASE_PATH
TITANIC_DATABASE_PATH = DATABASE_PATH

# Artifact filename aliases
MODEL_FILENAME = DEFAULT_MODEL_FILENAME
BEST_MODEL_FILENAME = DEFAULT_MODEL_FILENAME

MODEL_COMPARISON_FILENAME = (
    DEFAULT_MODEL_COMPARISON_FILENAME
)

TEST_METRICS_FILENAME = DEFAULT_TEST_METRICS_FILENAME
MODEL_METADATA_FILENAME = DEFAULT_MODEL_METADATA_FILENAME

# Artifact path aliases
MODEL_PATH = BEST_MODEL_PATH
DEFAULT_MODEL_PATH = BEST_MODEL_PATH

# Report filename aliases
CLASSIFICATION_REPORT_FILENAME = (
    DEFAULT_CLASSIFICATION_REPORT_FILENAME
)

TRAINING_SUMMARY_FILENAME = (
    DEFAULT_TRAINING_SUMMARY_FILENAME
)

CONFUSION_MATRIX_FILENAME = (
    DEFAULT_CONFUSION_MATRIX_FILENAME
)

FEATURE_IMPORTANCE_FILENAME = (
    DEFAULT_FEATURE_IMPORTANCE_FILENAME
)

PERMUTATION_IMPORTANCE_FILENAME = (
    DEFAULT_PERMUTATION_IMPORTANCE_FILENAME
)


# =============================================================================
# Directory creation
# =============================================================================

PROJECT_DIRECTORIES: tuple[Path, ...] = (
    CONFIG_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    MODELS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    NOTEBOOKS_DIR,
    SCRIPTS_DIR,
    SQL_DIR,
)


def ensure_project_directories() -> tuple[Path, ...]:
    """Create all standard project directories.

    The operation is idempotent, meaning it is safe to call more than once.

    Returns
    -------
    tuple[pathlib.Path, ...]
        The project's configured directories.
    """
    for directory in PROJECT_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)

    return PROJECT_DIRECTORIES