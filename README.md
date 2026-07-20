# 🚢 Titanic Passenger Class Prediction

> An end-to-end machine learning and analytics project for predicting passenger class from Titanic passenger data, built with reproducible pipelines, modular Python code, SQL analytics, automated testing, model persistence, and explainable evaluation.

---

## Project Overview

This project predicts a passenger's travel class (`Pclass`) from demographic, family, ticket, fare, cabin, and embarkation information.

The project began with notebook-based exploration but has since been redesigned as a production-inspired machine learning repository. The main workflow is implemented as a reusable Python package rather than being confined to a single notebook.

The repository demonstrates both machine learning and software engineering practices:

- deterministic data preparation;
- schema and data validation;
- reusable feature engineering;
- scikit-learn preprocessing pipelines;
- model comparison with stratified cross-validation;
- holdout evaluation;
- artifact persistence;
- SQL analytics with SQLite;
- automated testing;
- reusable visualization and reporting utilities.

The original Titanic notebook used during the early exploration stage is treated only as inspiration. The production implementation in this repository is independent of that notebook.

---

## Project Goals

- Build a reproducible data preparation workflow.
- Validate raw and processed data before modeling.
- Engineer meaningful passenger-level features.
- Create reusable preprocessing and modeling pipelines.
- Compare candidate classifiers with stratified cross-validation.
- Evaluate the selected model on unseen labeled data.
- Persist trained pipelines, metrics, and metadata.
- Produce model evaluation figures and reports.
- Build an inference workflow for new unlabeled passenger data.
- Present the final model through an interactive Streamlit application.

---

## Machine Learning Problem

This is a multiclass classification problem.

**Target**

```text
Pclass
```

Possible classes:

```text
1 = First class
2 = Second class
3 = Third class
```

The model uses passenger information such as age, sex, fare, family relationships, embarkation location, cabin information, and engineered features to predict the passenger's class.

---

## Data Usage

The repository contains two raw datasets:

```text
data/raw/train.csv
data/raw/test.csv
```

### `train.csv`

This is the labeled development dataset. It is used for:

- feature engineering;
- train/holdout splitting;
- cross-validation;
- model comparison;
- hyperparameter tuning;
- final model evaluation.

The development workflow creates an internal stratified holdout set from this labeled data.

### `test.csv`

This is treated as new unlabeled passenger data. It is not currently used to calculate model quality because it does not contain the target column.

A later inference workflow will load the saved fitted pipeline and generate:

```text
predicted class
prediction probabilities
```

for every row in `test.csv`.

This project is not structured around a competition submission workflow. The file represents production-style unseen input data.

---

## Project Architecture

```text
                       Raw CSV Files
                    train.csv / test.csv
                              │
                              ▼
                       Data Validation
                              │
                              ▼
                      Feature Engineering
                              │
                              ▼
                 Processed Dataset (.parquet)
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
         SQLite Analytics          Machine Learning
                │                           │
                ▼                           ▼
           SQL Views              Stratified Split
                                            │
                                            ▼
                                    Cross-Validation
                                            │
                                            ▼
                                    Model Comparison
                                            │
                                            ▼
                                     Selected Model
                                            │
                                            ▼
                                   Holdout Evaluation
                                            │
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
                Saved Pipeline         Metrics/Metadata      Evaluation Figures
                     │
                     ▼
               Inference Workflow
                     │
                     ▼
                  test.csv
                     │
                     ▼
                Predictions
```

---

## Repository Structure

```text
Titanic-Passenger-Class-Prediction/
│
├── app/                                # Streamlit application
├── artifacts/
│   ├── metrics/                        # Evaluation results and metadata
│   └── models/                         # Persisted fitted pipelines
│
├── config/                             # Optional external configuration
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed/
│   │   └── passengers.parquet
│   └── titanic.sqlite
│
├── notebooks/
│   └── 01_data_exploration.ipynb       # Project exploratory analysis
│
├── reports/
│   └── figures/
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       └── permutation_importance.png
│
├── scripts/
│   ├── build_database.py
│   ├── prepare_data.py
│   └── train.py
│
├── sql/
│   └── create_views.sql
│
├── src/
│   └── titanic_passenger_class_prediction/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── database.py
│       ├── evaluation.py
│       ├── features.py
│       ├── modeling.py
│       ├── persistence.py
│       ├── preprocessing.py
│       ├── validation.py
│       └── visualization.py
│
├── tests/
│   ├── test_data.py
│   ├── test_database.py
│   ├── test_evaluation.py
│   ├── test_features.py
│   ├── test_modeling.py
│   ├── test_persistence.py
│   ├── test_preprocessing.py
│   ├── test_train.py
│   ├── test_validation.py
│   └── test_visualization.py
│
├── README.md
├── pyproject.toml
└── requirements.txt
```

Generated files are created by the project workflows and may not all be committed to version control.

---

## Core Modules

### `config.py`

Centralizes:

- project directories;
- raw and processed data paths;
- artifact paths;
- report paths;
- random seeds;
- split and cross-validation settings;
- compatibility aliases used by existing modules.

### `data.py`

Provides reusable functions for loading raw and processed datasets.

### `validation.py`

Checks dataset structure and assumptions before downstream processing.

Examples include:

- required columns;
- target validity;
- missing values;
- duplicate records;
- expected data types;
- numeric value constraints.

### `features.py`

Contains deterministic passenger feature engineering.

Engineered features may include:

- family size;
- whether the passenger traveled alone;
- extracted title;
- cabin deck;
- ticket-related features;
- fare and group-based features.

### `preprocessing.py`

Builds reusable scikit-learn transformations for:

- numeric imputation;
- categorical imputation;
- scaling where appropriate;
- one-hot encoding;
- unknown category handling.

### `modeling.py`

Defines candidate estimators and complete preprocessing-plus-model pipelines.

### `evaluation.py`

Handles:

- stratified cross-validation;
- candidate model comparison;
- accuracy;
- balanced accuracy;
- macro F1;
- holdout evaluation;
- confusion matrices;
- classification reports.

### `persistence.py`

Saves:

- fitted model pipelines;
- model comparison tables;
- test metrics;
- training metadata.

### `visualization.py`

Creates reusable model evaluation figures:

- confusion matrix;
- estimator-derived feature importance;
- permutation importance.

---

## End-to-End Workflow

### 1. Install the project

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

If the project defines a development dependency group:

```bash
python -m pip install -e ".[dev]"
```

---

### 2. Prepare the data

```bash
python scripts/prepare_data.py
```

Expected workflow:

```text
data/raw/train.csv
        │
        ▼
validation
        │
        ▼
feature engineering
        │
        ▼
data/processed/passengers.parquet
```

---

### 3. Build the analytics database

```bash
python scripts/build_database.py
```

Expected output:

```text
data/titanic.sqlite
```

The SQLite database supports reusable SQL analysis independently of the Python modeling workflow.

---

### 4. Train and evaluate models

```bash
python scripts/train.py
```

The training workflow:

1. loads the processed labeled data;
2. separates features and target;
3. creates a stratified training and holdout split;
4. evaluates candidate pipelines with stratified cross-validation;
5. ranks models using macro F1;
6. fits the selected model;
7. evaluates it on the unseen holdout set;
8. saves the fitted pipeline, metrics, metadata, and comparison table.

---

## Evaluation Strategy

Because the target contains three classes, accuracy alone is not sufficient.

The project reports:

| Metric | Purpose |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Balanced accuracy | Average recall across classes |
| Macro F1 | Equal-weighted F1 score across all classes |
| Confusion matrix | Class-by-class error analysis |
| Classification report | Precision, recall, F1, and support by class |

The primary model-selection metric is:

```text
macro F1
```

Macro F1 prevents the largest passenger class from dominating the model comparison.

---

## Latest Model Benchmark

The latest successful training run selected a Random Forest classifier.

Approximate results from that run:

| Evaluation | Macro F1 | Accuracy |
|---|---:|---:|
| Cross-validation | 0.909 | — |
| Holdout set | 0.908 | 0.916 |

These values may change as feature engineering, model tuning, and the evaluation workflow evolve.

---

## Model Artifacts

A successful training run produces files similar to:

```text
artifacts/
├── models/
│   └── best_model.joblib
└── metrics/
    ├── model_comparison.csv
    ├── model_metadata.json
    └── test_metrics.json
```

Planned and partially implemented evaluation outputs include:

```text
reports/
├── classification_report.csv
├── training_summary.md
└── figures/
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── permutation_importance.png
```

---

## Feature Importance

The project distinguishes between two forms of importance.

### Model-derived importance

For supported estimators, the project extracts internal model importance values such as:

```python
feature_importances_
```

or absolute coefficient magnitudes.

These values operate on transformed features, including one-hot encoded columns.

### Permutation importance

Permutation importance measures how much holdout macro F1 decreases when an original input column is shuffled.

This evaluates the complete fitted pipeline and produces passenger-level feature names that are easier to interpret.

---

## Testing

Run the complete test suite:

```bash
python -m pytest
```

Run only the visualization tests:

```bash
python -m pytest tests/test_visualization.py -v
```

Optional code-quality checks:

```bash
python -m ruff check .
python -m ruff format .
```

The tests cover:

- configuration;
- data loading;
- validation;
- feature engineering;
- preprocessing;
- model construction;
- cross-validation and evaluation;
- persistence;
- SQLite database creation;
- training orchestration;
- visualization utilities.

---

## Current Status

| Component | Status |
|---|:---:|
| Modular project structure | ✅ |
| Central configuration | ✅ |
| Raw data loading | ✅ |
| Data validation | ✅ |
| Feature engineering | ✅ |
| Processed Parquet dataset | ✅ |
| SQLite analytics database | ✅ |
| Reusable preprocessing pipeline | ✅ |
| Candidate model registry | ✅ |
| Stratified cross-validation | ✅ |
| Holdout evaluation | ✅ |
| Model persistence | ✅ |
| Metrics and metadata persistence | ✅ |
| Automated tests | ✅ |
| Visualization utilities | ✅ |
| Automated report integration | 🚧 |
| Hyperparameter tuning | 🚧 |
| Unlabeled-data inference workflow | 🚧 |
| Project notebook completion | 🚧 |
| Streamlit application | 🚧 |
| Deployment | 🚧 |

---

## Development Roadmap

### Model reporting

- automatically generate the confusion matrix;
- save the classification report as CSV;
- generate model and permutation importance figures;
- create a Markdown training summary;
- integrate all reporting into `scripts/train.py`.

### Model improvement

- tune the strongest candidate with `RandomizedSearchCV`;
- compare tuned and untuned performance;
- persist the tuned pipeline only when evaluation supports the change;
- document the selected hyperparameters.

### Inference

- load `data/raw/test.csv`;
- validate its feature schema;
- apply the saved fitted pipeline;
- generate predicted classes and probabilities;
- save a reusable predictions file.

### Explainability and application

- add richer model explanations;
- complete the project EDA notebook;
- build a Streamlit prediction interface;
- display model information and feature importance;
- add deployment instructions and screenshots.

---

## Reproducibility

The project uses centralized settings such as:

```python
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
```

Preprocessing and estimation are combined into scikit-learn pipelines. This ensures the same transformations are learned during training and applied during evaluation or inference, reducing leakage and training-serving inconsistencies.

---

## Why This Project?

A notebook can demonstrate an analysis, but a maintainable machine learning system requires more:

- reliable data assumptions;
- reusable transformations;
- repeatable evaluation;
- testable modules;
- saved artifacts;
- clear separation between training and inference;
- understandable model outputs.

This repository is intended to demonstrate that broader workflow.

---

## Author

**Pontus Björkell**

Data Analytics & AI · Python · SQL · Machine Learning
