# Titanic Passenger Class Prediction

> A production-style machine learning project that predicts passenger class
> from Titanic passenger information using feature engineering, model
> comparison, hyperparameter tuning, automated reporting, and extensive
> software engineering practices.

---

## Project Overview

This project demonstrates how to build a complete machine learning system
rather than a single notebook.

Instead of focusing only on predictive performance, the repository
emphasizes

- clean software architecture
- reusable Python modules
- automated testing
- reproducible machine learning
- model comparison
- experiment tracking
- automated report generation

The final solution follows a production-inspired workflow from raw CSV files
to a fully trained model and automatically generated evaluation reports.

---

## Project Goals

The project was designed to demonstrate practical machine learning engineering
skills that are commonly expected in data analyst and junior machine learning
roles.

Specifically, it demonstrates:

- Data validation
- Feature engineering
- Pipeline-based preprocessing
- Cross-validation
- Hyperparameter optimization
- Model comparison
- Model persistence
- Automated reporting
- SQLite integration
- Unit testing with pytest

---

## Repository Structure

```text
Titanic-Passenger-Class-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│   ├── models/
│   ├── metrics/
│   └── figures/
│
├── scripts/
│   ├── prepare_data.py
│   ├── build_database.py
│   └── train.py
│
├── sql/
│   ├── create_schema.sql
│   └── create_views.sql
│
├── src/
│   └── titanic_passenger_class_prediction/
│       ├── config.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── validation.py
│       ├── features.py
│       ├── modeling.py
│       ├── tuning.py
│       ├── evaluation.py
│       ├── visualization.py
│       ├── reporting.py
│       ├── persistence.py
│       └── database.py
│
├── tests/
│
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

## Machine Learning Workflow

```text
Raw CSV
    │
    ▼
Data Validation
    │
    ▼
Preprocessing
    │
    ▼
Feature Engineering
    │
    ▼
Candidate Models
    │
    ▼
5-Fold Stratified Cross Validation
    │
    ▼
Randomized Hyperparameter Search
    │
    ▼
Best Model Selection
    │
    ▼
Holdout Evaluation
    │
    ▼
Automatic Reports
    │
    ▼
Saved Model
```

---

## Feature Engineering

Several domain-inspired features were created before model training.

Examples include

- FamilySize
- IsAlone
- CabinCount
- CabinDeck
- FareLog
- Passenger Title
- Ticket Prefix
- FamilySizeGroup

These engineered variables significantly improve predictive performance over
using the raw dataset alone.

---

## Candidate Models

The project compares multiple classification algorithms.

| Model | Purpose |
|-------|---------|
| Dummy Classifier | Baseline |
| Logistic Regression | Linear baseline |
| Random Forest | Final production model |

The best model is selected using **Macro F1** obtained from stratified
cross-validation.

---

## Hyperparameter Tuning

Random Forest hyperparameters are optimized using
RandomizedSearchCV.

Optimized parameters include

- Number of trees
- Maximum tree depth
- Minimum samples split
- Minimum samples leaf
- Bootstrap
- Feature selection
- Class weighting

Model selection is performed **only** using cross-validation.
The holdout dataset is never used during tuning.

---

## Final Model Performance

| Metric | Value |
|---------|-------|
| Accuracy | **88.83%** |
| Balanced Accuracy | **86.00%** |
| Macro F1 | **86.41%** |

The final model achieved approximately **91.4% Macro F1 during
cross-validation** before being evaluated once on the unseen holdout set.

---

## Feature Importance

The trained Random Forest identified the following features as the most
important predictors.

1. Fare
2. FareLog
3. CabinCount
4. CabinDeck
5. HasCabin

Permutation importance confirmed that ticket price and family-related
features contributed most strongly to predictive performance.

---

## Generated Reports

The training pipeline automatically produces

- Model comparison table
- Classification report
- Confusion matrix
- Feature importance plot
- Permutation importance plot
- Training summary
- Model metadata
- Hyperparameter search results
- Saved model

---

## Confusion Matrix

![Confusion Matrix](artifacts/figures/confusion_matrix.png)

---

## Feature Importance

![Feature Importance](artifacts/figures/feature_importance.png)

---

## Permutation Importance

![Permutation Importance](artifacts/figures/permutation_importance.png)

---

## Testing

The repository includes a comprehensive pytest suite covering

- Data loading
- Validation
- Feature engineering
- Preprocessing
- Modeling
- Cross-validation
- Hyperparameter tuning
- Persistence
- Reporting
- Visualization
- SQLite utilities
- Training pipeline

This helps ensure correctness and reproducibility across the entire project.

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- SQLite
- pytest
- joblib

---

## Future Improvements

Potential extensions include

- Streamlit application
- SHAP explainability
- Docker support
- GitHub Actions CI
- Batch inference
- Prediction CLI
- REST API

---

## Author

Pontus Björkell

Master's degree in Philosophy

Minor studies in

- Statistics
- Computer Science
- Mathematics

Currently building a portfolio of production-style machine learning projects
for data analyst and machine learning positions.