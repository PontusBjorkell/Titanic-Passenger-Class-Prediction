# 🚢 Titanic Passenger Class Prediction

> **An end-to-end analytics and machine learning project demonstrating reproducible data preparation, feature engineering, SQL analytics, automated testing, and predictive modeling using the Titanic dataset.**

---

## Project Overview

This project began as a notebook-based exploration and has been redesigned into a production-inspired machine learning workflow.

Instead of relying on a single notebook, the project is structured as a reusable Python package with separate modules for data loading, validation, feature engineering, preprocessing, database creation, model training, and application development.

The goal is to demonstrate software engineering practices alongside data analytics and machine learning.

---

## Objectives

- Build a reproducible data preparation pipeline
- Perform exploratory data analysis
- Engineer meaningful passenger features
- Create a reusable preprocessing pipeline
- Build an SQLite analytics database
- Train and evaluate machine learning models
- Deploy predictions through a Streamlit application
- Follow software engineering best practices

---

# Project Architecture

```text
                    Raw CSV Files
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
             ┌────────────┴────────────┐
             ▼                         ▼
      SQLite Analytics         Machine Learning
             │                         │
             ▼                         ▼
        SQL Analysis          Model Training Pipeline
                                        │
                                        ▼
                               Saved Model Artifacts
                                        │
                                        ▼
                                Streamlit Dashboard
```

---

# Repository Structure

```text
Titanic-Passenger-Class-Prediction
│
├── app/                     # Streamlit application
├── artifacts/               # Generated reports and trained models
├── config/                  # Configuration files
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/               # Exploratory analysis
├── reports/                 # Figures and evaluation reports
├── scripts/                 # Executable project scripts
├── sql/                     # SQL views and queries
├── src/
│   └── titanic_passenger_class_prediction/
│       ├── config.py
│       ├── data.py
│       ├── validation.py
│       ├── features.py
│       ├── preprocessing.py
│       ├── modeling.py
│
├── tests/                   # Automated tests
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

# Technologies

| Area | Tools |
|------|------|
| Programming | Python |
| Data Analysis | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Database | SQLite |
| Testing | pytest |
| Application | Streamlit *(coming next)* |
| Version Control | Git & GitHub |

---

# Workflow

## 1. Data Preparation

```
Raw CSV
    ↓
Validation
    ↓
Feature Engineering
    ↓
Processed Dataset
```

Run:

```bash
python scripts/prepare_data.py
```

---

## 2. Build Analytics Database

```
Processed Data
        ↓
SQLite Database
        ↓
SQL Views
```

Run:

```bash
python scripts/build_database.py
```

---

## 3. Model Training *(coming next)*

```
Processed Dataset
        ↓
Train/Test Split
        ↓
Preprocessing
        ↓
Model Training
        ↓
Evaluation
        ↓
Saved Model
```

---

## Testing

The project includes automated tests covering:

- Data loading
- Validation
- Feature engineering
- Preprocessing
- Modeling
- Database creation

Run all tests:

```bash
python -m pytest
```

Current status:

```
21 tests passing
```

---

# Current Progress

| Component | Status |
|-----------|:------:|
| Project Structure | ✅ |
| Data Pipeline | ✅ |
| Validation | ✅ |
| Feature Engineering | ✅ |
| SQLite Analytics | ✅ |
| Preprocessing Pipeline | ✅ |
| Automated Tests | ✅ |
| Model Training | 🚧 |
| Streamlit Dashboard | 🚧 |
| Deployment | 🚧 |

---

# Future Improvements

- Train and compare multiple classification models
- Hyperparameter tuning
- Model persistence
- Interactive Streamlit dashboard
- Docker support
- Cloud deployment

---

# Why This Project?

The purpose of this repository is to demonstrate not only machine learning techniques but also the engineering practices required to build maintainable and reproducible data science projects.

Rather than focusing on a single notebook, the project follows a modular architecture inspired by real-world analytics and machine learning workflows.

---

## Author

**Pontus Björkell**

Data Analytics & AI | Python | SQL | Machine Learning