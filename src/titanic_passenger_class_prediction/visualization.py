"""Visualization utilities for trained classification models.

This module generates reusable evaluation figures for the passenger-class
prediction project. The plotting functions save figures to disk and return the
created paths so that callers can include them in reports or application views.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline

from titanic_passenger_class_prediction.config import FIGURES_DIR


DEFAULT_FIGURE_DPI = 150
DEFAULT_TOP_FEATURES = 20
DEFAULT_PERMUTATION_REPEATS = 20
DEFAULT_RANDOM_STATE = 42


def ensure_figure_directory(directory: Path = FIGURES_DIR) -> Path:
    """Create the figure directory if it does not already exist.

    Parameters
    ----------
    directory:
        Directory in which generated figures will be stored.

    Returns
    -------
    pathlib.Path
        The created or existing directory.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _validate_output_path(output_path: Path) -> Path:
    """Validate a plot output path and create its parent directory."""
    output_path = Path(output_path)

    if output_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        raise ValueError(
            "Plot output path must use one of these extensions: "
            ".png, .jpg, .jpeg, .svg, or .pdf."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _validate_feature_limit(top_n: int) -> None:
    """Validate the requested number of displayed features."""
    if not isinstance(top_n, int):
        raise TypeError("top_n must be an integer.")

    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")


def _save_and_close_figure(
    figure: plt.Figure,
    output_path: Path,
    dpi: int = DEFAULT_FIGURE_DPI,
) -> Path:
    """Save and close a Matplotlib figure."""
    if dpi <= 0:
        raise ValueError("dpi must be greater than zero.")

    output_path = _validate_output_path(output_path)

    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    return output_path


def plot_confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    output_path: Path,
    *,
    labels: Sequence[Any] | None = None,
    display_labels: Sequence[str] | None = None,
    normalize: str | None = None,
    title: str = "Confusion Matrix",
    dpi: int = DEFAULT_FIGURE_DPI,
) -> Path:
    """Create and save a classification confusion matrix.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Predicted target values.
    output_path:
        Destination image path.
    labels:
        Optional ordered target labels used when constructing the matrix.
    display_labels:
        Optional labels displayed on the plot axes.
    normalize:
        Optional normalization mode accepted by
        ``sklearn.metrics.confusion_matrix``:
        ``None``, ``"true"``, ``"pred"``, or ``"all"``.
    title:
        Plot title.
    dpi:
        Saved image resolution.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if y_true_array.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")

    if y_pred_array.ndim != 1:
        raise ValueError("y_pred must be one-dimensional.")

    if len(y_true_array) == 0:
        raise ValueError("y_true cannot be empty.")

    if len(y_true_array) != len(y_pred_array):
        raise ValueError("y_true and y_pred must contain the same number of rows.")

    valid_normalization_values = {None, "true", "pred", "all"}
    if normalize not in valid_normalization_values:
        raise ValueError(
            "normalize must be one of None, 'true', 'pred', or 'all'."
        )

    matrix = confusion_matrix(
        y_true=y_true_array,
        y_pred=y_pred_array,
        labels=labels,
        normalize=normalize,
    )

    if display_labels is not None and len(display_labels) != matrix.shape[0]:
        raise ValueError(
            "display_labels must contain one value for each matrix class."
        )

    figure, axis = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=display_labels if display_labels is not None else labels,
    )
    display.plot(ax=axis, values_format=".2f" if normalize else "d")
    axis.set_title(title)

    return _save_and_close_figure(
        figure=figure,
        output_path=output_path,
        dpi=dpi,
    )


def get_transformed_feature_names(pipeline: Pipeline) -> np.ndarray:
    """Return output feature names from a fitted preprocessing pipeline.

    The fitted pipeline must contain a step named ``preprocessor`` whose
    transformer supports ``get_feature_names_out``.

    Parameters
    ----------
    pipeline:
        Fitted scikit-learn pipeline.

    Returns
    -------
    numpy.ndarray
        Transformed feature names.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a scikit-learn Pipeline.")

    if "preprocessor" not in pipeline.named_steps:
        raise ValueError(
            "The pipeline must contain a step named 'preprocessor'."
        )

    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(preprocessor, "get_feature_names_out"):
        raise TypeError(
            "The fitted preprocessor does not support get_feature_names_out()."
        )

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception as exc:
        raise ValueError(
            "Unable to obtain transformed feature names. "
            "Confirm that the pipeline has been fitted."
        ) from exc

    feature_names = np.asarray(feature_names, dtype=str)

    if feature_names.size == 0:
        raise ValueError("The preprocessor produced no feature names.")

    return feature_names


def _clean_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Remove common ColumnTransformer prefixes from feature names."""
    cleaned_names: list[str] = []

    for feature_name in feature_names:
        cleaned_name = str(feature_name)

        if "__" in cleaned_name:
            cleaned_name = cleaned_name.split("__", maxsplit=1)[1]

        cleaned_names.append(cleaned_name)

    return cleaned_names


def extract_model_feature_importance(
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Extract model-derived feature importance from a fitted pipeline.

    Models exposing ``feature_importances_`` are supported directly.
    Linear models exposing ``coef_`` are represented using the mean absolute
    coefficient magnitude across classes.

    Parameters
    ----------
    pipeline:
        Fitted preprocessing-and-model pipeline.

    Returns
    -------
    pandas.DataFrame
        Table containing ``feature`` and ``importance`` columns, ordered from
        highest to lowest importance.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a scikit-learn Pipeline.")

    if "model" not in pipeline.named_steps:
        raise ValueError("The pipeline must contain a step named 'model'.")

    feature_names = get_transformed_feature_names(pipeline)
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_, dtype=float)

        if coefficients.ndim == 1:
            importances = np.abs(coefficients)
        elif coefficients.ndim == 2:
            importances = np.mean(np.abs(coefficients), axis=0)
        else:
            raise ValueError(
                "The model coefficient array has an unsupported shape."
            )
    else:
        raise TypeError(
            "The fitted model does not expose feature_importances_ or coef_."
        )

    if len(feature_names) != len(importances):
        raise ValueError(
            "The number of transformed feature names does not match the "
            "number of model importance values."
        )

    importance_table = pd.DataFrame(
        {
            "feature": _clean_feature_names(feature_names),
            "importance": importances,
        }
    )

    importance_table = importance_table.sort_values(
        by="importance",
        ascending=False,
        ignore_index=True,
    )

    return importance_table


def plot_feature_importance(
    pipeline: Pipeline,
    output_path: Path,
    *,
    top_n: int = DEFAULT_TOP_FEATURES,
    title: str = "Model Feature Importance",
    dpi: int = DEFAULT_FIGURE_DPI,
) -> Path:
    """Create and save a model-derived feature-importance chart.

    Parameters
    ----------
    pipeline:
        Fitted scikit-learn pipeline.
    output_path:
        Destination image path.
    top_n:
        Maximum number of features displayed.
    title:
        Plot title.
    dpi:
        Saved image resolution.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    _validate_feature_limit(top_n)

    importance_table = extract_model_feature_importance(pipeline)
    displayed_table = importance_table.head(top_n).sort_values(
        by="importance",
        ascending=True,
    )

    figure_height = max(5.0, 0.35 * len(displayed_table))
    figure, axis = plt.subplots(figsize=(10, figure_height))

    axis.barh(
        displayed_table["feature"],
        displayed_table["importance"],
    )
    axis.set_title(title)
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")

    return _save_and_close_figure(
        figure=figure,
        output_path=output_path,
        dpi=dpi,
    )


def calculate_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: Sequence[Any],
    *,
    scoring: str = "f1_macro",
    n_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Calculate permutation importance using original input features.

    Unlike estimator-derived importance, permutation importance evaluates the
    complete fitted pipeline. The resulting feature names therefore correspond
    to the original input columns rather than one-hot encoded columns.

    Parameters
    ----------
    pipeline:
        Fitted model pipeline.
    X:
        Evaluation feature matrix.
    y:
        Evaluation target values.
    scoring:
        Scikit-learn scorer name.
    n_repeats:
        Number of random permutations per feature.
    random_state:
        Reproducibility seed.
    n_jobs:
        Number of parallel workers.

    Returns
    -------
    pandas.DataFrame
        Ordered table with mean importance and standard deviation.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a scikit-learn Pipeline.")

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X cannot be empty.")

    y_array = np.asarray(y)

    if y_array.ndim != 1:
        raise ValueError("y must be one-dimensional.")

    if len(X) != len(y_array):
        raise ValueError("X and y must contain the same number of rows.")

    if n_repeats <= 0:
        raise ValueError("n_repeats must be greater than zero.")

    result = permutation_importance(
        estimator=pipeline,
        X=X,
        y=y_array,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    if len(X.columns) != len(result.importances_mean):
        raise ValueError(
            "The number of input columns does not match the number of "
            "permutation-importance results."
        )

    importance_table = pd.DataFrame(
        {
            "feature": X.columns.astype(str),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    importance_table = importance_table.sort_values(
        by="importance_mean",
        ascending=False,
        ignore_index=True,
    )

    return importance_table


def plot_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: Sequence[Any],
    output_path: Path,
    *,
    top_n: int = DEFAULT_TOP_FEATURES,
    scoring: str = "f1_macro",
    n_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int | None = None,
    title: str = "Permutation Importance",
    dpi: int = DEFAULT_FIGURE_DPI,
) -> Path:
    """Create and save a permutation-importance chart.

    Parameters
    ----------
    pipeline:
        Fitted preprocessing-and-model pipeline.
    X:
        Evaluation feature matrix.
    y:
        Evaluation target values.
    output_path:
        Destination image path.
    top_n:
        Maximum number of input features displayed.
    scoring:
        Scikit-learn scorer used to measure performance degradation.
    n_repeats:
        Number of random permutations per feature.
    random_state:
        Reproducibility seed.
    n_jobs:
        Number of parallel workers.
    title:
        Plot title.
    dpi:
        Saved image resolution.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    _validate_feature_limit(top_n)

    importance_table = calculate_permutation_importance(
        pipeline=pipeline,
        X=X,
        y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    displayed_table = importance_table.head(top_n).sort_values(
        by="importance_mean",
        ascending=True,
    )

    figure_height = max(5.0, 0.45 * len(displayed_table))
    figure, axis = plt.subplots(figsize=(10, figure_height))

    axis.barh(
        displayed_table["feature"],
        displayed_table["importance_mean"],
        xerr=displayed_table["importance_std"],
        capsize=3,
    )
    axis.axvline(0.0, linewidth=1)
    axis.set_title(title)
    axis.set_xlabel(f"Decrease in {scoring}")
    axis.set_ylabel("Feature")

    return _save_and_close_figure(
        figure=figure,
        output_path=output_path,
        dpi=dpi,
    )