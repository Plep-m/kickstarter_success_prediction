"""Internal re-exports of sklearn metric functions.

Services import from here so swapping the metric backend is a one-file change.
"""
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)

__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "f1_score",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_score",
    "r2_score",
    "recall_score",
    "silhouette_score",
]
