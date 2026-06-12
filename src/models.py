"""Math layer — preprocessing, the ML models, training and scoring.

Everything sklearn/xgboost lives here. The business layer (`data`) knows nothing
about machine learning; this layer knows nothing about Kickstarter beyond the
column lists it imports.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import CATEGORICAL, NUMERIC


def _preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics; mode-impute + one-hot encode categoricals."""
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=50)),
        ]), CATEGORICAL),
    ])


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """The three classifiers, each a full preprocessing + model pipeline."""
    return {
        "logistic_regression": Pipeline([
            ("prep", _preprocessor()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=random_state,
            )),
        ]),
        "random_forest": Pipeline([
            ("prep", _preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_leaf=10,
                class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
            )),
        ]),
        "xgboost": Pipeline([
            ("prep", _preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.5,
                eval_metric="logloss", n_jobs=-1, verbosity=0, random_state=random_state,
            )),
        ]),
    }


def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Stratified train/test split — the one split both entry points share."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Accuracy and F1 (weighted + success-class) for a fitted model."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "f1_success": float(f1_score(y_test, y_pred, average="binary", pos_label=1)),
    }


def benchmark(X_train, X_test, y_train, y_test) -> list[tuple[str, float]]:
    """Fit every model, score weighted F1, return (name, score) sorted best-first."""
    scores = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        scores.append((name, evaluate(model, X_test, y_test)["f1_weighted"]))
    return sorted(scores, key=lambda r: r[1], reverse=True)
