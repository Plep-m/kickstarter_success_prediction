"""Math layer — preprocessing, models, cross-validation and scoring.

Mirrors the modelling and evaluation phases of the CRISP-DM notebooks:
5-fold stratified cross-validation with five metrics, then a final hold-out
evaluation (accuracy, precision, recall, F1, AUC-ROC) and feature importance.

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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate as sk_cross_validate
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import CATEGORICAL, NUMERIC

RANDOM_STATE = 42

# The five metrics tracked in the notebooks (Phase 4 & 5).
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


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


def build_models(random_state: int = RANDOM_STATE, pos_weight: float = 1.5) -> dict[str, Pipeline]:
    """The three classifiers, each a full preprocessing + model pipeline.

    `pos_weight` is XGBoost's scale_pos_weight (≈ #failures / #successes). Pass
    `class_balance(y_train)` to reproduce the notebook's data-driven value.
    """
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
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_weight,
                eval_metric="logloss", n_jobs=-1, verbosity=0, random_state=random_state,
            )),
        ]),
    }


def class_balance(y: pd.Series) -> float:
    """#negatives / #positives — the data-driven scale_pos_weight for XGBoost."""
    return float((y == 0).sum() / (y == 1).sum())


def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """Stratified train/test split — the one split both entry points share."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def cross_validate(X, y, n_splits: int = 5, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """5-fold stratified CV (Phase 4): mean of each metric per model, on the train set."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for name, model in build_models().items():
        scores = sk_cross_validate(model, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            row[metric] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def evaluate(model: Pipeline, X_test, y_test) -> dict[str, float]:
    """All five metrics for one fitted model (Phase 5)."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def benchmark(X_train, X_test, y_train, y_test) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    """Fit every model, evaluate on test, return (results sorted by AUC, fitted models)."""
    fitted: dict[str, Pipeline] = {}
    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        rows.append({"model": name, **evaluate(model, X_test, y_test)})
    table = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)
    return table, fitted


def feature_importances(pipeline: Pipeline, top_n: int = 15) -> pd.Series:
    """Tree-model importances mapped back to the encoded feature names (Phase 5.1).

    Works for random_forest and xgboost (logistic_regression has no importances_).
    """
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    names = prep.get_feature_names_out()
    return (
        pd.Series(clf.feature_importances_, index=names)
        .sort_values(ascending=False)
        .head(top_n)
    )
