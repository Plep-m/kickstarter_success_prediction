from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier as _RFC
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier as _XGB
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


class SklearnPipelineAdapter:
    """Wraps a full sklearn Pipeline (preprocessing + classifier) behind ClassifierPort.

    Accepts both numpy arrays and pandas DataFrames, making it compatible with
    ColumnTransformer pipelines that rely on column names.
    """

    def __init__(self, pipeline: Pipeline, name: str = "") -> None:
        self._pipeline = pipeline
        self.name = name or str(pipeline.steps[-1][0])

    def fit(self, X, y) -> "SklearnPipelineAdapter":
        self._pipeline.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self._pipeline.predict_proba(X)

    def __repr__(self) -> str:
        return f"SklearnPipelineAdapter({self.name})"


def kickstarter_pipelines(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    random_state: int = 42,
    pos_weight: float = 1.5,
) -> dict[str, SklearnPipelineAdapter]:
    """Ready-to-use classification pipelines for Kickstarter data (LR, RF, XGBoost).

    Each pipeline embeds its own ColumnTransformer:
    - Numerics: median imputation + StandardScaler
    - Categoricals: most-frequent imputation + OneHotEncoder (handle_unknown='ignore')

    pos_weight: approximate failed/successful ratio used as XGBoost's scale_pos_weight.
    Columns match KickstarterCleaner.FEATURES by default.
    """
    num_feats = numeric_features or [
        "goal_usd", "duration_days", "description_length", "launch_year", "launch_month"
    ]
    cat_feats = categorical_features or ["category", "country"]

    def _preprocessor() -> ColumnTransformer:
        return ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_feats),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, max_categories=50
                )),
            ]), cat_feats),
        ])

    pipes: dict[str, Pipeline] = {
        "logistic_regression": Pipeline([
            ("prep", _preprocessor()),
            ("clf", _LR(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("prep", _preprocessor()),
            ("clf", _RFC(
                n_estimators=200, max_depth=20, min_samples_leaf=10,
                class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
            )),
        ]),
    }

    if _XGB_AVAILABLE:
        pipes["xgboost"] = Pipeline([
            ("prep", _preprocessor()),
            ("clf", _XGB(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                scale_pos_weight=pos_weight, random_state=random_state,
                n_jobs=-1, verbosity=0,
            )),
        ])

    return {name: SklearnPipelineAdapter(pipe, name) for name, pipe in pipes.items()}
