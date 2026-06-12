from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor as _RFR
from sklearn.linear_model import LinearRegression as _LR
from sklearn.linear_model import Ridge as _Ridge
from sklearn.pipeline import make_pipeline as _pipeline
from sklearn.preprocessing import StandardScaler as _SS


class _BaseRegressorAdapter:
    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseRegressorAdapter":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


class LinearRegressionAdapter(_BaseRegressorAdapter):
    def __init__(self, scale: bool = True) -> None:
        lr = _LR()
        self._model = _pipeline(_SS(), lr) if scale else lr


class RidgeRegressionAdapter(_BaseRegressorAdapter):
    def __init__(self, alpha: float = 1.0, scale: bool = True) -> None:
        ridge = _Ridge(alpha=alpha)
        self._model = _pipeline(_SS(), ridge) if scale else ridge


class RandomForestRegressorAdapter(_BaseRegressorAdapter):
    def __init__(self, n_estimators: int = 100, random_state: int = 42, n_jobs: int = -1) -> None:
        self._model = _RFR(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
