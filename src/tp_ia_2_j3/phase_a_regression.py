from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RegressionMetrics:
    """Immutable container for R2 / MAE / RMSE scores."""

    r2: float
    mae: float
    rmse: float

    def __str__(self) -> str:
        return f"R2={self.r2:.3f}  MAE={self.mae:.3f}  RMSE={self.rmse:.3f}"


class HousingLoader:
    """Wraps fetch_california_housing and exposes X, y, feature names."""

    def __init__(self) -> None:
        _data = fetch_california_housing()
        self.X: np.ndarray = _data.data
        self.y: np.ndarray = _data.target
        self.feature_names: list[str] = list(_data.feature_names)

    def summary(self) -> None:
        print(
            f"California Housing : ({self.X.shape[0]}, {self.X.shape[1]}) variables, "
            "cible = prix médian en centaines de milliers de $"
        )


class RegressionSplit:
    """Train/test split with no-leakage StandardScaler."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        scale: bool = True,
    ) -> None:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
        self.X_train: np.ndarray = X_tr
        self.X_test: np.ndarray = X_te
        self.y_train: np.ndarray = np.asarray(y_tr)
        self.y_test: np.ndarray = np.asarray(y_te)


class RegressionEvaluator:
    """Fits a single regressor on a split and returns RegressionMetrics."""

    def __init__(self, model: BaseEstimator) -> None:
        self._model = model

    def fit_evaluate(self, split: RegressionSplit) -> RegressionMetrics:
        """Train on split.X_train, evaluate on split.X_test."""
        self._model.fit(split.X_train, split.y_train)
        y_pred = self._model.predict(split.X_test)
        return RegressionMetrics(
            r2=float(r2_score(split.y_test, y_pred)),
            mae=float(mean_absolute_error(split.y_test, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(split.y_test, y_pred))),
        )


class HousingBenchmark:
    """Compares LinearRegression (baseline) and RandomForest on California Housing."""

    _MODELS: dict[str, BaseEstimator] = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self._loader = HousingLoader()
        self._split = RegressionSplit(
            self._loader.X,
            self._loader.y,
            test_size=test_size,
            random_state=random_state,
        )
        self._results: dict[str, RegressionMetrics] = {}

    def run(self) -> "HousingBenchmark":
        for name, model in self._MODELS.items():
            self._results[name] = RegressionEvaluator(model).fit_evaluate(self._split)
        return self

    def report(self) -> None:
        self._loader.summary()
        for name, metrics in self._results.items():
            print(f"{name:<20} : {metrics}")

    @property
    def results(self) -> dict[str, RegressionMetrics]:
        """Access results after calling run()."""
        if not self._results:
            raise RuntimeError("Call run() first.")
        return self._results


if __name__ == "__main__":
    HousingBenchmark().run().report()
