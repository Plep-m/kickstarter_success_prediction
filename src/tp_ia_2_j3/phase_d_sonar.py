from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SONAR_URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/undocumented/connectionist-bench/sonar/sonar.all-data"
)


@dataclass(frozen=True)
class SonarMetrics:
    """Accuracy and binary F1 for one classifier run."""

    accuracy: float
    f1: float

    def __str__(self) -> str:
        return f"accuracy={self.accuracy:.2%}  F1={self.f1:.3f}"


class SonarLoader:
    """Loads the UCI Sonar dataset; maps M→1 (mine), R→0 (rock)."""

    def __init__(self, url_or_path: str = SONAR_URL) -> None:
        self._source = url_or_path

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) where y=1 for mine, y=0 for rock."""
        df = pd.read_csv(self._source, header=None)
        X = df.iloc[:, :60].values.astype(float)
        y = (df.iloc[:, 60] == "M").astype(int).values
        mines = int(y.sum())
        print(f"Sonar : {X.shape}, mines={mines}, rochers={len(y) - mines}")
        return X, y


class SonarEvaluator:
    """Fits one classifier (with optional scaling) and returns SonarMetrics."""

    def __init__(self, model: BaseEstimator, scale: bool = True) -> None:
        self._model = model
        self._scale = scale

    def fit_evaluate(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> SonarMetrics:
        X_tr, X_te = X_train, X_test
        if self._scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_train)
            X_te = sc.transform(X_test)
        self._model.fit(X_tr, y_train)
        y_pred = self._model.predict(X_te)
        return SonarMetrics(
            accuracy=float(accuracy_score(y_test, y_pred)),
            f1=float(f1_score(y_test, y_pred, average="binary")),
        )


class SonarBenchmark:
    """Benchmarks LogisticRegression, SVC(rbf), and RandomForest on Sonar."""

    _COMPETITORS: dict[str, tuple[BaseEstimator, bool]] = {
        "LogisticRegression": (LogisticRegression(max_iter=5000, random_state=42), True),
        "SVC (rbf)": (SVC(kernel="rbf", random_state=42), True),
        "RandomForest": (RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), False),
    }

    def __init__(
        self,
        url_or_path: str = SONAR_URL,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> None:
        self._loader = SonarLoader(url_or_path)
        self._test_size = test_size
        self._random_state = random_state
        self._results: dict[str, SonarMetrics] = {}

    def run(self) -> "SonarBenchmark":
        X, y = self._loader.load()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y,
        )
        for name, (model, scale) in self._COMPETITORS.items():
            self._results[name] = SonarEvaluator(model, scale).fit_evaluate(
                X_train, X_test, y_train, y_test
            )
        return self

    def report(self) -> None:
        for name, metrics in self._results.items():
            print(f"{name:<22} : {metrics}")

    @property
    def results(self) -> dict[str, SonarMetrics]:
        if not self._results:
            raise RuntimeError("Call run() first.")
        return self._results


if __name__ == "__main__":
    SonarBenchmark().run().report()
