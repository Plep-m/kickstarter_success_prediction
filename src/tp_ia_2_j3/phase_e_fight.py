from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tp_ia_2_j3.phase_d_sonar import SONAR_URL, SonarLoader

MetricFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class FightResult:
    """Score and training time for one competitor."""

    name: str
    score: float
    train_time: float

    def __str__(self) -> str:
        return f"{self.name:<22} : score={self.score:.4f}  ({self.train_time:.3f}s)"


class IAFight:
    """Benchmarks every registered competitor on a fixed train/test split."""

    _DEFAULT_COMPETITORS: dict[str, BaseEstimator] = {
        "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "SVC_rbf": SVC(kernel="rbf", random_state=42),
        "GaussianNB": GaussianNB(),
    }

    def __init__(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        metric_fn: MetricFn,
        scale: bool = True,
    ) -> None:
        scaler = StandardScaler()
        self._X_train = scaler.fit_transform(X_train) if scale else X_train
        self._X_test = scaler.transform(X_test) if scale else X_test
        self._y_train = y_train
        self._y_test = y_test
        self._metric_fn = metric_fn
        self._competitors: dict[str, BaseEstimator] = dict(self._DEFAULT_COMPETITORS)
        self._results: list[FightResult] = []

    def register(self, name: str, model: BaseEstimator) -> None:
        """Add or replace a competitor before calling run()."""
        self._competitors[name] = model

    def run(self) -> "IAFight":
        results: list[FightResult] = []
        for name, model in self._competitors.items():
            fresh = copy.deepcopy(model)
            t0 = time.perf_counter()
            fresh.fit(self._X_train, self._y_train)
            elapsed = time.perf_counter() - t0
            y_pred = fresh.predict(self._X_test)
            score = self._metric_fn(self._y_test, y_pred)
            results.append(FightResult(name=name, score=score, train_time=elapsed))
        self._results = sorted(results, key=lambda r: r.score, reverse=True)
        return self

    @property
    def results(self) -> list[FightResult]:
        if not self._results:
            raise RuntimeError("Call run() first.")
        return self._results

    def champion(self) -> FightResult:
        """Return the highest-scoring competitor."""
        return self.results[0]


class Leaderboard:
    """Renders a ranked fight results table."""

    def __init__(
        self,
        results: list[FightResult],
        dataset_name: str,
        metric_name: str,
    ) -> None:
        self._results = results
        self._dataset_name = dataset_name
        self._metric_name = metric_name

    def __str__(self) -> str:
        header = f"=== LEADERBOARD (jeu : {self._dataset_name}, métrique : {self._metric_name}) ==="
        rows = [header]
        for rank, result in enumerate(self._results, 1):
            rows.append(f"{rank}. {result}")
        return "\n".join(rows)


def f1_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 score — positive class = 1."""
    return float(f1_score(y_true, y_pred, average="binary"))


if __name__ == "__main__":
    X, y = SonarLoader(SONAR_URL).load()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    fight = IAFight(X_train, X_test, y_train, y_test, metric_fn=f1_binary, scale=True)
    fight.run()

    board = Leaderboard(fight.results, dataset_name="sonar", metric_name="F1")
    print(board)
    print(f"\nChampion : {fight.champion().name}")
