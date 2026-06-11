from __future__ import annotations

import copy
from typing import Callable

import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


@dataclass(frozen=True)
class BootstrapResult:
    """OOB scores from all bootstrap iterations with aggregate stats."""

    scores: tuple[float, ...]
    mean: float
    std: float

    def __str__(self) -> str:
        return (
            f"Score moyen sur {len(self.scores)} bootstraps : "
            f"{self.mean:.3f} (± {self.std:.3f})"
        )


class BootstrapEvaluator:
    """Estimates model stability via bootstrap with out-of-bag evaluation."""

    def __init__(
        self,
        model: BaseEstimator,
        n_iterations: int = 30,
        random_state: int = 42,
        scoring: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
    ) -> None:
        self._model = model
        self._n = n_iterations
        self._rng = np.random.default_rng(random_state)
        self._scoring = scoring

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> BootstrapResult:
        """Sample with replacement n_iterations times; score on out-of-bag points."""
        n = len(y)
        scores: list[float] = []
        for _ in range(self._n):
            idx_in = self._rng.choice(n, size=n, replace=True)
            idx_oob = np.setdiff1d(np.arange(n), idx_in)
            if len(idx_oob) == 0:
                continue
            m = copy.deepcopy(self._model)
            m.fit(X[idx_in], y[idx_in])
            scores.append(self._scoring(y[idx_oob], m.predict(X[idx_oob])))
        arr = np.array(scores)
        return BootstrapResult(
            scores=tuple(float(s) for s in arr),
            mean=float(arr.mean()),
            std=float(arr.std()),
        )


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_breast_cancer(return_X_y=True)
    result = BootstrapEvaluator(RandomForestClassifier(n_estimators=50, random_state=42)).evaluate(X, y)
    print(result)
