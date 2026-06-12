from __future__ import annotations

import copy
import time
from typing import Callable

import numpy as np

from ..domain.metrics import FightResult
from ..ports.classifier import ClassifierPort
from ..ports.scaler import ScalerPort

MetricFn = Callable[[np.ndarray, np.ndarray], float]


class Benchmark:
    """Benchmarks every registered ClassifierPort on a fixed train/test split.

    The metric function is user-supplied, making the benchmark task-agnostic.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        metric_fn: MetricFn,
        scaler: ScalerPort | None = None,
    ) -> None:
        if scaler is not None:
            self._X_train = scaler.fit_transform(X_train)
            self._X_test = scaler.transform(X_test)
        else:
            self._X_train = X_train
            self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._metric_fn = metric_fn
        self._competitors: dict[str, ClassifierPort] = {}
        self._results: list[FightResult] = []

    def register(self, name: str, model: ClassifierPort) -> "Benchmark":
        self._competitors[name] = model
        return self

    def register_many(self, models: dict[str, ClassifierPort]) -> "Benchmark":
        self._competitors.update(models)
        return self

    def run(self) -> "Benchmark":
        results: list[FightResult] = []
        for name, model in self._competitors.items():
            fresh = copy.deepcopy(model)
            t0 = time.perf_counter()
            fresh.fit(self._X_train, self._y_train)
            elapsed = time.perf_counter() - t0
            score = self._metric_fn(self._y_test, fresh.predict(self._X_test))
            results.append(FightResult(name=name, score=score, train_time=elapsed))
        self._results = sorted(results, key=lambda r: r.score, reverse=True)
        return self

    @property
    def results(self) -> list[FightResult]:
        if not self._results:
            raise RuntimeError("Call run() first.")
        return self._results

    def champion(self) -> FightResult:
        return self.results[0]

    def __str__(self) -> str:
        if not self._results:
            return "No results — call run() first."
        rows = ["=== BENCHMARK LEADERBOARD ==="]
        for rank, r in enumerate(self._results, 1):
            rows.append(f"{rank}. {r}")
        return "\n".join(rows)
