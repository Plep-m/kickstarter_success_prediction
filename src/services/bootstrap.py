from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from .._metrics import accuracy_score
from ..domain.metrics import BootstrapResult
from ..ports.classifier import ClassifierPort


class BootstrapEvaluator:
    """Estimates model stability via bootstrap with out-of-bag (OOB) evaluation."""

    def __init__(
        self,
        model: ClassifierPort,
        n_iterations: int = 30,
        random_state: int = 42,
        scoring: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
    ) -> None:
        self._model = model
        self._n = n_iterations
        self._rng = np.random.default_rng(random_state)
        self._scoring = scoring

    def evaluate(self, X, y) -> BootstrapResult:
        """Sample with replacement n_iterations times; score on OOB points."""
        n = len(y)
        scores: list[float] = []
        _sel = lambda arr, idx: arr.iloc[idx] if hasattr(arr, "iloc") else arr[idx]
        for _ in range(self._n):
            idx_in = self._rng.choice(n, size=n, replace=True)
            idx_oob = np.setdiff1d(np.arange(n), idx_in)
            if len(idx_oob) == 0:
                continue
            m = copy.deepcopy(self._model)
            m.fit(_sel(X, idx_in), _sel(y, idx_in))
            scores.append(float(self._scoring(_sel(y, idx_oob), m.predict(_sel(X, idx_oob)))))
        arr = np.array(scores)
        return BootstrapResult(
            scores=tuple(float(s) for s in arr),
            mean=float(arr.mean()),
            std=float(arr.std()),
        )
