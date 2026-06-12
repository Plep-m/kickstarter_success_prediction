from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from .._metrics import accuracy_score, f1_score
from ..domain.metrics import CrossValResult
from ..ports.classifier import ClassifierPort
from ..ports.splitter import KFoldSplitterPort

_SCORING_FNS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": accuracy_score,
    "f1": lambda yt, yp: f1_score(yt, yp, average="weighted"),
    "f1_binary": lambda yt, yp: f1_score(yt, yp, average="binary"),
}


class CrossValidator:
    """Stratified k-fold cross-validation using any KFoldSplitterPort."""

    def __init__(self, kfold: KFoldSplitterPort, scoring: str = "accuracy") -> None:
        if scoring not in _SCORING_FNS:
            raise ValueError(f"Unknown scoring '{scoring}'. Available: {list(_SCORING_FNS)}")
        self._kfold = kfold
        self._score_fn = _SCORING_FNS[scoring]

    def evaluate(
        self, model: ClassifierPort, X, y
    ) -> CrossValResult:
        scores: list[float] = []
        _sel = lambda arr, idx: arr.iloc[idx] if hasattr(arr, "iloc") else arr[idx]
        for train_idx, test_idx in self._kfold.splits(X, y):
            m = copy.deepcopy(model)
            m.fit(_sel(X, train_idx), _sel(y, train_idx))
            scores.append(float(self._score_fn(_sel(y, test_idx), m.predict(_sel(X, test_idx)))))
        arr = np.array(scores)
        return CrossValResult(
            scores=tuple(float(s) for s in arr),
            mean=float(arr.mean()),
            std=float(arr.std()),
            k=self._kfold.k,
        )
