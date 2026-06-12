from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold as _SKF


class StratifiedKFoldAdapter:
    """Yields (train_idx, test_idx) pairs for stratified k-fold cross-validation."""

    def __init__(self, k: int = 5, random_state: int = 42) -> None:
        self._k = k
        self._kf = _SKF(n_splits=k, shuffle=True, random_state=random_state)

    @property
    def k(self) -> int:
        return self._k

    def splits(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from self._kf.split(X, y)
