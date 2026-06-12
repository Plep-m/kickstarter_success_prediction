from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold as _SKF
from sklearn.model_selection import train_test_split as _tts

from ...domain.dataset import DataChunk, SplitResult


class StratifiedSplitterAdapter:
    """Three-way stratified split: train / val / test with no leakage."""

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self._test_size = test_size
        self._val_size = val_size
        self._random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray) -> SplitResult:
        X_temp, X_test, y_temp, y_test = _tts(
            X, y,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y,
        )
        adjusted_val = self._val_size / (1.0 - self._test_size)
        X_train, X_val, y_train, y_val = _tts(
            X_temp, y_temp,
            test_size=adjusted_val,
            random_state=self._random_state,
            stratify=y_temp,
        )
        return SplitResult(
            train=DataChunk(X_train, np.asarray(y_train)),
            val=DataChunk(X_val, np.asarray(y_val)),
            test=DataChunk(X_test, np.asarray(y_test)),
        )


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
