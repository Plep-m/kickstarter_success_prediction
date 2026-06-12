from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np

from ..domain.dataset import SplitResult


@runtime_checkable
class SplitterPort(Protocol):
    """Produces a three-way train/val/test split."""

    def split(self, X: np.ndarray, y: np.ndarray) -> SplitResult: ...


@runtime_checkable
class KFoldSplitterPort(Protocol):
    """Yields (train_indices, test_indices) pairs for k-fold cross-validation."""

    def splits(self, X: np.ndarray, y: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...

    @property
    def k(self) -> int: ...
