from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DataChunk:
    """Immutable container for a (features, labels) pair."""

    features: np.ndarray
    labels: np.ndarray

    @property
    def n_samples(self) -> int:
        return len(self.labels)

    @property
    def n_features(self) -> int:
        return self.features.shape[1] if self.features.ndim > 1 else 1

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.labels)


@dataclass
class SplitResult:
    """Three-way stratified split — train / validation / test."""

    train: DataChunk
    val: DataChunk
    test: DataChunk

    def sizes(self) -> tuple[int, int, int]:
        return self.train.n_samples, self.val.n_samples, self.test.n_samples

    def total(self) -> int:
        return sum(self.sizes())

    def __str__(self) -> str:
        tr, va, te = self.sizes()
        return f"SplitResult(train={tr}, val={va}, test={te}, total={tr + va + te})"
