from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClustererPort(Protocol):
    def fit(self, X: np.ndarray) -> "ClustererPort": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @property
    def labels_(self) -> np.ndarray: ...

    @property
    def inertia_(self) -> float: ...
