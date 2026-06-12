from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RegressorPort(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressorPort": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
