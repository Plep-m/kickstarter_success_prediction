from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ScalerPort(Protocol):
    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...
    def inverse_transform(self, X: np.ndarray) -> np.ndarray: ...
