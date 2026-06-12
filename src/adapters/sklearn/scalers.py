from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler as _MinMax
from sklearn.preprocessing import RobustScaler as _Robust
from sklearn.preprocessing import StandardScaler as _SS


class _BaseScalerAdapter:
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(X)


class StandardScalerAdapter(_BaseScalerAdapter):
    def __init__(self) -> None:
        self._scaler = _SS()


class MinMaxScalerAdapter(_BaseScalerAdapter):
    def __init__(self, feature_range: tuple[float, float] = (0, 1)) -> None:
        self._scaler = _MinMax(feature_range=feature_range)


class RobustScalerAdapter(_BaseScalerAdapter):
    def __init__(self) -> None:
        self._scaler = _Robust()
