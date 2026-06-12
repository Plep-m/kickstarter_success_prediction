from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans as _KMeans


class KMeansAdapter:
    """KMeans adapter exposing ClustererPort interface."""

    def __init__(self, n_clusters: int, n_init: int = 10, random_state: int = 42) -> None:
        self._model = _KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray) -> "KMeansAdapter":
        self._model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def labels_(self) -> np.ndarray:
        return self._model.labels_

    @property
    def inertia_(self) -> float:
        return float(self._model.inertia_)
