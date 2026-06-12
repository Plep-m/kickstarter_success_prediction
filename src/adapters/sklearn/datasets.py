from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from sklearn import datasets as _sk_datasets
from sklearn.utils import Bunch

SklearnDatasetName = Literal["iris", "wine", "breast_cancer", "digits", "california_housing"]

_LOADERS: dict[str, Callable[[], Bunch]] = {
    "iris": _sk_datasets.load_iris,
    "wine": _sk_datasets.load_wine,
    "breast_cancer": _sk_datasets.load_breast_cancer,
    "digits": _sk_datasets.load_digits,
    "california_housing": _sk_datasets.fetch_california_housing,
}


class SklearnDatasetAdapter:
    """Loads a bundled sklearn dataset and exposes (X, y) + metadata."""

    def __init__(self, name: SklearnDatasetName) -> None:
        if name not in _LOADERS:
            raise ValueError(f"Unknown dataset '{name}'. Available: {list(_LOADERS)}")
        self._name = name
        self._bunch: Bunch | None = None

    def _ensure_loaded(self) -> None:
        if self._bunch is None:
            self._bunch = _LOADERS[self._name]()

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_loaded()
        return self._bunch.data, self._bunch.target

    @property
    def feature_names(self) -> list[str]:
        self._ensure_loaded()
        return list(getattr(self._bunch, "feature_names", []))

    @property
    def target_names(self) -> list[str]:
        self._ensure_loaded()
        return list(getattr(self._bunch, "target_names", []))

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def available(cls) -> list[str]:
        return list(_LOADERS)
