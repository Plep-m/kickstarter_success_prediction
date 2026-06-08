from typing import Callable, Literal

import numpy as np
from sklearn import datasets
from sklearn.utils import Bunch

DatasetName = Literal["iris", "wine", "breast_cancer", "digits"]


class Dataset:
    _LOADERS: dict[str, Callable[[], Bunch]] = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "digits": datasets.load_digits,
    }

    def __init__(self, name: DatasetName):
        if name not in self._LOADERS:
            raise ValueError(f"Unknown dataset '{name}'. Available: {list(self._LOADERS)}")
        self.name: str = name
        self._bunch: Bunch = self._LOADERS[name]()

    @property
    def X(self) -> np.ndarray:
        return self._bunch.data

    @property
    def y(self) -> np.ndarray:
        return self._bunch.target

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._LOADERS)

    def __str__(self) -> str:
        rows, cols = self.X.shape
        lines = [f"Rows, columns: ({rows}, {cols})"]
        classes, counts = np.unique(self.y, return_counts=True)
        target_names = getattr(self._bunch, "target_names", classes)
        for cls_idx, count in zip(classes, counts):
            lines.append(f"Class {cls_idx} ({target_names[cls_idx]}): {count} cases")
        return "\n".join(lines)
