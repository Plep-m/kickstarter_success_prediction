from typing import Literal

import numpy as np
from sklearn import datasets

DatasetName = Literal["iris", "wine", "breast_cancer", "digits"]

_LOADERS: dict[str, callable] = {
    "iris": datasets.load_iris,
    "wine": datasets.load_wine,
    "breast_cancer": datasets.load_breast_cancer,
    "digits": datasets.load_digits,
}


def load_dataset(name: DatasetName) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays for a named sklearn dataset."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(_LOADERS)}")
    return _LOADERS[name](return_X_y=True)


def available_datasets() -> list[str]:
    return list(_LOADERS)
