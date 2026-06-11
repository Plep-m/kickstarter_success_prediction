from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataSplit:
    """Holds the six arrays from a stratified train / validation / test split."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    def sizes(self) -> tuple[int, int, int]:
        """Return (n_train, n_val, n_test)."""
        return len(self.y_train), len(self.y_val), len(self.y_test)

    def total(self) -> int:
        return sum(self.sizes())


class DataSplitter:
    """Stratified three-way split with no data leakage between sets."""

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        if val_size <= 0:
            raise ValueError("val_size must be > 0.")
        self._test_size = test_size
        self._val_size = val_size
        self._random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray) -> DataSplit:
        """Return a DataSplit with class proportions preserved in all three sets."""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y,
        )
        adjusted_val = self._val_size / (1.0 - self._test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=adjusted_val,
            random_state=self._random_state,
            stratify=y_temp,
        )
        return DataSplit(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=np.asarray(y_train), y_val=np.asarray(y_val), y_test=np.asarray(y_test),
        )

    def report(self, split: DataSplit) -> None:
        tr, va, te = split.sizes()
        print(f"Train : {tr} | Validation : {va} | Test : {te}")
        assert tr + va + te == split.total()
        all_present = all(
            len(np.unique(arr)) > 1
            for arr in (split.y_train, split.y_val, split.y_test)
        )
        print(f"Répartition des classes conservée dans chaque jeu : {'oui' if all_present else 'NON'}")


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True)
    splitter = DataSplitter()
    split = splitter.split(X, y)
    splitter.report(split)
