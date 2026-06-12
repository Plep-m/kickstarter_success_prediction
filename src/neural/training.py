from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpochState:
    """Weight snapshot and MSE loss at one training epoch."""

    epoch: int
    weight: float
    loss: float

    def __str__(self) -> str:
        return f"epoch {self.epoch:>3d} | w={self.weight:.4f} | loss={self.loss:.6f}"


class GradientDescentLoop:
    """Illustrates the forward→loss→backprop→update cycle for a single-weight model.

    Fits y ≈ w·X using MSE loss and plain gradient descent.
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 10) -> None:
        self._lr = learning_rate
        self._epochs = epochs
        self._history: list[EpochState] = []

    def fit(self, X: np.ndarray, y: np.ndarray, w_init: float = 0.0) -> "GradientDescentLoop":
        w = w_init
        for epoch in range(self._epochs):
            y_pred = w * X
            loss = float(np.mean((y_pred - y) ** 2))
            grad = float(np.mean(2.0 * (y_pred - y) * X))
            w -= self._lr * grad
            self._history.append(EpochState(epoch=epoch, weight=w, loss=loss))
        return self

    @property
    def history(self) -> list[EpochState]:
        if not self._history:
            raise RuntimeError("Call fit() first.")
        return self._history

    @property
    def final_weight(self) -> float:
        return self.history[-1].weight

    def report(self) -> None:
        for state in self._history:
            print(state)
