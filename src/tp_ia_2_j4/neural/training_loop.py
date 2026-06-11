from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochState:
    """Weight and MSE loss captured at one training epoch."""

    epoch: int
    weight: float
    loss: float

    def __str__(self) -> str:
        return f"epoch {self.epoch} | w = {self.weight:.3f} | loss = {self.loss:.3f}"


class GradientDescentLoop:
    """Manual single-weight SGD to illustrate the forward/loss/backprop/update cycle."""

    def __init__(self, learning_rate: float = 0.01, epochs: int = 5) -> None:
        self._lr = learning_rate
        self._epochs = epochs
        self._history: list[EpochState] = []

    def fit(self, X: np.ndarray, y: np.ndarray, w_init: float = 0.0) -> "GradientDescentLoop":
        """Learn scalar w in y ≈ w·X using MSE and gradient descent."""
        w = w_init
        for epoch in range(self._epochs):
            y_pred = w * X
            loss = float(np.mean((y_pred - y) ** 2))
            grad = float(np.mean(2.0 * (y_pred - y) * X))
            w = w - self._lr * grad
            self._history.append(EpochState(epoch=epoch, weight=w, loss=loss))
        return self

    @property
    def history(self) -> list[EpochState]:
        if not self._history:
            raise RuntimeError("Call fit() first.")
        return self._history

    def report(self) -> None:
        for state in self._history:
            print(state)


if __name__ == "__main__":
    X = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    GradientDescentLoop(learning_rate=0.01, epochs=5).fit(X, y).report()
