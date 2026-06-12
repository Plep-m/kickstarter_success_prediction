from __future__ import annotations

import numpy as np


class Activations:
    """Collection of activation functions used in a multi-layer perceptron."""

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, np.asarray(z, dtype=float))

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(z, dtype=float))

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        arr = np.asarray(z, dtype=float)
        e = np.exp(arr - arr.max())
        return e / e.sum()

    @staticmethod
    def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        arr = np.asarray(z, dtype=float)
        return np.where(arr > 0, arr, alpha * arr)
