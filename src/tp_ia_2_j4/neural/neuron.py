from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class Neuron:
    """Single artificial neuron: weighted sum + bias (no activation)."""

    weights: np.ndarray
    bias: float

    def forward(self, x: np.ndarray) -> float:
        """Compute z = x·w + b."""
        return float(np.dot(x, self.weights) + self.bias)


class Activations:
    """Static collection of the four activation functions used in a PMC."""

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Squash z → (0, 1). Output layer for binary classification."""
        return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """max(0, z). Default activation for hidden layers."""
        return np.maximum(0.0, np.asarray(z, dtype=float))

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent → (-1, 1), zero-centred gradients."""
        return np.tanh(np.asarray(z, dtype=float))

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Normalised exponentials summing to 1. Output for multiclass."""
        arr = np.asarray(z, dtype=float)
        e = np.exp(arr - arr.max())
        return e / e.sum()


if __name__ == "__main__":
    n = Neuron(weights=np.array([0.8, 0.4]), bias=0.1)
    x = np.array([0.5, -0.3])
    print(f"z = {n.forward(x):.4f}")

    probe = np.array([-20.0, -5.0, 0.0, 5.0, 20.0])
    print("sigmoid:", Activations.sigmoid(probe).round(4))
    print("relu:   ", Activations.relu(probe))
    print("softmax:", Activations.softmax(np.array([2.0, 1.0, 0.1])).round(4))
