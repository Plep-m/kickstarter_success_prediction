from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Neuron:
    """Single artificial neuron: weighted sum + bias (pre-activation)."""

    weights: np.ndarray
    bias: float

    def forward(self, x: np.ndarray) -> float:
        """Compute z = x·w + b."""
        return float(np.dot(x, self.weights) + self.bias)

    @classmethod
    def random(cls, n_inputs: int, seed: int | None = None) -> "Neuron":
        rng = np.random.default_rng(seed)
        return cls(weights=rng.standard_normal(n_inputs), bias=0.0)
